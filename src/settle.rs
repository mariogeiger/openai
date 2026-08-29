//! Accumulating a stream, and the boundary where it becomes a finished
//! response.
//!
//! # Why "settled" is a type and not a flag
//!
//! A streamed response is only trustworthy once a terminal event arrives. A
//! connection can drop mid-answer, and the text collected so far looks exactly
//! like a complete answer — same field, same characters, no error anywhere.
//! Anything that reports "finished" with a boolean invites the caller to read
//! the half-finished case as the finished one.
//!
//! So the two states are two types. [`Settling`] accumulates and has no method
//! that yields a response. [`Settled`] is a finished response and has no method
//! that accepts more events. The only bridge is [`Settling::settle`], which
//! consumes the accumulator and returns a `Result`: a truncated stream yields
//! [`SettleError::Truncated`], and there is no other way to obtain a `Settled`.
//! A caller who forgets to check gets a compile error, not a plausible answer.
//!
//! # Cost of accumulation
//!
//! Text arrives as many small deltas. Each is appended to a `String` for its
//! part, which amortizes to linear in the total bytes — `String::push_str`
//! doubles capacity, so *n* deltas cost O(*n*) copying rather than the O(*n*²)
//! of rebuilding a joined string per delta. Parts are keyed by
//! `(output_index, kind, part_index)` in a `BTreeMap`, so a repeated part
//! overwrites instead of duplicating, and iteration is already in document
//! order.

use std::collections::BTreeMap;

use crate::stream::{
    CalledFunction, FrameError, OutputItem, PartKey, PartKind, ReasoningItem, ResponseError, ResponseSnapshot,
    StreamEvent, joined, joined_at,
};
use crate::usage::Usage;
use crate::values::IncompleteReason;

// ── Settling ─────────────────────────────────────────────────────────────────

/// Why a stream did not produce a finished response.
#[derive(Debug)]
pub enum SettleError {
    /// The stream ended without a terminal event.
    ///
    /// The connection dropped, the reader stopped early, or the server hung up.
    /// Whatever text had accumulated is *not* returned: handing back a partial
    /// answer typed as a whole one is the mistake this module exists to
    /// prevent. The counts are here so the failure can be logged usefully.
    Truncated {
        /// How many events had been consumed.
        events: usize,
        /// How many characters of answer text had accumulated.
        text_len: usize,
    },
    /// A frame could not be decoded. Carries the frame error unchanged.
    Frame(FrameError),
}

impl std::fmt::Display for SettleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettleError::Truncated { events, text_len } => write!(
                f,
                "stream ended without a terminal event after {events} event(s) and {text_len} character(s) of text"
            ),
            SettleError::Frame(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SettleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SettleError::Frame(error) => Some(error),
            SettleError::Truncated { .. } => None,
        }
    }
}

impl From<FrameError> for SettleError {
    fn from(error: FrameError) -> Self {
        SettleError::Frame(error)
    }
}

/// A stream still being read.
///
/// Deliberately offers no way to read a response out of itself. Feed it events
/// with [`Self::consume`]; when the stream is exhausted call [`Self::settle`],
/// which either produces a [`Settled`] or fails.
///
/// An accumulator is not a response, and the compiler enforces it rather than
/// the documentation asking politely. There is no `text` on a `Settling`:
///
/// ```compile_fail
/// let settling = openai::settle::Settling::new();
/// let _: String = settling.text;
/// ```
///
/// And a `Settled` cannot be assembled by hand to bypass the check, because its
/// only constructor is [`Self::settle`]:
///
/// ```compile_fail
/// let _ = openai::settle::Settled {
///     outcome: openai::settle::Outcome::Completed,
///     id: None,
///     text: "invented".to_owned(),
///     reasoning_summary: String::new(),
///     items: Vec::new(),
///     usage: None,
///     events: 0,
/// };
/// ```
#[derive(Debug, Default)]
pub struct Settling {
    text_parts: BTreeMap<PartKey, String>,
    items: BTreeMap<u32, OutputItem>,
    terminal: Option<Terminal>,
    events: usize,
}

/// The terminal event, kept only so [`Settling::settle`] can turn it into an
/// [`Outcome`].
#[derive(Debug)]
enum Terminal {
    Completed(ResponseSnapshot),
    Failed(ResponseSnapshot),
    Incomplete(ResponseSnapshot),
    Error(ResponseError),
}

impl Settling {
    /// A stream with nothing in it yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// One raw `data:` payload, decoded and folded in.
    ///
    /// The convenience path: equivalent to [`StreamEvent::decode`] followed by
    /// [`Self::consume`], returning the decoded event so a caller can also
    /// forward deltas to a display as they arrive.
    pub fn consume_payload(&mut self, payload: &str) -> Result<StreamEvent, SettleError> {
        let event = StreamEvent::decode(payload)?;
        self.consume(event.clone());
        Ok(event)
    }

    /// One decoded event, folded in.
    ///
    /// Infallible on purpose. Every failure a frame can have already happened
    /// during decoding, and an unmodeled event is not a failure — so nothing
    /// here can go wrong, and the signature says so.
    ///
    /// A second terminal event is ignored: the first one is what ended the
    /// response, and a duplicate must not overwrite a success with a later
    /// spurious failure or the reverse.
    pub fn consume(&mut self, event: StreamEvent) {
        self.events += 1;
        match event {
            StreamEvent::OutputTextDelta { output_index, content_index, delta } => {
                self.append(PartKind::OutputText, output_index, content_index, &delta);
            }
            StreamEvent::ReasoningSummaryTextDelta { output_index, summary_index, delta } => {
                self.append(PartKind::ReasoningSummary, output_index, summary_index, &delta);
            }
            // An announcement records the item's existence and position; the
            // `done` form is authoritative and replaces it.
            StreamEvent::OutputItemAdded { output_index, item }
            | StreamEvent::OutputItemDone { output_index, item } => {
                self.items.insert(output_index, item);
            }
            StreamEvent::Completed(snapshot) => self.finish(Terminal::Completed(snapshot)),
            StreamEvent::Failed(snapshot) => self.finish(Terminal::Failed(snapshot)),
            StreamEvent::Incomplete(snapshot) => self.finish(Terminal::Incomplete(snapshot)),
            StreamEvent::Error(error) => self.finish(Terminal::Error(error)),
            StreamEvent::Unmodeled { .. } => {}
        }
    }

    fn append(&mut self, kind: PartKind, output_index: u32, part_index: u32, delta: &str) {
        self.text_parts.entry(PartKey { output_index, kind, part_index }).or_default().push_str(delta);
    }

    fn finish(&mut self, terminal: Terminal) {
        if self.terminal.is_none() {
            self.terminal = Some(terminal);
        }
    }

    /// The answer text so far, for a live display.
    ///
    /// Named for what it is. Reading it does not settle the stream and does not
    /// claim the answer is complete — that is what [`Settled`] is for.
    pub fn text_so_far(&self) -> String {
        joined(&self.text_parts, PartKind::OutputText)
    }

    /// The reasoning summary so far, for a live display.
    pub fn reasoning_summary_so_far(&self) -> String {
        joined(&self.text_parts, PartKind::ReasoningSummary)
    }

    /// How many events have been consumed.
    pub fn event_count(&self) -> usize {
        self.events
    }

    /// Whether a terminal event has arrived, so [`Self::settle`] will succeed.
    ///
    /// For deciding when to stop reading, not for deciding to trust the text: a
    /// `true` here still gives you nothing without calling `settle`.
    pub fn is_terminated(&self) -> bool {
        self.terminal.is_some()
    }

    /// The finished response, or a failure explaining why there is none.
    ///
    /// Consumes the accumulator, which is what makes the two states exclusive:
    /// after settling there is no `Settling` left to append to, and without
    /// settling there is no [`Settled`] at all.
    pub fn settle(self) -> Result<Settled, SettleError> {
        let text = joined(&self.text_parts, PartKind::OutputText);
        let terminal = self.terminal.ok_or(SettleError::Truncated { events: self.events, text_len: text.len() })?;
        let reasoning_summary = joined(&self.text_parts, PartKind::ReasoningSummary);

        // A message item is announced empty and its text arrives as deltas, so
        // the deltas at that item's index *are* its text. Filling it here is
        // what keeps `items` a faithful record of the answer in document order:
        // without it a message announced beside a function call reads as empty,
        // and a caller iterating items alone loses the prose the model said
        // before it called anything.
        let mut streamed = self.items;
        for (output_index, item) in &mut streamed {
            if let OutputItem::Message { text, .. } = item
                && text.is_empty()
            {
                *text = joined_at(&self.text_parts, PartKind::OutputText, *output_index);
            }
        }

        // A terminal snapshot's own `output` array is authoritative when it has
        // one: it is the server's final word. Streamed items fill in when the
        // snapshot omits it, which is the ordinary case.
        let mut items: Vec<OutputItem> = streamed.into_values().collect();
        let (outcome, usage, id) = match terminal {
            Terminal::Completed(snapshot) => (Outcome::Completed, snapshot.usage, take_output(snapshot, &mut items)),
            Terminal::Incomplete(snapshot) => {
                let outcome = Outcome::Incomplete { reason: snapshot.incomplete_reason };
                (outcome, snapshot.usage, take_output(snapshot, &mut items))
            }
            Terminal::Failed(snapshot) => {
                let outcome = Outcome::Failed {
                    error: snapshot.error.clone().unwrap_or(ResponseError {
                        code: None,
                        message: "the response failed without an error object".to_owned(),
                        param: None,
                    }),
                };
                (outcome, snapshot.usage, take_output(snapshot, &mut items))
            }
            Terminal::Error(error) => (Outcome::Errored { error }, None, None),
        };

        Ok(Settled { outcome, id, text, reasoning_summary, items, usage, events: self.events })
    }
}

/// Moves a snapshot's `output` into `items` when it has one, and yields the
/// response id.
fn take_output(mut snapshot: ResponseSnapshot, items: &mut Vec<OutputItem>) -> Option<String> {
    if !snapshot.output.is_empty() {
        *items = std::mem::take(&mut snapshot.output);
    }
    snapshot.id
}

// ── Settled ──────────────────────────────────────────────────────────────────

/// How a stream ended.
///
/// Not a status string: each ending carries exactly the data that ending has,
/// so there is no `error` field to read on a success and no reason to check for
/// on one either.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The model answered in full.
    Completed,
    /// Generation stopped short. Whatever was produced is still present in the
    /// [`Settled`], because a truncated answer is often still useful — the
    /// caller just has to know it is truncated, which this variant tells them.
    Incomplete {
        /// Why, when the API named a reason this crate knows.
        reason: Option<IncompleteReason>,
    },
    /// The response failed, as `response.failed`.
    Failed {
        /// What the API reported.
        error: ResponseError,
    },
    /// The stream delivered a bare `error` event. Distinct from
    /// [`Self::Failed`], which carries a whole failed response object.
    Errored {
        /// What the API reported.
        error: ResponseError,
    },
}

/// A stream that reached a terminal event.
///
/// Obtainable only from [`Settling::settle`], so holding one is proof the
/// stream finished. Every field is final.
///
/// `#[non_exhaustive]` is what makes "only from `settle`" true rather than
/// merely intended: it forbids the struct literal outside this crate, so no
/// caller can fabricate a finished response from an unfinished stream. Reading
/// every field still works.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Settled {
    /// How the stream ended, with that ending's own data.
    pub outcome: Outcome,
    /// The response's identifier, worth logging for OpenAI support.
    pub id: Option<String>,
    /// The answer text, every `output_text` delta in document order.
    pub text: String,
    /// The reasoning summary, when one was requested.
    pub reasoning_summary: String,
    /// The output items, in `output_index` order.
    pub items: Vec<OutputItem>,
    /// What the response cost. `None` when the API did not report it, which is
    /// the normal case for a bare `error` event.
    pub usage: Option<Usage>,
    /// How many events the stream delivered.
    pub events: usize,
}

impl Settled {
    /// Whether the model answered in full.
    pub fn is_completed(&self) -> bool {
        matches!(self.outcome, Outcome::Completed)
    }

    /// The error, on either failing outcome. `None` on the two that carry none.
    pub fn error(&self) -> Option<&ResponseError> {
        match &self.outcome {
            Outcome::Failed { error } | Outcome::Errored { error } => Some(error),
            Outcome::Completed | Outcome::Incomplete { .. } => None,
        }
    }

    /// The function calls the model made, in output order.
    ///
    /// The list a tool-running loop iterates. Arguments are still undecoded —
    /// see [`crate::stream::FunctionArguments::decode`], which fails per call
    /// rather than per response.
    pub fn function_calls(&self) -> impl Iterator<Item = &CalledFunction> {
        self.items.iter().filter_map(|item| match item {
            OutputItem::FunctionCall(call) => Some(call),
            _ => None,
        })
    }

    /// The reasoning items, in output order, for replay into the next request.
    pub fn reasoning_items(&self) -> impl Iterator<Item = &ReasoningItem> {
        self.items.iter().filter_map(|item| match item {
            OutputItem::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text_delta(output_index: u32, content_index: u32, delta: &str) -> StreamEvent {
        StreamEvent::OutputTextDelta { output_index, content_index, delta: delta.to_owned() }
    }

    fn usage_frame() -> serde_json::Value {
        json!({
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 100},
            "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 30},
            "total_tokens": 1240
        })
    }

    #[test]
    fn deltas_accumulate_and_a_terminal_event_settles_them() {
        let mut settling = Settling::new();
        settling.consume(text_delta(0, 0, "Hel"));
        settling.consume(text_delta(0, 0, "lo, "));
        settling.consume(text_delta(0, 0, "world"));
        assert_eq!(settling.text_so_far(), "Hello, world");
        assert!(!settling.is_terminated());

        settling.consume(StreamEvent::Completed(ResponseSnapshot {
            id: Some("resp_1".to_owned()),
            usage: Some(serde_json::from_value(usage_frame()).unwrap()),
            error: None,
            incomplete_reason: None,
            output: Vec::new(),
        }));
        assert!(settling.is_terminated());

        let settled = settling.settle().unwrap();
        assert!(settled.is_completed());
        assert_eq!(settled.text, "Hello, world");
        assert_eq!(settled.id.as_deref(), Some("resp_1"));
        assert_eq!(settled.usage.unwrap().input_tokens_details.cached_tokens, 1_000);
        assert_eq!(settled.error(), None);
        assert_eq!(settled.events, 4);
    }

    /// The point of the whole module: without a terminal event there is no
    /// `Settled`, and the text that did arrive is not handed over.
    #[test]
    fn a_truncated_stream_does_not_settle() {
        let mut settling = Settling::new();
        settling.consume(text_delta(0, 0, "half an ans"));
        assert_eq!(settling.text_so_far(), "half an ans", "readable as in-progress");
        assert!(!settling.is_terminated());

        let error = settling.settle().unwrap_err();
        let SettleError::Truncated { events, text_len } = error else { panic!("expected truncation") };
        assert_eq!(events, 1);
        assert_eq!(text_len, 11);
    }

    /// An empty stream is the degenerate truncation, and behaves the same.
    #[test]
    fn an_empty_stream_does_not_settle() {
        let error = Settling::new().settle().unwrap_err();
        assert!(matches!(error, SettleError::Truncated { events: 0, text_len: 0 }));
    }

    /// A full happy path from raw frames: reasoning, text, a function call, and
    /// completion with usage.
    #[test]
    fn a_captured_happy_path_settles_into_text_and_one_function_call() {
        let frames = [
            r#"{"type":"response.created","sequence_number":0,"response":{"id":"resp_9","status":"in_progress"}}"#
                .to_owned(),
            r#"{"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_9"}}"#.to_owned(),
            r#"{"type":"response.output_item.added","sequence_number":2,"output_index":0,
                "item":{"type":"reasoning","id":"rs_1","summary":[]}}"#
                .to_owned(),
            r#"{"type":"response.reasoning_summary_part.added","sequence_number":3,"output_index":0,
                "summary_index":0,"item_id":"rs_1","part":{"type":"summary_text","text":""}}"#
                .to_owned(),
            r#"{"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_1",
                "output_index":0,"summary_index":0,"delta":"Reading the file"}"#
                .to_owned(),
            r#"{"type":"response.output_item.done","sequence_number":5,"output_index":0,
                "item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text",
                "text":"Reading the file"}],"encrypted_content":"gAAAAA","status":"completed"}}"#
                .to_owned(),
            r#"{"type":"response.output_item.added","sequence_number":6,"output_index":1,
                "item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress",
                "content":[]}}"#
                .to_owned(),
            r#"{"type":"response.content_part.added","sequence_number":7,"item_id":"msg_1",
                "output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}"#
                .to_owned(),
            r#"{"type":"response.output_text.delta","sequence_number":8,"item_id":"msg_1",
                "output_index":1,"content_index":0,"delta":"Reading "}"#
                .to_owned(),
            r#"{"type":"response.output_text.delta","sequence_number":9,"item_id":"msg_1",
                "output_index":1,"content_index":0,"delta":"it now."}"#
                .to_owned(),
            r#"{"type":"response.output_text.done","sequence_number":10,"item_id":"msg_1",
                "output_index":1,"content_index":0,"text":"Reading it now."}"#
                .to_owned(),
            r#"{"type":"response.output_item.done","sequence_number":11,"output_index":1,
                "item":{"type":"message","id":"msg_1","role":"assistant","status":"completed",
                "phase":"commentary","content":[{"type":"output_text","text":"Reading it now.",
                "annotations":[]}]}}"#
                .to_owned(),
            r#"{"type":"response.output_item.added","sequence_number":12,"output_index":2,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_x","name":"read_file",
                "arguments":"","status":"in_progress"}}"#
                .to_owned(),
            r#"{"type":"response.function_call_arguments.delta","sequence_number":13,"item_id":"fc_1",
                "output_index":2,"delta":"{\"path\":"}"#
                .to_owned(),
            r#"{"type":"response.function_call_arguments.done","sequence_number":14,"item_id":"fc_1",
                "output_index":2,"arguments":"{\"path\":\"src/lib.rs\"}"}"#
                .to_owned(),
            r#"{"type":"response.output_item.done","sequence_number":15,"output_index":2,
                "item":{"type":"function_call","id":"fc_1","call_id":"call_x","name":"read_file",
                "arguments":"{\"path\":\"src/lib.rs\"}","status":"completed"}}"#
                .to_owned(),
            format!(
                r#"{{"type":"response.completed","sequence_number":16,"response":{{"id":"resp_9",
                    "status":"completed","usage":{}}}}}"#,
                usage_frame()
            ),
        ];

        let mut settling = Settling::new();
        for frame in &frames {
            settling.consume_payload(frame).unwrap();
        }
        let settled = settling.settle().unwrap();

        assert_eq!(settled.outcome, Outcome::Completed);
        assert_eq!(settled.text, "Reading it now.");
        assert_eq!(settled.reasoning_summary, "Reading the file");
        assert_eq!(settled.items.len(), 3, "reasoning, message, function call");

        let calls: Vec<_> = settled.function_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_x");
        assert_eq!(calls[0].arguments.decode().unwrap(), json!({"path": "src/lib.rs"}));

        let reasoning: Vec<_> = settled.reasoning_items().collect();
        assert_eq!(reasoning.len(), 1);
        assert_eq!(reasoning[0].replayable().unwrap().encrypted_content, "gAAAAA");

        let usage = settled.usage.unwrap();
        assert_eq!(usage.input_tokens_details.cached_tokens, 1_000);
        assert_eq!(usage.uncached_input_tokens(), 100);
    }

    #[test]
    fn a_failed_stream_settles_as_failed_with_its_error() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"parti"}"#,
            )
            .unwrap();
        settling
            .consume_payload(
                r#"{"type":"response.failed","response":{"id":"resp_e","status":"failed",
                    "error":{"code":"server_error","message":"upstream fell over"}}}"#,
            )
            .unwrap();

        let settled = settling.settle().unwrap();
        assert!(!settled.is_completed());
        let error = settled.error().unwrap();
        assert_eq!(error.code.as_deref(), Some("server_error"));
        assert_eq!(settled.text, "parti", "what arrived is kept; the outcome says it failed");
    }

    /// A `failed` event with no error object still settles, with a message that
    /// says the object was missing rather than pretending there was no error.
    #[test]
    fn a_failure_without_an_error_object_still_names_itself_a_failure() {
        let mut settling = Settling::new();
        settling.consume_payload(r#"{"type":"response.failed","response":{"id":"r"}}"#).unwrap();
        let settled = settling.settle().unwrap();
        assert!(!settled.is_completed());
        assert_eq!(settled.error().unwrap().code, None);
        assert!(settled.error().unwrap().message.contains("without an error object"));
    }

    #[test]
    fn an_incomplete_stream_settles_with_its_reason_and_its_partial_text() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,
                    "delta":"as far as I got"}"#,
            )
            .unwrap();
        settling
            .consume_payload(&format!(
                r#"{{"type":"response.incomplete","response":{{"id":"resp_i","status":"incomplete",
                    "incomplete_details":{{"reason":"max_output_tokens"}},"usage":{}}}}}"#,
                usage_frame()
            ))
            .unwrap();

        let settled = settling.settle().unwrap();
        assert_eq!(settled.outcome, Outcome::Incomplete { reason: Some(IncompleteReason::MaxOutputTokens) });
        assert_eq!(settled.text, "as far as I got");
        assert!(settled.usage.is_some(), "an incomplete response still cost money");
        assert_eq!(settled.error(), None, "incomplete is not an error");
    }

    /// An event type this crate has never seen passes through the middle of a
    /// stream without disturbing it. This is the server-side-release case.
    #[test]
    fn an_unknown_event_interleaved_changes_nothing() {
        let mut settling = Settling::new();
        settling
            .consume_payload(r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"be"}"#)
            .unwrap();
        let unknown =
            settling.consume_payload(r#"{"type":"response.telepathy.delta","output_index":0,"delta":"???"}"#).unwrap();
        assert_eq!(unknown, StreamEvent::Unmodeled { kind: "response.telepathy.delta".to_owned() });
        settling
            .consume_payload(
                r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"fore"}"#,
            )
            .unwrap();
        settling.consume_payload(r#"{"type":"response.completed","response":{"id":"r"}}"#).unwrap();

        let settled = settling.settle().unwrap();
        assert!(settled.is_completed());
        assert_eq!(settled.text, "before", "the unknown event contributed nothing and broke nothing");
        assert_eq!(settled.events, 4, "it was still counted");
    }

    /// A truncated stream that had already produced a whole function call still
    /// does not settle. Partial structure is no substitute for a terminal event.
    #[test]
    fn a_truncated_stream_with_finished_items_still_does_not_settle() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type":"response.output_item.done","output_index":0,
                    "item":{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}}"#,
            )
            .unwrap();
        assert!(matches!(settling.settle(), Err(SettleError::Truncated { events: 1, text_len: 0 })));
    }

    /// The bare `error` event ends the stream as its own outcome, not as
    /// `Failed`, because it carries no response object.
    #[test]
    fn a_bare_error_event_settles_as_errored() {
        let mut settling = Settling::new();
        settling
            .consume_payload(r#"{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":null}"#)
            .unwrap();
        let settled = settling.settle().unwrap();
        assert!(matches!(settled.outcome, Outcome::Errored { .. }));
        assert_eq!(settled.error().unwrap().code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(settled.usage, None);
    }

    /// The first terminal event wins, so a duplicate cannot rewrite history.
    #[test]
    fn a_second_terminal_event_is_ignored() {
        let mut settling = Settling::new();
        settling.consume_payload(r#"{"type":"response.completed","response":{"id":"first"}}"#).unwrap();
        settling.consume_payload(r#"{"type":"response.failed","response":{"error":{"message":"too late"}}}"#).unwrap();
        let settled = settling.settle().unwrap();
        assert!(settled.is_completed());
        assert_eq!(settled.id.as_deref(), Some("first"));
    }

    /// The `done` form of an item replaces the announcement at the same index,
    /// so arguments are the finished ones and the item is not duplicated.
    #[test]
    fn a_finished_item_replaces_its_announcement() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type":"response.output_item.added","output_index":0,
                    "item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}"#,
            )
            .unwrap();
        settling
            .consume_payload(
                r#"{"type":"response.output_item.done","output_index":0,
                    "item":{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"k\":1}"}}"#,
            )
            .unwrap();
        settling.consume_payload(r#"{"type":"response.completed","response":{}}"#).unwrap();

        let settled = settling.settle().unwrap();
        assert_eq!(settled.items.len(), 1, "one item, not two");
        assert_eq!(settled.function_calls().next().unwrap().arguments.decode().unwrap(), json!({"k": 1}));
    }

    /// Items settle in `output_index` order however the frames arrive.
    #[test]
    fn items_settle_in_output_order() {
        let mut settling = Settling::new();
        for (index, name) in [(2u32, "third"), (0, "first"), (1, "second")] {
            settling
                .consume_payload(&format!(
                    r#"{{"type":"response.output_item.done","output_index":{index},
                        "item":{{"type":"function_call","call_id":"c{index}","name":"{name}",
                        "arguments":"{{}}"}}}}"#
                ))
                .unwrap();
        }
        settling.consume_payload(r#"{"type":"response.completed","response":{}}"#).unwrap();
        let settled = settling.settle().unwrap();
        let names: Vec<&str> = settled.function_calls().map(|call| call.name.as_str()).collect();
        assert_eq!(names, ["first", "second", "third"]);
    }

    /// Text from several items and parts joins in document order, not arrival
    /// order, and a repeated part does not double.
    #[test]
    fn text_joins_in_document_order() {
        let mut settling = Settling::new();
        settling.consume(text_delta(1, 0, "second"));
        settling.consume(text_delta(0, 1, "part-two "));
        settling.consume(text_delta(0, 0, "part-one "));
        settling.consume(StreamEvent::Completed(ResponseSnapshot {
            id: None,
            usage: None,
            error: None,
            incomplete_reason: None,
            output: Vec::new(),
        }));
        assert_eq!(settling.settle().unwrap().text, "part-one part-two second");
    }

    /// A broken frame surfaces as a frame error through the payload path, and
    /// does not corrupt what has accumulated.
    #[test]
    fn a_broken_frame_does_not_disturb_the_accumulator() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"kept"}"#,
            )
            .unwrap();
        assert!(matches!(settling.consume_payload("{ not json"), Err(SettleError::Frame(_))));
        assert_eq!(settling.text_so_far(), "kept");
        assert_eq!(settling.event_count(), 1, "a frame that never decoded was never an event");
    }

    /// A terminal snapshot that repeats the whole `output` array is the
    /// server's final word and supersedes the streamed items.
    #[test]
    fn a_terminal_output_array_supersedes_streamed_items() {
        let mut settling = Settling::new();
        settling
            .consume_payload(
                r#"{"type":"response.output_item.added","output_index":0,
                    "item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}"#,
            )
            .unwrap();
        settling
            .consume_payload(
                r#"{"type":"response.completed","response":{"id":"r","output":[
                    {"type":"function_call","call_id":"c1","name":"f","arguments":"{\"final\":true}"}]}}"#,
            )
            .unwrap();
        let settled = settling.settle().unwrap();
        assert_eq!(settled.items.len(), 1);
        assert_eq!(settled.function_calls().next().unwrap().arguments.decode().unwrap(), json!({"final": true}));
    }
    /// A message item is announced empty and filled by deltas, so the deltas at
    /// its index are its text. Without that, prose said beside a function call
    /// reads as an empty message and a caller iterating items loses it.
    #[test]
    fn a_streamed_message_item_carries_the_text_of_its_own_deltas() {
        let mut settling = Settling::new();
        for frame in [
            r#"{"type":"response.output_item.added","output_index":0,
                "item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"I will read it."}"#,
            r#"{"type":"response.output_item.added","output_index":1,
                "item":{"type":"function_call","call_id":"c1","name":"read","arguments":""}}"#,
            r#"{"type":"response.output_item.done","output_index":1,
                "item":{"type":"function_call","call_id":"c1","name":"read","arguments":"{\"path\":\"a\"}"}}"#,
            r#"{"type":"response.completed","response":{"id":"resp_1"}}"#,
        ] {
            settling.consume_payload(frame).unwrap();
        }
        let settled = settling.settle().unwrap();

        assert_eq!(settled.text, "I will read it.");
        assert_eq!(settled.items.len(), 2);
        let OutputItem::Message { text, .. } = &settled.items[0] else {
            panic!("the first item is the message");
        };
        assert_eq!(text, "I will read it.", "the item carries the text its own deltas built");
        assert_eq!(settled.function_calls().count(), 1);
    }
}
