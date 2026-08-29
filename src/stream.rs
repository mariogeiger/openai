//! One streamed frame becomes one typed event.
//!
//! A streaming response is a sequence of Server-Sent Events, each carrying a
//! JSON object whose `type` field names it. This module turns one such payload
//! into a [`StreamEvent`]; [`crate::settle`] turns a sequence of them into a
//! finished response.
//!
//! # Why an unknown event is not an error
//!
//! Adding a new streaming event type is, by OpenAI's own compatibility promise,
//! a *backwards-compatible* change. A decoder that returns an error for an
//! event it has never seen is therefore a decoder that a routine server-side
//! release will break. So the unrecognized case is a variant —
//! [`StreamEvent::Unmodeled`] — and never a [`FrameError`]. The type states the
//! policy: you must match a variant that means "ignore me", and you cannot get
//! an error out of a well-formed frame you simply do not know.
//!
//! The same variant covers events this crate models on purpose and events it
//! does not model on purpose. Both are "well-formed, nothing to do here", which
//! is why one variant is enough.
//!
//! # What *is* an error
//!
//! Bytes that are not JSON, a payload that is not an object, a missing `type`,
//! a field whose type contradicts the schema, and a `usage` object that will
//! not deserialize. Those are broken frames, not new ones.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::content::ReplayedReasoning;
use crate::usage::Usage;
use crate::values::{AssistantPhase, IncompleteReason};

// ── Frame errors ─────────────────────────────────────────────────────────────

/// Why one frame could not be decoded.
///
/// Every variant describes a frame that contradicts the documented schema. An
/// event type this crate does not model is deliberately absent from this list —
/// see [`StreamEvent::Unmodeled`].
#[derive(Debug)]
pub enum FrameError {
    /// The payload was not JSON at all.
    NotJson(serde_json::Error),
    /// The payload parsed, but was not a JSON object.
    NotAnObject,
    /// A field the event cannot be interpreted without was absent.
    MissingField {
        /// Dotted path of the absent field, from the payload root.
        field: &'static str,
    },
    /// A field was present with the wrong JSON type.
    WrongType {
        /// Dotted path of the offending field, from the payload root.
        field: &'static str,
        /// What the schema says it should have been.
        expected: &'static str,
    },
    /// A `usage` object was present but would not deserialize into [`Usage`].
    ///
    /// Reported rather than dropped: cache accounting is this crate's reason to
    /// exist, so usage that silently read as absent would hide exactly the
    /// measurement the caller is here for.
    UndecodableUsage(serde_json::Error),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::NotJson(error) => write!(f, "streamed frame is not JSON: {error}"),
            FrameError::NotAnObject => write!(f, "streamed frame is not a JSON object"),
            FrameError::MissingField { field } => write!(f, "streamed frame has no `{field}`"),
            FrameError::WrongType { field, expected } => {
                write!(f, "streamed frame field `{field}` is not {expected}")
            }
            FrameError::UndecodableUsage(error) => write!(f, "streamed `usage` object is unusable: {error}"),
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FrameError::NotJson(error) | FrameError::UndecodableUsage(error) => Some(error),
            FrameError::NotAnObject | FrameError::MissingField { .. } | FrameError::WrongType { .. } => None,
        }
    }
}

// ── Server-Sent Events framing ───────────────────────────────────────────────

/// The value of a `data:` field, for one line of an SSE body.
///
/// `None` for everything else a stream contains — the `event:` line that names
/// the type redundantly, comment lines, and the blank line that ends a frame —
/// so a caller can pass every line through and act on what comes back.
///
/// Per the SSE grammar one optional space after the colon is part of the
/// framing, not the data, and is removed. OpenAI sends each event as a single
/// `data:` line; a payload split across several `data:` lines would have to be
/// rejoined with newlines by the caller before decoding.
pub fn data_payload(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("data:")?;
    Some(rest.strip_prefix(' ').unwrap_or(rest))
}

// ── Function call arguments ──────────────────────────────────────────────────

/// The `arguments` of a function call, exactly as the model emitted them.
///
/// A JSON *string* on the wire, and a string here, because the bytes are what
/// the model sees when the call is replayed as input — re-serializing a parsed
/// value could reorder keys or change spacing and cost the prompt cache. See
/// [`crate::content::FunctionCall::arguments`], which holds the same bytes for
/// the same reason.
///
/// [`Self::decode`] is therefore a separate step, and one that can fail without
/// destroying the call it belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FunctionArguments(String);

impl FunctionArguments {
    /// The arguments as they arrived.
    pub fn from_wire(arguments: impl Into<String>) -> Self {
        Self(arguments.into())
    }

    /// The exact bytes, for replay.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The arguments as a JSON value.
    ///
    /// Three outcomes, and the type keeps them apart:
    ///
    /// * Well-formed JSON decodes to it.
    /// * Empty or blank arguments decode to the empty object. A function taking
    ///   no arguments is streamed with `arguments` absent on
    ///   `response.output_item.added` and, observed in practice, as `""` on
    ///   the corresponding `done`. Absent arguments mean the empty argument
    ///   set, so that is what they decode to.
    /// * Anything else is `Err`, carrying the byte offset from `serde_json`.
    ///   The malformed case cannot be confused with the empty case, and it
    ///   never fails the surrounding stream: the call, its `call_id`, its name,
    ///   and these raw bytes all survive, so a caller can answer the model with
    ///   a tool error instead of dropping the turn.
    pub fn decode(&self) -> Result<Value, serde_json::Error> {
        if self.0.trim().is_empty() {
            return Ok(Value::Object(serde_json::Map::new()));
        }
        serde_json::from_str(&self.0)
    }
}

// ── Output items ─────────────────────────────────────────────────────────────

/// A function call the model made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalledFunction {
    /// The item's own identifier, absent on providers that omit it.
    pub id: Option<String>,
    /// The identifier the matching `function_call_output` must repeat.
    pub call_id: String,
    /// Which function to run.
    pub name: String,
    /// The arguments, undecoded.
    pub arguments: FunctionArguments,
}

/// A reasoning item from the response.
///
/// Reasoning models work better when their own reasoning items are handed back
/// alongside function outputs, so this exists to be fed into the next turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningItem {
    /// The item's identifier.
    pub id: String,
    /// The opaque payload that makes the item replayable, present only when the
    /// request asked for `reasoning.encrypted_content` to be included.
    pub encrypted_content: Option<String>,
}

impl ReasoningItem {
    /// The item in the form the next request needs, or `None` when it cannot be
    /// replayed at all.
    ///
    /// Without `encrypted_content` there is nothing to send in stateless mode,
    /// and this says so rather than handing back an item the API would reject.
    pub fn replayable(&self) -> Option<ReplayedReasoning> {
        self.encrypted_content
            .as_ref()
            .map(|content| ReplayedReasoning { id: self.id.clone(), encrypted_content: content.clone() })
    }
}

/// One entry of the response's `output` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputItem {
    /// An assistant message.
    Message {
        /// The item's identifier, absent on providers that omit it.
        id: Option<String>,
        /// Interim commentary or the final answer. OpenAI asks that this be
        /// preserved and resent on replay.
        phase: Option<AssistantPhase>,
        /// Its `output_text` blocks, concatenated. Empty while the item is
        /// still streaming, since the text arrives as deltas.
        text: String,
    },
    /// A function call.
    FunctionCall(CalledFunction),
    /// Reasoning.
    Reasoning(ReasoningItem),
    /// An item kind this crate does not model, such as a hosted tool call.
    ///
    /// Present for the same reason as [`StreamEvent::Unmodeled`]: OpenAI adds
    /// output item kinds, and an unrecognized one is not a broken frame.
    Unmodeled {
        /// The item's `type`, for logging.
        kind: String,
    },
}

// ── The response object a terminal event carries ─────────────────────────────

/// An error the API reported for a response.
///
/// `code` is a plain string, not an enum, because the documented list is not
/// closed: codes outside it are observed in practice, and a variant per code
/// would turn a new one into a decode failure. For the separate vocabulary of
/// HTTP error-body types see [`crate::values::ErrorType`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseError {
    /// The error code, absent when the API did not give one.
    pub code: Option<String>,
    /// The human-readable message.
    pub message: String,
    /// Which request parameter was at fault, on the events that report it.
    pub param: Option<String>,
}

/// The `response` object delivered by a terminal event.
///
/// Only the fields a streaming consumer needs are decoded. The rest of the
/// response object — `instructions`, `tools`, `temperature` and the others — is
/// what the caller sent, so it is not read back.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseSnapshot {
    /// The response's identifier, worth recording for OpenAI support.
    pub id: Option<String>,
    /// What the response cost. `None` when absent or explicitly `null`.
    pub usage: Option<Usage>,
    /// Why it failed, on a failure.
    pub error: Option<ResponseError>,
    /// Why it stopped short, on an incomplete response. `None` when absent or
    /// when the API names a reason this crate does not know.
    pub incomplete_reason: Option<IncompleteReason>,
    /// The final `output` array, when the terminal event repeats it.
    pub output: Vec<OutputItem>,
}

// ── Events ───────────────────────────────────────────────────────────────────

/// One decoded streaming event.
///
/// The seven `response.*` variants are the ones a consumer of text, reasoning
/// summaries and function calls needs; [`Self::Error`] covers the bare `error`
/// event; [`Self::Unmodeled`] covers everything else, now and in future.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// `response.output_text.delta`: more answer text.
    OutputTextDelta {
        /// Which output item this text belongs to.
        output_index: u32,
        /// Which content block within that item.
        content_index: u32,
        /// The new text, to append.
        delta: String,
    },
    /// `response.reasoning_summary_text.delta`: more summarized reasoning.
    ReasoningSummaryTextDelta {
        /// Which output item this summary belongs to.
        output_index: u32,
        /// Which summary part within that item.
        summary_index: u32,
        /// The new text, to append.
        delta: String,
    },
    /// `response.output_item.added`: an item has begun.
    ///
    /// Announcement only. A function call arrives here without its arguments,
    /// and a reasoning item's `encrypted_content` may be truncated — OpenAI
    /// documents that the replayable form is the one on `done`.
    OutputItemAdded {
        /// Its position in the `output` array.
        output_index: u32,
        /// The item as far as it exists.
        item: OutputItem,
    },
    /// `response.output_item.done`: an item is complete.
    ///
    /// The authoritative form: a function call's `arguments` and a reasoning
    /// item's `encrypted_content` are whole here.
    OutputItemDone {
        /// Its position in the `output` array.
        output_index: u32,
        /// The finished item.
        item: OutputItem,
    },
    /// `response.completed`: the model answered. Terminal.
    Completed(ResponseSnapshot),
    /// `response.failed`: the response failed. Terminal.
    Failed(ResponseSnapshot),
    /// `response.incomplete`: generation stopped short, usually on
    /// `max_output_tokens`. Terminal, and whatever was produced is still
    /// present — a truncated answer is often still useful.
    Incomplete(ResponseSnapshot),
    /// The bare `error` event, emitted when generation itself errors.
    ///
    /// Distinct from [`Self::Failed`], which delivers a whole response object
    /// whose status is `failed`. This one carries only the error.
    Error(ResponseError),
    /// A well-formed event this crate does not model.
    ///
    /// Never an error, by design: see the module documentation. Ignoring it is
    /// correct, and its `kind` is worth logging once.
    Unmodeled {
        /// The event's `type`.
        kind: String,
    },
}

impl StreamEvent {
    /// The event's wire `type`.
    ///
    /// The one accessor that works uniformly across variants, so logging and
    /// metrics need no match.
    pub fn kind(&self) -> &str {
        match self {
            StreamEvent::OutputTextDelta { .. } => "response.output_text.delta",
            StreamEvent::ReasoningSummaryTextDelta { .. } => "response.reasoning_summary_text.delta",
            StreamEvent::OutputItemAdded { .. } => "response.output_item.added",
            StreamEvent::OutputItemDone { .. } => "response.output_item.done",
            StreamEvent::Completed(_) => "response.completed",
            StreamEvent::Failed(_) => "response.failed",
            StreamEvent::Incomplete(_) => "response.incomplete",
            StreamEvent::Error(_) => "error",
            StreamEvent::Unmodeled { kind } => kind,
        }
    }

    /// Whether this event ends the stream.
    ///
    /// True for the three terminal `response.*` events and for `error`. A
    /// stream that never delivers one of them never settles — see
    /// [`crate::settle::Settling`].
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            StreamEvent::Completed(_) | StreamEvent::Failed(_) | StreamEvent::Incomplete(_) | StreamEvent::Error(_)
        )
    }

    /// One `data:` payload, decoded.
    pub fn decode(payload: &str) -> Result<Self, FrameError> {
        let value: Value = serde_json::from_str(payload).map_err(FrameError::NotJson)?;
        Self::from_json(&value)
    }

    /// The same, for a caller who already holds the parsed frame.
    pub fn from_json(frame: &Value) -> Result<Self, FrameError> {
        if !frame.is_object() {
            return Err(FrameError::NotAnObject);
        }
        let kind = require_str(frame, "type")?;
        Ok(match kind {
            "response.output_text.delta" => StreamEvent::OutputTextDelta {
                output_index: require_u32(frame, "output_index")?,
                content_index: require_u32(frame, "content_index")?,
                delta: require_str(frame, "delta")?.to_owned(),
            },
            "response.reasoning_summary_text.delta" => StreamEvent::ReasoningSummaryTextDelta {
                output_index: require_u32(frame, "output_index")?,
                summary_index: require_u32(frame, "summary_index")?,
                delta: require_str(frame, "delta")?.to_owned(),
            },
            "response.output_item.added" => StreamEvent::OutputItemAdded {
                output_index: require_u32(frame, "output_index")?,
                item: decode_output_item(require(frame, "item")?)?,
            },
            "response.output_item.done" => StreamEvent::OutputItemDone {
                output_index: require_u32(frame, "output_index")?,
                item: decode_output_item(require(frame, "item")?)?,
            },
            "response.completed" => StreamEvent::Completed(decode_snapshot(require(frame, "response")?)?),
            "response.failed" => StreamEvent::Failed(decode_snapshot(require(frame, "response")?)?),
            "response.incomplete" => StreamEvent::Incomplete(decode_snapshot(require(frame, "response")?)?),
            "error" => StreamEvent::Error(decode_error(frame)),
            other => StreamEvent::Unmodeled { kind: other.to_owned() },
        })
    }
}

// ── Decoding ─────────────────────────────────────────────────────────────────

fn require<'a>(object: &'a Value, field: &'static str) -> Result<&'a Value, FrameError> {
    object.get(field).ok_or(FrameError::MissingField { field })
}

fn require_str<'a>(object: &'a Value, field: &'static str) -> Result<&'a str, FrameError> {
    require(object, field)?.as_str().ok_or(FrameError::WrongType { field, expected: "a string" })
}

fn require_u32(object: &Value, field: &'static str) -> Result<u32, FrameError> {
    let wrong = FrameError::WrongType { field, expected: "a non-negative integer" };
    let number = require(object, field)?.as_u64().ok_or(wrong)?;
    u32::try_from(number).map_err(|_| FrameError::WrongType { field, expected: "a non-negative integer" })
}

fn optional_string(object: &Value, field: &str) -> Option<String> {
    object.get(field).and_then(Value::as_str).map(str::to_owned)
}

fn decode_output_item(item: &Value) -> Result<OutputItem, FrameError> {
    if !item.is_object() {
        return Err(FrameError::WrongType { field: "item", expected: "an object" });
    }
    Ok(match require_str(item, "type")? {
        "message" => OutputItem::Message {
            id: optional_string(item, "id"),
            phase: item.get("phase").and_then(Value::as_str).and_then(AssistantPhase::from_str),
            text: concatenated_output_text(item),
        },
        "function_call" => OutputItem::FunctionCall(CalledFunction {
            id: optional_string(item, "id"),
            call_id: require_str(item, "call_id")?.to_owned(),
            name: require_str(item, "name")?.to_owned(),
            arguments: FunctionArguments(optional_string(item, "arguments").unwrap_or_default()),
        }),
        "reasoning" => OutputItem::Reasoning(ReasoningItem {
            id: require_str(item, "id")?.to_owned(),
            encrypted_content: item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .filter(|content| !content.is_empty())
                .map(str::to_owned),
        }),
        other => OutputItem::Unmodeled { kind: other.to_owned() },
    })
}

/// The item's `output_text` blocks, joined. Refusal blocks are not text and are
/// not folded in: `response.refusal.delta` is outside this crate's coverage,
/// and a refusal silently concatenated into an answer would read as one.
fn concatenated_output_text(item: &Value) -> String {
    let Some(blocks) = item.get("content").and_then(Value::as_array) else {
        return String::new();
    };
    let texts = blocks
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str));
    let mut text = String::new();
    for piece in texts {
        text.push_str(piece);
    }
    text
}

fn decode_error(object: &Value) -> ResponseError {
    ResponseError {
        code: optional_string(object, "code"),
        message: optional_string(object, "message").unwrap_or_default(),
        param: optional_string(object, "param"),
    }
}

fn decode_snapshot(response: &Value) -> Result<ResponseSnapshot, FrameError> {
    if !response.is_object() {
        return Err(FrameError::WrongType { field: "response", expected: "an object" });
    }
    let usage = match response.get("usage") {
        None | Some(Value::Null) => None,
        Some(usage) => Some(serde_json::from_value(usage.clone()).map_err(FrameError::UndecodableUsage)?),
    };
    let output = match response.get("output") {
        Some(Value::Array(items)) => items.iter().map(decode_output_item).collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };
    Ok(ResponseSnapshot {
        id: optional_string(response, "id"),
        usage,
        error: response.get("error").filter(|error| error.is_object()).map(decode_error),
        incomplete_reason: response
            .get("incomplete_details")
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
            .and_then(IncompleteReason::from_str),
        output,
    })
}

/// Streamed text, gathered per item and per part.
///
/// Keyed rather than concatenated so that the same text can never be counted
/// twice: a part's identity is `(output_index, kind, part_index)`, and a source
/// that repeats it overwrites rather than appends. Derived `Ord` orders keys by
/// those fields in that order, which is document order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PartKey {
    pub(crate) output_index: u32,
    pub(crate) kind: PartKind,
    pub(crate) part_index: u32,
}

/// Which stream a text part belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PartKind {
    OutputText,
    ReasoningSummary,
}

/// The parts of one kind belonging to one output item, in document order.
///
/// The index is part of a part's identity, so this is a filter rather than a
/// second bookkeeping scheme: one item's text is exactly the parts keyed to it.
pub(crate) fn joined_at(parts: &BTreeMap<PartKey, String>, kind: PartKind, output_index: u32) -> String {
    let of_item = || {
        parts
            .iter()
            .filter(move |(key, _)| key.kind == kind && key.output_index == output_index)
            .map(|(_, text)| text.as_str())
    };
    let mut joined = String::with_capacity(of_item().map(str::len).sum());
    for piece in of_item() {
        joined.push_str(piece);
    }
    joined
}

/// Concatenates the parts of one kind, in document order, with one allocation.
pub(crate) fn joined(parts: &BTreeMap<PartKey, String>, kind: PartKind) -> String {
    let of_kind = || parts.iter().filter(move |(key, _)| key.kind == kind).map(|(_, text)| text.as_str());
    let mut joined = String::with_capacity(of_kind().map(str::len).sum());
    for piece in of_kind() {
        joined.push_str(piece);
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_data_lines_carry_a_payload() {
        assert_eq!(data_payload("data: {\"type\":\"x\"}"), Some("{\"type\":\"x\"}"));
        assert_eq!(data_payload("data:{\"type\":\"x\"}"), Some("{\"type\":\"x\"}"), "the space is optional");
        assert_eq!(data_payload("data:  two spaces"), Some(" two spaces"), "only one space is framing");
        assert_eq!(data_payload("event: response.completed"), None);
        assert_eq!(data_payload(": keep-alive comment"), None);
        assert_eq!(data_payload(""), None);
    }

    /// The captured shape of a text delta, decoded field by field.
    #[test]
    fn a_text_delta_decodes() {
        let event = StreamEvent::decode(
            r#"{"type":"response.output_text.delta","sequence_number":7,"item_id":"msg_1",
                "output_index":0,"content_index":0,"delta":"Hel","logprobs":[]}"#,
        )
        .unwrap();
        assert_eq!(event, StreamEvent::OutputTextDelta { output_index: 0, content_index: 0, delta: "Hel".to_owned() });
        assert_eq!(event.kind(), "response.output_text.delta");
        assert!(!event.is_terminal());
    }

    #[test]
    fn a_reasoning_summary_delta_decodes() {
        let event = StreamEvent::from_json(&json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1", "output_index": 0, "summary_index": 1, "delta": "weighing",
            "sequence_number": 3
        }))
        .unwrap();
        assert_eq!(
            event,
            StreamEvent::ReasoningSummaryTextDelta { output_index: 0, summary_index: 1, delta: "weighing".to_owned() }
        );
    }

    /// An event type nobody has taught this crate about is a variant, not an
    /// error: OpenAI may add one in any release.
    #[test]
    fn an_unknown_event_is_ignorable_rather_than_an_error() {
        let event = StreamEvent::decode(r#"{"type":"response.crystal_ball.delta","delta":"?"}"#).unwrap();
        assert_eq!(event, StreamEvent::Unmodeled { kind: "response.crystal_ball.delta".to_owned() });
        assert_eq!(event.kind(), "response.crystal_ball.delta");
        assert!(!event.is_terminal());
    }

    /// A documented event this crate does not model reads the same way, because
    /// "well-formed, nothing to do" is one situation, not two.
    #[test]
    fn a_documented_but_unmodeled_event_is_also_ignorable() {
        for kind in ["response.created", "response.in_progress", "response.content_part.added", "response.queued"] {
            let event = StreamEvent::from_json(&json!({"type": kind, "sequence_number": 1})).unwrap();
            assert_eq!(event, StreamEvent::Unmodeled { kind: kind.to_owned() });
        }
    }

    #[test]
    fn a_broken_frame_is_an_error() {
        assert!(matches!(StreamEvent::decode("not json at all"), Err(FrameError::NotJson(_))));
        assert!(matches!(StreamEvent::decode("[1,2]"), Err(FrameError::NotAnObject)));
        assert!(matches!(StreamEvent::decode("{}"), Err(FrameError::MissingField { field: "type" })));
        assert!(matches!(
            StreamEvent::from_json(&json!({"type": "response.output_text.delta", "output_index": 0})),
            Err(FrameError::MissingField { field: "content_index" })
        ));
        assert!(matches!(
            StreamEvent::from_json(&json!({
                "type": "response.output_text.delta", "output_index": -1, "content_index": 0, "delta": "x"
            })),
            Err(FrameError::WrongType { field: "output_index", .. })
        ));
    }

    #[test]
    fn a_function_call_item_keeps_its_arguments_as_bytes() {
        let event = StreamEvent::from_json(&json!({
            "type": "response.output_item.done", "output_index": 1, "sequence_number": 12,
            "item": {
                "type": "function_call", "id": "fc_1", "call_id": "call_abc", "name": "read_file",
                "arguments": "{\"path\":\"src/lib.rs\"}", "status": "completed"
            }
        }))
        .unwrap();
        let StreamEvent::OutputItemDone { output_index: 1, item: OutputItem::FunctionCall(call) } = event else {
            panic!("expected a finished function call");
        };
        assert_eq!(call.call_id, "call_abc");
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments.as_str(), r#"{"path":"src/lib.rs"}"#);
        assert_eq!(call.arguments.decode().unwrap(), json!({"path": "src/lib.rs"}));
    }

    /// Malformed arguments stay readable and stay malformed. The bytes survive
    /// so the caller can answer the model, and the failure is not silently an
    /// empty argument set.
    #[test]
    fn malformed_arguments_fail_only_where_they_are_decoded() {
        let arguments = FunctionArguments::from_wire(r#"{"path": "src/lib.rs"#);
        assert_eq!(arguments.as_str(), r#"{"path": "src/lib.rs"#);
        let error = arguments.decode().unwrap_err();
        assert!(error.to_string().contains("EOF"), "{error}");
    }

    /// A function taking no arguments is streamed with them absent, and once
    /// finished as the empty string. Both mean the empty argument set.
    #[test]
    fn absent_arguments_decode_to_the_empty_object() {
        assert_eq!(FunctionArguments::default().decode().unwrap(), json!({}));
        assert_eq!(FunctionArguments::from_wire("").decode().unwrap(), json!({}));
        assert_eq!(FunctionArguments::from_wire("  ").decode().unwrap(), json!({}));

        let event = StreamEvent::from_json(&json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "call_id": "c1", "name": "finish"}
        }))
        .unwrap();
        let StreamEvent::OutputItemAdded { item: OutputItem::FunctionCall(call), .. } = event else {
            panic!("expected an announced function call");
        };
        assert_eq!(call.arguments, FunctionArguments::default());
    }

    #[test]
    fn a_reasoning_item_is_replayable_only_with_its_encrypted_content() {
        let with = ReasoningItem { id: "rs_1".to_owned(), encrypted_content: Some("opaque".to_owned()) };
        assert_eq!(
            with.replayable(),
            Some(ReplayedReasoning { id: "rs_1".to_owned(), encrypted_content: "opaque".to_owned() })
        );
        let without = ReasoningItem { id: "rs_1".to_owned(), encrypted_content: None };
        assert_eq!(without.replayable(), None);

        let event = StreamEvent::from_json(&json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": ""}
        }))
        .unwrap();
        let StreamEvent::OutputItemDone { item: OutputItem::Reasoning(item), .. } = event else {
            panic!("expected a reasoning item");
        };
        assert_eq!(item.encrypted_content, None, "an empty payload is no payload");
    }

    #[test]
    fn an_output_item_kind_this_crate_ignores_is_not_an_error() {
        let event = StreamEvent::from_json(&json!({
            "type": "response.output_item.done", "output_index": 2,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "completed"}
        }))
        .unwrap();
        assert_eq!(
            event,
            StreamEvent::OutputItemDone {
                output_index: 2,
                item: OutputItem::Unmodeled { kind: "web_search_call".to_owned() }
            }
        );
    }

    #[test]
    fn a_message_item_carries_its_finished_text_and_phase() {
        let event = StreamEvent::from_json(&json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "phase": "final_answer",
                "content": [
                    {"type": "output_text", "text": "Hello ", "annotations": []},
                    {"type": "output_text", "text": "world", "annotations": []},
                    {"type": "refusal", "refusal": "I cannot help with that"}
                ]
            }
        }))
        .unwrap();
        let StreamEvent::OutputItemDone { item: OutputItem::Message { text, phase, id }, .. } = event else {
            panic!("expected a finished message");
        };
        assert_eq!(text, "Hello world", "output_text blocks join; a refusal is not text");
        assert_eq!(phase, Some(AssistantPhase::FinalAnswer));
        assert_eq!(id.as_deref(), Some("msg_1"));
    }

    #[test]
    fn the_terminal_events_carry_usage_and_their_reason() {
        let completed = StreamEvent::from_json(&json!({
            "type": "response.completed", "sequence_number": 40,
            "response": {
                "id": "resp_1", "status": "completed",
                "usage": {
                    "input_tokens": 15000,
                    "input_tokens_details": {"cached_tokens": 12000, "cache_write_tokens": 3000},
                    "output_tokens": 500, "output_tokens_details": {"reasoning_tokens": 400},
                    "total_tokens": 15500
                }
            }
        }))
        .unwrap();
        let StreamEvent::Completed(snapshot) = completed else { panic!("expected completion") };
        assert!(completed_is_terminal());
        assert_eq!(snapshot.id.as_deref(), Some("resp_1"));
        let usage = snapshot.usage.unwrap();
        assert_eq!(usage.input_tokens_details.cached_tokens, 12_000);
        assert_eq!(usage.input_tokens_details.cache_write_tokens, 3_000);
        assert_eq!(usage.cache_hit_rate(), Some(0.8));

        let incomplete = StreamEvent::from_json(&json!({
            "type": "response.incomplete",
            "response": {"id": "resp_2", "status": "incomplete",
                         "incomplete_details": {"reason": "max_output_tokens"}}
        }))
        .unwrap();
        let StreamEvent::Incomplete(snapshot) = incomplete else { panic!("expected an incomplete response") };
        assert_eq!(snapshot.incomplete_reason, Some(IncompleteReason::MaxOutputTokens));

        let failed = StreamEvent::from_json(&json!({
            "type": "response.failed",
            "response": {"id": "resp_3", "status": "failed",
                         "error": {"code": "server_error", "message": "upstream fell over"}}
        }))
        .unwrap();
        let StreamEvent::Failed(snapshot) = failed else { panic!("expected a failure") };
        let error = snapshot.error.unwrap();
        assert_eq!(error.code.as_deref(), Some("server_error"));
        assert_eq!(error.message, "upstream fell over");
    }

    fn completed_is_terminal() -> bool {
        StreamEvent::Completed(ResponseSnapshot {
            id: None,
            usage: None,
            error: None,
            incomplete_reason: None,
            output: Vec::new(),
        })
        .is_terminal()
    }

    /// An incomplete reason outside the documented pair reads as absent rather
    /// than failing the frame, since the response itself is still usable.
    #[test]
    fn an_unrecognized_incomplete_reason_reads_as_absent() {
        let event = StreamEvent::from_json(&json!({
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "some_new_reason"}}
        }))
        .unwrap();
        let StreamEvent::Incomplete(snapshot) = event else { panic!("expected an incomplete response") };
        assert_eq!(snapshot.incomplete_reason, None);
    }

    /// A `null` or absent usage is absent, and a usage object reporting only the
    /// totals is those totals: an omitted breakdown is zero of that kind, which
    /// is what a gateway that reports less than OpenAI does actually means. What
    /// remains an error is an object whose counts contradict the schema, because
    /// quietly reporting no cost would hide the one number this crate exists to
    /// measure.
    #[test]
    fn usage_is_absent_or_as_much_of_it_as_was_reported() {
        let null = StreamEvent::from_json(&json!({"type": "response.completed", "response": {"usage": null}}));
        let StreamEvent::Completed(snapshot) = null.unwrap() else { panic!("expected completion") };
        assert_eq!(snapshot.usage, None);

        let missing = StreamEvent::from_json(&json!({"type": "response.completed", "response": {}}));
        let StreamEvent::Completed(snapshot) = missing.unwrap() else { panic!("expected completion") };
        assert_eq!(snapshot.usage, None);

        let totals_only = StreamEvent::from_json(&json!({
            "type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 2}}
        }));
        let StreamEvent::Completed(snapshot) = totals_only.unwrap() else { panic!("expected completion") };
        let usage = snapshot.usage.expect("two totals are two real numbers");
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);

        let corrupt = StreamEvent::from_json(&json!({
            "type": "response.completed", "response": {"usage": {"input_tokens": "five"}}
        }));
        assert!(matches!(corrupt, Err(FrameError::UndecodableUsage(_))));
    }

    #[test]
    fn the_bare_error_event_decodes() {
        let event = StreamEvent::decode(
            r#"{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":null,
                "sequence_number":2}"#,
        )
        .unwrap();
        assert_eq!(
            event,
            StreamEvent::Error(ResponseError {
                code: Some("rate_limit_exceeded".to_owned()),
                message: "slow down".to_owned(),
                param: None,
            })
        );
        assert!(event.is_terminal());
        assert_eq!(event.kind(), "error");
    }

    /// A terminal event may repeat the whole `output` array; it decodes with
    /// the same code as the item events.
    #[test]
    fn a_terminal_snapshot_decodes_its_output_array() {
        let event = StreamEvent::from_json(&json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "output": [
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "done"}]},
                {"type": "function_call", "call_id": "c1", "name": "stop", "arguments": "{}"}
            ]}
        }))
        .unwrap();
        let StreamEvent::Completed(snapshot) = event else { panic!("expected completion") };
        assert_eq!(snapshot.output.len(), 2);
        assert!(matches!(&snapshot.output[0], OutputItem::Message { text, .. } if text == "done"));
        assert!(matches!(&snapshot.output[1], OutputItem::FunctionCall(call) if call.call_id == "c1"));
    }

    #[test]
    fn parts_join_in_document_order() {
        let mut parts = BTreeMap::new();
        parts.insert(PartKey { output_index: 1, kind: PartKind::OutputText, part_index: 0 }, "second".to_owned());
        parts.insert(PartKey { output_index: 0, kind: PartKind::OutputText, part_index: 1 }, "first ".to_owned());
        parts.insert(
            PartKey { output_index: 0, kind: PartKind::ReasoningSummary, part_index: 0 },
            "thinking".to_owned(),
        );
        assert_eq!(joined(&parts, PartKind::OutputText), "first second");
        assert_eq!(joined(&parts, PartKind::ReasoningSummary), "thinking");
    }
}
