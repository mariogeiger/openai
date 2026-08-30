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

use crate::hosted::HostedToolPhase;
use crate::items::{FunctionArguments, OutputItem, ResponseError, ResponseSnapshot};
use crate::values::{HostedTool, ResponseStatus};

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
    /// A `usage` object was present but would not deserialize into [`Usage`](crate::usage::Usage).
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

// ── Which stream a piece of text belongs to ──────────────────────────────────

/// Which of the model's four text streams a delta belongs to.
///
/// The API sends four pairs of `delta`/`done` events that differ only in what
/// the text *is*: the answer, a refusal, the reasoning summary, and the raw
/// reasoning. All four carry an output index and a part index and accumulate the
/// same way, so they are one variant plus this discriminant rather than eight
/// variants — and the difference that matters is kept, because it decides where
/// the text may be shown.
///
/// The distinction is not cosmetic. Concatenating a refusal into the answer makes
/// the refusal read as the answer, which is the defect
/// [`Self::Refusal`] exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextStream {
    /// `output_text`: the answer. This is what a user reads.
    Output,
    /// `refusal`: the model declining. **Not** answer text — a refusal folded
    /// into the answer reads as the answer, and the two must stay apart.
    Refusal,
    /// `reasoning_summary_text`: the summarized reasoning, safe to show.
    ReasoningSummary,
    /// `reasoning_text`: the raw reasoning, on the models that stream it.
    Reasoning,
}

impl TextStream {
    /// The stream's wire name, the middle of its event types.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Output => "output_text",
            Self::Refusal => "refusal",
            Self::ReasoningSummary => "reasoning_summary_text",
            Self::Reasoning => "reasoning_text",
        }
    }

    /// Whether this text belongs in the answer a user reads.
    ///
    /// A method rather than a comparison a caller writes, because getting it
    /// wrong is silent: the only stream that is answer text is
    /// [`Self::Output`], and a caller who folds any other one in has produced a
    /// wrong answer with no error anywhere.
    pub fn is_answer_text(self) -> bool {
        matches!(self, Self::Output)
    }

    /// Which field name the `done` event uses for the whole text, and which
    /// field names its part index.
    ///
    /// The wire is not uniform here: a refusal's whole text arrives as
    /// `refusal` rather than `text`, and a reasoning summary indexes by
    /// `summary_index` rather than `content_index`. Stating both irregularities
    /// once, beside the stream they belong to, is what keeps them out of the
    /// decoder as special cases.
    pub(crate) fn wire_fields(self) -> (&'static str, &'static str) {
        match self {
            Self::Output => ("text", "content_index"),
            Self::Refusal => ("refusal", "content_index"),
            Self::ReasoningSummary => ("text", "summary_index"),
            Self::Reasoning => ("text", "content_index"),
        }
    }
}

/// Whether a part event opens a part or closes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PartBoundary {
    /// The part now exists, and its text is empty.
    Added,
    /// The part is complete, and its whole text is here.
    Done,
}

/// Which non-terminal stage a progress event announces.
///
/// Three stages rather than a [`ResponseStatus`], because `created` and `queued`
/// are two events that report the same status: a response is queued when it is
/// created. Reusing the status enum would have made the two indistinguishable
/// after decoding, and an event that cannot name itself back is an event this
/// crate decoded lossily. The status is still available, on the snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProgressStage {
    /// `response.created`: the request was accepted. Every stream begins here.
    Created,
    /// `response.queued`: it is waiting for capacity.
    Queued,
    /// `response.in_progress`: the model is generating.
    InProgress,
}

impl ProgressStage {
    /// The stage's own wire event type.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Created => "response.created",
            Self::Queued => "response.queued",
            Self::InProgress => "response.in_progress",
        }
    }

    /// The status a response has at this stage.
    ///
    /// Two stages share one status, which is exactly why they are separate
    /// values: this direction is a function, and the other is not.
    pub fn status(self) -> ResponseStatus {
        match self {
            Self::Created | Self::Queued => ResponseStatus::Queued,
            Self::InProgress => ResponseStatus::InProgress,
        }
    }
}

/// Which half of a generated-audio response a frame carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AudioStream {
    /// The audio itself, base64-encoded.
    Audio,
    /// Its transcript, as text.
    Transcript,
}

// ── Events ───────────────────────────────────────────────────────────────────

/// One decoded streaming event.
///
/// The API documents 58 event types. They are not 58 shapes: most are one *kind*
/// of thing — a text delta, a lifecycle notice, a finished part — parameterized
/// by which stream it belongs to. So the variants here are the kinds, and the
/// parameter is a field:
///
/// * Growing text is [`Self::TextDelta`] plus a [`TextStream`] saying which of
///   the four text streams grew.
/// * A finished run of text is [`Self::TextDone`], the same way.
/// * All 18 hosted-tool lifecycle notices are [`Self::HostedToolLifecycle`],
///   carrying a [`HostedTool`] and a [`HostedToolPhase`] — twelve tools times six
///   phases from one variant.
/// * The response's own progress is [`Self::ResponseProgress`] plus a
///   [`ResponseStatus`].
///
/// [`Self::Unmodeled`] remains for anything OpenAI adds next, because a new
/// event type is a compatible change and must never be an error.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A run of text grew.
    ///
    /// One variant for all four text streams — the answer, a refusal, the
    /// reasoning summary, the raw reasoning — because they accumulate
    /// identically and differ only in [`TextStream`], which decides where the
    /// text may be shown. Four variants would have made the answer/refusal
    /// distinction a name a caller reads rather than a value it must match.
    TextDelta {
        /// Which stream grew.
        stream: TextStream,
        /// Which output item this text belongs to.
        output_index: u32,
        /// Which part within that item. `content_index` on the wire for three
        /// of the streams and `summary_index` for the reasoning summary; one
        /// field here, because it means the same thing.
        part_index: u32,
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
    /// A run of text finished, and this is the whole of it.
    ///
    /// The `done` counterpart of [`Self::TextDelta`], carrying the same
    /// [`TextStream`]. Redundant with the deltas by design — OpenAI sends both —
    /// and useful for exactly that reason: a consumer can check its own
    /// accumulation against what the server says it sent, and
    /// [`Settling`](crate::settle::Settling) does, so a dropped delta shows up as
    /// a mismatch rather than as silently missing text.
    TextDone {
        /// Which text stream finished.
        stream: TextStream,
        /// Which output item it belongs to.
        output_index: u32,
        /// Which part within that item.
        part_index: u32,
        /// The whole text of that part.
        text: String,
    },
    /// A content part or reasoning summary part began or ended.
    ///
    /// `response.content_part.added` and `.done`, and
    /// `response.reasoning_summary_part.added` and `.done`. The part's text is
    /// also delivered as deltas, so this is a boundary marker: it says *a part
    /// exists here and it is of this kind*, which is what lets a consumer show
    /// several reasoning summary paragraphs as separate paragraphs rather than
    /// one run-on block.
    Part {
        /// Which stream the part belongs to.
        stream: TextStream,
        /// Whether it began or ended.
        boundary: PartBoundary,
        /// Which output item it belongs to.
        output_index: u32,
        /// Its index within that item.
        part_index: u32,
        /// Its text: empty on `added`, whole on `done`.
        text: String,
    },
    /// A function call's arguments grew, or finished.
    ///
    /// `response.function_call_arguments.delta` and `.done`. A consumer that
    /// only needs the finished call can ignore both and read
    /// [`Self::OutputItemDone`], which carries the same arguments; these exist
    /// for showing a call being composed, and for the case the `done` item is
    /// never sent.
    FunctionArgumentsDelta {
        /// Which output item is the call.
        output_index: u32,
        /// The item's identifier, absent where a gateway omits it.
        item_id: Option<String>,
        /// The new fragment, to append. Not JSON on its own.
        delta: String,
    },
    /// A function call's arguments are complete.
    FunctionArgumentsDone {
        /// Which output item is the call.
        output_index: u32,
        /// The item's identifier, absent where a gateway omits it.
        item_id: Option<String>,
        /// The whole arguments string, exactly as the model emitted it.
        arguments: FunctionArguments,
    },
    /// A hosted tool's call reached a new phase.
    ///
    /// One variant for all 18 documented hosted-tool lifecycle events, because
    /// they are one shape: a tool, a phase, and which output item is the call.
    /// A consumer showing "searching the web…" matches
    /// `(HostedTool::WebSearch, _)` once rather than three event types, and a
    /// tool OpenAI adds later arrives here rather than in
    /// [`Self::Unmodeled`].
    HostedToolLifecycle {
        /// Which built-in tool.
        tool: HostedTool,
        /// What it is now doing.
        phase: HostedToolPhase,
        /// Which output item is the call.
        output_index: u32,
        /// The call item's identifier, absent where a gateway omits it.
        item_id: Option<String>,
    },
    /// A hosted tool's input grew: code being written, MCP arguments being
    /// composed, a custom tool's free-text input.
    ///
    /// `response.code_interpreter_call_code.delta`,
    /// `response.mcp_call_arguments.delta`, and
    /// `response.custom_tool_call_input.delta`, which differ only in which tool
    /// is composing. The `done` form arrives with [`Self::HostedToolInputDone`].
    HostedToolInputDelta {
        /// Which built-in tool is composing.
        tool: HostedTool,
        /// Which output item is the call.
        output_index: u32,
        /// The call item's identifier, absent where a gateway omits it.
        item_id: Option<String>,
        /// The new fragment, to append.
        delta: String,
    },
    /// A hosted tool's input is complete.
    HostedToolInputDone {
        /// Which built-in tool composed it.
        tool: HostedTool,
        /// Which output index is the call.
        output_index: u32,
        /// The call item's identifier, absent where a gateway omits it.
        item_id: Option<String>,
        /// The whole input.
        input: String,
    },
    /// A partial image from the image-generation tool.
    ///
    /// Sent repeatedly at increasing quality, so a consumer can show the image
    /// resolving. Its own variant rather than a phase of
    /// [`Self::HostedToolLifecycle`] because it is the one hosted-tool event
    /// carrying a payload keyed by its own index.
    PartialImage {
        /// Which output item is the call.
        output_index: u32,
        /// The call item's identifier, absent where a gateway omits it.
        item_id: Option<String>,
        /// Which partial image this is, counting from zero.
        partial_image_index: u32,
        /// The image, base64-encoded. Kept encoded: decoding it is the caller's
        /// choice, and this crate takes no image dependency to do it.
        partial_image_base64: String,
    },
    /// A citation or file path was attached to a run of output text.
    ///
    /// Only hosted tools produce these, and only `include` asks for some of
    /// them. The annotation is kept as the raw value rather than typed into the
    /// four documented shapes, because a consumer that renders citations needs
    /// the whole object and the shapes grow with the tools.
    Annotation {
        /// Which output item the annotated text belongs to.
        output_index: u32,
        /// Which content part within that item.
        content_index: u32,
        /// Which annotation on that part.
        annotation_index: u32,
        /// The annotation object, undecoded.
        annotation: Value,
    },
    /// Generated audio, or its transcript, grew or finished.
    ///
    /// Keyed by the response rather than by an output index, which is how the
    /// wire sends it: audio is one stream per response, not one per item.
    Audio {
        /// Which stream: the audio itself or its transcript.
        stream: AudioStream,
        /// The response's identifier, as the frame states it.
        response_id: Option<String>,
        /// The new fragment, base64 for audio and text for a transcript. Empty
        /// on the `done` form, which carries no payload.
        delta: String,
        /// Whether this is the final frame of that stream.
        done: bool,
    },
    /// The response's own generation reached a new stage.
    ///
    /// `response.created`, `.queued`, and `.in_progress` — three events carrying
    /// the same payload and differing only in the stage they announce. None is
    /// terminal. The terminal three are [`Self::Completed`], [`Self::Failed`],
    /// and [`Self::Incomplete`], which stay separate variants because settling
    /// depends on them and a stage field would let a caller forget one.
    ResponseProgress {
        /// Which stage was announced.
        stage: ProgressStage,
        /// The response so far. Its `output` is usually empty here.
        response: ResponseSnapshot,
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
    /// metrics need no match. Reconstructed from the variant's own fields rather
    /// than stored, so it cannot drift from what the variant means — and the two
    /// directions are tested against each other for every documented event.
    ///
    /// `Cow` because most variants name a `&'static str` from the wire
    /// vocabulary, while [`Self::Unmodeled`] carries a string of its own and the
    /// factored variants compose one from their fields.
    pub fn kind(&self) -> std::borrow::Cow<'_, str> {
        use std::borrow::Cow::{Borrowed, Owned};
        match self {
            StreamEvent::TextDelta { stream, .. } => Owned(format!("response.{}.delta", stream.as_str())),
            StreamEvent::TextDone { stream, .. } => Owned(format!("response.{}.done", stream.as_str())),
            StreamEvent::Part { stream, boundary, .. } => {
                let part = match stream {
                    TextStream::ReasoningSummary => "reasoning_summary_part",
                    // The other three streams all report their parts through
                    // `content_part`, which is why this is not a per-stream
                    // table.
                    TextStream::Output | TextStream::Refusal | TextStream::Reasoning => "content_part",
                };
                Owned(format!("response.{part}.{}", boundary.as_str()))
            }
            StreamEvent::FunctionArgumentsDelta { .. } => Borrowed("response.function_call_arguments.delta"),
            StreamEvent::FunctionArgumentsDone { .. } => Borrowed("response.function_call_arguments.done"),
            StreamEvent::HostedToolLifecycle { tool, phase, .. } => {
                crate::hosted::lifecycle_event_type(*tool, *phase).map_or_else(
                    // A pair OpenAI does not send has no event type to name, and
                    // inventing one would be a string that decodes to nothing.
                    || Owned(format!("response.{}.{}", tool.as_str(), phase.as_str())),
                    Borrowed,
                )
            }
            StreamEvent::HostedToolInputDelta { tool, .. } => Borrowed(hosted_input_event_type(*tool, false)),
            StreamEvent::HostedToolInputDone { tool, .. } => Borrowed(hosted_input_event_type(*tool, true)),
            StreamEvent::PartialImage { .. } => Borrowed("response.image_generation_call.partial_image"),
            StreamEvent::Annotation { .. } => Borrowed("response.output_text.annotation.added"),
            StreamEvent::Audio { stream, done, .. } => Borrowed(match (stream, done) {
                (AudioStream::Audio, false) => "response.audio.delta",
                (AudioStream::Audio, true) => "response.audio.done",
                (AudioStream::Transcript, false) => "response.audio.transcript.delta",
                (AudioStream::Transcript, true) => "response.audio.transcript.done",
            }),
            StreamEvent::ResponseProgress { stage, .. } => Borrowed(stage.as_str()),
            StreamEvent::OutputItemAdded { .. } => Borrowed("response.output_item.added"),
            StreamEvent::OutputItemDone { .. } => Borrowed("response.output_item.done"),
            StreamEvent::Completed(_) => Borrowed("response.completed"),
            StreamEvent::Failed(_) => Borrowed("response.failed"),
            StreamEvent::Incomplete(_) => Borrowed("response.incomplete"),
            StreamEvent::Error(_) => Borrowed("error"),
            StreamEvent::Unmodeled { kind } => Borrowed(kind.as_str()),
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

    /// The answer text this event contributes, if any.
    ///
    /// The question a display asks of every frame, answered in one place so a
    /// caller cannot get it wrong. `None` for a refusal and for both reasoning
    /// streams: their text is real and must not be shown as the answer.
    pub fn answer_text_delta(&self) -> Option<&str> {
        match self {
            StreamEvent::TextDelta { stream, delta, .. } if stream.is_answer_text() => Some(delta),
            _ => None,
        }
    }

    /// One raw `data:` payload, decoded.
    pub fn decode(payload: &str) -> Result<Self, FrameError> {
        let value: Value = serde_json::from_str(payload).map_err(FrameError::NotJson)?;
        Self::from_json(&value)
    }

    /// The same, for a caller who already holds the parsed frame.
    pub fn from_json(frame: &Value) -> Result<Self, FrameError> {
        crate::decode::event_from_json(frame)
    }
}

impl PartBoundary {
    /// The boundary's wire word, the tail of its event type.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Done => "done",
        }
    }
}

/// Which event type a hosted tool's input delta or done frame carries.
///
/// Three tools compose input, and each names its event differently, so this is a
/// table rather than a format string.
fn hosted_input_event_type(tool: HostedTool, done: bool) -> &'static str {
    match (tool, done) {
        (HostedTool::CodeInterpreter, false) => "response.code_interpreter_call_code.delta",
        (HostedTool::CodeInterpreter, true) => "response.code_interpreter_call_code.done",
        (HostedTool::Mcp, false) => "response.mcp_call_arguments.delta",
        (HostedTool::Mcp, true) => "response.mcp_call_arguments.done",
        (HostedTool::Custom, false) => "response.custom_tool_call_input.delta",
        // Only these three tools stream composed input; anything else reaching
        // here is a `Custom` done frame or a variant built by hand.
        _ => "response.custom_tool_call_input.done",
    }
}

/// Streamed text, gathered per stream, item, and part.
///
/// Keyed rather than concatenated so that the same text can never be counted
/// twice: a part's identity is `(output_index, stream, part_index)`, and a source
/// that repeats it overwrites rather than appends. Derived `Ord` orders keys by
/// those fields in that order, which is document order.
///
/// The stream is part of the key, which is what keeps the four kinds of text
/// apart in one map: an answer and a refusal at the same indices are two parts,
/// not one part written twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartKey {
    /// Which output item the part belongs to.
    pub output_index: u32,
    /// Which of the four text streams it is.
    pub stream: TextStream,
    /// Its index within that item and stream.
    pub part_index: u32,
}

/// The parts of one stream belonging to one output item, in document order.
///
/// The index is part of a part's identity, so this is a filter rather than a
/// second bookkeeping scheme: one item's text is exactly the parts keyed to it.
pub(crate) fn joined_at(parts: &BTreeMap<PartKey, String>, stream: TextStream, output_index: u32) -> String {
    joined_where(parts, move |key| key.stream == stream && key.output_index == output_index)
}

/// Concatenates the parts of one stream, in document order, with one allocation.
pub(crate) fn joined(parts: &BTreeMap<PartKey, String>, stream: TextStream) -> String {
    joined_where(parts, move |key| key.stream == stream)
}

/// The selected parts, joined in key order with one allocation.
///
/// One implementation for both selections: the `BTreeMap` is already in document
/// order, so joining is a filter plus a sum of lengths however the filter is
/// spelled.
fn joined_where(parts: &BTreeMap<PartKey, String>, keep: impl Fn(&PartKey) -> bool + Copy) -> String {
    let selected = || parts.iter().filter(move |(key, _)| keep(key)).map(|(_, text)| text.as_str());
    let mut joined = String::with_capacity(selected().map(str::len).sum());
    for piece in selected() {
        joined.push_str(piece);
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_join_in_document_order() {
        let mut parts = BTreeMap::new();
        parts.insert(PartKey { output_index: 1, stream: TextStream::Output, part_index: 0 }, "second".to_owned());
        parts.insert(PartKey { output_index: 0, stream: TextStream::Output, part_index: 1 }, "first ".to_owned());
        parts.insert(
            PartKey { output_index: 0, stream: TextStream::ReasoningSummary, part_index: 0 },
            "thinking".to_owned(),
        );
        assert_eq!(joined(&parts, TextStream::Output), "first second");
        assert_eq!(joined(&parts, TextStream::ReasoningSummary), "thinking");
    }
}
