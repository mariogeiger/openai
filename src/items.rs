//! What one response is made of: its output items, and the snapshot of the
//! response object a frame carries.
//!
//! Separate from [`crate::stream`], which states the *events*, because these are
//! the nouns and those are the verbs: an item is a thing the model produced, and
//! an event is news about one. The two change for different reasons — OpenAI adds
//! events far more often than it changes what an item is.

use serde_json::Value;

use crate::content::ReplayedReasoning;
use crate::usage::Usage;
use crate::values::{AssistantPhase, HostedTool, IncompleteReason, ResponseStatus, ServiceTier};

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
    /// request asked for `reasoning.encrypted_content` to be included — see
    /// [`Include::ReasoningEncryptedContent`](crate::values::Include::ReasoningEncryptedContent),
    /// which is how a caller asks.
    pub encrypted_content: Option<String>,
    /// The item's `summary` parts, concatenated, and empty when the request
    /// asked for no summary.
    ///
    /// A different array from the raw reasoning: OpenAI puts the summary in
    /// `summary`, as `summary_text` parts, and the raw reasoning in `content`.
    /// Only the summary is safe to show a user, which is why it is the half read
    /// here.
    pub summary: String,
}

impl ReasoningItem {
    /// The item in the form the next request needs, or `None` when it cannot be
    /// replayed at all.
    ///
    /// Without `encrypted_content` there is nothing to send in stateless mode,
    /// and this says so rather than handing back an item the API would reject.
    pub fn replayable(&self) -> Option<ReplayedReasoning> {
        crate::decode::replayable(self)
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
        /// Its `refusal` blocks, concatenated, and empty when it refused
        /// nothing.
        ///
        /// A separate field rather than folded into `text`, because a refusal
        /// concatenated into an answer *reads* as the answer: the user sees "I
        /// can't help with that" as though the model had answered. Two fields
        /// make the caller choose which one to show.
        refusal: String,
    },
    /// A function call.
    FunctionCall(CalledFunction),
    /// Reasoning.
    Reasoning(ReasoningItem),
    /// A hosted tool's call, named by which tool made it.
    ///
    /// The tool is decoded because it is what a consumer branches on; the rest
    /// of the item travels along undecoded because the useful fields differ per
    /// tool — a file search has `queries`, a code interpreter has a container, a
    /// shell call has a command list — and a consumer that turned the tool on
    /// knows which of them it wants. Typing all twelve shapes would be the
    /// crate guessing which fields matter.
    HostedToolCall {
        /// Which built-in tool.
        tool: HostedTool,
        /// The item's identifier, absent where a gateway omits it.
        id: Option<String>,
        /// Its `status`, as the wire spells it. A plain string because the
        /// values differ per tool: a file search reports `searching`, which no
        /// other tool sends.
        status: Option<String>,
        /// The whole item, for the fields this crate does not name.
        item: Value,
    },
    /// An item kind this crate does not model.
    ///
    /// Present for the same reason as [`StreamEvent::Unmodeled`](crate::stream::StreamEvent::Unmodeled): OpenAI adds
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

/// The `response` object delivered by a progress or terminal event.
///
/// The fields a streaming consumer acts on. The rest of the response object —
/// `instructions`, `tools`, `temperature` and the others — is what the caller
/// sent, so it is not read back here.
///
/// `Default` gives the all-absent snapshot, which is what a frame carrying only
/// `{"response": {}}` decodes to. Every field is independently absent, because a
/// gateway may report any subset and none of them is required to read the rest.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResponseSnapshot {
    /// The response's identifier, worth recording for OpenAI support.
    pub id: Option<String>,
    /// Where generation had got to when this frame was sent. `None` when the
    /// frame omits it or names a status this crate does not know.
    ///
    /// Redundant with which event carried the snapshot, and read anyway: a
    /// progress event's status is the only place `cancelled` appears, and a
    /// caller polling a background response has nothing else to read.
    pub status: Option<ResponseStatus>,
    /// What the response cost. `None` when absent or explicitly `null`.
    pub usage: Option<Usage>,
    /// Why it failed, on a failure.
    pub error: Option<ResponseError>,
    /// Why it stopped short, on an incomplete response. `None` when absent or
    /// when the API names a reason this crate does not know.
    pub incomplete_reason: Option<IncompleteReason>,
    /// Which processing tier actually served the request, which may differ from
    /// the one asked for: `fast` is reported as `priority`.
    pub service_tier: Option<ServiceTier>,
    /// The final `output` array, when the terminal event repeats it.
    pub output: Vec<OutputItem>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::StreamEvent;
    use serde_json::json;

    #[test]
    fn a_reasoning_item_is_replayable_only_with_its_encrypted_content() {
        let with = ReasoningItem {
            id: "rs_1".to_owned(),
            encrypted_content: Some("opaque".to_owned()),
            summary: String::new(),
        };
        assert_eq!(
            with.replayable(),
            Some(ReplayedReasoning { id: "rs_1".to_owned(), encrypted_content: "opaque".to_owned() })
        );
        let without = ReasoningItem { id: "rs_1".to_owned(), encrypted_content: None, summary: String::new() };
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
}
