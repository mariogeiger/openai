//! The whole response body, for a request that did not stream.
//!
//! The other way to read an answer. A `stream: false` request answers with one
//! JSON object rather than a sequence of events, and until now this crate
//! decoded only the events — so a caller who did not want a stream had to
//! hand-match the body, which is the re-derivation this crate exists to prevent.
//!
//! # Why this is not a `Settled`
//!
//! [`Settled`](crate::settled::Settled) means *a stream that reached a terminal
//! event*, and its whole value is that a truncated stream cannot be read as one.
//! A buffered body has no such failure mode: it either arrived whole or the HTTP
//! request failed, and the caller already knows which. Reusing the type would
//! have meant either weakening that guarantee or pretending to a check there is
//! nothing to check.
//!
//! What the two share is [`OutputItem`], so [`Response::text`] and
//! [`Response::function_calls`] read the same items the same way.
//!
//! # What is decoded, and what is echo
//!
//! The response repeats most of the request back. Those fields are decoded where
//! the server may have *changed* one — `service_tier` reports the tier that
//! actually served the request, and `model` reports the resolved name — and
//! skipped where the answer is only what the caller sent. The rest of the body is
//! kept in [`Response::raw`], so nothing is lost to a caller who needs a field
//! this type does not name.

use serde_json::Value;

use crate::decode;
use crate::items::{CalledFunction, OutputItem, ResponseError, ResponseSnapshot};
use crate::stream::FrameError;
use crate::usage::Usage;
use crate::values::{IncompleteReason, ResponseStatus, ServiceTier};

/// One complete response body.
///
/// `#[non_exhaustive]` because the API adds response fields, and adding one here
/// must not break a caller who matched the struct.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Response {
    /// The response's identifier. Worth logging: it is what OpenAI support needs
    /// to look a request up.
    pub id: Option<String>,
    /// Where generation ended up. `None` when the body omits it or names a status
    /// this crate does not know — which is not an error, since the output is
    /// still readable.
    pub status: Option<ResponseStatus>,
    /// The model that actually answered, which a gateway may resolve differently
    /// from the name sent.
    pub model: Option<String>,
    /// Everything the model produced, in order.
    pub output: Vec<OutputItem>,
    /// What it cost. `None` when the body reported none.
    pub usage: Option<Usage>,
    /// Why it failed, on a failure.
    pub error: Option<ResponseError>,
    /// Why it stopped short, on an incomplete response.
    pub incomplete_reason: Option<IncompleteReason>,
    /// Which processing tier served it, which may differ from the one asked for.
    pub service_tier: Option<ServiceTier>,
    /// The body as it arrived, for every field this type does not name.
    ///
    /// Kept rather than dropped because the response echoes far more than it
    /// decides, and a caller needing `created_at` or a moderation verdict should
    /// not have to parse the body a second time. Naming it `raw` says plainly
    /// that reading it is reaching past the typed surface.
    pub raw: Value,
}

impl Response {
    /// One response body, decoded.
    ///
    /// # Errors
    ///
    /// Fails on a body that contradicts the schema: not an object, or an output
    /// item or `usage` object of the wrong shape. A field this crate does not
    /// model is not a failure.
    pub fn decode(body: &str) -> Result<Self, FrameError> {
        let value: Value = serde_json::from_str(body).map_err(FrameError::NotJson)?;
        Self::from_json(&value)
    }

    /// The same, for a caller who already parsed the body.
    ///
    /// Shares its whole implementation with the snapshot a terminal streaming
    /// event carries, because they are the same object: `response.completed`
    /// delivers exactly this body. One decoder means the two paths cannot drift.
    pub fn from_json(body: &Value) -> Result<Self, FrameError> {
        let ResponseSnapshot { id, status, usage, error, incomplete_reason, service_tier, output } =
            decode::snapshot(body)?;
        Ok(Self {
            id,
            status,
            model: body.get("model").and_then(Value::as_str).map(str::to_owned),
            output,
            usage,
            error,
            incomplete_reason,
            service_tier,
            raw: body.clone(),
        })
    }

    /// The answer text: every message item's `output_text`, in order.
    ///
    /// Answer text only. A refusal is [`Self::refusal`] and reasoning is
    /// [`Self::reasoning_summary`], for the reason those fields exist on
    /// [`Settled`](crate::settled::Settled) too: a refusal folded into the answer
    /// reads as the answer.
    pub fn text(&self) -> String {
        self.joined(|item| match item {
            OutputItem::Message { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// The refusal, when the model declined, and empty otherwise.
    pub fn refusal(&self) -> String {
        self.joined(|item| match item {
            OutputItem::Message { refusal, .. } => Some(refusal.as_str()),
            _ => None,
        })
    }

    /// The reasoning summary, when one was requested.
    pub fn reasoning_summary(&self) -> String {
        self.joined(|item| match item {
            OutputItem::Reasoning(item) => Some(item.summary.as_str()),
            _ => None,
        })
    }

    /// The selected text of every output item, joined in order.
    fn joined(&self, select: impl Fn(&OutputItem) -> Option<&str> + Copy) -> String {
        let selected = || self.output.iter().filter_map(select);
        let mut joined = String::with_capacity(selected().map(str::len).sum());
        for piece in selected() {
            joined.push_str(piece);
        }
        joined
    }

    /// The function calls the model made, in order.
    ///
    /// The list a tool-running loop iterates. Arguments are still undecoded — see
    /// [`FunctionArguments::decode`](crate::items::FunctionArguments::decode),
    /// which fails per call rather than per response.
    pub fn function_calls(&self) -> impl Iterator<Item = &CalledFunction> {
        self.output.iter().filter_map(|item| match item {
            OutputItem::FunctionCall(call) => Some(call),
            _ => None,
        })
    }

    /// The reasoning items, in the form the next request needs.
    ///
    /// Only the ones that can be replayed: an item without `encrypted_content`
    /// has nothing to send in stateless mode, and this skips it rather than
    /// handing back an item the API would refuse. Ask for the payload with
    /// [`Include::ReasoningEncryptedContent`](crate::values::Include::ReasoningEncryptedContent).
    pub fn replayable_reasoning(&self) -> impl Iterator<Item = crate::content::ReplayedReasoning> {
        self.output.iter().filter_map(|item| match item {
            OutputItem::Reasoning(item) => item.replayable(),
            _ => None,
        })
    }

    /// Whether the model answered in full.
    pub fn is_completed(&self) -> bool {
        self.status == Some(ResponseStatus::Completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A body captured from `POST https://inference-api.nvidia.com/v1/responses`
    /// with model `openai/openai/gpt-5.6-sol` on 2026-08-30, `stream` absent.
    /// Its identifiers are replaced with fixed ones, because no assertion here
    /// reads them and a fixture should carry only what it is for. The streamed
    /// captures keep theirs, because a test there asserts the real shape of a
    /// reasoning payload.
    ///
    /// Kept whole rather than trimmed to what the assertions read, because the
    /// fields it carries that this crate does not model are the point: a real
    /// body has `billing`, `tool_usage`, `frequency_penalty` and a dozen nulls,
    /// and every one of them must decode as "not my business".
    const CAPTURED: &str = include_str!("../tests/data/captured_buffered_response.json");

    #[test]
    fn a_captured_buffered_body_decodes() {
        let response = Response::decode(CAPTURED).expect("a real body decodes");
        assert!(response.is_completed());
        assert_eq!(response.text(), "OK");
        assert!(response.refusal().is_empty());
        assert_eq!(response.service_tier, Some(ServiceTier::Default));
        assert_eq!(response.model.as_deref(), Some("openai/openai/gpt-5.6-sol"));
        assert_eq!(response.function_calls().count(), 0);

        let usage = response.usage.expect("a real body reports usage");
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 16);

        // The fields this type does not name are still reachable, undecoded.
        assert_eq!(response.raw["metadata"]["probe"], "buffered");
        assert_eq!(response.raw["billing"]["payer"], "developer");
    }

    /// A gateway that spells a count `null` reports zero of that kind, not a
    /// broken body.
    ///
    /// Measured on the captured body above, which sends `"audio_tokens": null`
    /// inside a breakdown whose other counts are real numbers. Before this, one
    /// `null` failed the whole `usage` object and took the cache accounting with
    /// it — the one measurement the crate exists to provide.
    #[test]
    fn a_null_count_reports_zero_rather_than_failing() {
        let response = Response::from_json(&json!({
            "id": "resp_1", "status": "completed", "output": [],
            "usage": {
                "input_tokens": null,
                "input_tokens_details": {"cached_tokens": null, "cache_write_tokens": 7, "audio_tokens": null},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": null},
                "total_tokens": null
            }
        }))
        .expect("nulls are absences, not errors");
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.input_tokens_details.cache_write_tokens, 7, "a real number beside a null still counts");
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
    }

    /// A refusal is kept out of the answer here for the same reason it is in a
    /// settled stream.
    #[test]
    fn a_buffered_refusal_is_not_answer_text() {
        let response = Response::from_json(&json!({
            "id": "resp_1", "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "I cannot help with that."}]}]
        }))
        .unwrap();
        assert_eq!(response.text(), "", "a refusal is not an answer");
        assert_eq!(response.refusal(), "I cannot help with that.");
    }

    /// A failed body carries its error, and an incomplete one its reason.
    #[test]
    fn a_failure_and_a_truncation_each_say_why() {
        let failed = Response::from_json(&json!({
            "id": "resp_1", "status": "failed",
            "error": {"code": "server_error", "message": "upstream fell over"}
        }))
        .unwrap();
        assert!(!failed.is_completed());
        assert_eq!(failed.status, Some(ResponseStatus::Failed));
        assert_eq!(failed.error.unwrap().message, "upstream fell over");

        let incomplete = Response::from_json(&json!({
            "id": "resp_2", "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}
        }))
        .unwrap();
        assert_eq!(incomplete.incomplete_reason, Some(IncompleteReason::MaxOutputTokens));
    }

    /// A body that is not an object is a broken body; one carrying fields this
    /// crate does not model is not.
    #[test]
    fn a_broken_body_fails_and_an_unknown_field_does_not() {
        assert!(matches!(Response::decode("[1, 2]"), Err(FrameError::WrongType { .. })));
        assert!(matches!(Response::decode("not json"), Err(FrameError::NotJson(_))));
        assert!(Response::from_json(&json!({"id": "r", "invented_next_year": {"deeply": [1]}})).is_ok());
    }
}
