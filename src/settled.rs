//! A finished response, and the four ways a stream can end.
//!
//! Separate from [`crate::settle`], which accumulates, because these are the two
//! halves of one deliberate boundary: that file holds the type that cannot yield
//! a response, and this one holds the type that cannot take more events. Keeping
//! them apart in the source keeps the boundary visible.
//!
//! [`Settled`] is reachable only through
//! [`Settling::settle`](crate::settle::Settling::settle)(crate::settle::Settling::settle), and `#[non_exhaustive]`
//! is what makes that structural rather than advisory: a caller outside the crate
//! cannot write the literal, so a finished response can only come from a finished
//! stream.

use std::collections::BTreeMap;

use crate::items::{CalledFunction, OutputItem, ReasoningItem, ResponseError};
use crate::settle::PartDisagreement;
use crate::usage::Usage;
use crate::values::{HostedTool, IncompleteReason};

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
/// Obtainable only from [`Settling::settle`](crate::settle::Settling::settle), so holding one is proof the
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
    ///
    /// Answer text only. A refusal is in [`Self::refusal`] and reasoning is in
    /// the two reasoning fields, because folding any of them in here would make
    /// them read as the answer.
    pub text: String,
    /// The refusal, when the model declined, and empty otherwise.
    ///
    /// Its own field for the reason [`Self::text`] states. A caller that shows
    /// this as the answer has chosen to; one that concatenated it would not have
    /// noticed.
    pub refusal: String,
    /// The reasoning summary, when one was requested.
    pub reasoning_summary: String,
    /// The raw reasoning, on models that stream it. Usually empty: most models
    /// send only the summary.
    pub reasoning: String,
    /// The output items, in `output_index` order.
    pub items: Vec<OutputItem>,
    /// What the response cost. `None` when the API did not report it, which is
    /// the normal case for a bare `error` event.
    pub usage: Option<Usage>,
    /// How many events the stream delivered.
    pub events: usize,
    /// Which hosted tools ran, and how many events each sent.
    ///
    /// Present because a hosted tool call is a cost and a latency a caller can
    /// otherwise only guess at: nothing else in a settled response says that a
    /// web search happened.
    pub hosted_tool_events: BTreeMap<HostedTool, usize>,
    /// Every place a `done` frame contradicted the deltas accumulated for it.
    ///
    /// Empty on a healthy stream. Non-empty means a delta was dropped,
    /// duplicated, or misordered — the server states each run of text twice, and
    /// this is the free check that comparison buys.
    pub part_disagreements: Vec<PartDisagreement>,
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
    /// see [`FunctionArguments::decode`](crate::items::FunctionArguments::decode), which fails per call
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
