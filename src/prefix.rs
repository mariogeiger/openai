//! Everything that determines the bytes OpenAI hashes.
//!
//! OpenAI caches a *rendered prefix*: hidden OpenAI content, then `tools`, then
//! developer instructions, then the input items. A cache read requires that
//! prefix to be byte-identical, so a setting that changes it is categorically
//! different from one that does not — the first must be held constant across a
//! thread, the second may vary every request.
//!
//! This module is that first category, gathered into [`PrefixSettings`]. The
//! second lives on [`Request`](crate::request::Request). Naming the split is the
//! point: in hand-rolled JSON these settings sit side by side and look
//! interchangeable, which is how `reasoning.effort` quietly becomes a per-request
//! knob and every cached token disappears.
//!
//! `tools` belongs to this category too, and lives on
//! [`Context`](crate::context::Context) because it is stable across turns rather
//! than chosen per call.

use crate::model::Model;
use crate::values::{ReasoningEffort, ReasoningSummary, Verbosity};
use serde_json::Value;

// ── Sampling ─────────────────────────────────────────────────────────────────

/// Sampling temperature, in the API-accepted range `[0.0, 2.0]`.
///
/// A validating newtype so the range is proved once, at construction, and never
/// re-checked downstream.
///
/// A caveat the type cannot express: every model this crate models is a
/// reasoning model, and reasoning models reject any temperature but the default.
/// [`Request`](crate::request::Request) therefore has no temperature field at all, and this type exists
/// for callers extending the crate to a non-reasoning model. The validation is
/// the reusable part; where it may be sent is a per-model fact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperature(f32);

/// Why a temperature was refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureError {
    /// NaN or infinite. Neither survives JSON, which has no way to spell them.
    NotFinite,
    /// Finite but outside `[0.0, 2.0]`.
    OutOfRange(f32),
}

impl std::fmt::Display for TemperatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemperatureError::NotFinite => write!(f, "temperature must be finite"),
            TemperatureError::OutOfRange(v) => write!(f, "temperature {v} is outside [0.0, 2.0]"),
        }
    }
}

impl std::error::Error for TemperatureError {}

impl Temperature {
    /// Check a temperature once, so nothing downstream has to.
    pub fn new(value: f32) -> Result<Self, TemperatureError> {
        if !value.is_finite() {
            Err(TemperatureError::NotFinite)
        } else if !(0.0..=2.0).contains(&value) {
            Err(TemperatureError::OutOfRange(value))
        } else {
            Ok(Self(value))
        }
    }

    /// The validated value.
    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for Temperature {
    /// The API default, `1.0`.
    fn default() -> Self {
        Self(1.0)
    }
}

// ── Structured outputs ───────────────────────────────────────────────────────

/// `text.format` — plain text, or a JSON Schema the answer must satisfy.
///
/// Part of the hashed prefix: OpenAI turns the schema into instructions the
/// model reads, so changing it mid-thread costs the prefix from that point.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum TextFormat {
    /// Ordinary prose. The API default.
    #[default]
    Text,
    /// Structured Outputs: the answer is guaranteed to match `schema`.
    JsonSchema {
        /// Name of the format, `[a-zA-Z0-9_-]{1,64}`.
        name: String,
        /// The schema itself. A `Value` because JSON Schema is an open
        /// language, and a Rust mirror would be a lossier second one.
        schema: Value,
        /// Whether the model must follow the schema exactly. With `true`, only
        /// the strict-mode subset of JSON Schema is allowed.
        strict: bool,
    },
}

impl TextFormat {
    /// A schema the answer is guaranteed to match. Strict is on: the point of
    /// Structured Outputs is not having to validate the answer yourself.
    pub fn json_schema(name: impl Into<String>, schema: Value) -> Self {
        Self::JsonSchema { name: name.into(), schema, strict: true }
    }
}

// ── Compaction ───────────────────────────────────────────────────────────────

/// `context_management` — let OpenAI compact the conversation when it grows past
/// a threshold.
///
/// Compaction replaces earlier context with a shorter rendering, so it changes
/// the prefix and resets reuse from the first changed token. That is why it
/// lives in [`PrefixSettings`] and not beside `max_output_tokens`: it is a
/// property of the prefix, not of one call. Fewer input tokens can still be
/// cheaper than a cache hit on more of them — measure both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compaction {
    /// Token count at which compaction triggers. `None` lets OpenAI choose.
    pub compact_threshold: Option<u32>,
}

// ── Prefix-affecting settings ────────────────────────────────────────────────

/// Every per-call setting that changes the bytes OpenAI hashes.
///
/// One value, held constant across a thread, is a prefix that cannot drift.
/// The grouping is not this crate's invention: it is the table of
/// prefix-affecting settings from OpenAI's prompt-caching guide, made into a
/// type so it cannot be forgotten one field at a time.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefixSettings {
    /// The model and its accepted parameters — including its reasoning effort
    /// and its caching field, both of which differ by generation.
    pub model: Model,
    /// Whether the model may call several tools in one turn. In the prefix
    /// because OpenAI turns it into instructions the model reads.
    pub parallel_tool_calls: bool,
    /// Plain text or a JSON Schema.
    pub text_format: TextFormat,
    /// How long the answer should be.
    pub verbosity: Verbosity,
    /// Compaction, if enabled.
    pub context_management: Option<Compaction>,
    /// Whether to return a summary of the model's reasoning.
    ///
    /// Present here for grouping only — summaries are output, so this does not
    /// affect the prefix.
    pub reasoning_summary: Option<ReasoningSummary>,
}

impl PrefixSettings {
    /// The documented defaults for a model: parallel tool calls on, plain text,
    /// medium verbosity, no compaction, no reasoning summary.
    ///
    /// All of them are emitted explicitly rather than omitted. Relying on
    /// server-side defaults would mean the prefix silently changes the day
    /// OpenAI changes a default — the exact failure the type is built to stop.
    pub fn new(model: impl Into<Model>) -> Self {
        Self {
            model: model.into(),
            parallel_tool_calls: true,
            text_format: TextFormat::Text,
            verbosity: Verbosity::Medium,
            context_management: None,
            reasoning_summary: None,
        }
    }

    /// Require the answer to match a JSON Schema.
    pub fn with_text_format(mut self, format: TextFormat) -> Self {
        self.text_format = format;
        self
    }

    /// Set the answer length.
    pub fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Forbid calling several tools in one turn.
    pub fn with_serial_tool_calls(mut self) -> Self {
        self.parallel_tool_calls = false;
        self
    }

    /// Enable compaction at a token threshold.
    pub fn with_compaction(mut self, compact_threshold: Option<u32>) -> Self {
        self.context_management = Some(Compaction { compact_threshold });
        self
    }

    /// Ask for a summary of the model's reasoning.
    pub fn with_reasoning_summary(mut self, summary: ReasoningSummary) -> Self {
        self.reasoning_summary = Some(summary);
        self
    }

    /// The effort as the shared vocabulary spells it, widened from whichever
    /// per-model effort type the model carries.
    ///
    /// Widening is safe and narrowing is not: every per-model effort is a
    /// [`ReasoningEffort`], but not every `ReasoningEffort` is accepted by every
    /// model. So this direction is a total function and the reverse is not
    /// offered at all.
    pub fn effort(&self) -> ReasoningEffort {
        use crate::model::{EffortMediumToXhigh as Pro, EffortNoneToMax as Six};
        match &self.model {
            Model::Gpt5_6(m) => match m.effort {
                Six::None => ReasoningEffort::None,
                Six::Low => ReasoningEffort::Low,
                Six::Medium => ReasoningEffort::Medium,
                Six::High => ReasoningEffort::High,
                Six::Xhigh => ReasoningEffort::Xhigh,
                Six::Max => ReasoningEffort::Max,
            },
            Model::Gpt5_5(m) => five_effort(m.effort),
            Model::Gpt5_4(m) => five_effort(m.effort),
            Model::Gpt5_5Pro(m) => match m.effort {
                Pro::Medium => ReasoningEffort::Medium,
                Pro::High => ReasoningEffort::High,
                Pro::Xhigh => ReasoningEffort::Xhigh,
            },
        }
    }
}

fn five_effort(effort: crate::model::EffortNoneToXhigh) -> ReasoningEffort {
    use crate::model::EffortNoneToXhigh as Five;
    match effort {
        Five::None => ReasoningEffort::None,
        Five::Low => ReasoningEffort::Low,
        Five::Medium => ReasoningEffort::Medium,
        Five::High => ReasoningEffort::High,
        Five::Xhigh => ReasoningEffort::Xhigh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EffortNoneToMax;

    #[test]
    fn temperature_validates_once() {
        assert_eq!(Temperature::new(f32::NAN), Err(TemperatureError::NotFinite));
        assert_eq!(Temperature::new(f32::INFINITY), Err(TemperatureError::NotFinite));
        assert_eq!(Temperature::new(f32::NEG_INFINITY), Err(TemperatureError::NotFinite));
        assert_eq!(Temperature::new(-0.5), Err(TemperatureError::OutOfRange(-0.5)));
        assert_eq!(Temperature::new(2.5), Err(TemperatureError::OutOfRange(2.5)));
        assert_eq!(Temperature::new(0.0).unwrap().get(), 0.0);
        assert_eq!(Temperature::new(2.0).unwrap().get(), 2.0);
        assert_eq!(Temperature::default().get(), 1.0);
    }

    /// Widening a per-model effort into the shared vocabulary is total; the
    /// reverse would not be, so it is not offered.
    #[test]
    fn effort_widens_from_every_model_type() {
        assert_eq!(PrefixSettings::new(Model::gpt_5_6_sol()).effort(), ReasoningEffort::Medium);
        assert_eq!(
            PrefixSettings::new(Model::gpt_5_6_luna().with_effort(EffortNoneToMax::None)).effort(),
            ReasoningEffort::None
        );
        assert_eq!(
            PrefixSettings::new(Model::gpt_5_6_sol().with_effort(EffortNoneToMax::Max)).effort(),
            ReasoningEffort::Max
        );
        assert_eq!(PrefixSettings::new(Model::gpt_5_5()).effort(), ReasoningEffort::Medium);
        assert_eq!(PrefixSettings::new(Model::gpt_5_5_pro()).effort(), ReasoningEffort::High);
        assert_eq!(PrefixSettings::new(Model::gpt_5_4()).effort(), ReasoningEffort::None);
    }

    #[test]
    fn the_documented_defaults_are_stated_not_omitted() {
        let prefix = PrefixSettings::new(Model::gpt_5_6_sol());
        assert!(prefix.parallel_tool_calls);
        assert_eq!(prefix.text_format, TextFormat::Text);
        assert_eq!(prefix.verbosity, Verbosity::Medium);
        assert_eq!(prefix.context_management, None);
        assert_eq!(prefix.reasoning_summary, None);
    }

    #[test]
    fn a_json_schema_format_is_strict_by_default() {
        let schema = serde_json::json!({"type": "object"});
        assert_eq!(
            TextFormat::json_schema("verdict", schema.clone()),
            TextFormat::JsonSchema { name: "verdict".into(), schema, strict: true }
        );
    }
}
