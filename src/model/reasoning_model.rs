//! The parameter set GPT-5.6 introduced, which every GPT model since shares.
//!
//! GPT-5.6, GPT-6 and GPT-6.1 take the same request fields: a reasoning effort,
//! `reasoning.mode`, `reasoning.context`, and `prompt_cache_options`. They differ
//! in one thing a type can see, the range of efforts. Astra and GPT-6.1 Sol
//! refuse `none`; the rest accept it. So the parameter set is one generic
//! struct over the effort it takes, and the models that share a range are the
//! tiers of one alias. Which model a value routes to is data (a tier), which
//! range it takes is the type, and a model that arrives with a known range is a
//! new tier, not a new struct.

use super::{EffortLowToMax, EffortNoneToMax, ModelId};
use crate::values::{CacheMode, CacheTtl, ReasoningContext, ReasoningMode};

/// `prompt_cache_options` — the caching control of GPT-5.6 and later.
///
/// Only those models carry this. Reaching for it on an earlier model does not
/// compile, because [`Gpt5_5`](super::Gpt5_5) has no such field — it has
/// `prompt_cache_retention` instead, and the two can never be sent together:
///
/// ```compile_fail
/// use openai::model::Model;
/// let _ = Model::gpt_5_5().caching;
/// ```
///
/// [`mode`](Self::mode) is not cosmetic: `Implicit` spends one of the four
/// per-request cache-write slots on a breakpoint OpenAI places for you, leaving
/// three for your own. [`crate::request::Request::new`] enforces that budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheOptions {
    /// Whether OpenAI adds its own breakpoint at the end of the latest
    /// eligible message.
    pub mode: CacheMode,
    /// Minimum lifetime of every breakpoint this request writes.
    pub ttl: CacheTtl,
}

impl Default for CacheOptions {
    /// The API's own documented defaults: `mode` "Defaults to `implicit`", and
    /// `ttl` "Defaults to `30m`". Both are plain fields carrying the documented
    /// value, so the request states what the model sees rather than inheriting
    /// a server-side default that could shift.
    fn default() -> Self {
        Self { mode: CacheMode::Implicit, ttl: CacheTtl::ThirtyMinutes }
    }
}

/// A model that takes the [`ReasoningModel`] parameter set, named by the value
/// that selects it.
pub trait ModelTier: Copy {
    /// The model this tier routes to.
    fn model_id(self) -> ModelId;
}

/// Which model of the none-to-max parameter set to route to.
///
/// These accept identical parameters and differ in price, limits and
/// capability, which is why they are values of one type rather than types that
/// would be textually identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GptNoneToMaxTier {
    /// GPT-5.6 Sol: the `gpt-5.6` alias points here.
    Sol5_6,
    /// GPT-5.6 Terra: balanced intelligence and cost.
    Terra5_6,
    /// GPT-5.6 Luna: cost-sensitive, high-volume workloads.
    Luna5_6,
    /// GPT-6 Sol.
    Sol6,
    /// GPT-6.1 Terra: visual reasoning and precise computer use, fast and
    /// efficient.
    Terra6_1,
    /// GPT-6 Luna: the most efficient, for focused high-volume tasks.
    Luna6,
}

impl ModelTier for GptNoneToMaxTier {
    fn model_id(self) -> ModelId {
        match self {
            GptNoneToMaxTier::Sol5_6 => ModelId::Gpt5_6Sol,
            GptNoneToMaxTier::Terra5_6 => ModelId::Gpt5_6Terra,
            GptNoneToMaxTier::Luna5_6 => ModelId::Gpt5_6Luna,
            GptNoneToMaxTier::Sol6 => ModelId::Gpt6Sol,
            GptNoneToMaxTier::Terra6_1 => ModelId::Gpt6_1Terra,
            GptNoneToMaxTier::Luna6 => ModelId::Gpt6Luna,
        }
    }
}

/// Which model of the low-to-max parameter set to route to: the GPT-6 models
/// that refuse `none`.
///
/// Astra and GPT-6.1 Sol accept identical parameters and differ in price and
/// capability, so which one is a value rather than two textually identical
/// types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gpt6LowToMaxTier {
    /// GPT-6 Astra, the most capable.
    Astra,
    /// GPT-6.1 Sol, near Astra at a lower cost.
    Sol6_1,
}

impl ModelTier for Gpt6LowToMaxTier {
    fn model_id(self) -> ModelId {
        match self {
            Gpt6LowToMaxTier::Astra => ModelId::Gpt6Astra,
            Gpt6LowToMaxTier::Sol6_1 => ModelId::Gpt6_1Sol,
        }
    }
}

/// The parameter set of GPT-5.6 and later, over the efforts the model takes.
///
/// Use the aliases [`GptNoneToMax`] and [`Gpt6LowToMax`]: they fix the two
/// pairs of tier and effort the API has, and the effort type is what keeps an
/// effort a model refuses from being written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningModel<Tier, Effort> {
    /// Which model to route to.
    pub tier: Tier,
    /// How much the model reasons before answering, when the caller says.
    ///
    /// `None` means the field is not sent, and the model reasons at whatever
    /// level it documents for itself — [`ModelId::default_effort`] states that
    /// level without imposing it. The API documents no default for
    /// `reasoning.effort` and reports `"effort": null` on a response that never
    /// carried one, so "told a level" and "not told" are two runtime states,
    /// not one value and its default.
    pub effort: Option<Effort>,
    /// Standard or Pro execution, independent of effort. `None` is not sent:
    /// the API documents no default for `reasoning.mode`.
    pub mode: Option<ReasoningMode>,
    /// Which earlier reasoning is rendered into this turn. Always sent, because
    /// the API documents a default — `auto`, which the model then resolves.
    pub reasoning_context: ReasoningContext,
    /// Breakpoint mode and minimum cache lifetime.
    pub caching: CacheOptions,
}

/// GPT-5.6, GPT-6 Sol, GPT-6.1 Terra and GPT-6 Luna: `none` through `max`.
pub type GptNoneToMax = ReasoningModel<GptNoneToMaxTier, EffortNoneToMax>;

/// GPT-6 Astra and GPT-6.1 Sol: `low` through `max`.
///
/// The effort type starts at `low`, so `none` cannot reach the wire.
pub type Gpt6LowToMax = ReasoningModel<Gpt6LowToMaxTier, EffortLowToMax>;

impl<Tier: ModelTier, Effort> ReasoningModel<Tier, Effort> {
    /// Every documented default stated explicitly, and nothing else chosen.
    ///
    /// `reasoning.context` is `auto` and the caching fields are `implicit` /
    /// `30m`, because the API documents those defaults and emitting them keeps
    /// the request a complete record of the prefix the model will see — stable
    /// even if OpenAI's defaults shift. `reasoning.effort` and `reasoning.mode`
    /// have no documented default, so they start absent: the crate does not
    /// decide how hard the model thinks.
    pub fn new(tier: Tier) -> Self {
        Self {
            tier,
            effort: None,
            mode: None,
            reasoning_context: ReasoningContext::Auto,
            caching: CacheOptions::default(),
        }
    }

    /// The model this value routes to.
    pub fn model_id(&self) -> ModelId {
        self.tier.model_id()
    }

    /// Set the reasoning effort.
    pub fn with_effort(mut self, effort: Effort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Leave `reasoning.effort` unsent, so the model reasons at its own
    /// documented level. See [`ModelId::default_effort`] for what that is.
    pub fn without_effort(mut self) -> Self {
        self.effort = None;
        self
    }

    /// Select standard or Pro execution.
    pub fn with_mode(mut self, mode: ReasoningMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Choose which earlier reasoning the model may render.
    pub fn with_reasoning_context(mut self, context: ReasoningContext) -> Self {
        self.reasoning_context = context;
        self
    }

    /// Turn off OpenAI's implicit breakpoint, freeing all four write slots for
    /// breakpoints you place yourself. With no explicit breakpoint placed, a
    /// request in this mode uses no caching at all.
    pub fn with_explicit_cache_only(mut self) -> Self {
        self.caching.mode = CacheMode::Explicit;
        self
    }

    /// Choose the breakpoint mode and minimum cache lifetime outright.
    ///
    /// Caching is the caller's decision, so the whole object is settable and
    /// not only the one corner [`Self::with_explicit_cache_only`] reaches.
    pub fn with_caching(mut self, caching: CacheOptions) -> Self {
        self.caching = caching;
        self
    }
}
