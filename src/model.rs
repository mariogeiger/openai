//! One type per accepted-parameter set, plus the facts each model carries.
//!
//! Models differ in three ways that a single shared struct would blur:
//!
//! * **Which reasoning efforts they accept.** GPT-5.6 adds `max`; GPT-5.5 Pro
//!   refuses everything below `medium`.
//! * **Which field controls cache lifetime.** GPT-5.6 and later use
//!   `prompt_cache_options.ttl`; earlier models use `prompt_cache_retention`.
//!   These are different fields with different value sets, and sending the
//!   wrong one is a 400.
//! * **Whether explicit cache breakpoints exist at all.** Only GPT-5.6 and
//!   later honor them.
//!
//! So the type boundary follows the parameter set, not the model name: the
//! three GPT-5.6 tiers accept exactly the same parameters and share one type
//! that carries a [`Gpt5_6Tier`], while GPT-5.5, GPT-5.5 Pro, and GPT-5.4 each
//! get their own type because each accepts something the others do not.

#![allow(non_camel_case_types)]

use crate::values::{CacheMode, CacheRetention, CacheTtl, ReasoningContext, ReasoningMode, api_enum};

api_enum! {
    /// The reasoning efforts GPT-5.6 models accept. `max` exists only here.
    EffortNoneToMax {
        /// No reasoning tokens.
        None => "none",
        /// Modest reasoning.
        Low => "low",
        /// The GPT-5.6 default.
        Medium => "medium",
        /// Deeper reasoning.
        High => "high",
        /// Long runs.
        Xhigh => "xhigh",
        /// The most reasoning available.
        Max => "max",
    }
}

api_enum! {
    /// The reasoning efforts GPT-5.5 and GPT-5.4 accept. `max` is absent
    /// because those models refuse it, so it cannot be written.
    ///
    /// GPT-5.6 accepts `max`, and so takes a different effort type. Handing
    /// GPT-5.6's effort to GPT-5.5 does not compile:
    ///
    /// ```compile_fail
    /// use openai::model::{EffortNoneToMax, Model};
    /// let _ = Model::gpt_5_5().with_effort(EffortNoneToMax::Max);
    /// ```
    ///
    /// Nor does the reverse: GPT-5.6 will not take GPT-5.5's effort type, even
    /// for a level both models accept. The types are the accepted sets, and two
    /// different sets are two different types.
    ///
    /// ```compile_fail
    /// use openai::model::{EffortNoneToXhigh, Model};
    /// let _ = Model::gpt_5_6_sol().with_effort(EffortNoneToXhigh::High);
    /// ```
    EffortNoneToXhigh {
        /// No reasoning tokens.
        None => "none",
        /// Modest reasoning.
        Low => "low",
        /// Balanced reasoning.
        Medium => "medium",
        /// Deeper reasoning.
        High => "high",
        /// Long runs.
        Xhigh => "xhigh",
    }
}

api_enum! {
    /// The reasoning efforts GPT-5.5 Pro accepts. Pro always reasons, so
    /// `none` and `low` are absent rather than rejected at runtime.
    ///
    /// There is no `EffortMediumToXhigh::None` to write, so "Pro without
    /// reasoning" is not a runtime error — it is not a sentence:
    ///
    /// ```compile_fail
    /// use openai::model::EffortMediumToXhigh;
    /// let _ = EffortMediumToXhigh::None;
    /// ```
    EffortMediumToXhigh {
        /// The least reasoning Pro offers.
        Medium => "medium",
        /// The Pro default.
        High => "high",
        /// Long runs.
        Xhigh => "xhigh",
    }
}

api_enum! {
    /// `prompt_cache_retention` on models where only extended retention is
    /// supported. One variant, because `in_memory` is a 400 on these models.
    ///
    /// ```compile_fail
    /// use openai::model::ExtendedRetentionOnly;
    /// let _ = ExtendedRetentionOnly::InMemory;
    /// ```
    ExtendedRetentionOnly {
        /// Up to 24 hours.
        TwentyFourHours => "24h",
    }
}

/// Which GPT-5.6 tier to route to. The tiers accept identical parameters and
/// differ only in price, so tier is a value on one model type rather than three
/// types that would be textually identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gpt5_6Tier {
    /// Flagship: the `gpt-5.6` alias points here.
    Sol,
    /// Balanced intelligence and cost.
    Terra,
    /// Cost-sensitive, high-volume workloads.
    Luna,
}

/// `prompt_cache_options` — the GPT-5.6 caching control.
///
/// Only GPT-5.6 and later carry this. Reaching for it on an earlier model does
/// not compile, because [`Gpt5_5`] has no such field — it has
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
pub struct Gpt5_6Caching {
    /// Whether OpenAI adds its own breakpoint at the end of the latest
    /// eligible message.
    pub mode: CacheMode,
    /// Minimum lifetime of every breakpoint this request writes.
    pub ttl: CacheTtl,
}

impl Default for Gpt5_6Caching {
    /// The API's own defaults: an implicit breakpoint, 30-minute minimum life.
    fn default() -> Self {
        Self { mode: CacheMode::Implicit, ttl: CacheTtl::ThirtyMinutes }
    }
}

/// GPT-5.6 Sol, Terra, or Luna.
///
/// Carries `reasoning.mode` and `reasoning.context`, which earlier models do
/// not accept, and [`Gpt5_6Caching`] rather than a retention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt5_6 {
    /// Which tier to route to.
    pub tier: Gpt5_6Tier,
    /// How much the model reasons before answering.
    pub effort: EffortNoneToMax,
    /// Standard or Pro execution, independent of effort.
    pub mode: ReasoningMode,
    /// Which earlier reasoning is rendered into this turn.
    pub reasoning_context: ReasoningContext,
    /// Breakpoint mode and minimum cache lifetime.
    pub caching: Gpt5_6Caching,
}

impl Gpt5_6 {
    /// The documented runtime defaults, stated explicitly: `medium` effort,
    /// `standard` mode, `all_turns` context, implicit caching at `30m`.
    ///
    /// Emitting them rather than relying on omission keeps the request body a
    /// complete record of the prefix the model will see — and keeps that prefix
    /// stable if OpenAI's defaults ever shift.
    pub fn new(tier: Gpt5_6Tier) -> Self {
        Self {
            tier,
            effort: EffortNoneToMax::Medium,
            mode: ReasoningMode::Standard,
            reasoning_context: ReasoningContext::AllTurns,
            caching: Gpt5_6Caching::default(),
        }
    }

    /// Set the reasoning effort.
    pub fn with_effort(mut self, effort: EffortNoneToMax) -> Self {
        self.effort = effort;
        self
    }

    /// Select standard or Pro execution.
    pub fn with_mode(mut self, mode: ReasoningMode) -> Self {
        self.mode = mode;
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
}

/// GPT-5.5. No explicit breakpoints, no `max` effort, and extended retention is
/// the only retention this model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt5_5 {
    /// How much the model reasons before answering.
    pub effort: EffortNoneToXhigh,
    /// Maximum cache retention. `in_memory` is refused on this model, so it is
    /// not expressible.
    pub retention: ExtendedRetentionOnly,
}

impl Default for Gpt5_5 {
    /// `medium` effort — the documented default for this model.
    fn default() -> Self {
        Self { effort: EffortNoneToXhigh::Medium, retention: ExtendedRetentionOnly::TwentyFourHours }
    }
}

impl Gpt5_5 {
    /// Default parameters. Chain `with_*` to change them.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reasoning effort.
    pub fn with_effort(mut self, effort: EffortNoneToXhigh) -> Self {
        self.effort = effort;
        self
    }
}

/// GPT-5.5 Pro. Responses-only, always reasoning, and billed with no cached
/// discount — see [`Pricing`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt5_5Pro {
    /// How much the model reasons. `none` and `low` do not exist here.
    pub effort: EffortMediumToXhigh,
    /// Maximum cache retention.
    pub retention: ExtendedRetentionOnly,
}

impl Default for Gpt5_5Pro {
    /// `high` effort — the documented default for Pro.
    fn default() -> Self {
        Self { effort: EffortMediumToXhigh::High, retention: ExtendedRetentionOnly::TwentyFourHours }
    }
}

impl Gpt5_5Pro {
    /// Default parameters. Chain `with_*` to change them.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reasoning effort.
    pub fn with_effort(mut self, effort: EffortMediumToXhigh) -> Self {
        self.effort = effort;
        self
    }
}

/// GPT-5.4. The one modeled generation that accepts both retention policies,
/// which is why retention is a full [`CacheRetention`] here and narrower
/// elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt5_4 {
    /// How much the model reasons before answering.
    pub effort: EffortNoneToXhigh,
    /// Maximum cache retention. Both documented values are accepted.
    pub retention: CacheRetention,
}

impl Default for Gpt5_4 {
    /// `none` effort — the documented default for this model, unlike GPT-5.5.
    /// Retention defaults to `24h`, which is OpenAI's default for organizations
    /// without Zero Data Retention; set `in_memory` explicitly under ZDR.
    fn default() -> Self {
        Self { effort: EffortNoneToXhigh::None, retention: CacheRetention::TwentyFourHours }
    }
}

impl Gpt5_4 {
    /// Default parameters. Chain `with_*` to change them.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reasoning effort.
    pub fn with_effort(mut self, effort: EffortNoneToXhigh) -> Self {
        self.effort = effort;
        self
    }

    /// Set the maximum retention policy.
    pub fn with_retention(mut self, retention: CacheRetention) -> Self {
        self.retention = retention;
        self
    }
}

/// A model's identity, without per-call parameters. Everything here is a
/// documented fact about the model rather than something the caller chooses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    /// `gpt-5.6-sol`.
    Gpt5_6Sol,
    /// `gpt-5.6-terra`.
    Gpt5_6Terra,
    /// `gpt-5.6-luna`.
    Gpt5_6Luna,
    /// `gpt-5.5`.
    Gpt5_5,
    /// `gpt-5.5-pro`.
    Gpt5_5Pro,
    /// `gpt-5.4`.
    Gpt5_4,
}

/// List price, as an exact integer count of nanodollars per token.
///
/// Nanodollars are chosen so no rounding is ever needed: every published
/// per-million-token price, and every cache multiplier applied to it, lands on
/// a whole nanodollar. Cost arithmetic is therefore integer arithmetic — see
/// [`crate::usage::Usage::cost_nanodollars`].
///
/// Every field is a real rate, never an absent one. A model with no cached
/// discount reports its full input rate as the cached rate; a model with no
/// cache-write surcharge reports its full input rate as the write rate. That
/// keeps the cost formula uniform instead of branching per model.
///
/// Not represented: the 2× input / 1.5× output surcharge above 272K input
/// tokens, batch and flex discounts, regional-processing uplift, and the fact
/// that promotional prices expire. Verify against the pricing page before
/// billing anyone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pricing {
    /// Ordinary, uncached input tokens.
    pub input_nanodollars_per_token: u64,
    /// Tokens served from the cache.
    pub cached_input_nanodollars_per_token: u64,
    /// Tokens written to the cache.
    pub cache_write_nanodollars_per_token: u64,
    /// Generated tokens, reasoning tokens included.
    pub output_nanodollars_per_token: u64,
}

/// A calendar month. Knowledge cutoffs are published to the day, but the day is
/// not a fact anyone should depend on, so this crate stops at the month.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct YearMonth {
    /// Four-digit year.
    pub year: u16,
    /// Month, 1 through 12.
    pub month: u8,
}

impl ModelId {
    /// The string sent in the `model` field.
    pub fn api_id(self) -> &'static str {
        match self {
            ModelId::Gpt5_6Sol => "gpt-5.6-sol",
            ModelId::Gpt5_6Terra => "gpt-5.6-terra",
            ModelId::Gpt5_6Luna => "gpt-5.6-luna",
            ModelId::Gpt5_5 => "gpt-5.5",
            ModelId::Gpt5_5Pro => "gpt-5.5-pro",
            ModelId::Gpt5_4 => "gpt-5.4",
        }
    }

    /// Whether `prompt_cache_breakpoint` does anything on this model.
    ///
    /// GPT-5.6 introduced explicit breakpoints; earlier models place implicit
    /// ones only. [`crate::request::Request::new`] refuses a context holding
    /// explicit breakpoints together with a model that ignores them, because
    /// the alternative is paying for a prefix nobody can reuse.
    pub fn supports_explicit_cache_breakpoints(self) -> bool {
        matches!(self, ModelId::Gpt5_6Sol | ModelId::Gpt5_6Terra | ModelId::Gpt5_6Luna)
    }

    /// Shortest visible prefix this model will cache, in tokens.
    ///
    /// Below it, caching is a silent no-op: no error, no cache write, and
    /// `cached_tokens` stays zero. Hidden OpenAI content does not count toward
    /// the threshold. A prefix a little under the threshold can cost *more*
    /// than one padded up to it — the guide calls this the cost trap.
    pub fn min_cacheable_prefix_tokens(self) -> u32 {
        if self.supports_explicit_cache_breakpoints() { 1_024 } else { 2_048 }
    }

    /// Total context window in tokens, shared by input and output.
    pub fn context_window_tokens(self) -> u32 {
        1_050_000
    }

    /// Largest `max_output_tokens` this model accepts, reasoning tokens
    /// included. [`crate::request::Request::new`] rejects anything above it.
    pub fn max_output_tokens(self) -> u32 {
        128_000
    }

    /// The month through which the model's knowledge is reliable.
    pub fn knowledge_cutoff(self) -> YearMonth {
        let (year, month) = match self {
            ModelId::Gpt5_6Sol | ModelId::Gpt5_6Terra | ModelId::Gpt5_6Luna => (2026, 2),
            ModelId::Gpt5_5 | ModelId::Gpt5_5Pro => (2025, 12),
            ModelId::Gpt5_4 => (2025, 8),
        };
        YearMonth { year, month }
    }

    /// List price per token (see [`Pricing`] for what is not represented).
    pub fn pricing(self) -> Pricing {
        let (input, cached, write, output) = match self {
            // GPT-5.6: cached reads at 0.1x, cache writes at 1.25x.
            ModelId::Gpt5_6Sol => (4_000, 400, 5_000, 20_000),
            ModelId::Gpt5_6Terra => (2_000, 200, 2_500, 12_000),
            ModelId::Gpt5_6Luna => (200, 20, 250, 1_200),
            // Earlier generations levy no cache-write charge, so writes bill at
            // the ordinary input rate.
            ModelId::Gpt5_5 => (5_000, 500, 5_000, 30_000),
            // Pro offers no cached discount: reads bill at the full rate.
            ModelId::Gpt5_5Pro => (30_000, 30_000, 30_000, 180_000),
            ModelId::Gpt5_4 => (2_500, 250, 2_500, 15_000),
        };
        Pricing {
            input_nanodollars_per_token: input,
            cached_input_nanodollars_per_token: cached,
            cache_write_nanodollars_per_token: write,
            output_nanodollars_per_token: output,
        }
    }
}

/// A model together with the parameters that model accepts.
///
/// Every variant's payload is a distinct type, so a parameter one model refuses
/// is not merely rejected at runtime — it does not exist on that variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// A GPT-5.6 tier.
    Gpt5_6(Gpt5_6),
    /// GPT-5.5.
    Gpt5_5(Gpt5_5),
    /// GPT-5.5 Pro.
    Gpt5_5Pro(Gpt5_5Pro),
    /// GPT-5.4.
    Gpt5_4(Gpt5_4),
}

impl Model {
    /// Identity without per-call parameters.
    pub fn id(&self) -> ModelId {
        match self {
            Model::Gpt5_6(m) => match m.tier {
                Gpt5_6Tier::Sol => ModelId::Gpt5_6Sol,
                Gpt5_6Tier::Terra => ModelId::Gpt5_6Terra,
                Gpt5_6Tier::Luna => ModelId::Gpt5_6Luna,
            },
            Model::Gpt5_5(_) => ModelId::Gpt5_5,
            Model::Gpt5_5Pro(_) => ModelId::Gpt5_5Pro,
            Model::Gpt5_4(_) => ModelId::Gpt5_4,
        }
    }

    /// The string sent in the `model` field.
    pub fn api_id(&self) -> &'static str {
        self.id().api_id()
    }

    /// GPT-5.6 Sol with documented defaults.
    pub fn gpt_5_6_sol() -> Gpt5_6 {
        Gpt5_6::new(Gpt5_6Tier::Sol)
    }

    /// GPT-5.6 Terra with documented defaults.
    pub fn gpt_5_6_terra() -> Gpt5_6 {
        Gpt5_6::new(Gpt5_6Tier::Terra)
    }

    /// GPT-5.6 Luna with documented defaults.
    pub fn gpt_5_6_luna() -> Gpt5_6 {
        Gpt5_6::new(Gpt5_6Tier::Luna)
    }

    /// GPT-5.5 with documented defaults.
    pub fn gpt_5_5() -> Gpt5_5 {
        Gpt5_5::default()
    }

    /// GPT-5.5 Pro with documented defaults.
    pub fn gpt_5_5_pro() -> Gpt5_5Pro {
        Gpt5_5Pro::default()
    }

    /// GPT-5.4 with documented defaults.
    pub fn gpt_5_4() -> Gpt5_4 {
        Gpt5_4::default()
    }
}

impl From<Gpt5_6> for Model {
    fn from(m: Gpt5_6) -> Self {
        Model::Gpt5_6(m)
    }
}

impl From<Gpt5_5> for Model {
    fn from(m: Gpt5_5) -> Self {
        Model::Gpt5_5(m)
    }
}

impl From<Gpt5_5Pro> for Model {
    fn from(m: Gpt5_5Pro) -> Self {
        Model::Gpt5_5Pro(m)
    }
}

impl From<Gpt5_4> for Model {
    fn from(m: Gpt5_4) -> Self {
        Model::Gpt5_4(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_ids_match_the_model_pages() {
        assert_eq!(ModelId::Gpt5_6Sol.api_id(), "gpt-5.6-sol");
        assert_eq!(ModelId::Gpt5_6Terra.api_id(), "gpt-5.6-terra");
        assert_eq!(ModelId::Gpt5_6Luna.api_id(), "gpt-5.6-luna");
        assert_eq!(ModelId::Gpt5_5.api_id(), "gpt-5.5");
        assert_eq!(ModelId::Gpt5_5Pro.api_id(), "gpt-5.5-pro");
        assert_eq!(ModelId::Gpt5_4.api_id(), "gpt-5.4");
    }

    #[test]
    fn tier_choice_selects_the_identity() {
        assert_eq!(Model::from(Model::gpt_5_6_sol()).id(), ModelId::Gpt5_6Sol);
        assert_eq!(Model::from(Model::gpt_5_6_terra()).id(), ModelId::Gpt5_6Terra);
        assert_eq!(Model::from(Model::gpt_5_6_luna()).id(), ModelId::Gpt5_6Luna);
        assert_eq!(Model::from(Model::gpt_5_5()).api_id(), "gpt-5.5");
    }

    #[test]
    fn explicit_breakpoints_are_a_gpt_5_6_feature() {
        assert!(ModelId::Gpt5_6Sol.supports_explicit_cache_breakpoints());
        assert!(ModelId::Gpt5_6Luna.supports_explicit_cache_breakpoints());
        assert!(!ModelId::Gpt5_5.supports_explicit_cache_breakpoints());
        assert!(!ModelId::Gpt5_5Pro.supports_explicit_cache_breakpoints());
        assert!(!ModelId::Gpt5_4.supports_explicit_cache_breakpoints());
    }

    /// The minimum halves at GPT-5.6, and it tracks the breakpoint feature
    /// exactly — both changed in the same generation.
    #[test]
    fn minimum_cacheable_prefix_follows_the_generation() {
        assert_eq!(ModelId::Gpt5_6Sol.min_cacheable_prefix_tokens(), 1_024);
        assert_eq!(ModelId::Gpt5_5.min_cacheable_prefix_tokens(), 2_048);
        assert_eq!(ModelId::Gpt5_4.min_cacheable_prefix_tokens(), 2_048);
    }

    #[test]
    fn documented_facts() {
        assert_eq!(ModelId::Gpt5_6Sol.max_output_tokens(), 128_000);
        assert_eq!(ModelId::Gpt5_6Sol.context_window_tokens(), 1_050_000);
        assert_eq!(ModelId::Gpt5_6Sol.knowledge_cutoff(), YearMonth { year: 2026, month: 2 });
        assert_eq!(ModelId::Gpt5_4.knowledge_cutoff(), YearMonth { year: 2025, month: 8 });
        assert!(ModelId::Gpt5_4.knowledge_cutoff() < ModelId::Gpt5_6Sol.knowledge_cutoff());
    }

    /// A cache read costs a tenth of ordinary input, and a write costs a
    /// quarter more, on every GPT-5.6 tier. Checking the ratios rather than the
    /// numbers is what catches a mistyped price.
    #[test]
    fn gpt_5_6_cache_multipliers_hold_exactly() {
        for id in [ModelId::Gpt5_6Sol, ModelId::Gpt5_6Terra, ModelId::Gpt5_6Luna] {
            let p = id.pricing();
            assert_eq!(p.cached_input_nanodollars_per_token * 10, p.input_nanodollars_per_token);
            assert_eq!(p.cache_write_nanodollars_per_token * 4, p.input_nanodollars_per_token * 5);
        }
    }

    /// Earlier models levy no write surcharge, and Pro grants no read discount.
    /// Both are expressed as a rate equal to the ordinary input rate.
    #[test]
    fn earlier_generations_price_caching_differently() {
        let five_five = ModelId::Gpt5_5.pricing();
        assert_eq!(five_five.cache_write_nanodollars_per_token, five_five.input_nanodollars_per_token);
        assert_eq!(five_five.cached_input_nanodollars_per_token * 10, five_five.input_nanodollars_per_token);

        let pro = ModelId::Gpt5_5Pro.pricing();
        assert_eq!(pro.cached_input_nanodollars_per_token, pro.input_nanodollars_per_token);
        assert_eq!(pro.cache_write_nanodollars_per_token, pro.input_nanodollars_per_token);
    }

    #[test]
    fn documented_default_effort_differs_per_model() {
        assert_eq!(Model::gpt_5_6_sol().effort, EffortNoneToMax::Medium);
        assert_eq!(Model::gpt_5_5().effort, EffortNoneToXhigh::Medium);
        assert_eq!(Model::gpt_5_5_pro().effort, EffortMediumToXhigh::High);
        assert_eq!(Model::gpt_5_4().effort, EffortNoneToXhigh::None);
    }
}
