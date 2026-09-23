//! GPT-6 Sol request parameters and cache controls.

use super::{EffortNoneToMax, Gpt5_6Caching};
use crate::values::{CacheMode, ReasoningContext, ReasoningMode};

/// The cache controls GPT-6 Sol shares with GPT-5.6.
///
/// The alias names the model-facing contract while keeping one representation
/// for the identical `prompt_cache_options` object.
pub type Gpt6SolCaching = Gpt5_6Caching;

/// GPT-6 Sol's accepted parameter set.
///
/// Sol accepts `none` through `max` and shares the explicit breakpoint and
/// cache-TTL controls of GPT-5.6. Its identity keeps its prices and limits
/// separate from that generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt6Sol {
    /// How much the model reasons before answering, when the caller says.
    pub effort: Option<EffortNoneToMax>,
    /// Standard or Pro execution, independent of effort.
    pub mode: Option<ReasoningMode>,
    /// Which earlier reasoning is rendered into this turn.
    pub reasoning_context: ReasoningContext,
    /// Breakpoint mode and minimum cache lifetime.
    pub caching: Gpt6SolCaching,
}

impl Default for Gpt6Sol {
    /// Documented defaults are emitted; fields with no documented default stay
    /// absent so the crate does not decide how Sol thinks.
    fn default() -> Self {
        Self { effort: None, mode: None, reasoning_context: ReasoningContext::Auto, caching: Gpt6SolCaching::default() }
    }
}

impl Gpt6Sol {
    /// Sol with documented defaults and no invented reasoning choice.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a reasoning level Sol accepts.
    pub fn with_effort(mut self, effort: EffortNoneToMax) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Leave `reasoning.effort` unsent.
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

    /// Turn off OpenAI's implicit breakpoint, leaving all four slots available
    /// for explicit breakpoints.
    pub fn with_explicit_cache_only(mut self) -> Self {
        self.caching.mode = CacheMode::Explicit;
        self
    }

    /// Choose the breakpoint mode and minimum cache lifetime outright.
    pub fn with_caching(mut self, caching: Gpt6SolCaching) -> Self {
        self.caching = caching;
        self
    }
}
