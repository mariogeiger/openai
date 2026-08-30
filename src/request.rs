//! The `POST /v1/responses` body, and the per-call settings that do not touch
//! the hashed prefix.
//!
//! Every public field on [`Request`] is one OpenAI does *not* hash —
//! `tool_choice`, `prompt_cache_key`, `max_output_tokens`, `stream`, `store`,
//! `instructions` — so any of them may change from request to request at no
//! cache cost. Anything that would change the prefix lives in
//! [`crate::prefix::PrefixSettings`] or on the [`crate::context::Context`].
//!
//! [`Request::new`] is the only construction path, so the two checks the type
//! system cannot express — the cache-write budget, and whether the model honors
//! explicit breakpoints at all — cannot be skipped.

use crate::context::{CACHE_WRITE_SLOTS, Context};
use crate::model::{Gpt5_6, Model};
use crate::prefix::PrefixSettings;
use crate::tools::ToolChoice;
use crate::values::{CacheMode, Include, Metadata, ServiceTier, Truncation};

// ── Instructions ─────────────────────────────────────────────────────────────

/// Top-level `instructions`, which **cannot carry a cache breakpoint**.
///
/// A newtype rather than a bare `String`, because the name is the warning. The
/// API accepts this field but refuses a breakpoint on it, so instructions you
/// intend to reuse must instead live in an `input_text` block inside a developer
/// message — see
/// [`Context::push_anchored_developer_text`](crate::context::Context::push_anchored_developer_text).
///
/// It remains useful for the opposite case: per-request instructions you
/// *want* outside the reusable prefix, such as a timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UncacheableInstructions(pub String);

impl UncacheableInstructions {
    /// Instructions that will not be cached, and are not meant to be.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }
}

// ── Cache write budget ───────────────────────────────────────────────────────

/// How many of the four per-request cache writes the caller's own breakpoints
/// may use.
///
/// The arithmetic is short and easy to get wrong. A request writes at most four
/// cache entries. Under [`CacheMode::Implicit`] OpenAI spends one on a
/// breakpoint it places itself, leaving three. Under [`CacheMode::Explicit`] all
/// four are the caller's — but with none placed, the request caches nothing at
/// all. On models older than GPT-5.6 explicit breakpoints do not exist, so the
/// budget is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheWriteBudget {
    /// Explicit breakpoints this request may carry.
    pub explicit_slots: usize,
    /// Whether OpenAI adds one of its own, spending a slot.
    pub implicit_breakpoint: bool,
}

impl CacheWriteBudget {
    /// The budget a model and its caching mode allow.
    pub fn of(model: &Model) -> Self {
        match model {
            Model::Gpt5_6(Gpt5_6 { caching, .. }) => match caching.mode {
                CacheMode::Implicit => Self { explicit_slots: CACHE_WRITE_SLOTS - 1, implicit_breakpoint: true },
                CacheMode::Explicit => Self { explicit_slots: CACHE_WRITE_SLOTS, implicit_breakpoint: false },
            },
            // Earlier generations place implicit breakpoints only.
            Model::Gpt5_5(_) | Model::Gpt5_5Pro(_) | Model::Gpt5_4(_) => {
                Self { explicit_slots: 0, implicit_breakpoint: true }
            }
        }
    }
}

// ── Request ──────────────────────────────────────────────────────────────────

/// Why a request was refused before it could be sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestError {
    /// `max_output_tokens` outside `1..=` the model's maximum. Reasoning tokens
    /// count toward it, so a small value can be spent entirely on reasoning and
    /// return an `incomplete` response with nothing visible in it.
    MaxOutputTokensOutOfRange {
        /// What was asked for.
        max_output_tokens: u32,
        /// What the model allows.
        model_maximum: u32,
    },
    /// More explicit breakpoints than this model and caching mode can write.
    ///
    /// Under `implicit` mode OpenAI's own breakpoint takes one of the four
    /// slots, so a fourth explicit breakpoint has nowhere to go. Refused here
    /// rather than left to the server, which would decide *for* you which
    /// breakpoints to drop.
    TooManyExplicitBreakpoints {
        /// How many the context holds.
        placed: usize,
        /// How many this model and mode allow.
        budget: usize,
    },
    /// Explicit breakpoints on a model that ignores them. The context would
    /// carry markers nobody reads, and the caller would believe a prefix was
    /// reusable when it is not.
    ExplicitBreakpointsUnsupported {
        /// The model's wire identifier.
        model: &'static str,
        /// How many the context holds.
        placed: usize,
    },
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::MaxOutputTokensOutOfRange { max_output_tokens, model_maximum } => {
                write!(f, "max_output_tokens ({max_output_tokens}) must be in 1..={model_maximum} for this model")
            }
            RequestError::TooManyExplicitBreakpoints { placed, budget } => {
                write!(f, "{placed} explicit cache breakpoints exceed this request's budget of {budget}")
            }
            RequestError::ExplicitBreakpointsUnsupported { model, placed } => {
                write!(f, "{model} ignores explicit cache breakpoints, but {placed} are placed")
            }
        }
    }
}

impl std::error::Error for RequestError {}

// ── Transport ────────────────────────────────────────────────────────────────

/// How the response comes back, with the options that only that way accepts.
///
/// The API pairs `stream` with `stream_options` and refuses the second without
/// the first — measured live, *"The 'stream_options' parameter is only allowed
/// when 'stream' is enabled."* Two independent fields would let a caller write
/// exactly that 400; one sum type makes the pairing the only shape there is.
///
/// This is also the whole difference between the two ways of reading an answer.
/// A buffered request answers with one response body; a streamed one answers
/// with the events [`Settling`](crate::settle::Settling) accumulates. The
/// request body is otherwise identical, which is why this is a field rather than
/// two request types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transport {
    /// The whole response in one body. `stream` is emitted as `false`.
    #[default]
    Buffered,
    /// Server-sent events. `stream` is emitted as `true`.
    Streamed {
        /// Whether deltas carry `obfuscation` padding, which normalizes payload
        /// sizes so a packet length cannot reveal a token length.
        ///
        /// `None` sends no `stream_options` at all and leaves OpenAI's default,
        /// which is to include the padding. `Some` states the choice, and
        /// `Some(false)` trades the mitigation for bandwidth — sound only over
        /// links the caller trusts.
        include_obfuscation: Option<bool>,
    },
}

/// A borrowed [`Context`] plus per-call settings. Serializes to the
/// `POST /v1/responses` body.
///
/// Every public field here is one the API does *not* hash, so any of them may
/// change from request to request without costing a cached token. Anything that
/// would change the prefix lives in [`PrefixSettings`] or on the `Context`.
#[derive(Debug)]
pub struct Request<'a> {
    /// Conversation state: the frozen tool array, the items, the breakpoints.
    pub context: &'a Context,
    /// The settings that determine the hashed prefix.
    pub prefix: PrefixSettings,
    /// Which tools may be called this turn. Prefer
    /// [`ToolChoice::Allowed`] or [`ToolChoice::None`] over editing the array.
    pub tool_choice: ToolChoice,
    /// A routing hint that groups requests sharing a prefix onto the same
    /// machine. Not part of content identity: it cannot cause or prevent a match
    /// by itself, only make reaching a machine that holds one more likely. Keep
    /// it stable; a fresh key per request defeats it.
    pub prompt_cache_key: Option<String>,
    /// Upper bound on generated tokens, reasoning included. `None` leaves the
    /// model free up to the context window.
    pub max_output_tokens: Option<u32>,
    /// How the response comes back, and — inseparably — what the stream carries.
    ///
    /// One field rather than `stream` beside `stream_options`, because the API
    /// refuses the second without the first: measured live,
    /// `"stream_options" without "stream": true` answers *"The 'stream_options'
    /// parameter is only allowed when 'stream' is enabled."* A sum type makes
    /// that pairing the only one expressible, so the 400 is not a value.
    pub transport: Transport,
    /// Whether OpenAI stores the response for later retrieval.
    ///
    /// Always sent, carrying OpenAI's documented behavior: responses are "saved
    /// for 30 days by default", disabled "by setting `store` to `false`". So
    /// `true` is the default this field states, and stating it is the point —
    /// a caller who wants OpenAI to retain nothing writes
    /// [`Self::without_storage`] and can read the `false` back off the request.
    ///
    /// With `false`, reasoning items come back with `encrypted_content` so they
    /// can be replayed — see
    /// [`Context::push_reasoning`](crate::context::Context::push_reasoning).
    pub store: bool,
    /// Per-request instructions that will not be cached. For reusable ones, use
    /// a developer message instead; see [`UncacheableInstructions`].
    pub instructions: Option<UncacheableInstructions>,
    /// Extra output to send back that the response otherwise omits.
    ///
    /// Empty by default and then absent from the body, because every entry asks
    /// for something extra and the API asks for nothing extra by default. One
    /// entry is load-bearing rather than diagnostic:
    /// [`Include::ReasoningEncryptedContent`] is what makes a reasoning item
    /// replayable in stateless mode, so without it a reasoning model loses its
    /// own train of thought across a tool call.
    ///
    /// A `Vec` rather than a set because the API takes an array; duplicates are
    /// the caller's to avoid, and [`Self::including`] does not introduce them.
    pub include: Vec<Include>,
    /// Which processing pool serves the request. `None` leaves OpenAI's
    /// documented `auto` behavior — the project's configured tier — unstated,
    /// because "the project decides" and "I chose auto" are the same request
    /// and the crate does not name a value the caller did not.
    pub service_tier: Option<ServiceTier>,
    /// The caller's own key-value pairs, stored on the response and queryable
    /// later. Absent when empty: `{}` and no field say the same thing here, and
    /// the shorter one is what "no metadata" means.
    pub metadata: Option<Metadata>,
    /// A stable pseudonymous identifier for the end user, so OpenAI can detect
    /// abuse without the caller sending anything identifying. Hash a username or
    /// an email; do not send the address itself.
    ///
    /// Distinct from [`Self::prompt_cache_key`] on purpose: this one is for
    /// safety attribution and that one is for cache routing. The older `user`
    /// field, which conflated the two, is deliberately not modeled — the
    /// reference names it replaced by exactly these two.
    pub safety_identifier: Option<String>,
    /// What to do when the input exceeds the context window.
    ///
    /// `None` leaves the documented default, `disabled`, which fails the request
    /// with a 400. [`Truncation::Auto`] drops items from the front of the
    /// conversation instead — which is a prefix change by definition, so the
    /// turn that first truncates pays to write the whole prefix again.
    pub truncation: Option<Truncation>,
    /// A cap on how many built-in tool calls one response may make, across all
    /// hosted tools rather than per tool. Further calls are ignored rather than
    /// erroring. No effect on function tools, which the model calls through the
    /// caller.
    pub max_tool_calls: Option<u32>,
    /// Whether OpenAI generates the response asynchronously, so the caller polls
    /// or reconnects rather than holding a socket open for the whole answer.
    ///
    /// Always emitted, carrying the `false` every response object reports, so
    /// the body records which transport the caller chose. Background mode needs
    /// [`Self::store`] to stay true: a response nobody stored is one nobody can
    /// come back for.
    pub background: bool,
}

impl<'a> Request<'a> {
    /// The single construction path, so its checks cannot be skipped.
    ///
    /// It validates what the type system cannot: that the context's explicit
    /// breakpoints fit the budget this model and caching mode allow, and that
    /// the model honors explicit breakpoints at all.
    pub fn new(context: &'a Context, prefix: PrefixSettings) -> Result<Self, RequestError> {
        let placed = context.breakpoint_count();
        let budget = CacheWriteBudget::of(&prefix.model);
        if placed > 0 && !prefix.model.id().supports_explicit_cache_breakpoints() {
            return Err(RequestError::ExplicitBreakpointsUnsupported { model: prefix.model.api_id(), placed });
        }
        if placed > budget.explicit_slots {
            return Err(RequestError::TooManyExplicitBreakpoints { placed, budget: budget.explicit_slots });
        }
        Ok(Self {
            context,
            prefix,
            tool_choice: ToolChoice::default(),
            prompt_cache_key: None,
            max_output_tokens: None,
            transport: Transport::Buffered,
            store: true,
            instructions: None,
            include: Vec::new(),
            service_tier: None,
            metadata: None,
            safety_identifier: None,
            truncation: None,
            max_tool_calls: None,
            background: false,
        })
    }

    /// Cap generated tokens, reasoning included.
    ///
    /// Validated here, against this model's maximum, because the bound is a fact
    /// about the model and `0` is refused outright.
    pub fn with_max_output_tokens(mut self, max_output_tokens: u32) -> Result<Self, RequestError> {
        let model_maximum = self.prefix.model.id().max_output_tokens();
        if max_output_tokens == 0 || max_output_tokens > model_maximum {
            return Err(RequestError::MaxOutputTokensOutOfRange { max_output_tokens, model_maximum });
        }
        self.max_output_tokens = Some(max_output_tokens);
        Ok(self)
    }

    /// Restrict what may be called this turn.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Set the routing hint. Group by prompt version plus a stable user,
    /// session, or workspace identifier.
    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    /// Stream the response as server-sent events, saying nothing about
    /// obfuscation and so leaving OpenAI's default of including it.
    pub fn streaming(mut self) -> Self {
        self.transport = Transport::Streamed { include_obfuscation: None };
        self
    }

    /// Stream, and state whether the deltas carry `obfuscation` padding.
    ///
    /// `false` trades a side-channel mitigation for bandwidth: the padding
    /// normalizes payload sizes so a packet length cannot reveal a token length.
    /// Sound only over links the caller trusts.
    pub fn streaming_with_obfuscation(mut self, include_obfuscation: bool) -> Self {
        self.transport = Transport::Streamed { include_obfuscation: Some(include_obfuscation) };
        self
    }

    /// Return the whole response in one body rather than streaming it.
    pub fn without_streaming(mut self) -> Self {
        self.transport = Transport::Buffered;
        self
    }

    /// Whether this request asks for a stream.
    ///
    /// A reader, not a setter: the transport is a sum type because the stream
    /// and its options are one decision, and this answers the one question a
    /// caller asks of it without letting the pair come apart.
    pub fn is_streaming(&self) -> bool {
        matches!(self.transport, Transport::Streamed { .. })
    }

    /// Ask OpenAI not to store the response. Reasoning items then arrive with
    /// `encrypted_content`, so they can still be replayed.
    pub fn without_storage(mut self) -> Self {
        self.store = false;
        self
    }

    /// Let OpenAI store the response for later retrieval, its documented
    /// behavior.
    pub fn with_storage(mut self) -> Self {
        self.store = true;
        self
    }

    /// Add per-request instructions that are deliberately outside the reusable
    /// prefix — a timestamp, a user's name.
    pub fn with_instructions(mut self, instructions: UncacheableInstructions) -> Self {
        self.instructions = Some(instructions);
        self
    }

    /// Ask for one extra piece of output, unless it was already asked for.
    ///
    /// Idempotent because the API takes an array and a repeated entry is a
    /// duplicate on the wire rather than an emphasis. The counterpart is
    /// [`Self::excluding`].
    pub fn including(mut self, include: Include) -> Self {
        if !self.include.contains(&include) {
            self.include.push(include);
        }
        self
    }

    /// Stop asking for one extra piece of output.
    pub fn excluding(mut self, include: Include) -> Self {
        self.include.retain(|held| *held != include);
        self
    }

    /// Ask for the reasoning payload that makes a reasoning item replayable.
    ///
    /// Named for what it buys rather than for the wire string it sets, because
    /// this is the one `include` entry a stateless multi-turn caller needs: pair
    /// it with [`Context::push_reasoning`](crate::context::Context::push_reasoning)
    /// and the model keeps its own train of thought across a tool call.
    pub fn with_replayable_reasoning(self) -> Self {
        self.including(Include::ReasoningEncryptedContent)
    }

    /// Choose the processing pool.
    pub fn with_service_tier(mut self, service_tier: ServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    /// Leave the pool to the project's configuration, sending no `service_tier`.
    pub fn without_service_tier(mut self) -> Self {
        self.service_tier = None;
        self
    }

    /// Attach the caller's own key-value pairs. An empty map is no metadata, so
    /// it is stored as absence rather than as `{}`.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = (!metadata.is_empty()).then_some(metadata);
        self
    }

    /// Send no metadata.
    pub fn without_metadata(mut self) -> Self {
        self.metadata = None;
        self
    }

    /// Attribute the request to a pseudonymous end user. Hash the real
    /// identifier before calling this.
    pub fn with_safety_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(identifier.into());
        self
    }

    /// Send no safety identifier.
    pub fn without_safety_identifier(mut self) -> Self {
        self.safety_identifier = None;
        self
    }

    /// Choose what an over-long input does.
    pub fn with_truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Leave the documented default, which fails an over-long input outright.
    pub fn without_truncation(mut self) -> Self {
        self.truncation = None;
        self
    }

    /// Cap how many hosted-tool calls one response may make.
    pub fn with_max_tool_calls(mut self, max_tool_calls: u32) -> Self {
        self.max_tool_calls = Some(max_tool_calls);
        self
    }

    /// Lift the hosted-tool call cap.
    pub fn without_max_tool_calls(mut self) -> Self {
        self.max_tool_calls = None;
        self
    }

    /// Generate the response asynchronously, so it is polled for rather than
    /// waited on. Needs [`Self::store`] left true to be retrievable.
    pub fn in_background(mut self) -> Self {
        self.background = true;
        self
    }

    /// Generate the response on this request, the documented behavior.
    pub fn in_foreground(mut self) -> Self {
        self.background = false;
        self
    }

    /// The cache-write budget this request runs under.
    pub fn cache_write_budget(&self) -> CacheWriteBudget {
        CacheWriteBudget::of(&self.prefix.model)
    }
}
