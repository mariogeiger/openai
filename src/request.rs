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

use crate::content::InputItem;
use crate::context::{CACHE_WRITE_SLOTS, Context};
use crate::model::{Gpt5_6, Model};
use crate::prefix::{PrefixSettings, TextFormat};
use crate::tools::{FunctionTool, ToolChoice};
use crate::values::{
    CacheMode, Include, Metadata, ReasoningEffort, ReasoningSummary, ServiceTier, Truncation, Verbosity,
};
use serde::Serialize;
use serde_json::Value;

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

// ── Serialization ────────────────────────────────────────────────────────────
// Hand-written so the emitted body is readable in one place. An `Option` here is
// a genuine runtime absence, never a default elided to save bytes: a field the
// API documents a default for is a plain field carrying that value, and it is
// always emitted. An enclosing object whose every field is absent vanishes
// entirely, because `{}` and no field are two different requests.

#[derive(Serialize)]
struct TextFormatWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Serialize)]
struct TextWire<'a> {
    format: TextFormatWire<'a>,
    verbosity: Verbosity,
}

/// `reasoning`, whose four fields are independently present or absent.
///
/// The object is only ever built by [`ReasoningWire::of`], which returns `None`
/// when all four are absent — an empty `"reasoning": {}` is a different request
/// from no `reasoning` at all, and only one of them is what "the caller
/// configured no reasoning" means.
#[derive(Serialize)]
struct ReasoningWire {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<crate::values::ReasoningMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<crate::values::ReasoningContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<ReasoningSummary>,
}

impl ReasoningWire {
    /// The `reasoning` object, or `None` when it would be empty.
    fn of(
        effort: Option<ReasoningEffort>,
        mode: Option<crate::values::ReasoningMode>,
        context: Option<crate::values::ReasoningContext>,
        summary: Option<ReasoningSummary>,
    ) -> Option<Self> {
        let wire = Self { effort, mode, context, summary };
        (!wire.is_empty()).then_some(wire)
    }

    /// Whether every field inside is absent.
    fn is_empty(&self) -> bool {
        self.effort.is_none() && self.mode.is_none() && self.context.is_none() && self.summary.is_none()
    }
}

#[derive(Serialize)]
struct CacheOptionsWire {
    mode: CacheMode,
    ttl: crate::values::CacheTtl,
}

#[derive(Serialize)]
struct CompactionWire {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    compact_threshold: Option<u32>,
}

/// `stream_options`, whose single field is independently present or absent, so
/// the object vanishes when it is: an empty `"stream_options": {}` says nothing
/// that no `stream_options` does not already say.
#[derive(Serialize)]
struct StreamOptionsWire {
    include_obfuscation: bool,
}

#[derive(Serialize)]
struct RequestWire<'a> {
    model: &'static str,
    // `tools` first: it is the first thing OpenAI hashes after its own hidden
    // content. Field order does not affect the hash — OpenAI renders the prompt
    // from the parsed body — but writing the struct in prefix order keeps the
    // code readable against the guide's diagram.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [FunctionTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    input: &'a [InputItem],
    parallel_tool_calls: bool,
    text: TextWire<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningWire>,
    tool_choice: &'a ToolChoice,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<CacheOptionsWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<crate::values::CacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_management: Option<[CompactionWire; 1]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a [Include]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a Metadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_identifier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<Truncation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tool_calls: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptionsWire>,
    store: bool,
    background: bool,
}

impl Serialize for Request<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use crate::model::ExtendedRetentionOnly;
        let prefix = &self.prefix;

        // Exactly one caching field per model generation: `prompt_cache_options`
        // on GPT-5.6 and later, `prompt_cache_retention` before. Building both
        // from one match is what makes sending both impossible.
        let (prompt_cache_options, prompt_cache_retention) = match &prefix.model {
            Model::Gpt5_6(m) => (Some(CacheOptionsWire { mode: m.caching.mode, ttl: m.caching.ttl }), None),
            Model::Gpt5_5(m) => (None, Some(retention_of(m.retention))),
            Model::Gpt5_5Pro(m) => (None, Some(retention_of(m.retention))),
            // The one model that accepts both values, and the one whose default
            // is the organization's data-retention policy rather than a value
            // OpenAI names. So `None` here stays absent.
            Model::Gpt5_4(m) => (None, m.retention),
        };
        fn retention_of(r: ExtendedRetentionOnly) -> crate::values::CacheRetention {
            match r {
                ExtendedRetentionOnly::TwentyFourHours => crate::values::CacheRetention::TwentyFourHours,
            }
        }

        // `mode` and `context` exist only on GPT-5.6; earlier models 400 on them.
        // `context` carries its documented default and is always sent there;
        // `mode` has no documented default and is sent only when chosen.
        let (mode, context) = match &prefix.model {
            Model::Gpt5_6(m) => (m.mode, Some(m.reasoning_context)),
            Model::Gpt5_5(_) | Model::Gpt5_5Pro(_) | Model::Gpt5_4(_) => (None, None),
        };

        let format = match &prefix.text_format {
            TextFormat::Text => TextFormatWire { kind: "text", name: None, schema: None, strict: None },
            TextFormat::JsonSchema { name, schema, strict } => {
                TextFormatWire { kind: "json_schema", name: Some(name), schema: Some(schema), strict: Some(*strict) }
            }
        };

        let tools = self.context.tools();
        RequestWire {
            model: prefix.model.api_id(),
            tools: (!tools.is_empty()).then_some(tools),
            instructions: self.instructions.as_ref().map(|i| i.0.as_str()),
            input: self.context.items(),
            parallel_tool_calls: prefix.parallel_tool_calls,
            text: TextWire { format, verbosity: prefix.verbosity },
            reasoning: ReasoningWire::of(prefix.effort(), mode, context, prefix.reasoning_summary),
            tool_choice: &self.tool_choice,
            prompt_cache_options,
            prompt_cache_retention,
            prompt_cache_key: self.prompt_cache_key.as_deref(),
            context_management: prefix
                .context_management
                .map(|c| [CompactionWire { kind: "compaction", compact_threshold: c.compact_threshold }]),
            max_output_tokens: self.max_output_tokens,
            include: (!self.include.is_empty()).then_some(&self.include),
            service_tier: self.service_tier,
            metadata: self.metadata.as_ref(),
            safety_identifier: self.safety_identifier.as_deref(),
            truncation: self.truncation,
            max_tool_calls: self.max_tool_calls,
            stream: self.is_streaming(),
            stream_options: match self.transport {
                Transport::Streamed { include_obfuscation: Some(include_obfuscation) } => {
                    Some(StreamOptionsWire { include_obfuscation })
                }
                Transport::Streamed { include_obfuscation: None } | Transport::Buffered => None,
            },
            store: self.store,
            background: self.background,
        }
        .serialize(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::InputBlock;
    use crate::context::{BreakpointSlot, Context};
    use crate::model::{EffortMediumToXhigh, EffortNoneToMax, EffortNoneToXhigh, Gpt5_6Tier};
    use crate::tools::AllowedToolsMode;
    use crate::values::{CacheRetention, ReasoningContext, ReasoningMode};
    use serde_json::json;

    fn tools() -> Vec<FunctionTool> {
        vec![
            FunctionTool::new("read_file", json!({"type": "object"})),
            FunctionTool::new("write_file", json!({"type": "object"})),
        ]
    }

    fn body(context: &Context, prefix: PrefixSettings) -> Value {
        serde_json::to_value(Request::new(context, prefix).unwrap()).unwrap()
    }

    /// The whole body, exactly, for the commonest request — every field OpenAI
    /// documents a default for, carrying that default, and nothing else.
    ///
    /// This is the test the design rests on. Each field below is here because
    /// the reference names a default for it: `store` "saved for 30 days by
    /// default", `parallel_tool_calls` typed non-null on the response,
    /// `text.format` "the default format is `{\"type\": \"text\"}`",
    /// `text.verbosity` "the default is `medium`", `reasoning.context` "if
    /// omitted or set to `auto`, the model determines", `prompt_cache_options`
    /// "defaults to `implicit`" / "defaults to `30m`", `background` typed
    /// non-null `false` on the response object and confirmed live. Nothing else
    /// appears, because nothing else has one — and if a field drifts into or out
    /// of this literal, this test names it.
    #[test]
    fn a_default_gpt_5_6_body_is_exactly_this() {
        let mut context = Context::new(vec![]);
        context.push_user_text("hello");
        assert_eq!(
            body(&context, PrefixSettings::new(Model::gpt_5_6_sol())),
            json!({
                "model": "gpt-5.6-sol",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
                "parallel_tool_calls": true,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "reasoning": {"context": "auto"},
                "tool_choice": "auto",
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                "stream": false,
                "store": true,
                "background": false,
            })
        );
    }

    /// The minimal body: a model that documents a default for nothing but its
    /// caching field, asked for nothing. Every optional field is absent, and
    /// `reasoning` is absent *as an object* rather than present and empty.
    ///
    /// GPT-5.4 is the sharpest case because its `prompt_cache_retention`
    /// default "depends on your organization's data retention policy" — a
    /// default the crate cannot know, so the only honest rendering is silence.
    #[test]
    fn a_request_asking_for_nothing_sends_only_what_has_a_documented_default() {
        let mut context = Context::new(vec![]);
        context.push_user_text("hello");
        assert_eq!(
            body(&context, PrefixSettings::new(Model::gpt_5_4())),
            json!({
                "model": "gpt-5.4",
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
                "parallel_tool_calls": true,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "tool_choice": "auto",
                "stream": false,
                "store": true,
                "background": false,
            })
        );
    }

    /// `reasoning` vanishes rather than arriving empty. An empty object is a
    /// field the caller never asked for, and the two are not the same request.
    #[test]
    fn an_empty_reasoning_object_is_no_reasoning_object() {
        let context = Context::new(vec![]);
        let bare = body(&context, PrefixSettings::new(Model::gpt_5_5()));
        assert!(bare.get("reasoning").is_none(), "{bare}");
        assert!(!serde_json::to_string(&bare).unwrap().contains("reasoning"), "{bare}");

        // One field inside is enough to bring the object back, and it then holds
        // exactly that field.
        let asked = body(&context, PrefixSettings::new(Model::gpt_5_5().with_effort(EffortNoneToXhigh::High)));
        assert_eq!(asked["reasoning"], json!({"effort": "high"}));

        let summarized =
            body(&context, PrefixSettings::new(Model::gpt_5_5()).with_reasoning_summary(ReasoningSummary::Concise));
        assert_eq!(summarized["reasoning"], json!({"summary": "concise"}));
    }

    /// The caller decides how hard the model thinks, including deciding not to.
    #[test]
    fn effort_is_sent_only_when_chosen() {
        let context = Context::new(vec![]);
        for model in [Model::gpt_5_6_sol(), Model::gpt_5_6_terra(), Model::gpt_5_6_luna()] {
            assert!(body(&context, PrefixSettings::new(model))["reasoning"].get("effort").is_none());
        }
        assert!(body(&context, PrefixSettings::new(Model::gpt_5_5())).get("reasoning").is_none());
        assert!(body(&context, PrefixSettings::new(Model::gpt_5_5_pro())).get("reasoning").is_none());
        assert!(body(&context, PrefixSettings::new(Model::gpt_5_4())).get("reasoning").is_none());

        let chosen = PrefixSettings::new(Model::gpt_5_6_sol().with_effort(EffortNoneToMax::Max));
        assert_eq!(chosen.effort(), Some(crate::values::ReasoningEffort::Max));
        assert_eq!(body(&context, chosen)["reasoning"]["effort"], "max");

        // And a chosen effort can be taken back off again.
        let withdrawn = PrefixSettings::new(Model::gpt_5_6_sol().with_effort(EffortNoneToMax::Max).without_effort());
        assert_eq!(withdrawn.effort(), None);
        assert!(body(&context, withdrawn)["reasoning"].get("effort").is_none());
    }

    /// `store` proves the other half of the rule: OpenAI documents that a
    /// response is retained unless told otherwise, so the field is always on the
    /// wire — and a caller who wants nothing retained says so and can read it
    /// back.
    #[test]
    fn store_is_always_sent_and_always_the_caller_s() {
        let context = Context::new(vec![]);
        assert_eq!(body(&context, PrefixSettings::new(Model::gpt_5_6_sol()))["store"], true);

        let stateless = Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol())).unwrap().without_storage();
        assert!(!stateless.store);
        assert_eq!(serde_json::to_value(&stateless).unwrap()["store"], false);
        assert_eq!(serde_json::to_value(stateless.with_storage()).unwrap()["store"], true);
    }

    /// Each model's documented effort is readable without being imposed: the
    /// request says nothing, and the fact says `medium`.
    #[test]
    fn the_model_s_own_default_effort_is_a_readable_fact() {
        use crate::model::ModelId;
        use crate::values::ReasoningEffort;
        assert_eq!(ModelId::Gpt5_6Sol.default_effort(), ReasoningEffort::Medium);
        assert_eq!(ModelId::Gpt5_6Terra.default_effort(), ReasoningEffort::Medium);
        assert_eq!(ModelId::Gpt5_6Luna.default_effort(), ReasoningEffort::Medium);
        assert_eq!(ModelId::Gpt5_5.default_effort(), ReasoningEffort::Medium);
        assert_eq!(ModelId::Gpt5_5Pro.default_effort(), ReasoningEffort::High);
        assert_eq!(ModelId::Gpt5_4.default_effort(), ReasoningEffort::None);

        // Stated, never imposed: the body still carries no effort.
        let context = Context::new(vec![]);
        assert!(body(&context, PrefixSettings::new(Model::gpt_5_5_pro())).get("reasoning").is_none());
    }

    /// An empty tool array is omitted, not sent as `[]`: absent and empty render
    /// differently, so they would be two prefixes for one meaning.
    #[test]
    fn no_tools_means_no_tools_field() {
        let context = Context::new(vec![]);
        assert!(body(&context, PrefixSettings::new(Model::gpt_5_6_sol())).get("tools").is_none());

        let with = Context::new(tools());
        let v = body(&with, PrefixSettings::new(Model::gpt_5_6_sol()));
        assert_eq!(v["tools"].as_array().unwrap().len(), 2);
        assert_eq!(v["tools"][0]["name"], "read_file");
        assert_eq!(v["tools"][1]["name"], "write_file");
    }

    /// The measured cache-preserving pattern: array unchanged, availability
    /// narrowed by `tool_choice`.
    #[test]
    fn narrowing_availability_leaves_the_array_whole() {
        let context = Context::new(tools());
        let allowed = context.allow_tools(AllowedToolsMode::Auto, &["read_file"]).unwrap();
        let request = Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol()))
            .unwrap()
            .with_tool_choice(ToolChoice::Allowed(allowed));
        let v = serde_json::to_value(&request).unwrap();

        assert_eq!(v["tools"].as_array().unwrap().len(), 2, "the array must not shrink");
        assert_eq!(
            v["tool_choice"],
            json!({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "read_file"}]})
        );
    }

    /// GPT-5.6 sends `prompt_cache_options`; earlier models send
    /// `prompt_cache_retention`. Never both, because one match produces both.
    #[test]
    fn each_generation_sends_only_its_own_caching_field() {
        let context = Context::new(vec![]);

        let six = body(&context, PrefixSettings::new(Model::gpt_5_6_sol()));
        assert_eq!(six["prompt_cache_options"], json!({"mode": "implicit", "ttl": "30m"}));
        assert!(six.get("prompt_cache_retention").is_none());

        for prefix in [PrefixSettings::new(Model::gpt_5_5()), PrefixSettings::new(Model::gpt_5_5_pro())] {
            let v = body(&context, prefix);
            assert_eq!(v["prompt_cache_retention"], "24h");
            assert!(v.get("prompt_cache_options").is_none());
        }

        let in_memory = body(&context, PrefixSettings::new(Model::gpt_5_4().with_retention(CacheRetention::InMemory)));
        assert_eq!(in_memory["prompt_cache_retention"], "in_memory");
        assert!(in_memory.get("prompt_cache_options").is_none());

        // GPT-5.4 alone documents no value as its default — the organization's
        // data-retention policy decides — so unasked, the field is absent, and
        // asking can be taken back.
        let unasked = body(&context, PrefixSettings::new(Model::gpt_5_4()));
        assert!(unasked.get("prompt_cache_retention").is_none(), "{unasked}");
        let withdrawn =
            PrefixSettings::new(Model::gpt_5_4().with_retention(CacheRetention::InMemory).without_retention());
        assert!(body(&context, withdrawn).get("prompt_cache_retention").is_none());
    }

    /// `reasoning.mode` and `reasoning.context` are GPT-5.6-only; sending them
    /// to GPT-5.5 is a 400. `context` carries its documented `auto` default;
    /// `mode` has none, so it appears only when chosen.
    #[test]
    fn gpt_5_6_only_reasoning_fields_stay_on_gpt_5_6() {
        let context = Context::new(vec![]);
        let six = body(&context, PrefixSettings::new(Model::gpt_5_6_sol()));
        assert_eq!(six["reasoning"]["context"], "auto");
        assert!(six["reasoning"].get("mode").is_none());

        let chosen = body(&context, PrefixSettings::new(Model::gpt_5_6_sol().with_mode(ReasoningMode::Standard)));
        assert_eq!(chosen["reasoning"]["mode"], "standard");

        // On GPT-5.5 neither field can appear, chosen or not: the type carries
        // no `mode` and no `reasoning_context` at all.
        let five = body(&context, PrefixSettings::new(Model::gpt_5_5().with_effort(EffortNoneToXhigh::Medium)));
        assert!(five["reasoning"].get("mode").is_none());
        assert!(five["reasoning"].get("context").is_none());
        assert_eq!(five["reasoning"]["effort"], "medium");
    }

    #[test]
    fn pro_mode_and_max_effort_reach_the_wire() {
        let context = Context::new(vec![]);
        let model = Model::gpt_5_6_sol()
            .with_effort(EffortNoneToMax::Max)
            .with_mode(ReasoningMode::Pro)
            .with_reasoning_context(ReasoningContext::CurrentTurn);
        let v = body(&context, PrefixSettings::new(model));
        assert_eq!(v["reasoning"], json!({"effort": "max", "mode": "pro", "context": "current_turn"}));
        // Every one of the three is there because the caller asked for it.
    }

    /// Each model's own effort range still reaches the wire when chosen.
    #[test]
    fn a_chosen_effort_reaches_the_wire_on_every_model() {
        let context = Context::new(vec![]);
        let six = PrefixSettings::new(Model::gpt_5_6_sol().with_effort(EffortNoneToMax::Medium));
        assert_eq!(body(&context, six)["reasoning"]["effort"], "medium");
        let five = PrefixSettings::new(Model::gpt_5_5().with_effort(EffortNoneToXhigh::Medium));
        assert_eq!(body(&context, five)["reasoning"]["effort"], "medium");
        let pro = PrefixSettings::new(Model::gpt_5_5_pro().with_effort(EffortMediumToXhigh::High));
        assert_eq!(body(&context, pro)["reasoning"]["effort"], "high");
        let four = PrefixSettings::new(Model::gpt_5_4().with_effort(EffortNoneToXhigh::None));
        assert_eq!(body(&context, four)["reasoning"]["effort"], "none");
    }

    #[test]
    fn structured_outputs_and_verbosity_ride_in_text() {
        let context = Context::new(vec![]);
        let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
        let prefix = PrefixSettings::new(Model::gpt_5_6_sol())
            .with_text_format(TextFormat::json_schema("verdict", schema.clone()))
            .with_verbosity(Verbosity::Low);
        assert_eq!(
            body(&context, prefix)["text"],
            json!({
                "format": {"type": "json_schema", "name": "verdict", "schema": schema, "strict": true},
                "verbosity": "low",
            })
        );
    }

    #[test]
    fn compaction_serializes_as_a_one_entry_array() {
        let context = Context::new(vec![]);
        let v = body(&context, PrefixSettings::new(Model::gpt_5_6_sol()).with_compaction(Some(200_000)));
        assert_eq!(v["context_management"], json!([{"type": "compaction", "compact_threshold": 200_000}]));

        let auto = body(&context, PrefixSettings::new(Model::gpt_5_6_sol()).with_compaction(None));
        assert_eq!(auto["context_management"], json!([{"type": "compaction"}]));
    }

    #[test]
    fn per_call_settings_reach_the_wire() {
        let context = Context::new(vec![]);
        let request = Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol()))
            .unwrap()
            .with_max_output_tokens(4_096)
            .unwrap()
            .with_prompt_cache_key("agent_v1:user_7")
            .with_instructions(UncacheableInstructions::new("Today is Tuesday."))
            .streaming()
            .without_storage();
        let v = serde_json::to_value(&request).unwrap();
        assert_eq!(v["max_output_tokens"], 4_096);
        assert_eq!(v["prompt_cache_key"], "agent_v1:user_7");
        assert_eq!(v["instructions"], "Today is Tuesday.");
        assert_eq!(v["stream"], true);
        assert_eq!(v["store"], false);
    }

    #[test]
    fn max_output_tokens_is_checked_against_the_model() {
        let context = Context::new(vec![]);
        let prefix = || PrefixSettings::new(Model::gpt_5_6_sol());

        assert_eq!(
            Request::new(&context, prefix()).unwrap().with_max_output_tokens(0).err(),
            Some(RequestError::MaxOutputTokensOutOfRange { max_output_tokens: 0, model_maximum: 128_000 })
        );
        assert_eq!(
            Request::new(&context, prefix()).unwrap().with_max_output_tokens(128_001).err(),
            Some(RequestError::MaxOutputTokensOutOfRange { max_output_tokens: 128_001, model_maximum: 128_000 })
        );
        assert!(Request::new(&context, prefix()).unwrap().with_max_output_tokens(1).is_ok());
        assert!(Request::new(&context, prefix()).unwrap().with_max_output_tokens(128_000).is_ok());
        // Omitted means "up to the context window", not zero.
        assert!(body(&context, prefix()).get("max_output_tokens").is_none());
    }

    /// Implicit mode spends one of the four writes on OpenAI's own breakpoint,
    /// so three explicit ones fit and a fourth does not.
    #[test]
    fn implicit_mode_leaves_three_explicit_slots() {
        let budget = CacheWriteBudget::of(&Model::from(Model::gpt_5_6_sol()));
        assert_eq!(budget, CacheWriteBudget { explicit_slots: 3, implicit_breakpoint: true });

        let mut context = Context::new(vec![]);
        for (i, slot) in BreakpointSlot::ALL.iter().take(3).enumerate() {
            context.push_user_text(format!("turn {i}"));
            context.roll_breakpoint(*slot).unwrap();
        }
        assert!(Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol())).is_ok());

        context.push_user_text("turn 3");
        context.roll_breakpoint(BreakpointSlot::S3).unwrap();
        assert_eq!(
            Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol())).err(),
            Some(RequestError::TooManyExplicitBreakpoints { placed: 4, budget: 3 })
        );
    }

    /// Explicit-only mode frees the fourth slot, and the same context that was
    /// refused above is accepted.
    #[test]
    fn explicit_only_mode_grants_all_four_slots() {
        let model = Model::gpt_5_6_sol().with_explicit_cache_only();
        assert_eq!(
            CacheWriteBudget::of(&Model::from(model)),
            CacheWriteBudget { explicit_slots: 4, implicit_breakpoint: false }
        );

        let mut context = Context::new(vec![]);
        for (i, slot) in BreakpointSlot::ALL.iter().enumerate() {
            context.push_user_text(format!("turn {i}"));
            context.roll_breakpoint(*slot).unwrap();
        }
        let request = Request::new(&context, PrefixSettings::new(model)).unwrap();
        assert_eq!(request.cache_write_budget().explicit_slots, CACHE_WRITE_SLOTS);
        assert_eq!(serde_json::to_value(&request).unwrap()["prompt_cache_options"]["mode"], "explicit");
    }

    /// Older models ignore explicit breakpoints. Sending them anyway would mean
    /// believing in a reusable prefix that nothing reads.
    #[test]
    fn older_models_refuse_a_context_with_explicit_breakpoints() {
        let mut context = Context::new(vec![]);
        context.push_anchored_developer_text(BreakpointSlot::S0, "stable").unwrap();

        assert_eq!(
            Request::new(&context, PrefixSettings::new(Model::gpt_5_5())).err(),
            Some(RequestError::ExplicitBreakpointsUnsupported { model: "gpt-5.5", placed: 1 })
        );
        // Without breakpoints the same model is fine.
        let mut plain = Context::new(vec![]);
        plain.push_user_text("hello");
        assert!(Request::new(&plain, PrefixSettings::new(Model::gpt_5_5())).is_ok());
        assert_eq!(
            CacheWriteBudget::of(&Model::from(Model::gpt_5_5())),
            CacheWriteBudget { explicit_slots: 0, implicit_breakpoint: true }
        );
    }

    /// The documented shape for reusable instructions: a developer message
    /// block, because top-level `instructions` cannot be marked.
    #[test]
    fn reusable_instructions_live_in_a_developer_block() {
        let mut context = Context::new(vec![]);
        context.push_anchored_developer_text(BreakpointSlot::S0, "Stable instructions").unwrap();
        context.push_user_text("Dynamic question");
        let v = body(&context, PrefixSettings::new(Model::gpt_5_6_sol()));

        assert_eq!(v["input"][0]["role"], "developer");
        assert_eq!(v["input"][0]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
        assert!(v["input"][1]["content"][0].get("prompt_cache_breakpoint").is_none());
        // The uncacheable field stays absent unless asked for.
        assert!(v.get("instructions").is_none());
    }

    #[test]
    fn a_full_tool_turn_serializes_in_order() {
        let mut context = Context::new(tools());
        context.push_anchored_developer_text(BreakpointSlot::S0, "You edit files.").unwrap();
        context.push_user_text("Read a.rs");
        context.push_reasoning("rs_1", "opaque");
        context.push_function_call("call_1", "read_file", r#"{"path":"a.rs"}"#);
        context.push_function_call_output_blocks("call_1", vec![InputBlock::text("fn main() {}")]);
        context.roll_breakpoint(BreakpointSlot::S1).unwrap();
        context.push_assistant_text(crate::values::AssistantPhase::FinalAnswer, "It is a main function.");

        let v = body(&context, PrefixSettings::new(Model::gpt_5_6_sol()));
        let kinds: Vec<&str> = v["input"].as_array().unwrap().iter().map(|i| i["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["message", "message", "reasoning", "function_call", "function_call_output", "message"]);
        assert_eq!(v["input"][4]["output"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
        assert_eq!(v["input"][5]["phase"], "final_answer");
    }

    #[test]
    fn serial_tool_calls_and_reasoning_summary_reach_the_wire() {
        let context = Context::new(tools());
        let prefix = PrefixSettings::new(Model::gpt_5_6_terra())
            .with_serial_tool_calls()
            .with_reasoning_summary(ReasoningSummary::Auto);
        let v = body(&context, prefix);
        assert_eq!(v["parallel_tool_calls"], false);
        assert_eq!(v["reasoning"]["summary"], "auto");
        assert_eq!(v["model"], "gpt-5.6-terra");
    }

    /// Streaming changes one flag, not the body. Same prefix, same cache.
    #[test]
    fn streaming_differs_from_not_streaming_by_one_field() {
        let mut context = Context::new(tools());
        context.push_user_text("hello");
        let prefix = PrefixSettings::new(Model::gpt_5_6_sol());
        let mut plain = body(&context, prefix.clone());
        let streamed = serde_json::to_value(Request::new(&context, prefix).unwrap().streaming()).unwrap();

        assert_eq!(plain["stream"], false);
        assert_eq!(streamed["stream"], true);
        plain["stream"] = json!(true);
        assert_eq!(plain, streamed);
    }

    #[test]
    fn tool_choice_none_keeps_the_array() {
        let context = Context::new(tools());
        let request = Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol()))
            .unwrap()
            .with_tool_choice(ToolChoice::None);
        let v = serde_json::to_value(&request).unwrap();
        assert_eq!(v["tool_choice"], "none");
        assert_eq!(v["tools"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn every_tier_reaches_the_wire_with_its_own_id() {
        let context = Context::new(vec![]);
        for (model, id) in [
            (Model::gpt_5_6_sol(), "gpt-5.6-sol"),
            (Model::gpt_5_6_terra(), "gpt-5.6-terra"),
            (Model::gpt_5_6_luna(), "gpt-5.6-luna"),
        ] {
            assert_eq!(body(&context, PrefixSettings::new(model))["model"], id);
        }
        assert_eq!(Gpt5_6::new(Gpt5_6Tier::Luna).tier, Gpt5_6Tier::Luna);
    }

    /// `stream_options` reaches the wire only alongside `stream: true`, because
    /// the sum type admits no other pairing.
    ///
    /// The API's own words, measured live against the endpoint: *"The
    /// 'stream_options' parameter is only allowed when 'stream' is enabled."*
    /// With `stream` and `stream_options` as two independent fields, that 400
    /// was one line of caller code away.
    #[test]
    fn stream_options_cannot_be_sent_without_stream() {
        let context = Context::new(vec![]);
        let prefix = || PrefixSettings::new(Model::gpt_5_6_sol());
        let of = |request: Request<'_>| serde_json::to_value(&request).unwrap();

        let buffered = of(Request::new(&context, prefix()).unwrap());
        assert_eq!(buffered["stream"], false);
        assert!(buffered.get("stream_options").is_none(), "{buffered}");

        // Streaming without an obfuscation choice sends no options object: `{}`
        // would state a preference the caller never expressed.
        let streamed = of(Request::new(&context, prefix()).unwrap().streaming());
        assert_eq!(streamed["stream"], true);
        assert!(streamed.get("stream_options").is_none(), "{streamed}");

        let chosen = of(Request::new(&context, prefix()).unwrap().streaming_with_obfuscation(false));
        assert_eq!(chosen["stream"], true);
        assert_eq!(chosen["stream_options"], json!({"include_obfuscation": false}));

        // And back off the wire, which is the counterpart every setter has.
        let reverted =
            of(Request::new(&context, prefix()).unwrap().streaming_with_obfuscation(false).without_streaming());
        assert_eq!(reverted["stream"], false);
        assert!(reverted.get("stream_options").is_none(), "{reverted}");
    }

    /// The response-shaping fields are absent until asked for, and every one of
    /// them can be taken back off the wire.
    #[test]
    fn each_per_call_field_goes_on_and_comes_off_the_wire() {
        let context = Context::new(vec![]);
        let prefix = || PrefixSettings::new(Model::gpt_5_6_sol());
        let bare = body(&context, prefix());
        for field in
            ["include", "service_tier", "metadata", "safety_identifier", "truncation", "max_tool_calls", "instructions"]
        {
            assert!(bare.get(field).is_none(), "{field} was sent unasked: {bare}");
        }

        let metadata = Metadata::new([("thread", "42")]).unwrap();
        let asked = Request::new(&context, prefix())
            .unwrap()
            .with_replayable_reasoning()
            .with_service_tier(ServiceTier::Flex)
            .with_metadata(metadata)
            .with_safety_identifier("sha256:abc")
            .with_truncation(Truncation::Auto)
            .with_max_tool_calls(3)
            .in_background();
        let value = serde_json::to_value(&asked).unwrap();
        assert_eq!(value["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(value["service_tier"], "flex");
        assert_eq!(value["metadata"], json!({"thread": "42"}));
        assert_eq!(value["safety_identifier"], "sha256:abc");
        assert_eq!(value["truncation"], "auto");
        assert_eq!(value["max_tool_calls"], 3);
        assert_eq!(value["background"], true);

        let taken_back = asked
            .excluding(Include::ReasoningEncryptedContent)
            .without_service_tier()
            .without_metadata()
            .without_safety_identifier()
            .without_truncation()
            .without_max_tool_calls()
            .in_foreground();
        assert_eq!(serde_json::to_value(&taken_back).unwrap(), bare);
    }

    /// `include` is an array, so asking twice must not send twice: a duplicate
    /// entry is a duplicate on the wire, and the wire is the prefix.
    #[test]
    fn asking_for_the_same_extra_output_twice_sends_it_once() {
        let context = Context::new(vec![]);
        let request = Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol()))
            .unwrap()
            .with_replayable_reasoning()
            .including(Include::ReasoningEncryptedContent)
            .including(Include::FileSearchCallResults);
        assert_eq!(
            serde_json::to_value(&request).unwrap()["include"],
            json!(["reasoning.encrypted_content", "file_search_call.results"])
        );
    }

    /// An empty metadata map is no metadata, not `{}`.
    #[test]
    fn empty_metadata_is_absence() {
        let context = Context::new(vec![]);
        let request = Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol()))
            .unwrap()
            .with_metadata(Metadata::new(Vec::<(String, String)>::new()).unwrap());
        assert!(serde_json::to_value(&request).unwrap().get("metadata").is_none());
    }
}
