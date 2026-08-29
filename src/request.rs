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
use crate::values::{CacheMode, ReasoningEffort, ReasoningSummary, Verbosity};
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
    /// Whether to stream the response as server-sent events. The body is
    /// otherwise identical, which is why streaming is a flag here rather than a
    /// second request type.
    pub stream: bool,
    /// Whether OpenAI stores the response for later retrieval. With `false`,
    /// reasoning items come back with `encrypted_content` so they can be
    /// replayed — see
    /// [`Context::push_reasoning`](crate::context::Context::push_reasoning).
    pub store: bool,
    /// Per-request instructions that will not be cached. For reusable ones, use
    /// a developer message instead; see [`UncacheableInstructions`].
    pub instructions: Option<UncacheableInstructions>,
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
            stream: false,
            store: true,
            instructions: None,
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

    /// Stream the response as server-sent events.
    pub fn streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Ask OpenAI not to store the response. Reasoning items then arrive with
    /// `encrypted_content`, so they can still be replayed.
    pub fn without_storage(mut self) -> Self {
        self.store = false;
        self
    }

    /// Add per-request instructions that are deliberately outside the reusable
    /// prefix — a timestamp, a user's name.
    pub fn with_instructions(mut self, instructions: UncacheableInstructions) -> Self {
        self.instructions = Some(instructions);
        self
    }

    /// The cache-write budget this request runs under.
    pub fn cache_write_budget(&self) -> CacheWriteBudget {
        CacheWriteBudget::of(&self.prefix.model)
    }
}

// ── Serialization ────────────────────────────────────────────────────────────
// Hand-written so the emitted body is readable in one place. An `Option` here is
// a genuine runtime absence, never a default elided to save bytes.

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

#[derive(Serialize)]
struct ReasoningWire {
    effort: ReasoningEffort,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<crate::values::ReasoningMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<crate::values::ReasoningContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<ReasoningSummary>,
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
    reasoning: ReasoningWire,
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
    stream: bool,
    store: bool,
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
            Model::Gpt5_4(m) => (None, Some(m.retention)),
        };
        fn retention_of(r: ExtendedRetentionOnly) -> crate::values::CacheRetention {
            match r {
                ExtendedRetentionOnly::TwentyFourHours => crate::values::CacheRetention::TwentyFourHours,
            }
        }

        // `mode` and `context` exist only on GPT-5.6; earlier models 400 on them.
        let (mode, context) = match &prefix.model {
            Model::Gpt5_6(m) => (Some(m.mode), Some(m.reasoning_context)),
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
            reasoning: ReasoningWire { effort: prefix.effort(), mode, context, summary: prefix.reasoning_summary },
            tool_choice: &self.tool_choice,
            prompt_cache_options,
            prompt_cache_retention,
            prompt_cache_key: self.prompt_cache_key.as_deref(),
            context_management: prefix
                .context_management
                .map(|c| [CompactionWire { kind: "compaction", compact_threshold: c.compact_threshold }]),
            max_output_tokens: self.max_output_tokens,
            stream: self.stream,
            store: self.store,
        }
        .serialize(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::ContentBlock;
    use crate::context::{BreakpointSlot, Context};
    use crate::model::{EffortNoneToMax, Gpt5_6Tier};
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

    /// The whole body, exactly, for the commonest request. If any default drifts
    /// this test says which one.
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
                "reasoning": {"effort": "medium", "mode": "standard", "context": "all_turns"},
                "tool_choice": "auto",
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                "stream": false,
                "store": true,
            })
        );
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
    }

    /// `reasoning.mode` and `reasoning.context` are GPT-5.6-only; sending them
    /// to GPT-5.5 is a 400.
    #[test]
    fn gpt_5_6_only_reasoning_fields_stay_on_gpt_5_6() {
        let context = Context::new(vec![]);
        let six = body(&context, PrefixSettings::new(Model::gpt_5_6_sol()));
        assert_eq!(six["reasoning"]["mode"], "standard");
        assert_eq!(six["reasoning"]["context"], "all_turns");

        let five = body(&context, PrefixSettings::new(Model::gpt_5_5()));
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
    }

    /// Each model's documented default effort, on the wire.
    #[test]
    fn default_effort_is_the_model_s_own() {
        let context = Context::new(vec![]);
        assert_eq!(body(&context, PrefixSettings::new(Model::gpt_5_6_sol()))["reasoning"]["effort"], "medium");
        assert_eq!(body(&context, PrefixSettings::new(Model::gpt_5_5()))["reasoning"]["effort"], "medium");
        assert_eq!(body(&context, PrefixSettings::new(Model::gpt_5_5_pro()))["reasoning"]["effort"], "high");
        assert_eq!(body(&context, PrefixSettings::new(Model::gpt_5_4()))["reasoning"]["effort"], "none");
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
        context.push_function_call_output_blocks("call_1", vec![ContentBlock::text("fn main() {}")]);
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
}
