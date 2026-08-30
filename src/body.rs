//! The `POST /v1/responses` body, as bytes.
//!
//! Separate from [`crate::request`], which states what a request *is*, because
//! this file states how it is written down. Every type here is a `Wire` struct
//! beside a public type: the public type models the runtime concept, the wire
//! struct models the JSON. Hand-written rather than derived, so the emitted body
//! is readable in one place and every `type` tag is explicit.
//!
//! One rule governs the whole file. An `Option` here is a genuine runtime
//! absence, never a default elided to save bytes: a field the API documents a
//! default for is a plain field carrying that value, and it is always emitted. An
//! enclosing object whose every field is absent vanishes entirely, because `{}`
//! and no field are two different requests.

use serde::Serialize;
use serde_json::Value;

use crate::content::InputItem;
use crate::model::Model;
use crate::prefix::TextFormat;
use crate::request::{Request, Transport};
use crate::tools::{FunctionTool, ToolChoice};
use crate::values::{
    CacheMode, Include, Metadata, ReasoningEffort, ReasoningSummary, ServiceTier, Truncation, Verbosity,
};

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
