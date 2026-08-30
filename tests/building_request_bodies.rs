//! The request body a caller builds, asserted through the public API alone.
//!
//! Outside the crate rather than in a `mod tests`, and that is itself an
//! assertion: everything needed to build and inspect a body is reachable from
//! outside. A case that needed `pub(crate)` access would have found a type the
//! crate has not finished exposing.
//!
//! The two exhaustive-body tests are the ones the design rests on. Each states a
//! whole body as a literal, so a field drifting into or out of the wire is named
//! by a failing test rather than discovered in production.

use openai::content::InputBlock;
use openai::context::{BreakpointSlot, CACHE_WRITE_SLOTS, Context};
use openai::model::{EffortMediumToXhigh, EffortNoneToMax, EffortNoneToXhigh, Gpt5_6, Gpt5_6Tier, Model};
use openai::prefix::{PrefixSettings, TextFormat};
use openai::request::{CacheWriteBudget, Request, RequestError, UncacheableInstructions};
use openai::tools::{AllowedToolsMode, FunctionTool, ToolChoice};
use openai::values::{
    CacheRetention, Include, Metadata, ReasoningContext, ReasoningMode, ReasoningSummary, ServiceTier, Truncation,
    Verbosity,
};
use serde_json::{Value, json};

/// The two-tool array these tests share, so a body assertion is about the field
/// under test rather than about the tools beside it.
fn tools() -> Vec<FunctionTool> {
    vec![
        FunctionTool::new("read_file", json!({"type": "object"})),
        FunctionTool::new("write_file", json!({"type": "object"})),
    ]
}

/// The body a request serializes to, which is the only thing these tests read.
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
    assert_eq!(chosen.effort(), Some(openai::values::ReasoningEffort::Max));
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
    use openai::model::ModelId;
    use openai::values::ReasoningEffort;
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
    let withdrawn = PrefixSettings::new(Model::gpt_5_4().with_retention(CacheRetention::InMemory).without_retention());
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
    context.push_assistant_text(openai::values::AssistantPhase::FinalAnswer, "It is a main function.");

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
    let request =
        Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol())).unwrap().with_tool_choice(ToolChoice::None);
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
    let reverted = of(Request::new(&context, prefix()).unwrap().streaming_with_obfuscation(false).without_streaming());
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
