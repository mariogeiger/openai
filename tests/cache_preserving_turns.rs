//! Two consecutive turns of one agent thread, asserted as exact JSON.
//!
//! The unit tests check each type's promise in isolation. This checks the
//! property that only shows up between turns: that the second request's rendered
//! prefix *extends* the first's rather than differing from it. That is the whole
//! claim of the crate, and it is invisible to any single-request test.

use openai::content::InputBlock;
use openai::context::{BreakpointSlot, Context};
use openai::model::{EffortNoneToMax, Model};
use openai::prefix::PrefixSettings;
use openai::request::{Request, UncacheableInstructions};
use openai::tools::{AllowedToolsMode, FunctionTool, ToolChoice};
use openai::values::{AssistantPhase, ReasoningContext, ReasoningMode};
use serde_json::{Value, json};

fn tools() -> Vec<FunctionTool> {
    vec![
        FunctionTool::new("read_file", json!({"type": "object", "properties": {"path": {"type": "string"}}}))
            .with_description("Read a file")
            .with_strict_arguments(),
        FunctionTool::new("write_file", json!({"type": "object", "properties": {"path": {"type": "string"}}}))
            .with_description("Write a file")
            .with_strict_arguments(),
    ]
}

/// A thread's first turn, in full. Every field OpenAI documents a default for is
/// stated, so the prefix cannot shift under the caller's feet; every field it
/// documents none for is chosen here or absent, so the crate decides nothing.
#[test]
fn the_first_turn_of_a_thread_is_exactly_this_body() {
    let mut context = Context::new(tools());
    context.push_anchored_developer_text(BreakpointSlot::S0, "You edit files. Stable instructions.").unwrap();
    context.push_user_text("Read a.rs");

    let model = Model::gpt_5_6_sol()
        .with_effort(EffortNoneToMax::Medium)
        .with_mode(ReasoningMode::Standard)
        .with_reasoning_context(ReasoningContext::AllTurns);
    let request = Request::new(&context, PrefixSettings::new(model))
        .unwrap()
        .with_prompt_cache_key("agent_v1:user_42")
        .with_max_output_tokens(8_192)
        .unwrap();

    assert_eq!(
        serde_json::to_value(&request).unwrap(),
        json!({
            "model": "gpt-5.6-sol",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    "strict": true,
                },
                {
                    "type": "function",
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    "strict": true,
                },
            ],
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{
                        "type": "input_text",
                        "text": "You edit files. Stable instructions.",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Read a.rs"}],
                },
            ],
            "parallel_tool_calls": true,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "reasoning": {"effort": "medium", "mode": "standard", "context": "all_turns"},
            "tool_choice": "auto",
            "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
            "prompt_cache_key": "agent_v1:user_42",
            "max_output_tokens": 8_192,
            "stream": false,
            "store": true,
            "background": false,
        })
    );
}

/// The second turn must be the first plus appended items — every earlier byte
/// identical, including the anchor.
///
/// The assertion is structural rather than a second literal: strip the appended
/// items from turn two and it must equal turn one exactly. That catches drift
/// anywhere in the body, including in fields this test never names.
#[test]
fn the_second_turn_extends_the_first_prefix_byte_for_byte() {
    let mut context = Context::new(tools());
    context.push_anchored_developer_text(BreakpointSlot::S0, "You edit files. Stable instructions.").unwrap();
    context.push_user_text("Read a.rs");

    let prefix = PrefixSettings::new(Model::gpt_5_6_sol());
    let build = |context: &Context| -> Value {
        serde_json::to_value(
            Request::new(context, prefix.clone())
                .unwrap()
                .with_prompt_cache_key("agent_v1:user_42")
                .with_max_output_tokens(8_192)
                .unwrap(),
        )
        .unwrap()
    };
    let first = build(&context);
    let first_item_count = first["input"].as_array().unwrap().len();

    // The model answers with a tool call; we append the call, its result, and
    // the assistant turn, then roll a second breakpoint onto the tool result.
    context.push_reasoning("rs_1", "opaque-reasoning-payload");
    context.push_function_call("call_1", "read_file", r#"{"path":"a.rs"}"#);
    context.push_function_call_output_blocks("call_1", vec![InputBlock::text("fn main() {}")]);
    context.roll_breakpoint(BreakpointSlot::S1).unwrap();
    context.push_assistant_text(AssistantPhase::FinalAnswer, "It is a main function.");
    context.push_user_text("Now rename it.");

    let second = build(&context);

    let mut truncated = second.clone();
    truncated["input"] =
        Value::Array(second["input"].as_array().unwrap().iter().take(first_item_count).cloned().collect());
    assert_eq!(truncated, first, "turn two must extend turn one, not differ from it");

    // Two breakpoints, both within the three that implicit mode leaves free.
    let kinds: Vec<&str> = second["input"].as_array().unwrap().iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(
        kinds,
        ["message", "message", "reasoning", "function_call", "function_call_output", "message", "message"]
    );
    assert_eq!(second["input"][0]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
    assert_eq!(second["input"][4]["output"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
    assert_eq!(context.breakpoint_count(), 2);
}

/// The measured pair, as bytes.
///
/// Narrowing availability through `tool_choice` changes exactly one field and
/// leaves the hashed prefix identical — which is why it read 2,978 cached tokens
/// and wrote none, where removing a tool read 0 and paid to write 2,978.
#[test]
fn restricting_tools_changes_one_field_and_no_prefix_byte() {
    let mut context = Context::new(tools());
    context.push_anchored_developer_text(BreakpointSlot::S0, "You edit files.").unwrap();
    context.push_user_text("Read a.rs");

    let prefix = PrefixSettings::new(Model::gpt_5_6_sol());
    let unrestricted = serde_json::to_value(Request::new(&context, prefix.clone()).unwrap()).unwrap();

    let allowed = context.allow_tools(AllowedToolsMode::Auto, &["read_file"]).unwrap();
    let mut restricted =
        serde_json::to_value(Request::new(&context, prefix).unwrap().with_tool_choice(ToolChoice::Allowed(allowed)))
            .unwrap();

    assert_eq!(
        restricted["tool_choice"],
        json!({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "read_file"}]})
    );
    restricted["tool_choice"] = json!("auto");
    assert_eq!(restricted, unrestricted, "only tool_choice may differ");
}

/// Per-request instructions that are meant *not* to be cached go in the
/// top-level field, which cannot carry a breakpoint; reusable ones go in a
/// developer block, which can. Both appear here, correctly separated.
#[test]
fn stable_and_volatile_instructions_land_in_different_places() {
    let mut context = Context::new(vec![]);
    context.push_anchored_developer_text(BreakpointSlot::S0, "Stable rubric and examples.").unwrap();
    context.push_user_text("Interaction to judge.");

    let request = Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol().with_explicit_cache_only()))
        .unwrap()
        .with_instructions(UncacheableInstructions::new("Today is 2026-08-28."));
    let v = serde_json::to_value(&request).unwrap();

    assert_eq!(v["instructions"], "Today is 2026-08-28.");
    assert!(v["instructions"].get("prompt_cache_breakpoint").is_none());
    assert_eq!(v["input"][0]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
    // Explicit-only mode: no implicit breakpoint writes through the volatile tail.
    assert_eq!(v["prompt_cache_options"], json!({"mode": "explicit", "ttl": "30m"}));
}
