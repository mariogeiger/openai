# openai

Rust bindings for the [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) — strictly typed, cache-safe, bindings-only.

Bring your own HTTP client. This crate hands you a validated `Serialize` body; you POST it.

## Mission

Two kinds of mistake cost real money on this API, and neither shows up as a
compile error in a hand-rolled `serde_json::json!`:

1. **Parameter combinations the API refuses.** `max` reasoning effort exists on
   GPT-5.6 and not on GPT-5.5. `prompt_cache_options.ttl` is GPT-5.6's field;
   `prompt_cache_retention` is the earlier generation's. Sending the wrong one is
   a 400.
2. **Requests that silently destroy the prompt cache.** These are worse, because
   there is no error at all — only a bigger bill. OpenAI hashes the rendered
   prefix: hidden OpenAI content, then `tools`, then developer instructions, then
   the input items. Change any byte and every cached token after it is gone.

This crate makes the first kind unrepresentable and the second kind hard to
write by accident.

The tool array is the sharpest case. Measured live: an identical array read
2,969 cached tokens; the same request with one tool removed read 0 and paid to
write 2,978. Removing a tool does not save that tool's tokens — it costs the
whole prefix. So `Context` takes its tools once and never lets go of them, and
narrowing availability goes through `tool_choice`'s `allowed_tools`, which was
measured at cached 2,978 / written 0.

## Example

```rust
use openai::context::{BreakpointSlot, Context};
use openai::model::Model;
use openai::prefix::PrefixSettings;
use openai::request::Request;
use openai::tools::{AllowedToolsMode, FunctionTool, ToolChoice};
use openai::{API_BASE, HEADER_AUTHORIZATION, RESPONSES_PATH};
use serde_json::json;

// Tools are frozen at construction: the first bytes of the prefix cannot drift.
let mut context = Context::new(vec![
    FunctionTool::new("read_file", json!({"type": "object"})),
    FunctionTool::new("write_file", json!({"type": "object"})),
]);

// Reusable instructions live in a developer message, because top-level
// `instructions` cannot carry a breakpoint. The slot is anchored: never moved.
context.push_anchored_developer_text(BreakpointSlot::S0, "Stable instructions…")?;
context.push_user_text("What changed in this file?");

let prefix = PrefixSettings::new(Model::gpt_5_6_sol());
let request = Request::new(&context, prefix)?
    // Narrow availability without touching the array, and so without paying
    // to write the prefix again.
    .with_tool_choice(ToolChoice::Allowed(
        context.allow_tools(AllowedToolsMode::Auto, &["read_file"])?,
    ))
    .with_prompt_cache_key("agent_v1:user_42")
    .with_max_output_tokens(8_192)?;

reqwest::Client::new()
    .post(format!("{API_BASE}{RESPONSES_PATH}"))
    .header(HEADER_AUTHORIZATION, format!("Bearer {}", std::env::var("OPENAI_API_KEY")?))
    .json(&serde_json::to_value(&request)?)
    .send()
    .await?;
```

## What the types make impossible

| Mistake | What stops it |
| --- | --- |
| `max` effort on a model that refuses it | `EffortNoneToMax` and `EffortNoneToXhigh` are different types |
| `prompt_cache_options` and `prompt_cache_retention` together | one `match` on the model produces exactly one of them |
| A fifth cache breakpoint | `BreakpointSlot` has four variants |
| A fourth explicit breakpoint under `implicit` mode | `Request::new` returns `TooManyExplicitBreakpoints` — OpenAI's own breakpoint takes one of the four writes |
| Explicit breakpoints on a model that ignores them | `Request::new` returns `ExplicitBreakpointsUnsupported` |
| Shrinking or reordering the tool array | `Context` exposes no setter; use `allow_tools` |
| Allowing a tool that is not in the array | `AllowedTools` is built only by `Context::allow_tools` |
| Moving a breakpoint anchored on stable instructions | anchored slots return `SlotIsAnchored` |
| A breakpoint on top-level `instructions` | the field is `UncacheableInstructions`, and breakpoints attach only to content blocks |
| An out-of-range temperature or `max_output_tokens` | validating constructors returning named errors |

## Modeled

`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.5-pro`,
`gpt-5.4`. Each carries its own `max_output_tokens`, context window, knowledge
cutoff, minimum cacheable prefix, and exact per-token pricing including the
cache read and write rates.

## Not modeled

No HTTP client, no async runtime, no streaming decoder. No response
deserializer beyond `usage`, which exists because the cache accounting is the
crate's reason to be. No Chat Completions. No built-in or MCP tools — function
tools only. No stateful continuation (`previous_response_id`, `conversation`):
only the stateless path lets the caller control the prefix byte for byte.

## Install

```toml
[dependencies]
openai = { git = "https://github.com/mariogeiger/openai" }
```

## License

[MIT](LICENSE).
