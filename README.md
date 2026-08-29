# openai

Rust bindings for the [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) — strictly typed, cache-safe, bindings-only.

Bring your own HTTP client. This crate hands you a validated `Serialize` body; you
POST it, then feed the streamed frames back in to get a typed response.

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

A third kind is the crate's own to avoid: **deciding a value the caller should
decide.** Every value the API accepts is settable here, in both directions, and
the crate invents no defaults. Which fields go on the wire is read off OpenAI's
reference rather than chosen — see [Who chooses a value](#who-chooses-a-value).

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
| An empty `"reasoning": {}` where no reasoning was configured | the object is built by `ReasoningWire::of`, which returns nothing when every field inside is absent |
| Reading a half-finished stream as a finished response | `Settling` has no method returning one, and `Settled` comes only from `settle()` |
| Fabricating a `Settled` to bypass that | the struct is `#[non_exhaustive]`, so the literal will not compile outside the crate |
| A new server-side event type breaking the decoder | unknown types decode as `StreamEvent::Unmodeled`, never an error |
| Mistaking malformed function arguments for no arguments | `FunctionArguments::decode` returns `Err`; empty arguments return `{}` |
| Re-serializing function arguments and losing the prefix | `FunctionArguments` keeps the model's bytes; decoding is a separate step |

## Who chooses a value

The caller, always. The crate emits no value the caller did not choose, and
invents no default of its own. Two shapes carry that, and which one a field gets
comes from OpenAI's reference:

**The API documents a default** → plain field, `Default` equal to the documented
value, **always emitted**. The body is then a complete record of what the model
sees, and stays that way the day OpenAI changes a default.

| field | what the reference says |
| --- | --- |
| `store` | "saved for 30 days by default", disabled "by setting `store` to `false`" |
| `parallel_tool_calls` | typed non-null on the response; examples show `true` |
| `text.format` | "the default format is `{ \"type\": \"text\" }`" |
| `text.verbosity` | "the default is `medium`" |
| `reasoning.context` | "if omitted or set to `auto`, the model determines" |
| `prompt_cache_options.mode` / `.ttl` | "defaults to `implicit`" / "defaults to `30m`" |
| `prompt_cache_retention` (GPT-5.5, Pro) | "only `24h` is supported" |
| `tool_choice`, `stream` | `auto`; and streaming is a two-state transport |

**The API documents no default** → `Option`, omitted when absent, because
presence is a real runtime distinction:
`reasoning.effort`, `reasoning.mode`, `reasoning.summary`, `context_management`,
`max_output_tokens`, `instructions`, `prompt_cache_key`, a tool's `strict`, and
GPT-5.4's `prompt_cache_retention`.

`reasoning.effort` is the field this matters most for. The reference names no
default, a response that never carried one reports `"effort": null`, and the four
*models* document four different levels for themselves. So the crate sends
nothing unless told, and `ModelId::default_effort` states what each model does
with silence — a readable fact, not an imposed value:

```rust
use openai::model::{EffortNoneToMax, Model, ModelId};
use openai::prefix::PrefixSettings;

// Nothing chosen: no `reasoning.effort` on the wire at all.
assert_eq!(PrefixSettings::new(Model::gpt_5_6_sol()).effort(), None);
// What the model will do anyway, readable without being sent.
assert_eq!(ModelId::Gpt5_6Sol.default_effort(), openai::ReasoningEffort::Medium);
// Or say it outright.
let pinned = Model::gpt_5_6_sol().with_effort(EffortNoneToMax::Xhigh);
```

GPT-5.4's `prompt_cache_retention` shows the rule is about documentation rather
than types: it is `Option` on that model alone, because its default "depends on
your organization's data retention policy". That is a default the crate cannot
know, so silence is the only honest rendering.

An enclosing object vanishes when every field inside is absent. An empty
`"reasoning": {}` is a different request from no `reasoning`.

## Modeled

`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.5-pro`,
`gpt-5.4`. Each carries its own `max_output_tokens`, context window, knowledge
cutoff, minimum cacheable prefix, and exact per-token pricing including the
cache read and write rates.

## Reading the stream back

A request is half the API; the streamed answer is the other half. `stream`
decodes one Server-Sent Event payload into a `StreamEvent`, and `settle`
accumulates a sequence of them.

```rust
use openai::settle::{Outcome, Settling};
use openai::stream::data_payload;

let mut settling = Settling::new();
for line in sse_body.lines() {
    if let Some(payload) = data_payload(line) {
        settling.consume_payload(payload)?;   // decode and fold in
        print!("{}", settling.text_so_far()); // readable while in flight
    }
}

// The only way to obtain a finished response, and it fails on a stream that
// never sent a terminal event.
let settled = settling.settle()?;
match &settled.outcome {
    Outcome::Completed => println!("{}", settled.text),
    Outcome::Incomplete { reason } => println!("cut short: {reason:?}"),
    Outcome::Failed { error } | Outcome::Errored { error } => println!("{}", error.message),
}
for call in settled.function_calls() {
    let arguments = call.arguments.decode()?; // the JSON string, decoded here
    println!("{} {arguments}", call.name);
}
```

Two decisions carry the weight.

**An unknown event is a variant, never an error.** OpenAI's compatibility
promise lists "adding new event types in streaming APIs" as backwards
compatible, so a decoder that errors on an unrecognized event is one a routine
server-side release will break. `StreamEvent::Unmodeled` is that case, and it is
also how documented-but-uncovered events read — "well formed, nothing to do" is
one situation, not two.

**An unfinished stream has a different type from a finished one.** A dropped
connection leaves text that looks exactly like a complete answer. `Settling`
therefore has no method that returns a response, and `Settled` is reachable only
through `Settling::settle`, which returns `SettleError::Truncated` when no
terminal event ever arrived.

Covered events: `response.output_text.delta`,
`response.reasoning_summary_text.delta`, `response.output_item.added`,
`response.output_item.done`, `response.completed`, `response.failed`,
`response.incomplete`, and the bare `error` event. The other 50 documented
`response.*` events decode as `Unmodeled`; `CHANGELOG.md` says why for each
group.

## Not modeled

No HTTP client, no async runtime, no SSE transport — you own the socket and hand
this crate one `data:` payload at a time. Of the response body, only `usage` and
the streamed shapes are deserialized. No Chat Completions. No built-in or MCP
tools — function tools only. No stateful continuation (`previous_response_id`,
`conversation`): only the stateless path lets the caller control the prefix byte
for byte.

## Install

```toml
[dependencies]
openai = { git = "https://github.com/mariogeiger/openai" }
```

## License

[MIT](LICENSE).
