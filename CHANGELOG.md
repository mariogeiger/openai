# Changelog

All notable changes to this crate are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-29

The caller chooses every value, and the crate invents none.

`Request::new` used to render `reasoning.effort`, `reasoning.mode`,
`prompt_cache_retention` on GPT-5.4, and a function tool's `strict` whether the
caller had asked for them or not — and there was no way at all to say
"send nothing here". An imposed `reasoning.effort` made a model reason at
`medium` when the caller meant to let the model use its own documented level.

The rule now written down in `CLAUDE.md` §3 and §5, shared with the sibling
`anthropic` crate: a field OpenAI **documents a default for** is a plain field
carrying that documented value and is *always* emitted, so the body is a
complete record of what the model sees and stays stable if OpenAI's defaults
shift. A field OpenAI **documents no default for** is an `Option`, omitted when
absent, because presence is then a real runtime distinction. Which shape a field
gets is read off the API reference, not chosen here.

### Changed — breaking

- `reasoning.effort` is now sent only when chosen. Every model type's `effort`
  is `Option<_>` and starts `None`; `with_effort` still takes a bare effort, and
  `without_effort` takes it back off. `PrefixSettings::effort` returns
  `Option<ReasoningEffort>`.

  **Migration:** a caller who wants the old behavior names the level, which was
  always that model's own documented default — `Model::gpt_5_6_sol().with_effort(EffortNoneToMax::Medium)`,
  `Model::gpt_5_5().with_effort(EffortNoneToXhigh::Medium)`,
  `Model::gpt_5_5_pro().with_effort(EffortMediumToXhigh::High)`,
  `Model::gpt_5_4().with_effort(EffortNoneToXhigh::None)`. A caller who wants the
  model to decide changes nothing. `ModelId::default_effort` states each level as
  a readable fact, so omission is legible without being imposed.

- `reasoning.mode` is now sent only when chosen. `Gpt5_6::mode` is
  `Option<ReasoningMode>` and starts `None`. **Migration:**
  `.with_mode(ReasoningMode::Standard)` to keep sending `standard`.

- `reasoning.context` now defaults to the value OpenAI documents. The reference
  says "if omitted or set to `auto`, the model determines the context mode", so
  `ReasoningContext` gains an `Auto` variant and `Gpt5_6::reasoning_context`
  starts there instead of at `AllTurns` — the GPT-5.6 family resolves `auto` to
  `all_turns` anyway, so behavior is unchanged while the choice stops being the
  crate's. **Migration:** `.with_reasoning_context(ReasoningContext::AllTurns)`
  to pin the value on the wire.

- The whole `reasoning` object now vanishes when every field inside it is absent.
  An empty `"reasoning": {}` is a different request from no `reasoning`, and only
  the second means "no reasoning was configured".

- `prompt_cache_retention` on GPT-5.4 is now sent only when chosen.
  `Gpt5_4::retention` is `Option<CacheRetention>` and starts `None`, with
  `without_retention` to clear it. GPT-5.4 alone documents no value as its
  default — the reference says it "depends on your organization's data retention
  policy", `in_memory` under Zero Data Retention and `24h` otherwise — so any
  value the crate sent would override a policy it cannot see. GPT-5.5 and
  GPT-5.5 Pro are unchanged: `24h` is the only value they accept, so it stays a
  plain always-emitted field. **Migration:**
  `Model::gpt_5_4().with_retention(CacheRetention::TwentyFourHours)` to keep
  sending `24h`.

- A function tool's `strict` is now sent only when chosen. `FunctionTool::new`
  no longer imposes `true`, because omission has its own documented behavior:
  "Responses attempts to use strict validation when the schema is compatible,
  and falls back to non-strict validation otherwise" — a third state that
  neither `true` nor `false` can spell. **Migration:** `.with_strict_arguments()`
  to keep sending `true`; `with_loose_arguments` is unchanged, and
  `with_inferred_argument_strictness` returns to omission.

### Added

- Every setting now has a way to say it in both directions, so no choice is
  one-way: `Request::with_storage` and `without_streaming` beside
  `without_storage` and `streaming`; `PrefixSettings::with_parallel_tool_calls`,
  `without_compaction`, `without_reasoning_summary`; `Gpt5_6::with_caching` for
  the whole `prompt_cache_options` object rather than only its `explicit`
  corner; `without_effort` on all four model types; `Gpt5_4::without_retention`;
  `FunctionTool::with_strict_arguments` and
  `with_inferred_argument_strictness`.

- `ModelId::default_effort` — the effort each model's own page documents for a
  request that sends none. Stated so a caller can see what omission means;
  deliberately not applied, because four models with four different levels is
  precisely why the field has no default at the field level.

- `ReasoningContext::Auto`, the value the reference names as this field's
  default.

- `CLAUDE.md`, stating the design philosophy this crate shares with the sibling
  `anthropic` crate, adapted to the Responses API.

### Unchanged, and deliberately so

`store`, `parallel_tool_calls`, `text.format`, `text.verbosity`,
`prompt_cache_options.mode` and `.ttl`, `tool_choice`, and `stream` remain plain
fields that are always emitted, because OpenAI documents a default for each and
a request should be a complete record of what the model sees. `store` is the case
that proves the rule: it stays `true` by default and always on the wire, and a
caller who wants nothing retained writes `without_storage` and can read the
`false` back off the request.

## [0.2.2] — 2026-08-29

### Fixed

- A streamed message item now carries the text its own deltas built. An item is
  announced with empty content and filled by `response.output_text.delta`, so
  the deltas keyed to an item's `output_index` *are* that item's text — the key
  already knew the index, and `settle` now uses it.

  Before this, prose said beside a function call settled as an empty message
  item. `Settled::text` had the words, but a caller reading `items` — which is
  how a caller keeps the answer in document order, one text part and one call at
  a time — saw an empty message and lost the sentence the model said before it
  called anything. Observed live: a model asked to speak before calling a tool
  sends the message at index 0 and the call at index 1.

  Both halves of "the deltas at an index are the item at that index" now hold:
  an announced message is filled with its own deltas, and text at an index
  nothing announced becomes a message item there. A provider that streams deltas
  and never repeats its `output` array is the ordinary case, so a caller reading
  `items` finds the answer either way.

## [0.2.1] — 2026-08-29

### Fixed

- A streamed `usage` object reporting only `input_tokens` and `output_tokens`
  now decodes, instead of failing the frame that carried it. `Usage` and both
  breakdown structs are `#[serde(default)]`, so an omitted `input_tokens_details`
  or `output_tokens_details` reads as zero of that kind — the same rule that
  already applied to a count *inside* a breakdown, now applied to the breakdown
  itself.

  A gateway in front of OpenAI reports fewer fields than OpenAI does, and
  refusing the object turned a thin cost report into a broken response: the whole
  turn failed over an omitted cache count. Zero is the honest reading of an
  absent counter, and an object whose counts contradict the schema is still
  `FrameError::UndecodableUsage`.

## [0.2.0] — 2026-08-29

The other half of the API: a streamed response is now decoded, not hand-matched
on `event["type"]`.

### Added

- `stream`: `StreamEvent` is one Server-Sent Event payload, decoded. `decode`
  takes the raw `data:` bytes, `from_json` takes an already-parsed frame, and
  `data_payload` strips SSE framing so a caller can pass every line through.
  `FrameError` names the ways a frame can be broken. `OutputItem` covers
  assistant messages, function calls and reasoning; `FunctionArguments` holds a
  function call's `arguments` as the JSON *string* the model emitted, with
  `decode` as a separate step. `ResponseSnapshot` carries what a terminal event
  reports, reusing `usage::Usage` rather than a second set of token types.
- `settle`: `Settling` accumulates events and `Settled` is a finished response.
  `Outcome` is how the stream ended, carrying that ending's own data.
  `Settling::text_so_far` and `reasoning_summary_so_far` serve a live display;
  `Settled::function_calls` and `reasoning_items` serve the next turn.
- `values`: `AssistantPhase` now round-trips, since it is read back off a
  streamed `message` output item as well as sent.

### What the new types make impossible

- **Reading a half-finished stream as a finished response.** `Settling` has no
  method that returns a response, and `Settled` is reachable only through
  `Settling::settle`, which returns `SettleError::Truncated` when no terminal
  event arrived. `Settled` is `#[non_exhaustive]`, so the struct literal will
  not compile outside this crate and the check cannot be walked around.
- **A server-side release breaking the decoder.** An unrecognized event type is
  `StreamEvent::Unmodeled`, never a `FrameError`. OpenAI's compatibility promise
  names new streaming event types as backwards-compatible, so this is a
  correctness requirement rather than leniency.
- **Confusing malformed function arguments with absent ones.** `decode` returns
  the empty object for empty or blank arguments and `Err` for anything else. A
  malformed argument string never fails the surrounding stream: the call, its
  `call_id`, its name and its raw bytes all survive, so the caller can answer
  the model with a tool error instead of losing the turn.
- **Losing the prompt cache by re-serializing arguments.** `FunctionArguments`
  keeps the model's own bytes, so a replayed call is byte-identical.
- **Silently under-reporting cost.** A `usage` object that will not deserialize
  is `FrameError::UndecodableUsage`, not an absent usage; an explicit `null` or
  an omitted field is `None`.

### Streaming events covered

Verified against the `response.*` list in OpenAI's streaming-events reference,
which documents 57 of them plus the bare `error` event.

Modeled as their own variants — the eight a text-and-tools consumer needs:
`response.output_text.delta`, `response.reasoning_summary_text.delta`,
`response.output_item.added`, `response.output_item.done`,
`response.completed`, `response.failed`, `response.incomplete`, and `error`.

Decoded as `Unmodeled`, deliberately, in these groups:

- **Lifecycle duplicates** — `response.created`, `response.in_progress`,
  `response.queued`. They announce a status the terminal event states
  authoritatively, so acting on them adds nothing.
- **`done` and part events for text already assembled from deltas** —
  `response.output_text.done`, `response.content_part.added`,
  `response.content_part.done`, `response.reasoning_summary_text.done`,
  `response.reasoning_summary_part.added`,
  `response.reasoning_summary_part.done`, `response.reasoning_text.delta`,
  `response.reasoning_text.done`. The deltas plus
  `response.output_item.done` already carry the final text; consuming both
  would risk counting it twice.
- **Function-call argument deltas** — `response.function_call_arguments.delta`
  and `.done`. Arguments are taken whole from `response.output_item.done`,
  which is the authoritative form. Streaming a half-built argument string
  invites parsing it while incomplete.
- **Hosted tools this crate does not model at all** — the `file_search_call`,
  `web_search_call`, `code_interpreter_call`, `code_interpreter_call_code`,
  `image_generation_call`, `mcp_call`, `mcp_call_arguments`, `mcp_list_tools`,
  `custom_tool_call_input` and `shell_call*` families. The request side models
  function tools only, so these cannot occur for a request this crate built;
  the corresponding `output_item` kinds read as `OutputItem::Unmodeled`.
- **Audio** — `response.audio.delta`, `response.audio.done`,
  `response.audio.transcript.delta`, `response.audio.transcript.done`. Out of
  scope with the rest of the audio surface.
- **Refusals and annotations** — `response.refusal.delta`,
  `response.refusal.done`, `response.output_text.annotation.added`. A refusal
  is not answer text and is deliberately not concatenated into it; surfacing it
  as a distinct outcome is a product decision this crate has not made yet.

### Changed

- The crate documentation and `README.md` no longer say a streaming decoder is
  out of scope, because it no longer is. What remains out of scope is the
  transport: no HTTP client, no async runtime, no SSE reader.

## [0.1.0] — 2026-08-28

First release. Covers creating a response, streaming or not, with the caching
controls as the organizing concern.

### Added

- `model`: one type per accepted-parameter set — `Gpt5_6` over the Sol, Terra
  and Luna tiers, plus `Gpt5_5`, `Gpt5_5Pro`, and `Gpt5_4`. Each carries only
  the reasoning efforts and the caching field its generation accepts, and
  `ModelId` carries the documented facts: `max_output_tokens`,
  `context_window_tokens`, `knowledge_cutoff`, `min_cacheable_prefix_tokens`,
  and exact per-token `Pricing` including cache read and write rates.
- `context`: append-only conversation state with a tool array frozen at
  construction, and four named cache-write slots matching the API's four-writes
  limit one-to-one. Slots may be anchored, which refuses every later move.
  `allow_tools` builds a checked `AllowedTools` — the way to vary availability
  without disturbing the array.
- `content`: input items and content blocks. `prompt_cache_breakpoint` has no
  public constructor, so slots are the only route to one.
- `tools`: `FunctionTool` with a JSON-Schema parameter object, strict by
  default, and `ToolChoice` in all its forms including `allowed_tools`.
- `prefix`: `PrefixSettings` gathers every setting that changes the hashed
  prefix — model, `parallel_tool_calls`, `text.format`, `reasoning.effort`,
  `text.verbosity`, `context_management` — into one value to hold constant
  across a thread. `Temperature` and `TextFormat` validate here.
- `request`: `Request` holds only settings OpenAI does not hash, so they may
  vary every call. `Request::new` validates the cache-write budget against the
  model and its caching mode. `UncacheableInstructions` names the top-level
  field that cannot carry a breakpoint.
- `usage`: the `usage` shape with `cached_tokens` and `cache_write_tokens`,
  plus `cache_hit_rate` and exact integer `cost_nanodollars`.
- `values`: the wire vocabulary, with `from_str` and an HTTP-status table for
  the response-side enums.
