# Changelog

All notable changes to this crate are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
