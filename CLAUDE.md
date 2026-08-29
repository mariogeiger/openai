# CLAUDE.md

Design notes for the `openai` crate — Rust bindings for the Responses API.

The crate enforces one idea: *make invalid requests and broken caches unrepresentable in the type system*, so the compiler catches what the API would otherwise reject with a 400 — or worse, accept while silently charging to rewrite a prefix it could have reused.

It is the sibling of the `anthropic` crate and shares its design philosophy. Where a section here restates one there, that is deliberate: one philosophy, two wire formats. Where it differs, the difference is a fact about the Responses API and is named as such.

## 1. Prompt caching is hard to break by construction

OpenAI caches a *rendered prefix*: hidden OpenAI content, then `tools`, then developer instructions, then the input items. A cache read requires that prefix to be byte-identical, so any operation that could rewrite committed content is an operation that silently invalidates the cache. Conversation state is therefore append-only, and `Context` exposes no way to edit what it already holds.

The tool array is the sharpest case, because it is the first thing hashed after OpenAI's own content. Measured live: an identical array read 2,969 cached tokens; the same request with one tool removed read 0 and paid to write 2,978. Removing a tool does not save that tool's tokens — it costs the whole prefix. So `Context` takes its tools once, at construction, and never lets go of them; narrowing availability goes through `tool_choice`'s `allowed_tools`, measured at cached 2,978 / written 0.

Cache breakpoints live in a fixed, named set of slots (`BreakpointSlot`, four variants) that mirrors the API's four-writes-per-request limit one-to-one, so more breakpoints than the API accepts is not a value that exists. `PromptCacheBreakpoint` has no public constructor: the slots *are* the budget, and a budget you can route around is not one.

Two rules the type system cannot state are checked in `Request::new`, the single construction path: that the context's explicit breakpoints fit the budget this model and caching mode allow — `implicit` mode spends one of the four writes on OpenAI's own breakpoint — and that the model honors explicit breakpoints at all. A model that ignores them and a context full of them is a caller believing in a reusable prefix nothing reads.

Before adding any operation that mutates conversation state, convince yourself it cannot invalidate a previous cache prefix.

## 2. Unrepresentable requests are unrepresentable

Each model accepts a different subset of parameters, and the API returns 400 for invalid combinations. Model-specific parameters are carried on model-specific types, so a parameter a given model rejects does not exist on that model. The type boundary follows the *accepted parameter set*, not the model name: the three GPT-5.6 tiers accept identical parameters and share one type carrying a `Gpt5_6Tier`, while GPT-5.5, GPT-5.5 Pro, and GPT-5.4 each get their own because each accepts something the others do not.

Three concrete splits this buys:

- Reasoning effort. `EffortNoneToMax` (GPT-5.6, has `max`), `EffortNoneToXhigh` (GPT-5.5, GPT-5.4), `EffortMediumToXhigh` (Pro, which always reasons) are different types. "Pro without reasoning" is not a runtime error; it is not a sentence.
- The cache-lifetime field. `prompt_cache_options` is GPT-5.6's; `prompt_cache_retention` is the earlier generation's. One `match` on the model produces exactly one of them, which is what makes sending both impossible.
- `reasoning.mode` and `reasoning.context` exist only on GPT-5.6, so only `Gpt5_6` carries them.

Mutually exclusive settings are sum types, not independent optional fields the caller must keep in sync — `ImageSource` is `Url` or `FileId`, never two fields to reconcile; `TextFormat` is `Text` or `JsonSchema`, never a type tag beside a nullable schema.

An invariant is held by the type or it is not held. A doc comment saying a field is "set by" some method, or a private constructor that checks what a public field lets a caller assign around, is a convention the compiler does not know about. Two rules follow, and they are not in tension:

*A closed API vocabulary is an enum, never a string.* A `&'static str` field accepts any string; an enum with no invalid variant accepts only what the API does. So the enum goes in the field and the string comes out at serialization time — see `api_enum!`, which puts each wire literal in the crate exactly once. A public field of such an enum is safe and preferable: it keeps pattern matching available for free.

*A cross-field invariant means private fields plus readers.* Where validity is a relation between fields or against a model fact (`max_output_tokens` against *this model's* maximum, breakpoints against *this mode's* budget), no single field's type carries it, so the checking constructor must be the only way in. Where there is no such relation, do not hide the field.

The `compile_fail` doctest is how a claim of impossibility gets tested. A claim only a comment makes is the failure mode this section exists to prevent.

## 3. Model runtime behavior, not HTTP field presence

Types describe what the model actually *sees*, not which JSON fields happen to appear on the wire. Optional fields represent real runtime distinctions — something is present or not, configured or not — never "the field was omitted from the JSON."

When the wire format offers several shapes for one runtime concept, the type models the concept and the serializer picks the shape. Callers should not think about wire-format variants. The API accepts `"content": "hello"` as shorthand for a one-block array; this crate always emits the array, because two spellings of one message are two prefixes for one meaning.

Defaults come from the provider's documentation. **The crate invents no defaults and normalizes nothing on the caller's behalf.** Every value the API accepts is the caller's to choose, in both directions: every `with_*` that turns something on has a counterpart that turns it off or takes it back off the wire.

Which shape a field gets is *read off OpenAI's reference*, never decided here:

**A field the API documents a default for is a plain, non-`Option` field** whose `Default::default()` mirrors the documented value, and it is **always emitted**. Emitting explicitly makes the request body a complete record of what the model sees, and shields callers from silent behavior changes if OpenAI's defaults shift. These, with the documentation that puts them here:

| field | what the reference says |
| --- | --- |
| `store` | responses are "saved for 30 days by default", disabled "by setting `store` to `false`" |
| `parallel_tool_calls` | typed non-null `boolean` on the response object; every example body shows `true` |
| `text.format` | "The default format is `{ \"type\": \"text\" }` with no additional options." |
| `text.verbosity` | "The default is `medium`." |
| `reasoning.context` | "If omitted or set to `auto`, the model determines the context mode." |
| `prompt_cache_options.mode` | "Defaults to `implicit`." |
| `prompt_cache_options.ttl` | "Defaults to `30m`, which is currently the only supported value." |
| `prompt_cache_retention` on GPT-5.5 and Pro | "only `24h` is supported" — the one accepted value is the documented one |
| `tool_choice` | example bodies show `"auto"` |
| `stream` | no documented default, but a bimodal transport: a caller either reads a stream or reads a body, and there is no third state to represent |
| `ImageDetail` on an image block | "Defaults to `auto`" — and constructor-supplied, so it is stated per image |

**A field the API documents no default for is an `Option`, omitted when absent**, because presence is then a genuine runtime distinction. `reasoning.effort` is the clearest: the reference names no default, a response that never carried one reports `"effort": null`, and the *models* document four different levels for themselves. "Told a level" and "not told" are two states, and picking one would be the crate deciding how hard a model thinks. Likewise `reasoning.mode`, `reasoning.summary`, `context_management`, `max_output_tokens`, `instructions`, `prompt_cache_key`, and a function tool's `strict` — whose omission has its own documented behavior: "Responses attempts to use strict validation when the schema is compatible, and falls back to non-strict validation otherwise", a third state neither `true` nor `false` can spell.

GPT-5.4's `prompt_cache_retention` is the case that shows the rule is about *documentation*, not about types. It is `Option` on that model alone, because the reference says its default "depends on your organization's data retention policy" — `in_memory` under Zero Data Retention, `24h` otherwise. That is a default the crate cannot know, so sending any value would override a policy it cannot see, and silence is the only honest rendering.

Omission is reserved for that runtime-distinction case. It is never used for "the value happens to equal the default."

**An enclosing object vanishes when every field inside it is absent.** An empty `"reasoning": {}` is a different request from no `reasoning`, and only the second one means "the caller configured no reasoning". `ReasoningWire::of` returns `Option<Self>` for exactly this reason. The same rule makes an empty tool array absent rather than `[]`.

A per-model documented default is a *readable fact*, not an imposed value: `ModelId::default_effort` says what each model does with no `reasoning.effort`, so a caller can see what omission means without the crate choosing it. Facts about the model belong on `ModelId`; choices about the request belong on the request.

## 4. Conversation state vs. per-call parameters, and the prefix split

Conversation state — the frozen tool array, the input items, the breakpoints — is stable across turns and lives in `Context`. Per-call parameters live on `Request`, which borrows it.

This API adds a second split the caching guide forces, and naming it is the point: some per-call settings are part of the hashed prefix and some are not. The ones that are live together in `PrefixSettings`, one value a caller holds constant across a thread. The ones that are not — `tool_choice`, `prompt_cache_key`, `max_output_tokens`, `stream`, `store`, `instructions` — sit on `Request` and may vary every call for free. In hand-rolled JSON these sit side by side and look interchangeable, which is how `reasoning.effort` quietly becomes a per-request knob and every cached token disappears.

`instructions` is a newtype, `UncacheableInstructions`, because the name is the warning: the API accepts the field but refuses a breakpoint on it, so instructions meant for reuse must live in a developer message instead.

## 5. Explicit serialization, no omit-if-default

Serialization emits whatever the value represents. There is no "omit if equal to default" optimization and no hidden normalization — reading a request value tells you exactly what the model will see. Fields with a documented server-side default are still emitted explicitly (see §3).

The one kind of omission the crate uses is for optional fields genuinely absent at runtime. An absent optional is a real runtime absence, not a default elided on the wire.

Serializers are hand-written rather than derived, so the emitted body is readable in one place and every `type` tag is explicit. A `Wire` struct beside the public type is the pattern: the public type models the runtime concept, the wire struct models the bytes.

Function-call `arguments` stay the exact string the model emitted. Re-serializing a parsed value could reorder keys or change spacing, and on replay that is the prefix gone.

## 6. Scope

Bindings for both halves of the wire: the crate produces a serializable request body and decodes what comes back. No HTTP client, no retry logic, no reconnection. Callers bring their own HTTP stack and hand the bytes over.

Decoding is in scope for the same reason the request half is: a consumer that hand-matches raw JSON re-derives the API's shape badly, and the failure modes that matter — a stream that stopped early, a cache that silently did nothing — are exactly the ones a type can rule out.

Three rules keep it honest:

*Unknown is not broken.* OpenAI's compatibility policy permits new event types, and adding one is a compatible change. An unrecognized event or item kind is a variant meaning "ignore me" (`StreamEvent::Unmodeled`), never an error. What *is* an error is a frame contradicting the schema: not JSON, not an object, no `type`, a field of the wrong type.

*Incomplete is a different type from complete.* A truncated stream must not be readable as a finished response. `Settling` accumulates and cannot yield one; `Settled` is finished and cannot take more events; `settle` is the only bridge and fails on a stream that never reached a terminal event. `#[non_exhaustive]` on `Settled` makes that structural rather than decorative — a caller cannot write the literal, so a finished response can only come from a finished stream.

*Cache accounting is not optional detail.* A prefix below the model's minimum cacheable length is a silent no-op, and the only evidence is `usage`. So usage decodes rather than being skipped, a gateway reporting fewer fields than OpenAI reads as zero of that kind rather than failing the frame, and cost arithmetic is exact integer arithmetic in nanodollars — every published price and cache multiplier lands on a whole nanodollar, so nothing rounds.

Static lookup tables for API-documented wire values are part of the wire vocabulary: enum `from_str` (inverse of `as_str`) and the documented HTTP-status-to-`ErrorType` mapping. Both are pure `match` on a primitive.

The crate tracks the current GPT tiers. Older models are not wired up by default; adding one is a normal extension — follow the per-model-type approach in §2.

Out of scope for now: hosted tools (web search, file search, code interpreter, MCP), `conversation` and `previous_response_id` server-side state, `background` mode, `include`, `service_tier`, `truncation`, and the sampling parameters reasoning models refuse. `Temperature` exists as a validating newtype with no field to put it on, because every model modeled here rejects any temperature but the default; the validation is the reusable part, and where it may be sent is a per-model fact.

## 7. Details

Work in the main branch. Add every user-visible change to `CHANGELOG.md`, and keep `README.md` truthful about the surface.
