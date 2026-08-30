# Repository instructions

## Authority

- Treat [`SOUL.md`](SOUL.md) as the authority for the mission and the design
  principles. When a change and a principle disagree, the principle wins or the
  principle changes explicitly — never silently.
- `README.md` is the concise entrance: what the crate is, one worked example,
  what it covers, what it does not. Keep it true after every change; a README
  that overstates coverage is worse than one that says nothing.
- `CHANGELOG.md` records user-visible changes, including the reasoning behind a
  breaking one and the migration for it.

## Where things live

| file | its one job |
| --- | --- |
| `src/lib.rs` | the crate's own documentation, the endpoint constants, module list |
| `src/values.rs` | closed API vocabularies, via `api_enum!` — one wire literal, stated once |
| `src/model.rs` | per-model types: accepted parameter sets, effort ranges, limits, pricing |
| `src/content.rs` | input items and content blocks, and the role/vocabulary pairing |
| `src/context.rs` | append-only conversation state, the frozen tool array, breakpoint slots |
| `src/prefix.rs` | per-call settings that are *inside* the hashed prefix |
| `src/request.rs` | per-call settings outside the prefix, validation, body serialization |
| `src/tools.rs` | function tools and `tool_choice` |
| `src/stream.rs` | one streamed frame becomes one typed event |
| `src/settle.rs` | a sequence of events becomes a finished response |
| `src/response.rs` | the non-streaming response body |
| `src/usage.rs` | the `usage` object and exact cost arithmetic over it |

Give every file one bounded mission and split it before 1,000 lines. A public
serialization shape lives beside the public type it renders, as a `Wire` struct:
the public type models the runtime concept, the wire struct models the bytes.

## Required checks

Before every commit, all four, none weakened:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo doc --no-deps` with no warnings

`#![deny(missing_docs)]` is on. Document **why an item exists**, not what it is:
"the identifier the matching `function_call_output` must repeat" beats "the call
id". A doc comment that only restates the name adds nothing and must not be
written to satisfy the lint.

`compile_fail` doctests are load-bearing, not decoration. When you add one,
verify it fails for the reason you claim — temporarily fix the stated cause and
watch it start compiling — because a `compile_fail` that fails for a typo passes
while proving nothing.

## Adding API surface

The mission is whole coverage, so a gap is a defect with a date on it, not a
design choice. When closing one:

1. **Read the reference, not memory.** Fetch
   <https://developers.openai.com/api/reference/resources/responses/methods/create>
   and the streaming-events page. Appending `.md` to any docs page URL yields
   markdown, which is far easier to diff against the crate.
2. **Decide the field's shape from the documentation**, per `SOUL.md`: a
   documented default means a plain always-emitted field, no documented default
   means an `Option`.
3. **Model the concept, not the JSON.** Several wire spellings of one runtime
   idea collapse into one type plus a serializer choice.
4. **Land it as its own commit**, with its own changelog entry. One coherent
   piece per commit; a single enormous change hides which part broke.

For a new streaming event, decode the fields a consumer acts on and let the rest
of the frame be ignored. Every event carries `sequence_number`, and most carry
`item_id`; decode them only where a consumer needs them, and never *require* a
field the wire may omit.

## Versioning

Semantic versioning, and the crate is pre-1.0, so a breaking change bumps the
minor. Bump the version, add the changelog section, and update `Cargo.lock` in
the same commit as the change that needs it — a released version whose changelog
is a commit behind is a version nobody can trust.

## Verifying against the live API

A captured real frame beats an invented fixture, and this rule was earned: a
hand-written fixture omitted an `output_index` the wire always sends, and an
assistant `input_text` block returned 400 where the tests said 200. So when a
shape is in doubt, capture it.

- Never print, log, or commit a credential, and never put one in a test, an
  error message, or a doc example.
- Live traffic costs money and capacity. Capture once, save the frames, and test
  against the saved capture afterwards.
- A captured frame goes into `tests/` as data with its provenance stated: which
  endpoint, which model, what was asked.
- Gateways are not OpenAI. A capture through one may carry extra fields, omit
  optional ones, and report a thinner `usage`. Decoding must tolerate all three,
  which is why absent counters read as zero rather than failing the frame.

## The remaining gap, in priority order

The mission is whole coverage, so what is missing is written down rather than
left to be rediscovered. Priority is by what a real consumer needs.

1. **Hosted-tool definitions.** Their events and call items decode; the
   request-side definitions that turn them on do not exist. 15 tool types, each
   with its own configuration: `web_search`, `file_search`, `code_interpreter`,
   `image_generation`, `mcp`, `shell`, `local_shell`, `computer_use_preview`,
   `apply_patch`, `custom`, `tool_search`, `namespace`, `programmatic_tool_calling`,
   `additional_tools`, `web_search_preview`. The highest-value first, and each is
   its own commit. Note the caching consequence: a hosted tool joins the `tools`
   array, so adding one rewrites the whole prefix — the same fact that froze the
   array in the first place.
2. **`prompt`, the reusable prompt template.** `id`, `version`, and a variables
   map whose values are strings or input blocks. Server-side content the caller
   does not see, so its prefix implications need measuring before it is modeled.
3. **The remaining input item kinds.** The request side models messages, function
   calls, function outputs, and replayed reasoning. The reference lists 32 item
   kinds; the rest are hosted-tool calls and their outputs, plus `item_reference`
   and the compaction items. Model them alongside the tool whose calls they are.
4. **`moderation`.** A model name plus per-direction `score` or `block` policies.
5. **Five shell-call streaming events.** Their payload is a structured command
   list a caller running commands has to agree with exactly; model it with the
   `shell` tool definition, not before.
6. **`Response` for the non-streaming path, more deeply typed.** `Response::raw`
   keeps every field, and the ones a caller reads through it — `created_at`,
   `tool_usage`, `billing` — are candidates for names of their own once it is
   clear which are load-bearing.
7. **`scoped_content` and `input_audio`.** Both are in the documented input
   vocabulary and neither is modeled. Confirm live what each accepts first: the
   reference does not describe `scoped_content` at all.

## Deliberately out of scope

`SOUL.md` states these with reasons; do not add them without changing that file
first. Chat Completions and every other endpoint. `previous_response_id` and
`conversation`, because stateful continuation hands prefix control to the server.
An HTTP client, an async runtime, SSE transport, retries, reconnection.

## Repository workflow

- Work on `main`. A multi-file change goes through a worktree on a branch,
  fast-forwarded onto `main` when its checks pass.
- Preserve unrelated concurrent changes; inspect the working tree again before
  merging.
- The repository is public. Push only with the owner's authority.
