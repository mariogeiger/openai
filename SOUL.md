# Soul

## Mission

**Represent the entire OpenAI Responses API faithfully in types.**

That is why the crate exists. Every parameter the endpoint accepts, every item
it takes as input, every event it streams back, and every field it reports —
each expressed as a Rust type that admits what the API admits and refuses what
the API refuses. The measure of the crate is not how convenient it is; it is how
much of the API is unreachable through it and how little of what it reaches is
wrong.

Faithful cuts both ways. The crate must not be *narrower* than the API — a
parameter it cannot express is a caller who has to leave it and hand-roll JSON,
which loses every guarantee below. And it must not be *wider* — a request the
API refuses must not be a value that exists.

Two consequences of faithfulness deserve their own names, because they are what
the type work buys:

1. **A request the API answers with 400 does not compile.** Each model accepts a
   different parameter set, and the parameters interact. Model-specific
   parameters therefore live on model-specific types, and mutually exclusive
   settings are sum types rather than optional fields a caller must keep in
   sync.
2. **A request that silently destroys the prompt cache is hard to write by
   accident.** This one has no error at all, only a larger bill, so the type
   system is the only place it can be caught.

## Design principles

### The tool array is the first bytes of the hashed prefix

OpenAI caches a *rendered* prefix, in this order: hidden OpenAI content, then
`tools`, then developer instructions and `instructions`, then the input items. A
cache read requires that prefix to be byte-identical, so the tool array is the
sharpest thing in the request: growing it, shrinking it, or merely reordering it
costs the entire prefix, not just the tokens of the tool that moved.

Measured live: an identical array read 2,969 cached tokens; the same request with
one tool removed read 0 and paid to write 2,978. So `Context` takes its tools
once, at construction, and never exposes a way to change them. Varying which
tools are callable goes through `ToolChoice::Allowed`, which restricts the
callable set while leaving the array intact — measured at cached 2,978,
written 0.

Conversation state is append-only for the same reason. Before adding any
operation that mutates it, convince yourself it cannot invalidate a prefix a
previous request already paid to write.

### Cache writes are a bounded resource

The API writes at most four cache breakpoints per request, and implicit caching
spends one of the four on OpenAI's own. So breakpoints live in a fixed, named
set of slots — `BreakpointSlot`, four variants — that mirrors the limit
one-to-one, and `PromptCacheBreakpoint` has no public constructor. The slots
*are* the budget, and a budget you can route around is not one: a fifth
breakpoint must be unwritable, not merely rejected.

Two rules the type system cannot state are checked in `Request::new`, the single
construction path: that the context's explicit breakpoints fit the budget this
model and caching mode allow, and that the model honors explicit breakpoints at
all. A model that ignores them plus a context full of them is a caller believing
in a reusable prefix nothing reads.

### The crate invents no defaults and normalizes nothing

Every value the API accepts is the caller's explicit choice. Which *shape* a
field gets is read off OpenAI's reference, never decided here:

- **A field the API documents a default for is a plain, always-emitted field**
  whose `Default` mirrors the documented value. The body is then a complete
  record of what the model sees, and it stays that way the day OpenAI changes a
  default.
- **A field the API documents no default for is an `Option`, omitted when
  absent**, because presence is then a genuine runtime distinction.
  `reasoning.effort` is the clearest: "told a level" and "not told" are two
  states, and choosing one would be the crate deciding how hard a model thinks.

Omission is reserved for that runtime-absence case. It is never used because a
value happens to equal a default. An enclosing object vanishes when every field
inside it is absent, because `"reasoning": {}` and no `reasoning` are two
different requests.

Where the wire offers several spellings of one concept, the type models the
concept and the serializer picks one spelling. The API accepts
`"content": "hello"` as shorthand for a one-block array; this crate always emits
the array, because two spellings of one message are two prefixes for one
meaning.

A per-model documented default is a *readable fact*, not an imposed value:
`ModelId::default_effort` says what a model does with no `reasoning.effort`, so a
caller can see what omission means without the crate choosing it. Facts about
the model belong on `ModelId`; choices about the request belong on the request.

### The role decides which content blocks it can hold

A field whose legal values depend on another field is not a separate field. One
shared content type once served two roles with different vocabularies, so an
assistant message spelled its text `input_text` and every request carrying an
assistant turn — nearly every real conversation — was refused with a live 400:

```text
Invalid value: 'input_text'. Supported values are: 'output_text' and 'refusal'.
```

A `match` on the role at serialization time would have fixed the bytes and left
the wrong pairing writable. The fix is that the role and its vocabulary are one
value: `Message::Input { role: InputRole, content: Vec<InputBlock> }` or
`Message::Assistant { phase, content: Vec<OutputBlock> }`. `InputRole` has no
`Assistant` variant, and no type holds an assistant role beside an input block.

| role | blocks it accepts |
| --- | --- |
| `developer`, `user` | `input_text`, `input_image`, `input_file`, `scoped_content`, `input_audio` |
| `assistant` | `output_text`, `refusal` |

The same rule puts `phase` on the assistant variant as a plain field rather than
an `Option` on a shared struct: where it exists it always means something, and
where it does not it is not a field.

### An invariant is held by the type or it is not held

A doc comment saying a field is "set by" some method, and a checking constructor
beside a public field a caller can assign around, are conventions the compiler
does not know about. Two rules follow:

- **A closed API vocabulary is an enum, never a string.** A `&'static str` field
  accepts any string; an enum with no invalid variant accepts only what the API
  does. The enum goes in the field and the wire literal comes out at
  serialization time — see `api_enum!`, which states each literal once.
- **A cross-field invariant means private fields plus readers.** Where validity
  is a relation between fields or against a model fact, no single field's type
  carries it, so the checking constructor must be the only way in. Where there
  is no such relation, do not hide the field.

A capability only some values have is a method returning `Option`, never a
`bool` beside an unchecked accessor.

### A claim of impossibility is tested

Every claim that something cannot be written is a `compile_fail` doctest, and
each one is verified to fail *for its stated reason* rather than for a typo. A
claim only a comment makes is the failure mode this section exists to prevent.

A claim of *validity* is tested against the reference, not against the crate. A
test that only serializes agrees with whatever the crate currently emits, which
is how the `input_text` defect survived a full suite: every assertion asked
whether the crate wrote what the crate writes. So the vocabulary test transcribes
the documented vocabularies and checks a rendered body against them, and it fails
when the crate and the reference disagree rather than when the crate changes.

A captured real frame beats an invented fixture. Hand-written fixtures once
omitted an `output_index` the wire always sends.

### Incomplete is a different type from complete

A truncated stream must not be readable as a finished response: a dropped
connection leaves text that looks exactly like a complete answer. `Settling`
accumulates and cannot yield a response; `Settled` is finished and cannot take
more events; `settle` is the only bridge and fails on a stream that never reached
a terminal event. `#[non_exhaustive]` makes that structural rather than
decorative — a caller cannot write the literal.

### Unknown is not broken

OpenAI's compatibility policy names adding streaming event types as a compatible
change. An unrecognized event or item kind is therefore a variant meaning "ignore
me", never an error. What *is* an error is a frame contradicting the schema: not
JSON, not an object, no `type`, a field of the wrong type.

### Cache accounting is not optional detail

A prefix below the model's minimum cacheable length is a silent no-op, and the
only evidence is `usage`. So usage decodes rather than being skipped, a gateway
reporting fewer fields than OpenAI reads as zero of that kind rather than
failing the frame, and cost arithmetic is exact integer arithmetic in
nanodollars — every published price and cache multiplier lands on a whole
nanodollar, so nothing rounds.

Function-call `arguments` stay the exact string the model emitted.
Re-serializing a parsed value could reorder keys or change spacing, and on
replay that is the prefix gone.

## Scope

Both halves of one endpoint, `POST /v1/responses`: the crate produces a
serializable request body and decodes the response and the streamed events. No
HTTP client, no async runtime, no SSE transport, no retry or reconnection logic
— the caller owns the socket and hands over one `data:` payload at a time.

Decoding is in scope for the same reason the request half is: a consumer that
hand-matches raw JSON re-derives the API's shape badly, and the failure modes
that matter — a stream that stopped early, a cache that silently did nothing —
are exactly the ones a type can rule out.

Deliberately outside the mission, each for a reason rather than by omission:

- **Chat Completions, and every other endpoint.** One endpoint, done wholly. The
  Responses API is the one with prompt-cache control, and covering a second
  wire format would dilute the first.
- **Stateful continuation** — `previous_response_id` and `conversation`. These
  hand prefix control to the server: the caller no longer knows which bytes
  precede its input, so it cannot reason about the cache at all. Only the
  stateless path, where the caller supplies every input item, lets the caller
  control the rendered prefix byte for byte. This is a mission-level exclusion,
  not a gap.
- **Sampling parameters every modeled model refuses.** `Temperature` exists as a
  validating newtype with no field to put it on: the validation is the reusable
  part, and where it may be sent is a per-model fact.

Older models are not wired up by default; adding one is a normal extension,
following the per-model-type approach above.
