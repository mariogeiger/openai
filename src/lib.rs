//! Strictly typed bindings for the OpenAI **Responses** API.
//!
//! The crate exists to make two classes of mistake impossible to write:
//! requests the API refuses with a 400, and requests that silently destroy the
//! prompt cache. It builds a validated, `Serialize` body; the caller owns the
//! HTTP stack.
//!
//! # Why the types are shaped this way
//!
//! OpenAI hashes the *rendered* prefix of a request, in this order: hidden
//! OpenAI content, then `tools`, then developer/`instructions`, then the input
//! items. A cache read requires that prefix to be byte-identical. Two
//! consequences drive the whole design:
//!
//! * **The tool array is the first bytes of every request.** Growing it,
//!   shrinking it, or merely reordering it costs the entire prefix. So
//!   [`Context`](context::Context) takes its tools once, at construction, and
//!   never exposes a way to change them. To vary which tools are callable you
//!   use [`ToolChoice::Allowed`](tools::ToolChoice::Allowed), which restricts
//!   the callable set while leaving the array — and the cache — intact.
//! * **Some per-call settings are part of the prefix and some are not.**
//!   The ones that are live together in [`PrefixSettings`](prefix::PrefixSettings),
//!   one value you can hold constant across turns. The ones that are not —
//!   `tool_choice`, `prompt_cache_key`, `max_output_tokens`, `stream`, `store` —
//!   are set on the [`Request`](request::Request) itself and may vary freely.
//!
//! Everything else follows from OpenAI's own rules: cache writes are a bounded
//! resource (four per request), breakpoints attach only to content blocks and
//! never to top-level `instructions`, and the field that controls cache
//! lifetime differs by model generation.
//!
//! # Who chooses a value
//!
//! Every value the API accepts is the caller's to choose, and the crate invents
//! none. Two shapes carry that, and which one a field gets is read off OpenAI's
//! reference rather than decided here:
//!
//! * **The API documents a default.** The field is a plain, non-`Option` field
//!   whose `Default` is the documented value, and it is *always* emitted. So the
//!   body is a complete record of what the model sees, and it stays that way the
//!   day OpenAI changes a default. `store`, `parallel_tool_calls`,
//!   `text.format`, `text.verbosity`, `reasoning.context`,
//!   `prompt_cache_options.mode` and `.ttl` are these.
//! * **The API documents no default.** Then presence is a real runtime
//!   distinction — told versus not told — and the field is an `Option` that is
//!   omitted when absent. `reasoning.effort`, `reasoning.mode`,
//!   `reasoning.summary`, `context_management`, `max_output_tokens`,
//!   `instructions`, `prompt_cache_key`, and GPT-5.4's
//!   `prompt_cache_retention` are these.
//!
//! An enclosing object disappears when every field inside it is absent: an empty
//! `"reasoning": {}` is a different request from no `reasoning`, and only the
//! second one means "the caller configured no reasoning".
//!
//! # Worked example
//!
//! ```
//! use openai::context::{BreakpointSlot, Context};
//! use openai::model::Model;
//! use openai::prefix::PrefixSettings;
//! use openai::request::Request;
//! use openai::tools::{AllowedToolsMode, FunctionTool, ToolChoice};
//! use serde_json::json;
//!
//! // Tools are frozen here: the first bytes of the prefix can never drift.
//! let tools = vec![
//!     FunctionTool::new("read_file", json!({"type": "object"})),
//!     FunctionTool::new("write_file", json!({"type": "object"})),
//! ];
//! let mut context = Context::new(tools);
//!
//! // Reusable instructions go in a developer message, the only place that can
//! // carry a breakpoint. The slot is anchored: it will never be moved.
//! context.push_anchored_developer_text(BreakpointSlot::S0, "Stable instructions…")?;
//! context.push_user_text("What changed in this file?");
//!
//! // The model carries its own effort range and its own caching field.
//! let prefix = PrefixSettings::new(Model::gpt_5_6_sol());
//! let mut request = Request::new(&context, prefix)?;
//!
//! // Restrict availability without touching the array, and so without paying
//! // to write the prefix again.
//! let allowed = context.allow_tools(AllowedToolsMode::Auto, &["read_file"])?;
//! request.tool_choice = ToolChoice::Allowed(allowed);
//! request.prompt_cache_key = Some("agent_v1:user_42".into());
//!
//! let body = serde_json::to_value(&request)?;
//! assert_eq!(body["tools"].as_array().unwrap().len(), 2);
//! assert_eq!(body["tool_choice"]["type"], "allowed_tools");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Reading the answer back
//!
//! A request is half the API. The other half is the stream of Server-Sent
//! Events a `stream: true` response returns, and it is decoded here too:
//! [`StreamEvent`](stream::StreamEvent) is one frame, and
//! [`Settling`](settle::Settling) accumulates a sequence of them.
//!
//! Two rules shape those types, both learned from what breaks in production:
//!
//! * **A new event type is not an error.** OpenAI's compatibility promise names
//!   "adding new event types in streaming APIs" as backwards-compatible, so a
//!   decoder that errors on an unrecognized event is one a routine server-side
//!   release will break. Hence [`StreamEvent::Unmodeled`](stream::StreamEvent::Unmodeled):
//!   the unknown case is a variant you ignore, never a failure.
//! * **An unfinished stream is a different type from a finished one.** A
//!   dropped connection leaves text that looks exactly like a complete answer.
//!   So [`Settling`](settle::Settling) has no method returning a response, and
//!   [`Settled`](settle::Settled) is reachable only through
//!   [`Settling::settle`](settle::Settling::settle), which fails on a stream
//!   that never sent a terminal event.
//!
//! ```
//! use openai::settle::{Outcome, Settling};
//!
//! let mut settling = Settling::new();
//! for frame in [
//!     r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hi"}"#,
//!     r#"{"type":"response.some_event_added_next_year","output_index":0}"#,
//!     r#"{"type":"response.completed","response":{"id":"resp_1"}}"#,
//! ] {
//!     settling.consume_payload(frame)?;
//! }
//! let settled = settling.settle()?;
//! assert_eq!(settled.outcome, Outcome::Completed);
//! assert_eq!(settled.text, "Hi");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Deliberately out of scope
//!
//! No HTTP client, no async runtime, and no SSE transport: the caller owns the
//! socket and hands this crate one `data:` payload at a time. Of the response
//! body only [`Usage`](usage::Usage) and the streamed shapes are deserialized,
//! since cache accounting is the crate's reason to exist. Chat Completions is
//! not modeled at all. Stateful continuation (`previous_response_id`,
//! `conversation`) is excluded on purpose: only the stateless path, where the
//! caller supplies every input item, lets the caller control the rendered
//! prefix byte for byte.
//!
//! Of the streaming events, the seven a text-and-tools consumer needs are
//! modeled and the rest read as
//! [`Unmodeled`](stream::StreamEvent::Unmodeled) — see the crate's
//! `CHANGELOG.md` for the full list and the reason for each omission.

#![deny(missing_docs)]

pub mod content;
pub mod context;
pub mod model;
pub mod prefix;
pub mod request;
pub mod settle;
pub mod stream;
pub mod tools;
pub mod usage;
pub mod values;

pub use values::*;

/// Base URL of the OpenAI REST API.
pub const API_BASE: &str = "https://api.openai.com";

/// Path of the endpoint this crate builds bodies for, `POST /v1/responses`.
pub const RESPONSES_PATH: &str = "/v1/responses";

/// Name of the bearer-credential header. The value is an API key or a
/// short-lived workload-identity token; this crate never touches either.
pub const HEADER_AUTHORIZATION: &str = "Authorization";

/// Header selecting the organization, for keys that belong to several.
pub const HEADER_ORGANIZATION: &str = "OpenAI-Organization";

/// Header selecting the project, for keys that belong to several.
pub const HEADER_PROJECT: &str = "OpenAI-Project";

/// Response header carrying OpenAI's unique request identifier. Worth logging:
/// it is what OpenAI support needs to look a request up.
pub const HEADER_REQUEST_ID: &str = "x-request-id";
