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
//! # Deliberately out of scope
//!
//! No HTTP client, no async runtime, no streaming decoder, no response
//! deserializer beyond [`Usage`](usage::Usage) — the cache accounting is the
//! crate's reason to exist, so it is the one response shape modeled. Chat
//! Completions is not modeled at all. Stateful continuation
//! (`previous_response_id`, `conversation`) is excluded on purpose: only the
//! stateless path, where the caller supplies every input item, lets the caller
//! control the rendered prefix byte for byte.

#![deny(missing_docs)]

pub mod content;
pub mod context;
pub mod model;
pub mod prefix;
pub mod request;
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
