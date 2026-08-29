//! Function tools, and the `tool_choice` that varies availability without
//! disturbing them.
//!
//! The tool array is the first thing OpenAI hashes after its own hidden
//! content, so it is the first bytes of the prefix. Measured live: an unchanged
//! array read 2,969 cached tokens, while the same request with one tool removed
//! read 0 and paid to write 2,978. Removing a tool does not save the tokens that
//! tool cost — it costs the whole prefix.
//!
//! The remedy the API gives is [`ToolChoice::Allowed`]: the array stays whole,
//! and a separate field says which of its tools may actually be called.
//! Measured on the same prompt: cached 2,978, written 0. Because
//! `tool_choice` is not part of the hashed prefix, it can change every request
//! for free. That is why [`Context`](crate::context::Context) freezes the array
//! and [`AllowedTools`] can only be built from it.

use crate::values::api_enum;
use serde::Serialize;
use serde_json::Value;

/// A function the model may call.
///
/// Not `Clone`, and with no setter for `name` or `parameters`: a tool's bytes
/// are the start of the prefix, so this type is built once and read forever.
#[derive(Debug, PartialEq, Serialize)]
pub struct FunctionTool {
    #[serde(rename = "type")]
    kind: &'static str,
    /// The name the model calls, and the name `AllowedTools` refers to.
    pub name: String,
    /// A JSON Schema describing the arguments.
    ///
    /// Deliberately a `serde_json::Value`: JSON Schema is an open language, and
    /// a Rust mirror of it would be a second, lossier schema language that
    /// still had to fall back to raw JSON for anything unusual.
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

impl FunctionTool {
    /// A tool whose argument validation is left to Responses.
    ///
    /// `strict` starts absent, because absence is its own documented behavior
    /// rather than a default value: "if omitted, Responses attempts to use
    /// strict validation when the schema is compatible, and falls back to
    /// non-strict validation otherwise." Three states, and the crate picks
    /// none of them — say [`Self::with_strict_arguments`] to insist, or
    /// [`Self::with_loose_arguments`] to refuse.
    ///
    /// A tool's bytes are the start of the hashed prefix, so this is a choice
    /// worth making once and reading back.
    pub fn new(name: impl Into<String>, parameters: Value) -> Self {
        Self { kind: "function", name: name.into(), parameters, description: None, strict: None }
    }

    /// Add the description the model reads when deciding whether to call this.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Insist on strict validation, so arguments that do not match the schema
    /// never reach you.
    pub fn with_strict_arguments(mut self) -> Self {
        self.strict = Some(true);
        self
    }

    /// Turn strict validation off, for a schema that uses JSON Schema features
    /// strict mode does not support.
    pub fn with_loose_arguments(mut self) -> Self {
        self.strict = Some(false);
        self
    }

    /// Send no `strict`, letting Responses use strict validation where the
    /// schema allows it and fall back where it does not.
    pub fn with_inferred_argument_strictness(mut self) -> Self {
        self.strict = None;
        self
    }
}

api_enum! {
    /// Whether the model may answer directly or must call something, within a
    /// restricted set.
    AllowedToolsMode {
        /// Pick from the allowed tools, or answer in words.
        Auto => "auto",
        /// Call one or more of the allowed tools.
        Required => "required",
    }
}

/// A restriction to a subset of the tools already in the array.
///
/// Built only by [`Context::allow_tools`](crate::context::Context::allow_tools),
/// which checks each name against the frozen array. So a name that is not there
/// cannot reach the wire, and neither can the temptation to shrink the array
/// instead. There is no public constructor, so the check cannot be bypassed:
///
/// ```compile_fail
/// use openai::tools::{AllowedTools, AllowedToolsMode};
/// let _ = AllowedTools::new(AllowedToolsMode::Auto, vec!["ghost".into()]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedTools {
    pub(crate) mode: AllowedToolsMode,
    pub(crate) names: Vec<String>,
}

impl AllowedTools {
    pub(crate) fn new(mode: AllowedToolsMode, names: Vec<String>) -> Self {
        Self { mode, names }
    }

    /// Whether the model may also answer in words, or must call a tool.
    pub fn mode(&self) -> AllowedToolsMode {
        self.mode
    }

    /// The permitted tool names, in the order given.
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

/// Why a set of allowed tools was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllowedToolsError {
    /// A name that is not in the context's tool array. Allowing a tool that
    /// does not exist is a 400, and usually means the array was edited — the
    /// exact move that costs the prefix.
    UnknownTool(String),
    /// An empty allowed set. The API refuses it, and the thing it appears to
    /// mean already has a name: [`ToolChoice::None`], which keeps the array
    /// intact while forbidding every call.
    EmptyAllowedSet,
    /// A name given twice. Duplicates change the serialized bytes without
    /// changing the meaning, so they are refused rather than silently deduped.
    DuplicateTool(String),
}

impl std::fmt::Display for AllowedToolsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllowedToolsError::UnknownTool(name) => {
                write!(f, "tool {name:?} is not in this context's tool array")
            }
            AllowedToolsError::EmptyAllowedSet => {
                write!(f, "an empty allowed set is refused; use ToolChoice::None to forbid every call")
            }
            AllowedToolsError::DuplicateTool(name) => write!(f, "tool {name:?} was allowed twice"),
        }
    }
}

impl std::error::Error for AllowedToolsError {}

/// How the model chooses what to call.
///
/// None of these variants is part of the hashed prefix, so any of them may
/// change from one request to the next at no cache cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model picks: answer in words, or call any tool in the array.
    Auto,
    /// The model must not call anything. Prefer this over emptying the array
    /// when a turn should be answer-only.
    None,
    /// The model must call something, chosen from the whole array.
    Required,
    /// The model must call this exact function.
    Function(String),
    /// The model is restricted to a subset of the array.
    Allowed(AllowedTools),
}

impl Default for ToolChoice {
    /// The API default.
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Serialize)]
struct AllowedToolRefWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    name: &'a str,
}

#[derive(Serialize)]
struct AllowedToolsWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    mode: AllowedToolsMode,
    tools: Vec<AllowedToolRefWire<'a>>,
}

#[derive(Serialize)]
struct FunctionChoiceWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    name: &'a str,
}

impl Serialize for ToolChoice {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ToolChoice::Auto => s.serialize_str("auto"),
            ToolChoice::None => s.serialize_str("none"),
            ToolChoice::Required => s.serialize_str("required"),
            ToolChoice::Function(name) => FunctionChoiceWire { kind: "function", name }.serialize(s),
            ToolChoice::Allowed(allowed) => AllowedToolsWire {
                kind: "allowed_tools",
                mode: allowed.mode,
                tools: allowed.names.iter().map(|n| AllowedToolRefWire { kind: "function", name: n }).collect(),
            }
            .serialize(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `strict` has three states, and a fresh tool is in the third: absent, so
    /// Responses decides from the schema. Both other states are the caller's to
    /// name, and either can be taken back.
    #[test]
    fn argument_strictness_is_the_caller_s_three_way_choice() {
        let tool = FunctionTool::new("get_weather", json!({"type": "object"}));
        assert_eq!(
            serde_json::to_value(&tool).unwrap(),
            json!({"type": "function", "name": "get_weather", "parameters": {"type": "object"}})
        );

        let strict = FunctionTool::new("f", json!({})).with_strict_arguments();
        assert_eq!(serde_json::to_value(&strict).unwrap()["strict"], true);

        let loose = FunctionTool::new("f", json!({})).with_loose_arguments().with_description("does f");
        let v = serde_json::to_value(&loose).unwrap();
        assert_eq!(v["strict"], false);
        assert_eq!(v["description"], "does f");

        let inferred = FunctionTool::new("f", json!({})).with_strict_arguments().with_inferred_argument_strictness();
        assert!(serde_json::to_value(&inferred).unwrap().get("strict").is_none());
    }

    /// The three bare-string forms must stay bare strings; the API rejects them
    /// wrapped in an object.
    #[test]
    fn simple_choices_are_bare_strings() {
        assert_eq!(serde_json::to_value(ToolChoice::Auto).unwrap(), json!("auto"));
        assert_eq!(serde_json::to_value(ToolChoice::None).unwrap(), json!("none"));
        assert_eq!(serde_json::to_value(ToolChoice::Required).unwrap(), json!("required"));
    }

    #[test]
    fn forcing_one_function_names_it() {
        assert_eq!(
            serde_json::to_value(ToolChoice::Function("get_weather".into())).unwrap(),
            json!({"type": "function", "name": "get_weather"})
        );
    }

    /// The exact shape measured to keep a cache hit while narrowing
    /// availability: cached 2,978, written 0.
    #[test]
    fn allowed_tools_serializes_to_the_cache_preserving_shape() {
        let allowed = AllowedTools::new(AllowedToolsMode::Auto, vec!["read_file".into(), "list_dir".into()]);
        assert_eq!(
            serde_json::to_value(ToolChoice::Allowed(allowed)).unwrap(),
            json!({
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": "read_file"},
                    {"type": "function", "name": "list_dir"},
                ],
            })
        );
    }

    #[test]
    fn allowed_tools_can_demand_a_call() {
        let allowed = AllowedTools::new(AllowedToolsMode::Required, vec!["read_file".into()]);
        assert_eq!(serde_json::to_value(ToolChoice::Allowed(allowed)).unwrap()["mode"], "required");
    }

    #[test]
    fn the_default_choice_is_auto() {
        assert_eq!(ToolChoice::default(), ToolChoice::Auto);
    }
}
