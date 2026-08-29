//! Input items and the content blocks inside them — the part of the request
//! whose bytes must not drift.
//!
//! Two invariants are structural here.
//!
//! **A breakpoint is not something you can write.** `prompt_cache_breakpoint`
//! is a crate-private field with no public constructor, so the only way to mark
//! one is through a [`BreakpointSlot`](crate::context::BreakpointSlot) on the
//! [`Context`](crate::context::Context). That is what keeps the four-write
//! budget honest: the slots *are* the budget.
//!
//! **Content is always a block array, never a bare string.** The API accepts
//! `"content": "hello"` as shorthand, but a breakpoint can only ride on a
//! block, and two spellings of the same message are two different prefixes.
//! One shape means one set of bytes.

use crate::values::{AssistantPhase, ImageDetail, Role, api_enum};
use serde::Serialize;

api_enum! {
    /// The one value `prompt_cache_breakpoint.mode` accepts.
    BreakpointMode {
        /// A boundary you chose, as opposed to one OpenAI placed implicitly.
        Explicit => "explicit",
    }
}

/// A marker saying "a reusable prefix ends exactly here".
///
/// It has no public constructor on purpose. Breakpoints are a bounded resource
/// — four writes per request — and the bound is enforced by
/// [`BreakpointSlot`](crate::context::BreakpointSlot), which cannot be
/// circumvented if this is the only way to make one. The lifetime comes from
/// the request's `prompt_cache_options.ttl`, so nothing about TTL appears here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PromptCacheBreakpoint {
    mode: BreakpointMode,
}

impl PromptCacheBreakpoint {
    pub(crate) fn explicit() -> Self {
        Self { mode: BreakpointMode::Explicit }
    }
}

/// Where an image comes from. A `file_id` and a URL are alternatives, not two
/// optional fields to keep in sync, so they are variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSource {
    /// A URL, either fully qualified or a `data:` URL holding base64 bytes.
    Url(String),
    /// A file previously uploaded to the Files API.
    FileId(String),
}

/// Text the model reads. The workhorse block: the only one that can carry
/// reusable instructions, and hence the one breakpoints usually land on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InputText {
    /// The text itself.
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
}

/// An image the model looks at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputImage {
    /// Where the image comes from.
    pub source: ImageSource,
    /// How finely to render it. Higher detail costs more input tokens, and
    /// because the image sits inside the hashed prefix, changing this
    /// invalidates the cache from that image onward.
    pub detail: ImageDetail,
    pub(crate) prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
}

/// One piece of content inside a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    /// Text.
    Text(InputText),
    /// An image.
    Image(InputImage),
}

impl ContentBlock {
    /// A text block with no breakpoint. Adding one is the `Context`'s job.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(InputText { text: text.into(), prompt_cache_breakpoint: None })
    }

    /// An image block at the given detail level.
    pub fn image(source: ImageSource, detail: ImageDetail) -> Self {
        Self::Image(InputImage { source, detail, prompt_cache_breakpoint: None })
    }

    pub(crate) fn breakpoint_mut(&mut self) -> &mut Option<PromptCacheBreakpoint> {
        match self {
            Self::Text(b) => &mut b.prompt_cache_breakpoint,
            Self::Image(b) => &mut b.prompt_cache_breakpoint,
        }
    }

    /// Whether this block ends a reusable prefix.
    pub fn has_breakpoint(&self) -> bool {
        match self {
            Self::Text(b) => b.prompt_cache_breakpoint.is_some(),
            Self::Image(b) => b.prompt_cache_breakpoint.is_some(),
        }
    }
}

/// A message from the developer, the user, or the model.
///
/// `phase` belongs to assistant messages alone, and the API ignores it
/// elsewhere. It is kept as a plain `Option` on the shared struct rather than
/// splitting the type, because [`Context`](crate::context::Context) is the only
/// thing that builds messages and it sets `phase` only where it means
/// something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// Who is speaking.
    pub role: Role,
    /// The content blocks, in order.
    pub content: Vec<ContentBlock>,
    /// For assistant messages: commentary or final answer. OpenAI asks that
    /// this be preserved and resent on every replay; dropping it degrades
    /// tool-heavy flows.
    pub phase: Option<AssistantPhase>,
}

/// What a function returned. `Text` is the ordinary case; `Blocks` exists
/// because a breakpoint needs a block to sit on, and OpenAI recommends a
/// breakpoint after each tool result in long agent threads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionOutput {
    /// A single string, usually JSON.
    Text(String),
    /// Content blocks, one of which may carry a breakpoint.
    Blocks(Vec<ContentBlock>),
}

/// A model turn asking for a function call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    /// The identifier the model generated. The matching output must repeat it.
    pub call_id: String,
    /// Which function to run.
    pub name: String,
    /// The arguments, as the JSON *string* the model emitted.
    ///
    /// A string, not a parsed value, because the bytes are what the model sees
    /// on replay. Re-serializing a parsed value could reorder keys or change
    /// spacing and break the prefix.
    pub arguments: String,
}

/// The result of a function call, fed back for the next turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallOutput {
    /// The `call_id` of the call this answers.
    pub call_id: String,
    /// What the function returned.
    pub output: FunctionOutput,
}

/// A reasoning item from an earlier response, replayed verbatim.
///
/// Reasoning models work better when their own reasoning items are handed back
/// alongside function outputs. In stateless mode the payload arrives as
/// `encrypted_content`, opaque to everyone but OpenAI — so this type only
/// carries it, and offers no way to build one from scratch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayedReasoning {
    /// The item's `id` from the response that produced it.
    pub id: String,
    /// The opaque `encrypted_content` from that response.
    pub encrypted_content: String,
}

/// One entry in the `input` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputItem {
    /// A developer, user, or assistant message.
    Message(Message),
    /// A function call the model made.
    FunctionCall(FunctionCall),
    /// The result of that call.
    FunctionCallOutput(FunctionCallOutput),
    /// Reasoning from an earlier response.
    Reasoning(ReplayedReasoning),
}

impl InputItem {
    /// The blocks a breakpoint could attach to, in wire order. Empty for items
    /// that hold no blocks at all — a function call, or an output returned as a
    /// bare string.
    pub(crate) fn breakpointable_blocks_mut(&mut self) -> &mut [ContentBlock] {
        match self {
            InputItem::Message(m) => &mut m.content,
            InputItem::FunctionCallOutput(o) => match &mut o.output {
                FunctionOutput::Blocks(blocks) => blocks,
                FunctionOutput::Text(_) => &mut [],
            },
            InputItem::FunctionCall(_) | InputItem::Reasoning(_) => &mut [],
        }
    }

    /// How many breakpoints this item carries.
    ///
    /// Sums to the request's explicit-breakpoint total across all items, so it
    /// is how you audit an assembled context against the four-write budget
    /// without trusting the slot bookkeeping.
    pub fn breakpoint_count(&self) -> usize {
        match self {
            InputItem::Message(m) => m.content.iter().filter(|b| b.has_breakpoint()).count(),
            InputItem::FunctionCallOutput(o) => match &o.output {
                FunctionOutput::Blocks(blocks) => blocks.iter().filter(|b| b.has_breakpoint()).count(),
                FunctionOutput::Text(_) => 0,
            },
            InputItem::FunctionCall(_) | InputItem::Reasoning(_) => 0,
        }
    }
}

// ── Serialization ────────────────────────────────────────────────────────────
// Written by hand rather than derived, because every `type` tag is emitted
// explicitly and every optional field is skipped only when it is genuinely
// absent. What you read below is exactly what goes on the wire.

#[derive(Serialize)]
struct InputImageWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    detail: ImageDetail,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
}

#[derive(Serialize)]
struct InputTextWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
}

impl Serialize for ContentBlock {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ContentBlock::Text(t) => {
                InputTextWire { kind: "input_text", text: &t.text, prompt_cache_breakpoint: t.prompt_cache_breakpoint }
                    .serialize(s)
            }
            ContentBlock::Image(i) => {
                let (image_url, file_id) = match &i.source {
                    ImageSource::Url(url) => (Some(url.as_str()), None),
                    ImageSource::FileId(id) => (None, Some(id.as_str())),
                };
                InputImageWire {
                    kind: "input_image",
                    detail: i.detail,
                    image_url,
                    file_id,
                    prompt_cache_breakpoint: i.prompt_cache_breakpoint,
                }
                .serialize(s)
            }
        }
    }
}

#[derive(Serialize)]
struct MessageWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    role: Role,
    content: &'a Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<AssistantPhase>,
}

#[derive(Serialize)]
struct FunctionCallWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    call_id: &'a str,
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize)]
#[serde(untagged)]
enum FunctionOutputWire<'a> {
    Text(&'a str),
    Blocks(&'a Vec<ContentBlock>),
}

#[derive(Serialize)]
struct FunctionCallOutputWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    call_id: &'a str,
    output: FunctionOutputWire<'a>,
}

#[derive(Serialize)]
struct ReasoningWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    id: &'a str,
    encrypted_content: &'a str,
    summary: [(); 0],
}

impl Serialize for InputItem {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            InputItem::Message(m) => {
                MessageWire { kind: "message", role: m.role, content: &m.content, phase: m.phase }.serialize(s)
            }
            InputItem::FunctionCall(c) => {
                FunctionCallWire { kind: "function_call", call_id: &c.call_id, name: &c.name, arguments: &c.arguments }
                    .serialize(s)
            }
            InputItem::FunctionCallOutput(o) => FunctionCallOutputWire {
                kind: "function_call_output",
                call_id: &o.call_id,
                output: match &o.output {
                    FunctionOutput::Text(t) => FunctionOutputWire::Text(t),
                    FunctionOutput::Blocks(b) => FunctionOutputWire::Blocks(b),
                },
            }
            .serialize(s),
            InputItem::Reasoning(r) => {
                ReasoningWire { kind: "reasoning", id: &r.id, encrypted_content: &r.encrypted_content, summary: [] }
                    .serialize(s)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_fresh_text_block_carries_no_breakpoint() {
        let block = ContentBlock::text("hello");
        assert!(!block.has_breakpoint());
        assert_eq!(serde_json::to_value(&block).unwrap(), json!({"type": "input_text", "text": "hello"}));
    }

    #[test]
    fn an_image_emits_exactly_one_source_field() {
        let url = ContentBlock::image(ImageSource::Url("https://example.test/a.png".into()), ImageDetail::High);
        assert_eq!(
            serde_json::to_value(&url).unwrap(),
            json!({"type": "input_image", "detail": "high", "image_url": "https://example.test/a.png"})
        );

        let file = ContentBlock::image(ImageSource::FileId("file-123".into()), ImageDetail::Auto);
        assert_eq!(
            serde_json::to_value(&file).unwrap(),
            json!({"type": "input_image", "detail": "auto", "file_id": "file-123"})
        );
    }

    /// Content is always an array. The API's bare-string shorthand would be a
    /// second spelling of the same message, hence a second prefix.
    #[test]
    fn a_message_always_uses_a_block_array() {
        let item =
            InputItem::Message(Message { role: Role::User, content: vec![ContentBlock::text("hi")], phase: None });
        assert_eq!(
            serde_json::to_value(&item).unwrap(),
            json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]})
        );
    }

    #[test]
    fn phase_rides_only_on_assistant_messages() {
        let assistant = InputItem::Message(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::text("done")],
            phase: Some(AssistantPhase::FinalAnswer),
        });
        assert_eq!(serde_json::to_value(&assistant).unwrap()["phase"], "final_answer");

        let user =
            InputItem::Message(Message { role: Role::User, content: vec![ContentBlock::text("hi")], phase: None });
        assert!(serde_json::to_value(&user).unwrap().get("phase").is_none());
    }

    /// Arguments stay the exact string the model produced; re-serializing a
    /// parsed value could reorder keys and break the prefix on replay.
    #[test]
    fn function_call_arguments_are_bytes_not_values() {
        let item = InputItem::FunctionCall(FunctionCall {
            call_id: "call_1".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"a.rs"}"#.into(),
        });
        assert_eq!(
            serde_json::to_value(&item).unwrap(),
            json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": r#"{"path":"a.rs"}"#,
            })
        );
    }

    #[test]
    fn a_function_output_can_be_a_string_or_blocks() {
        let text = InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: "call_1".into(),
            output: FunctionOutput::Text("42".into()),
        });
        assert_eq!(serde_json::to_value(&text).unwrap()["output"], "42");

        let blocks = InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: "call_1".into(),
            output: FunctionOutput::Blocks(vec![ContentBlock::text("42")]),
        });
        assert_eq!(serde_json::to_value(&blocks).unwrap()["output"][0]["type"], "input_text");
    }

    #[test]
    fn replayed_reasoning_carries_only_what_openai_returned() {
        let item = InputItem::Reasoning(ReplayedReasoning { id: "rs_1".into(), encrypted_content: "opaque".into() });
        assert_eq!(
            serde_json::to_value(&item).unwrap(),
            json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []})
        );
    }

    /// Only blocks can hold a breakpoint. An item made of no blocks offers
    /// nowhere to put one, and reports so rather than silently accepting.
    #[test]
    fn items_without_blocks_offer_no_breakpoint_site() {
        let mut call = InputItem::FunctionCall(FunctionCall {
            call_id: "call_1".into(),
            name: "f".into(),
            arguments: "{}".into(),
        });
        assert!(call.breakpointable_blocks_mut().is_empty());

        let mut string_output = InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: "call_1".into(),
            output: FunctionOutput::Text("ok".into()),
        });
        assert!(string_output.breakpointable_blocks_mut().is_empty());
    }
}
