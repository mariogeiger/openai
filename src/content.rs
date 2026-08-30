//! Input items, and the content vocabulary each role is allowed.
//!
//! Three invariants are structural here.
//!
//! **The role decides the vocabulary.** OpenAI accepts `input_text` /
//! `input_image` inside a developer or user message, and `output_text` /
//! `refusal` inside an assistant one. Those are disjoint sets, so a message is
//! a [`Message`] variant carrying *its own* block type — [`InputBlock`] or
//! [`OutputBlock`] — rather than a role beside a vocabulary that may not match
//! it. Pairing the wrong two is not a request the API refuses; it is a sentence
//! this module cannot write.
//!
//! **A breakpoint is not something you can write.** `prompt_cache_breakpoint`
//! is a crate-private field with no public constructor, so the only way to mark
//! one is through a [`BreakpointSlot`](crate::context::BreakpointSlot) on the
//! [`Context`](crate::context::Context). That is what keeps the four-write
//! budget honest: the slots *are* the budget. A [`Refusal`] has no such field at
//! all, because the API rejects one there — measured live as
//! `Unknown parameter: 'input[0].content[0].prompt_cache_breakpoint'`.
//!
//! **Content is always a block array, never a bare string.** The API accepts
//! `"content": "hello"` as shorthand, but a breakpoint can only ride on a
//! block, and two spellings of the same message are two different prefixes.
//! One shape means one set of bytes.

use crate::values::{AssistantPhase, ImageDetail, InputRole, api_enum};
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

/// A content block, seen as "somewhere a breakpoint may or may not go".
///
/// Both vocabularies implement it, so [`Context`](crate::context::Context)'s
/// slot bookkeeping is written once and knows nothing about which block kinds
/// exist. A block with no site answers `None`, which is how [`Refusal`] stays
/// out of the budget without a special case anywhere else.
///
/// Two methods, because everything else follows from them: a site that exists
/// and holds something is a breakpoint, and a site that exists at all is a
/// place one could go.
pub(crate) trait BreakpointableBlock {
    fn breakpoint_site(&self) -> Option<&Option<PromptCacheBreakpoint>>;
    fn breakpoint_site_mut(&mut self) -> Option<&mut Option<PromptCacheBreakpoint>>;

    fn accepts_breakpoint(&self) -> bool {
        self.breakpoint_site().is_some()
    }

    fn has_breakpoint(&self) -> bool {
        self.breakpoint_site().is_some_and(Option::is_some)
    }
}

fn last_breakpoint_site<B: BreakpointableBlock>(blocks: &[B]) -> Option<usize> {
    blocks.iter().rposition(BreakpointableBlock::accepts_breakpoint)
}

fn breakpoint_count<B: BreakpointableBlock>(blocks: &[B]) -> usize {
    blocks.iter().filter(|b| b.has_breakpoint()).count()
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

/// Text the model reads, spelled `input_text`. The workhorse block: the only
/// one that can carry reusable instructions, and hence the one breakpoints
/// usually land on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputText {
    /// The text itself.
    pub text: String,
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

/// Text the model wrote, spelled `output_text`. A replayed assistant turn, and
/// a legal breakpoint site: measured live, a breakpoint here wrote 3,603 tokens
/// and read the same 3,603 back on the next request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputText {
    /// The text itself.
    pub text: String,
    pub(crate) prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
}

/// The model declining, replayed as context.
///
/// No breakpoint field, because the API has none to accept: sending
/// `prompt_cache_breakpoint` on a refusal is answered with
/// `Unknown parameter`. Mark the nearest legal block instead — the
/// [`Context`](crate::context::Context) does that for you.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// The refusal explanation the model gave.
    pub refusal: String,
}

/// One piece of content a developer or user message may hold: the vocabulary
/// the reference calls `ResponseInputContent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputBlock {
    /// Text, spelled `input_text`.
    Text(InputText),
    /// An image, spelled `input_image`.
    Image(InputImage),
}

impl InputBlock {
    /// A text block with no breakpoint. Adding one is the `Context`'s job.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(InputText { text: text.into(), prompt_cache_breakpoint: None })
    }

    /// An image block at the given detail level.
    pub fn image(source: ImageSource, detail: ImageDetail) -> Self {
        Self::Image(InputImage { source, detail, prompt_cache_breakpoint: None })
    }
}

impl BreakpointableBlock for InputBlock {
    fn breakpoint_site(&self) -> Option<&Option<PromptCacheBreakpoint>> {
        Some(match self {
            Self::Text(b) => &b.prompt_cache_breakpoint,
            Self::Image(b) => &b.prompt_cache_breakpoint,
        })
    }

    fn breakpoint_site_mut(&mut self) -> Option<&mut Option<PromptCacheBreakpoint>> {
        Some(match self {
            Self::Text(b) => &mut b.prompt_cache_breakpoint,
            Self::Image(b) => &mut b.prompt_cache_breakpoint,
        })
    }
}

/// One piece of content an assistant message may hold: the vocabulary the
/// reference calls the content of a `ResponseOutputMessage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputBlock {
    /// Text the model wrote, spelled `output_text`.
    Text(OutputText),
    /// The model declining, spelled `refusal`.
    Refusal(Refusal),
}

impl OutputBlock {
    /// A replayed answer with no breakpoint. Adding one is the `Context`'s job.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(OutputText { text: text.into(), prompt_cache_breakpoint: None })
    }

    /// A replayed refusal. Carries no breakpoint, and cannot be given one.
    pub fn refusal(refusal: impl Into<String>) -> Self {
        Self::Refusal(Refusal { refusal: refusal.into() })
    }
}

impl BreakpointableBlock for OutputBlock {
    fn breakpoint_site(&self) -> Option<&Option<PromptCacheBreakpoint>> {
        match self {
            Self::Text(b) => Some(&b.prompt_cache_breakpoint),
            Self::Refusal(_) => None,
        }
    }

    fn breakpoint_site_mut(&mut self) -> Option<&mut Option<PromptCacheBreakpoint>> {
        match self {
            Self::Text(b) => Some(&mut b.prompt_cache_breakpoint),
            Self::Refusal(_) => None,
        }
    }
}

/// A message from the developer, the user, or the model.
///
/// The variants are the roles grouped by the vocabulary they accept, which is
/// why `developer` and `user` share one: OpenAI takes the same content blocks
/// for both. An assistant message takes different blocks and carries a
/// `phase`, so it is its own variant — and the phase is a plain field rather
/// than an `Option`, because on this variant it always means something.
///
/// The wrong pairing does not compile:
///
/// ```compile_fail
/// use openai::content::{InputBlock, Message};
/// use openai::values::AssistantPhase;
///
/// let _ = Message::Assistant {
///     phase: AssistantPhase::FinalAnswer,
///     content: vec![InputBlock::text("what the model said")],
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// A developer or user message: the roles whose content is the input
    /// vocabulary.
    Input {
        /// Which of the two roles is speaking.
        role: InputRole,
        /// The content blocks, in order.
        content: Vec<InputBlock>,
    },
    /// A previous model turn, replayed as context.
    Assistant {
        /// Commentary or final answer. OpenAI asks that this be preserved and
        /// resent on every replay; dropping it degrades tool-heavy flows.
        phase: AssistantPhase,
        /// The content blocks, in order.
        content: Vec<OutputBlock>,
    },
}

/// What a function returned. `Text` is the ordinary case; `Blocks` exists
/// because a breakpoint needs a block to sit on, and OpenAI recommends a
/// breakpoint after each tool result in long agent threads.
///
/// The blocks are the *input* vocabulary: a tool result is something the model
/// reads, not something it wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionOutput {
    /// A single string, usually JSON.
    Text(String),
    /// Content blocks, one of which may carry a breakpoint.
    Blocks(Vec<InputBlock>),
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
    /// The index of the last block in this item that can carry a breakpoint.
    ///
    /// `None` for items offering no site at all — a function call, an output
    /// returned as a bare string, a reasoning item, or an assistant turn that
    /// is nothing but a refusal.
    pub(crate) fn last_breakpoint_site(&self) -> Option<usize> {
        match self {
            InputItem::Message(Message::Input { content, .. }) => last_breakpoint_site(content),
            InputItem::Message(Message::Assistant { content, .. }) => last_breakpoint_site(content),
            InputItem::FunctionCallOutput(o) => match &o.output {
                FunctionOutput::Blocks(blocks) => last_breakpoint_site(blocks),
                FunctionOutput::Text(_) => None,
            },
            InputItem::FunctionCall(_) | InputItem::Reasoning(_) => None,
        }
    }

    /// The breakpoint field of the `block`-th block, if that block has one.
    pub(crate) fn breakpoint_site_mut(&mut self, block: usize) -> Option<&mut Option<PromptCacheBreakpoint>> {
        match self {
            InputItem::Message(Message::Input { content, .. }) => content.get_mut(block)?.breakpoint_site_mut(),
            InputItem::Message(Message::Assistant { content, .. }) => content.get_mut(block)?.breakpoint_site_mut(),
            InputItem::FunctionCallOutput(o) => match &mut o.output {
                FunctionOutput::Blocks(blocks) => blocks.get_mut(block)?.breakpoint_site_mut(),
                FunctionOutput::Text(_) => None,
            },
            InputItem::FunctionCall(_) | InputItem::Reasoning(_) => None,
        }
    }

    /// How many breakpoints this item carries.
    ///
    /// Sums to the request's explicit-breakpoint total across all items, so it
    /// is how you audit an assembled context against the four-write budget
    /// without trusting the slot bookkeeping.
    pub fn breakpoint_count(&self) -> usize {
        match self {
            InputItem::Message(Message::Input { content, .. }) => breakpoint_count(content),
            InputItem::Message(Message::Assistant { content, .. }) => breakpoint_count(content),
            InputItem::FunctionCallOutput(o) => match &o.output {
                FunctionOutput::Blocks(blocks) => breakpoint_count(blocks),
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

/// One wire shape for both text blocks: the tag is the only difference between
/// `input_text` and `output_text`, so each literal appears once, at the one
/// place that knows which vocabulary it is in.
#[derive(Serialize)]
struct TextBlockWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
}

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

/// No `prompt_cache_breakpoint` field exists here, and that is the point: the
/// API answers `Unknown parameter` to one.
#[derive(Serialize)]
struct RefusalWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    refusal: &'a str,
}

impl Serialize for InputBlock {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            InputBlock::Text(t) => {
                TextBlockWire { kind: "input_text", text: &t.text, prompt_cache_breakpoint: t.prompt_cache_breakpoint }
                    .serialize(s)
            }
            InputBlock::Image(i) => {
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

impl Serialize for OutputBlock {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            OutputBlock::Text(t) => {
                TextBlockWire { kind: "output_text", text: &t.text, prompt_cache_breakpoint: t.prompt_cache_breakpoint }
                    .serialize(s)
            }
            OutputBlock::Refusal(r) => RefusalWire { kind: "refusal", refusal: &r.refusal }.serialize(s),
        }
    }
}

/// One wire shape for every message. `role` is a fixed tag here for the same
/// reason `kind` is: on the assistant variant the API documents exactly one
/// value, and a one-variant enum would state that less clearly than the
/// literal. The two-way choice keeps its enum, in
/// [`InputRole`](crate::values::InputRole).
#[derive(Serialize)]
struct MessageWire<'a, B: Serialize> {
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: &'a [B],
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
    Blocks(&'a Vec<InputBlock>),
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

impl Serialize for Message {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Message::Input { role, content } => {
                MessageWire { kind: "message", role: role.as_str(), content, phase: None }.serialize(s)
            }
            Message::Assistant { phase, content } => {
                MessageWire { kind: "message", role: "assistant", content, phase: Some(*phase) }.serialize(s)
            }
        }
    }
}

impl Serialize for InputItem {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            InputItem::Message(m) => m.serialize(s),
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
        let block = InputBlock::text("hello");
        assert!(!block.has_breakpoint());
        assert_eq!(serde_json::to_value(&block).unwrap(), json!({"type": "input_text", "text": "hello"}));
    }

    #[test]
    fn an_image_emits_exactly_one_source_field() {
        let url = InputBlock::image(ImageSource::Url("https://example.test/a.png".into()), ImageDetail::High);
        assert_eq!(
            serde_json::to_value(&url).unwrap(),
            json!({"type": "input_image", "detail": "high", "image_url": "https://example.test/a.png"})
        );

        let file = InputBlock::image(ImageSource::FileId("file-123".into()), ImageDetail::Auto);
        assert_eq!(
            serde_json::to_value(&file).unwrap(),
            json!({"type": "input_image", "detail": "auto", "file_id": "file-123"})
        );
    }

    /// The defect this split exists to prevent. An assistant turn spelled
    /// `input_text` is refused with
    /// `Invalid value: 'input_text'. Supported values are: 'output_text' and
    /// 'refusal'.` — reproduced live before the fix. There is now no way to
    /// write it, so the assertion is on the only spelling that exists.
    #[test]
    fn an_assistant_turn_speaks_the_output_vocabulary() {
        let item = InputItem::Message(Message::Assistant {
            phase: AssistantPhase::FinalAnswer,
            content: vec![OutputBlock::text("blue")],
        });
        assert_eq!(
            serde_json::to_value(&item).unwrap(),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "blue"}],
                "phase": "final_answer",
            })
        );
    }

    #[test]
    fn a_refusal_is_a_thing_an_assistant_can_have_said() {
        let item = InputItem::Message(Message::Assistant {
            phase: AssistantPhase::FinalAnswer,
            content: vec![OutputBlock::refusal("I can't help with that.")],
        });
        assert_eq!(
            serde_json::to_value(&item).unwrap(),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I can't help with that."}],
                "phase": "final_answer",
            })
        );
    }

    /// A refusal offers no breakpoint site, so it consumes none of the budget
    /// and nothing can be marked on it. The API answers `Unknown parameter` to
    /// a breakpoint there, which is why the field does not exist.
    #[test]
    fn a_refusal_offers_no_breakpoint_site() {
        let mut item = InputItem::Message(Message::Assistant {
            phase: AssistantPhase::Commentary,
            content: vec![OutputBlock::refusal("no")],
        });
        assert_eq!(item.last_breakpoint_site(), None);
        assert!(item.breakpoint_site_mut(0).is_none());
        assert_eq!(item.breakpoint_count(), 0);
    }

    /// Content is always an array. The API's bare-string shorthand would be a
    /// second spelling of the same message, hence a second prefix.
    #[test]
    fn a_message_always_uses_a_block_array() {
        let item = InputItem::Message(Message::Input { role: InputRole::User, content: vec![InputBlock::text("hi")] });
        assert_eq!(
            serde_json::to_value(&item).unwrap(),
            json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]})
        );
    }

    #[test]
    fn phase_rides_only_on_assistant_messages() {
        let assistant = InputItem::Message(Message::Assistant {
            phase: AssistantPhase::FinalAnswer,
            content: vec![OutputBlock::text("done")],
        });
        assert_eq!(serde_json::to_value(&assistant).unwrap()["phase"], "final_answer");

        let user = InputItem::Message(Message::Input { role: InputRole::User, content: vec![InputBlock::text("hi")] });
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

    /// A tool result is read by the model, not written by it, so its blocks are
    /// the input vocabulary.
    #[test]
    fn a_function_output_can_be_a_string_or_input_blocks() {
        let text = InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: "call_1".into(),
            output: FunctionOutput::Text("42".into()),
        });
        assert_eq!(serde_json::to_value(&text).unwrap()["output"], "42");

        let blocks = InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: "call_1".into(),
            output: FunctionOutput::Blocks(vec![InputBlock::text("42")]),
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
        let call = InputItem::FunctionCall(FunctionCall {
            call_id: "call_1".into(),
            name: "f".into(),
            arguments: "{}".into(),
        });
        assert_eq!(call.last_breakpoint_site(), None);

        let string_output = InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: "call_1".into(),
            output: FunctionOutput::Text("ok".into()),
        });
        assert_eq!(string_output.last_breakpoint_site(), None);
    }
}
