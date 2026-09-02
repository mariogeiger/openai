//! Append-only conversation state with a frozen tool array and four cache
//! breakpoint slots.
//!
//! Everything here exists to protect one property: **the rendered prefix of the
//! next request must extend the previous one byte for byte.** So the type offers
//! no way to rewrite what has already been said, and no way to change the tools.
//!
//! * **Tools are set once, at construction.** They are the first bytes OpenAI
//!   hashes, so a setter would be a setter for "throw the cache away". Vary
//!   availability with [`Context::allow_tools`] instead.
//! * **Items are appended, never edited.** Past items are readable but not
//!   mutable.
//! * **Breakpoints live in four named slots.** OpenAI writes at most four cache
//!   entries per request, so the slots *are* that budget: asking for a fifth is
//!   not an error to handle, it is a value that does not exist.
//! * **An anchored slot never moves.** A breakpoint at the end of stable
//!   instructions is the whole point of explicit caching; moving it would
//!   silently turn a cheap read into an expensive write. Anchored slots refuse
//!   to be rolled or cleared.

use crate::content::{
    FunctionCall, FunctionCallOutput, FunctionOutput, InputBlock, InputItem, Message, OutputBlock,
    PromptCacheBreakpoint, ReplayedReasoning,
};
use crate::tools::{AllowedTools, AllowedToolsError, AllowedToolsMode, FunctionTool};
use crate::values::{AssistantPhase, InputRole};

/// How many breakpoints a request may write. The API's hard ceiling, and hence
/// the number of slots.
pub const CACHE_WRITE_SLOTS: usize = 4;

/// How far back OpenAI looks for a *readable* breakpoint, in breakpoints.
///
/// Reads are far more generous than writes: a request writes at most
/// [`CACHE_WRITE_SLOTS`] entries but may match against many earlier ones,
/// choosing the longest prefix that hits. That asymmetry is why a long thread
/// keeps benefiting from breakpoints laid down many turns ago.
///
/// The prompt-caching guide states 50 and the API reference states 80; the
/// smaller figure is recorded here because relying on the larger one and being
/// wrong costs money. Nothing in this crate enforces it — it is a documented
/// fact about reuse, not a request-validity rule.
pub const CACHE_READ_BREAKPOINT_LOOKBACK: usize = 50;

/// One of the four cache-write slots.
///
/// Naming the slots rather than counting breakpoints is what makes the limit
/// structural: there is no fifth variant to pass. Asking for a fifth breakpoint
/// is not an error to handle — it cannot be written:
///
/// ```compile_fail
/// use openai::context::BreakpointSlot;
/// let _ = BreakpointSlot::S4;
/// ```
///
/// And a breakpoint cannot be built by hand and dropped into a block, so the
/// slots are the only route and therefore an exact accounting of the budget:
///
/// ```compile_fail
/// use openai::content::PromptCacheBreakpoint;
/// let _ = PromptCacheBreakpoint::explicit();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BreakpointSlot {
    /// First slot. Conventionally the stable-instructions anchor.
    S0,
    /// Second slot.
    S1,
    /// Third slot.
    S2,
    /// Fourth slot. Unusable under [`CacheMode::Implicit`](crate::CacheMode),
    /// where OpenAI's own breakpoint takes one of the four.
    S3,
}

impl BreakpointSlot {
    /// All four slots, in order.
    pub const ALL: [BreakpointSlot; CACHE_WRITE_SLOTS] =
        [BreakpointSlot::S0, BreakpointSlot::S1, BreakpointSlot::S2, BreakpointSlot::S3];

    fn index(self) -> usize {
        self as usize
    }
}

/// Where a breakpoint sits: which input item, and which block inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Position {
    item: usize,
    block: usize,
}

#[derive(Debug, Clone, Copy)]
struct SlotState {
    position: Position,
    /// An anchored slot is frozen for the context's life. Anchoring is how a
    /// caller says "this prefix is the reusable one" and means it.
    anchored: bool,
}

/// Why placing or moving a breakpoint was refused.
///
/// Every variant is a mistake that would otherwise show up as an unexplained
/// cache miss or an unexplained bill, so each is refused before the state
/// changes rather than reported by the server afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BreakpointError {
    /// The slot already holds a breakpoint. Slots never overwrite: silently
    /// reusing one would move a breakpoint the caller still believes is where
    /// they put it.
    SlotAlreadyInUse(BreakpointSlot),
    /// The slot is anchored, and anchors do not move or clear.
    SlotIsAnchored(BreakpointSlot),
    /// The slot is empty, so there is nothing to move.
    SlotIsEmpty(BreakpointSlot),
    /// No item yet holds a block a breakpoint could attach to. Function calls
    /// and string-valued function outputs have no blocks at all.
    NoBlockToMark,
    /// Another slot already marks that exact block. One block carries one wire
    /// breakpoint, so two slots there would consume two of four writes while
    /// producing one — an undercount of the budget.
    BlockAlreadyMarked,
}

impl std::fmt::Display for BreakpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BreakpointError::SlotAlreadyInUse(s) => write!(f, "cache slot {s:?} already holds a breakpoint"),
            BreakpointError::SlotIsAnchored(s) => write!(f, "cache slot {s:?} is anchored and cannot be moved"),
            BreakpointError::SlotIsEmpty(s) => write!(f, "cache slot {s:?} holds no breakpoint"),
            BreakpointError::NoBlockToMark => {
                write!(f, "no content block is available to carry a cache breakpoint")
            }
            BreakpointError::BlockAlreadyMarked => {
                write!(f, "another slot already marks that block; one block carries one breakpoint")
            }
        }
    }
}

impl std::error::Error for BreakpointError {}

/// The conversation: a frozen tool array, an append-only item list, and four
/// breakpoint slots.
///
/// Reusable across turns and across models. Per-call parameters live on
/// [`Request`](crate::request::Request), which borrows this.
#[derive(Debug)]
pub struct Context {
    tools: Vec<FunctionTool>,
    items: Vec<InputItem>,
    slots: [Option<SlotState>; CACHE_WRITE_SLOTS],
}

impl Context {
    /// A context over exactly these tools, in exactly this order.
    ///
    /// Order matters as much as content: reordering the array rewrites the first
    /// bytes of the prefix and costs every cached token. Pass an empty vector
    /// for a conversation with no tools; the `tools` field is then omitted
    /// rather than sent empty, since `[]` and absent render differently.
    pub fn new(tools: Vec<FunctionTool>) -> Self {
        Self { tools, items: Vec::new(), slots: [None; CACHE_WRITE_SLOTS] }
    }

    /// The frozen tool array.
    ///
    /// Read-only, and there is no setter. Changing the array is what costs the
    /// whole cached prefix, so the type offers no way to do it:
    ///
    /// ```compile_fail
    /// use openai::context::Context;
    /// use openai::tools::FunctionTool;
    /// use serde_json::json;
    ///
    /// let mut context = Context::new(vec![FunctionTool::new("f", json!({}))]);
    /// context.tools_mut().pop();
    /// ```
    pub fn tools(&self) -> &[FunctionTool] {
        &self.tools
    }

    /// The input items, in wire order.
    pub fn items(&self) -> &[InputItem] {
        &self.items
    }

    /// How many cache-write slots are filled. [`Request::new`](crate::request::Request::new)
    /// compares this against what the model and caching mode allow.
    pub fn breakpoint_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Restrict the callable set to these tools while leaving the array — and
    /// the cache — untouched.
    ///
    /// This is the answer to "these tools shouldn't be available this turn".
    /// Removing them from the array would answer the same question and cost the
    /// entire cached prefix. Every name is checked against the array, so a
    /// stale name surfaces here instead of as a 400.
    pub fn allow_tools(&self, mode: AllowedToolsMode, names: &[&str]) -> Result<AllowedTools, AllowedToolsError> {
        if names.is_empty() {
            return Err(AllowedToolsError::EmptyAllowedSet);
        }
        let mut allowed: Vec<String> = Vec::with_capacity(names.len());
        for name in names {
            if !self.tools.iter().any(|t| t.name == *name) {
                return Err(AllowedToolsError::UnknownTool((*name).to_string()));
            }
            if allowed.iter().any(|a| a == name) {
                return Err(AllowedToolsError::DuplicateTool((*name).to_string()));
            }
            allowed.push((*name).to_string());
        }
        Ok(AllowedTools::new(mode, allowed))
    }

    // ── Appending ───────────────────────────────────────────────────────────

    /// Append a developer message. Developer instructions outrank user input.
    pub fn push_developer(&mut self, blocks: Vec<InputBlock>) {
        self.push_input(InputRole::Developer, blocks);
    }

    /// Append a one-block developer message.
    pub fn push_developer_text(&mut self, text: impl Into<String>) {
        self.push_developer(vec![InputBlock::text(text)]);
    }

    /// Append a user message.
    pub fn push_user(&mut self, blocks: Vec<InputBlock>) {
        self.push_input(InputRole::User, blocks);
    }

    /// Append a one-block user message.
    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.push_user(vec![InputBlock::text(text)]);
    }

    /// Append a message from either input-vocabulary role, chosen at runtime.
    pub fn push_input(&mut self, role: InputRole, blocks: Vec<InputBlock>) {
        self.items.push(InputItem::Message(Message::Input { role, content: blocks }));
    }

    /// Append an assistant message, labelled commentary or final answer.
    ///
    /// The blocks are the *output* vocabulary — `output_text` or `refusal` —
    /// because that is what OpenAI accepts under this role. An `input_text`
    /// block here was the 400 this type split exists to prevent.
    ///
    /// The label is required rather than optional because OpenAI asks that it be
    /// preserved on every replayed assistant message, and a missing label makes
    /// a preamble look like a finished answer.
    pub fn push_assistant(&mut self, phase: AssistantPhase, blocks: Vec<OutputBlock>) {
        self.items.push(InputItem::Message(Message::Assistant { phase, content: blocks }));
    }

    /// Append one complete assistant item with its original phase and blocks.
    pub fn push_assistant_item(&mut self, phase: AssistantPhase, blocks: Vec<OutputBlock>) {
        self.push_assistant(phase, blocks);
    }

    /// Append a one-block assistant message holding what the model said.
    pub fn push_assistant_text(&mut self, phase: AssistantPhase, text: impl Into<String>) {
        self.push_assistant(phase, vec![OutputBlock::text(text)]);
    }

    /// Append a one-block assistant message holding a refusal the model gave.
    pub fn push_assistant_refusal(&mut self, phase: AssistantPhase, refusal: impl Into<String>) {
        self.push_assistant(phase, vec![OutputBlock::refusal(refusal)]);
    }

    /// Append a function call the model made, with its arguments exactly as the
    /// model emitted them.
    pub fn push_function_call(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) {
        self.items.push(InputItem::FunctionCall(FunctionCall {
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }));
    }

    /// Append a function result as a single string. Cheapest and commonest, but
    /// it holds no block, so no breakpoint can land on it — use
    /// [`Context::push_function_call_output_blocks`] where you want one.
    pub fn push_function_call_output(&mut self, call_id: impl Into<String>, output: impl Into<String>) {
        self.items.push(InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: call_id.into(),
            output: FunctionOutput::Text(output.into()),
        }));
    }

    /// Append a function result as content blocks, so a breakpoint can mark its
    /// end. OpenAI recommends exactly this in long agent threads: a breakpoint
    /// after each tool result preserves the earlier prefix when the thread forks.
    pub fn push_function_call_output_blocks(&mut self, call_id: impl Into<String>, blocks: Vec<InputBlock>) {
        self.items.push(InputItem::FunctionCallOutput(FunctionCallOutput {
            call_id: call_id.into(),
            output: FunctionOutput::Blocks(blocks),
        }));
    }

    /// Append a reasoning item from an earlier response.
    ///
    /// Reasoning models answer better when their own reasoning is handed back
    /// alongside function outputs. The content is opaque; this replays it.
    pub fn push_reasoning(&mut self, id: impl Into<String>, encrypted_content: impl Into<String>) {
        self.items.push(InputItem::Reasoning(ReplayedReasoning {
            id: id.into(),
            encrypted_content: encrypted_content.into(),
        }));
    }

    // ── Breakpoint slots ────────────────────────────────────────────────────

    /// Append a developer message whose text ends a reusable prefix, and anchor
    /// `slot` there for good.
    ///
    /// This is the shape OpenAI documents for reusable instructions, and the
    /// reason it is a method rather than advice: the top-level `instructions`
    /// field **cannot** carry a breakpoint, so instructions you intend to reuse
    /// have to live in an `input_text` block inside a developer message. See
    /// [`UncacheableInstructions`](crate::request::UncacheableInstructions) for
    /// the field this replaces.
    pub fn push_anchored_developer_text(
        &mut self,
        slot: BreakpointSlot,
        text: impl Into<String>,
    ) -> Result<(), BreakpointError> {
        if self.slots[slot.index()].is_some() {
            return Err(BreakpointError::SlotAlreadyInUse(slot));
        }
        self.push_developer_text(text);
        let position = self.tail_position().expect("a message was just pushed with one block");
        self.mark(position, true);
        self.slots[slot.index()] = Some(SlotState { position, anchored: true });
        Ok(())
    }

    /// Anchor `slot` at the end of the latest item, permanently.
    ///
    /// Use it after a tool result whose prefix you want to keep reusable no
    /// matter how the thread continues. Anchored means anchored: the slot will
    /// refuse to move or clear afterwards.
    pub fn anchor_breakpoint(&mut self, slot: BreakpointSlot) -> Result<(), BreakpointError> {
        self.place(slot, true)
    }

    /// Put `slot` at the end of the latest item, leaving it movable.
    ///
    /// A rolling slot follows the growing conversation, so each request writes a
    /// cache entry covering everything so far. Rolling it clears the old mark
    /// and sets a new one; the content itself is never touched.
    pub fn roll_breakpoint(&mut self, slot: BreakpointSlot) -> Result<(), BreakpointError> {
        if let Some(state) = self.slots[slot.index()] {
            if state.anchored {
                return Err(BreakpointError::SlotIsAnchored(slot));
            }
            let target = self.tail_position().ok_or(BreakpointError::NoBlockToMark)?;
            if target != state.position {
                self.reject_if_marked_by_another(slot, target)?;
                self.mark(state.position, false);
                self.mark(target, true);
                self.slots[slot.index()] = Some(SlotState { position: target, anchored: false });
            }
            Ok(())
        } else {
            self.place(slot, false)
        }
    }

    /// Empty a rolling slot and clear its mark. Anchored slots refuse.
    pub fn clear_breakpoint(&mut self, slot: BreakpointSlot) -> Result<(), BreakpointError> {
        let Some(state) = self.slots[slot.index()] else {
            return Err(BreakpointError::SlotIsEmpty(slot));
        };
        if state.anchored {
            return Err(BreakpointError::SlotIsAnchored(slot));
        }
        self.mark(state.position, false);
        self.slots[slot.index()] = None;
        Ok(())
    }

    /// Whether `slot` is anchored, and so frozen for this context's life.
    pub fn is_anchored(&self, slot: BreakpointSlot) -> bool {
        self.slots[slot.index()].is_some_and(|s| s.anchored)
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn place(&mut self, slot: BreakpointSlot, anchored: bool) -> Result<(), BreakpointError> {
        if self.slots[slot.index()].is_some() {
            return Err(BreakpointError::SlotAlreadyInUse(slot));
        }
        let position = self.tail_position().ok_or(BreakpointError::NoBlockToMark)?;
        self.reject_if_marked_by_another(slot, position)?;
        self.mark(position, true);
        self.slots[slot.index()] = Some(SlotState { position, anchored });
        Ok(())
    }

    /// The last block, of the last item, that can carry a breakpoint. Sites the
    /// API refuses are skipped — a function call, a string-valued output, a
    /// reasoning item, a refusal — because the nearest legal boundary is the one
    /// the caller means.
    fn tail_position(&self) -> Option<Position> {
        self.items
            .iter()
            .enumerate()
            .rev()
            .find_map(|(item, entry)| entry.last_breakpoint_site().map(|block| Position { item, block }))
    }

    fn reject_if_marked_by_another(&self, slot: BreakpointSlot, position: Position) -> Result<(), BreakpointError> {
        let clash = self
            .slots
            .iter()
            .enumerate()
            .any(|(i, other)| i != slot.index() && other.is_some_and(|s| s.position == position));
        if clash { Err(BreakpointError::BlockAlreadyMarked) } else { Ok(()) }
    }

    fn mark(&mut self, position: Position, present: bool) {
        let site = self.items[position.item]
            .breakpoint_site_mut(position.block)
            .expect("a Position is only ever built from a block that accepts a breakpoint");
        *site = present.then(PromptCacheBreakpoint::explicit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::{ImageSource, InputItem};
    use crate::values::ImageDetail;
    use serde_json::json;

    fn tools() -> Vec<FunctionTool> {
        vec![
            FunctionTool::new("read_file", json!({"type": "object"})),
            FunctionTool::new("write_file", json!({"type": "object"})),
        ]
    }

    fn breakpoints(context: &Context) -> usize {
        context.items().iter().map(InputItem::breakpoint_count).sum()
    }

    #[test]
    fn a_new_context_holds_its_tools_and_nothing_else() {
        let context = Context::new(tools());
        assert_eq!(context.tools().len(), 2);
        assert!(context.items().is_empty());
        assert_eq!(context.breakpoint_count(), 0);
    }

    #[test]
    fn allowed_tools_must_name_tools_that_exist() {
        let context = Context::new(tools());
        let ok = context.allow_tools(AllowedToolsMode::Auto, &["read_file"]).unwrap();
        assert_eq!(ok.names(), ["read_file"]);
        assert_eq!(ok.mode(), AllowedToolsMode::Auto);

        assert_eq!(
            context.allow_tools(AllowedToolsMode::Auto, &["nope"]),
            Err(AllowedToolsError::UnknownTool("nope".into()))
        );
        assert_eq!(context.allow_tools(AllowedToolsMode::Auto, &[]), Err(AllowedToolsError::EmptyAllowedSet));
        assert_eq!(
            context.allow_tools(AllowedToolsMode::Auto, &["read_file", "read_file"]),
            Err(AllowedToolsError::DuplicateTool("read_file".into()))
        );
    }

    #[test]
    fn an_anchored_developer_message_carries_the_breakpoint() {
        let mut context = Context::new(vec![]);
        context.push_anchored_developer_text(BreakpointSlot::S0, "stable").unwrap();
        assert_eq!(context.breakpoint_count(), 1);
        assert_eq!(breakpoints(&context), 1);
        assert!(context.is_anchored(BreakpointSlot::S0));

        let v = serde_json::to_value(context.items()).unwrap();
        assert_eq!(v[0]["role"], "developer");
        assert_eq!(v[0]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
    }

    /// An anchor is a promise. Breaking it silently would turn a cache read into
    /// a cache write with nothing in the logs to explain the bill.
    #[test]
    fn anchors_refuse_to_move_or_clear() {
        let mut context = Context::new(vec![]);
        context.push_anchored_developer_text(BreakpointSlot::S0, "stable").unwrap();
        context.push_user_text("question");

        assert_eq!(
            context.roll_breakpoint(BreakpointSlot::S0),
            Err(BreakpointError::SlotIsAnchored(BreakpointSlot::S0))
        );
        assert_eq!(
            context.clear_breakpoint(BreakpointSlot::S0),
            Err(BreakpointError::SlotIsAnchored(BreakpointSlot::S0))
        );
        // The anchor is still exactly where it was put.
        let v = serde_json::to_value(context.items()).unwrap();
        assert!(v[0]["content"][0].get("prompt_cache_breakpoint").is_some());
        assert!(v[1]["content"][0].get("prompt_cache_breakpoint").is_none());
    }

    #[test]
    fn a_slot_is_never_silently_reused() {
        let mut context = Context::new(vec![]);
        context.push_anchored_developer_text(BreakpointSlot::S0, "stable").unwrap();
        assert_eq!(
            context.push_anchored_developer_text(BreakpointSlot::S0, "other"),
            Err(BreakpointError::SlotAlreadyInUse(BreakpointSlot::S0))
        );
        context.push_user_text("q");
        assert_eq!(
            context.anchor_breakpoint(BreakpointSlot::S0),
            Err(BreakpointError::SlotAlreadyInUse(BreakpointSlot::S0))
        );
    }

    /// Rolling moves metadata only: the old mark clears, the new one appears,
    /// and no content byte changes.
    #[test]
    fn rolling_moves_the_mark_and_nothing_else() {
        let mut context = Context::new(vec![]);
        context.push_user_text("one");
        context.roll_breakpoint(BreakpointSlot::S1).unwrap();
        assert_eq!(breakpoints(&context), 1);

        context.push_assistant_text(AssistantPhase::FinalAnswer, "two");
        context.push_user_text("three");
        context.roll_breakpoint(BreakpointSlot::S1).unwrap();

        let v = serde_json::to_value(context.items()).unwrap();
        assert!(v[0]["content"][0].get("prompt_cache_breakpoint").is_none());
        assert_eq!(v[2]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
        assert_eq!(v[0]["content"][0]["text"], "one");
        assert_eq!(v[2]["content"][0]["text"], "three");
        assert_eq!(breakpoints(&context), 1);
        assert_eq!(context.breakpoint_count(), 1);
    }

    /// Rolling twice with no new content is a no-op, not an error: the slot is
    /// already where it would be put.
    #[test]
    fn rolling_to_the_same_place_is_idempotent() {
        let mut context = Context::new(vec![]);
        context.push_user_text("one");
        context.roll_breakpoint(BreakpointSlot::S0).unwrap();
        context.roll_breakpoint(BreakpointSlot::S0).unwrap();
        assert_eq!(breakpoints(&context), 1);
        assert_eq!(context.breakpoint_count(), 1);
    }

    #[test]
    fn clearing_frees_the_slot_and_the_mark() {
        let mut context = Context::new(vec![]);
        context.push_user_text("one");
        context.roll_breakpoint(BreakpointSlot::S2).unwrap();
        context.clear_breakpoint(BreakpointSlot::S2).unwrap();
        assert_eq!(context.breakpoint_count(), 0);
        assert_eq!(breakpoints(&context), 0);
        assert_eq!(context.clear_breakpoint(BreakpointSlot::S2), Err(BreakpointError::SlotIsEmpty(BreakpointSlot::S2)));
    }

    /// One block carries one wire breakpoint. Two slots on it would spend two
    /// of four writes to produce one, so the second is refused.
    #[test]
    fn two_slots_cannot_mark_one_block() {
        let mut context = Context::new(vec![]);
        context.push_user_text("one");
        context.roll_breakpoint(BreakpointSlot::S0).unwrap();
        assert_eq!(context.roll_breakpoint(BreakpointSlot::S1), Err(BreakpointError::BlockAlreadyMarked));
        assert_eq!(context.breakpoint_count(), 1);
    }

    #[test]
    fn there_is_nothing_to_mark_in_an_empty_context() {
        let mut context = Context::new(vec![]);
        assert_eq!(context.roll_breakpoint(BreakpointSlot::S0), Err(BreakpointError::NoBlockToMark));
        assert_eq!(context.anchor_breakpoint(BreakpointSlot::S0), Err(BreakpointError::NoBlockToMark));
    }

    /// A string-valued function output has no block, so the breakpoint falls
    /// back to the nearest earlier block rather than failing or inventing one.
    #[test]
    fn blockless_items_are_skipped_when_looking_for_a_boundary() {
        let mut context = Context::new(tools());
        context.push_user_text("go");
        context.push_function_call("call_1", "read_file", r#"{"path":"a"}"#);
        context.push_function_call_output("call_1", "contents");
        context.roll_breakpoint(BreakpointSlot::S0).unwrap();

        let v = serde_json::to_value(context.items()).unwrap();
        assert_eq!(v[0]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
        assert!(v[2].get("prompt_cache_breakpoint").is_none());
    }

    /// A block-valued output can carry one, which is what OpenAI recommends
    /// after each tool result in a long thread.
    #[test]
    fn a_block_valued_function_output_can_be_marked() {
        let mut context = Context::new(tools());
        context.push_user_text("go");
        context.push_function_call("call_1", "read_file", "{}");
        context.push_function_call_output_blocks("call_1", vec![InputBlock::text("contents")]);
        context.roll_breakpoint(BreakpointSlot::S0).unwrap();

        let v = serde_json::to_value(context.items()).unwrap();
        assert_eq!(v[2]["output"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
    }

    #[test]
    fn a_breakpoint_can_mark_the_last_block_of_a_multi_block_message() {
        let mut context = Context::new(vec![]);
        context.push_user(vec![
            InputBlock::text("look at this"),
            InputBlock::image(ImageSource::FileId("file-1".into()), ImageDetail::Low),
        ]);
        context.roll_breakpoint(BreakpointSlot::S0).unwrap();

        let v = serde_json::to_value(context.items()).unwrap();
        assert!(v[0]["content"][0].get("prompt_cache_breakpoint").is_none());
        assert_eq!(v[0]["content"][1]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
    }

    /// All four slots can be filled, and there is no fifth to try — the limit
    /// is the enum, not a runtime check.
    #[test]
    fn four_slots_fill_and_there_is_no_fifth() {
        let mut context = Context::new(vec![]);
        for (i, slot) in BreakpointSlot::ALL.iter().enumerate() {
            context.push_user_text(format!("turn {i}"));
            context.roll_breakpoint(*slot).unwrap();
        }
        assert_eq!(context.breakpoint_count(), CACHE_WRITE_SLOTS);
        assert_eq!(breakpoints(&context), CACHE_WRITE_SLOTS);
        assert_eq!(BreakpointSlot::ALL.len(), CACHE_WRITE_SLOTS);
    }

    #[test]
    fn reads_look_further_back_than_writes_reach() {
        const { assert!(CACHE_READ_BREAKPOINT_LOOKBACK > CACHE_WRITE_SLOTS) };
    }
}
