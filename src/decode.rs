//! One `data:` payload becomes one [`StreamEvent`].
//!
//! Separate from [`crate::stream`], which states what the events *are*, because
//! this file states how the bytes map onto them and the two grow at different
//! rates: OpenAI adds event types often and changes the shape of an existing one
//! almost never.
//!
//! # What the decoder refuses, and what it does not
//!
//! An event type this crate does not know is [`StreamEvent::Unmodeled`], never
//! an error: adding one is a compatible change by OpenAI's own policy, so a
//! decoder that errors is one a routine server release breaks.
//!
//! What *is* an error is a frame contradicting the schema: not JSON, not an
//! object, no `type`, or a field of the wrong type. And "required" here means
//! required *by the wire*, which is not the same as documented. `item_id` is
//! documented on nearly every event and omitted by some gateways, so it decodes
//! as `Option` — requiring it would fail frames that are fine. `output_index`
//! is required, because the wire always sends it: a hand-written fixture that
//! omitted it is what taught this crate the difference.

use serde_json::Value;

use crate::content::ReplayedReasoning;
use crate::hosted::lifecycle_of;
use crate::items::{CalledFunction, FunctionArguments, OutputItem, ReasoningItem, ResponseError, ResponseSnapshot};
use crate::stream::{AudioStream, FrameError, PartBoundary, ProgressStage, StreamEvent, TextStream};
use crate::values::{AssistantPhase, HostedTool, IncompleteReason, ResponseStatus};

/// The `(stream, is_done)` pair a text event names, or `None` when it is not one.
///
/// Eight event types, four streams, two phases each. Stating them as a table
/// keeps the decoder's body free of them.
fn text_event(kind: &str) -> Option<(TextStream, bool)> {
    Some(match kind {
        "response.output_text.delta" => (TextStream::Output, false),
        "response.output_text.done" => (TextStream::Output, true),
        "response.refusal.delta" => (TextStream::Refusal, false),
        "response.refusal.done" => (TextStream::Refusal, true),
        "response.reasoning_summary_text.delta" => (TextStream::ReasoningSummary, false),
        "response.reasoning_summary_text.done" => (TextStream::ReasoningSummary, true),
        "response.reasoning_text.delta" => (TextStream::Reasoning, false),
        "response.reasoning_text.done" => (TextStream::Reasoning, true),
        _ => return None,
    })
}

/// The `(stream, boundary)` pair a part event names, or `None`.
fn part_event(kind: &str) -> Option<(TextStream, PartBoundary)> {
    Some(match kind {
        "response.content_part.added" => (TextStream::Output, PartBoundary::Added),
        "response.content_part.done" => (TextStream::Output, PartBoundary::Done),
        "response.reasoning_summary_part.added" => (TextStream::ReasoningSummary, PartBoundary::Added),
        "response.reasoning_summary_part.done" => (TextStream::ReasoningSummary, PartBoundary::Done),
        _ => return None,
    })
}

/// The `(tool, is_done)` pair a hosted-tool input event names, or `None`.
///
/// Three tools compose input a character at a time, and each spells the finished
/// form differently — `code`, `arguments`, `input`. The field name travels with
/// the tool here so the decoder needs no case for it.
fn hosted_input_event(kind: &str) -> Option<(HostedTool, bool, &'static str)> {
    Some(match kind {
        "response.code_interpreter_call_code.delta" => (HostedTool::CodeInterpreter, false, "code"),
        "response.code_interpreter_call_code.done" => (HostedTool::CodeInterpreter, true, "code"),
        "response.mcp_call_arguments.delta" => (HostedTool::Mcp, false, "arguments"),
        "response.mcp_call_arguments.done" => (HostedTool::Mcp, true, "arguments"),
        "response.custom_tool_call_input.delta" => (HostedTool::Custom, false, "input"),
        "response.custom_tool_call_input.done" => (HostedTool::Custom, true, "input"),
        _ => return None,
    })
}

/// The `(stream, is_done)` pair an audio event names, or `None`.
fn audio_event(kind: &str) -> Option<(AudioStream, bool)> {
    Some(match kind {
        "response.audio.delta" => (AudioStream::Audio, false),
        "response.audio.done" => (AudioStream::Audio, true),
        "response.audio.transcript.delta" => (AudioStream::Transcript, false),
        "response.audio.transcript.done" => (AudioStream::Transcript, true),
        _ => return None,
    })
}

/// The non-terminal stage a progress event announces, or `None`.
///
/// The terminal three are decoded into their own variants, so they are not here:
/// settling depends on them, and a stage field would let a caller forget one.
fn progress_event(kind: &str) -> Option<ProgressStage> {
    Some(match kind {
        "response.created" => ProgressStage::Created,
        "response.queued" => ProgressStage::Queued,
        "response.in_progress" => ProgressStage::InProgress,
        _ => return None,
    })
}

/// One frame, decoded.
///
/// Ordered by how often a frame is each kind: text deltas dominate a real
/// stream by an order of magnitude, so they are matched first.
pub(crate) fn event_from_json(frame: &Value) -> Result<StreamEvent, FrameError> {
    if !frame.is_object() {
        return Err(FrameError::NotAnObject);
    }
    let kind = require_str(frame, "type")?;

    if let Some((stream, done)) = text_event(kind) {
        let (text_field, index_field) = stream.wire_fields();
        let output_index = require_u32(frame, "output_index")?;
        let part_index = require_u32(frame, index_field)?;
        return Ok(if done {
            StreamEvent::TextDone { stream, output_index, part_index, text: require_str(frame, text_field)?.to_owned() }
        } else {
            StreamEvent::TextDelta { stream, output_index, part_index, delta: require_str(frame, "delta")?.to_owned() }
        });
    }

    if let Some((stream, boundary)) = part_event(kind) {
        let (text_field, index_field) = stream.wire_fields();
        return Ok(StreamEvent::Part {
            stream,
            boundary,
            output_index: require_u32(frame, "output_index")?,
            part_index: require_u32(frame, index_field)?,
            // An `added` part is announced empty, and the `part` object is
            // present but has nothing in it. So an absent text is the empty
            // string rather than a broken frame.
            text: frame
                .get("part")
                .and_then(|part| part.get(text_field))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        });
    }

    if let Some((tool, phase)) = lifecycle_of(kind) {
        return Ok(StreamEvent::HostedToolLifecycle {
            tool,
            phase,
            output_index: require_u32(frame, "output_index")?,
            item_id: optional_string(frame, "item_id"),
        });
    }

    if let Some((tool, done, whole_field)) = hosted_input_event(kind) {
        let output_index = require_u32(frame, "output_index")?;
        let item_id = optional_string(frame, "item_id");
        return Ok(if done {
            StreamEvent::HostedToolInputDone {
                tool,
                output_index,
                item_id,
                input: require_str(frame, whole_field)?.to_owned(),
            }
        } else {
            StreamEvent::HostedToolInputDelta {
                tool,
                output_index,
                item_id,
                delta: require_str(frame, "delta")?.to_owned(),
            }
        });
    }

    if let Some((stream, done)) = audio_event(kind) {
        return Ok(StreamEvent::Audio {
            stream,
            response_id: optional_string(frame, "response_id"),
            // The `done` form carries no payload at all, which is why `delta` is
            // read as optional here and required on the delta form.
            delta: if done { String::new() } else { require_str(frame, "delta")?.to_owned() },
            done,
        });
    }

    if let Some(stage) = progress_event(kind) {
        return Ok(StreamEvent::ResponseProgress { stage, response: snapshot(require(frame, "response")?)? });
    }

    Ok(match kind {
        "response.output_item.added" => StreamEvent::OutputItemAdded {
            output_index: require_u32(frame, "output_index")?,
            item: output_item(require(frame, "item")?)?,
        },
        "response.output_item.done" => StreamEvent::OutputItemDone {
            output_index: require_u32(frame, "output_index")?,
            item: output_item(require(frame, "item")?)?,
        },
        "response.function_call_arguments.delta" => StreamEvent::FunctionArgumentsDelta {
            output_index: require_u32(frame, "output_index")?,
            item_id: optional_string(frame, "item_id"),
            delta: require_str(frame, "delta")?.to_owned(),
        },
        "response.function_call_arguments.done" => StreamEvent::FunctionArgumentsDone {
            output_index: require_u32(frame, "output_index")?,
            item_id: optional_string(frame, "item_id"),
            arguments: FunctionArguments::from_wire(require_str(frame, "arguments")?),
        },
        "response.image_generation_call.partial_image" => StreamEvent::PartialImage {
            output_index: require_u32(frame, "output_index")?,
            item_id: optional_string(frame, "item_id"),
            partial_image_index: require_u32(frame, "partial_image_index")?,
            partial_image_base64: require_str(frame, "partial_image_b64")?.to_owned(),
        },
        "response.output_text.annotation.added" => StreamEvent::Annotation {
            output_index: require_u32(frame, "output_index")?,
            content_index: require_u32(frame, "content_index")?,
            annotation_index: require_u32(frame, "annotation_index")?,
            annotation: require(frame, "annotation")?.clone(),
        },
        "response.completed" => StreamEvent::Completed(snapshot(require(frame, "response")?)?),
        "response.failed" => StreamEvent::Failed(snapshot(require(frame, "response")?)?),
        "response.incomplete" => StreamEvent::Incomplete(snapshot(require(frame, "response")?)?),
        "error" => StreamEvent::Error(error(frame)),
        // The shell-call events are deliberately here. Their payload is a
        // structured command list whose shape a caller running commands has to
        // agree with exactly, and modeling it thinly would be worse than saying
        // plainly that it is not modeled. The lifecycle of a shell call still
        // arrives, through `OutputItemAdded` and `OutputItemDone`.
        other => StreamEvent::Unmodeled { kind: other.to_owned() },
    })
}

// ── Field readers ────────────────────────────────────────────────────────────

fn require<'a>(object: &'a Value, field: &'static str) -> Result<&'a Value, FrameError> {
    object.get(field).ok_or(FrameError::MissingField { field })
}

fn require_str<'a>(object: &'a Value, field: &'static str) -> Result<&'a str, FrameError> {
    require(object, field)?.as_str().ok_or(FrameError::WrongType { field, expected: "a string" })
}

fn require_u32(object: &Value, field: &'static str) -> Result<u32, FrameError> {
    let wrong = FrameError::WrongType { field, expected: "a non-negative integer" };
    let number = require(object, field)?.as_u64().ok_or(wrong)?;
    u32::try_from(number).map_err(|_| FrameError::WrongType { field, expected: "a non-negative integer" })
}

fn optional_string(object: &Value, field: &str) -> Option<String> {
    object.get(field).and_then(Value::as_str).map(str::to_owned)
}

// ── Composite shapes ─────────────────────────────────────────────────────────

/// One entry of an `output` array, or of an `item` field.
pub(crate) fn output_item(item: &Value) -> Result<OutputItem, FrameError> {
    if !item.is_object() {
        return Err(FrameError::WrongType { field: "item", expected: "an object" });
    }
    Ok(match require_str(item, "type")? {
        "message" => OutputItem::Message {
            id: optional_string(item, "id"),
            phase: item.get("phase").and_then(Value::as_str).and_then(AssistantPhase::from_str),
            text: joined_output_text(item),
            refusal: joined_refusal(item),
        },
        "function_call" => OutputItem::FunctionCall(CalledFunction {
            id: optional_string(item, "id"),
            call_id: require_str(item, "call_id")?.to_owned(),
            name: require_str(item, "name")?.to_owned(),
            arguments: FunctionArguments::from_wire(optional_string(item, "arguments").unwrap_or_default()),
        }),
        "reasoning" => OutputItem::Reasoning(ReasoningItem {
            id: require_str(item, "id")?.to_owned(),
            encrypted_content: item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .filter(|content| !content.is_empty())
                .map(str::to_owned),
            summary: joined_summary(item),
        }),
        other => match hosted_call_tool(other) {
            // A hosted tool's call item, named by the tool it belongs to. The
            // whole item travels along because its useful fields differ per tool
            // — queries, a container, a command list — and a consumer that
            // enabled the tool knows which ones it needs.
            Some(tool) => OutputItem::HostedToolCall {
                tool,
                id: optional_string(item, "id"),
                status: optional_string(item, "status"),
                item: item.clone(),
            },
            None => OutputItem::Unmodeled { kind: other.to_owned() },
        },
    })
}

/// Which hosted tool an output item's `type` names, or `None` when it names none.
///
/// The item types and the event families are two vocabularies for one set of
/// tools — `web_search_call` the item, `response.web_search_call.*` the events —
/// so this is where the two are tied together.
fn hosted_call_tool(kind: &str) -> Option<HostedTool> {
    Some(match kind {
        "file_search_call" => HostedTool::FileSearch,
        "web_search_call" => HostedTool::WebSearch,
        "code_interpreter_call" => HostedTool::CodeInterpreter,
        "image_generation_call" => HostedTool::ImageGeneration,
        "mcp_call" => HostedTool::Mcp,
        "mcp_list_tools" => HostedTool::McpListTools,
        "shell_call" => HostedTool::Shell,
        "local_shell_call" => HostedTool::LocalShell,
        "computer_call" => HostedTool::Computer,
        "apply_patch_call" => HostedTool::ApplyPatch,
        "custom_tool_call" => HostedTool::Custom,
        "tool_search_call" => HostedTool::ToolSearch,
        _ => return None,
    })
}

/// The item's blocks of one type, joined.
fn joined_blocks(item: &Value, block_type: &str, text_field: &str) -> String {
    let Some(blocks) = item.get("content").and_then(Value::as_array) else {
        return String::new();
    };
    let mut joined = String::new();
    for block in blocks.iter().filter(|block| block.get("type").and_then(Value::as_str) == Some(block_type)) {
        if let Some(text) = block.get(text_field).and_then(Value::as_str) {
            joined.push_str(text);
        }
    }
    joined
}

/// The item's `output_text` blocks, joined. Refusals are deliberately not folded
/// in: a refusal concatenated into an answer reads as the answer.
fn joined_output_text(item: &Value) -> String {
    joined_blocks(item, "output_text", "text")
}

/// The item's `refusal` blocks, joined, and empty when it refused nothing.
fn joined_refusal(item: &Value) -> String {
    joined_blocks(item, "refusal", "refusal")
}

/// A reasoning item's `summary` parts, joined.
///
/// A different array from `content`: reasoning items carry their summary in
/// `summary`, as `summary_text` parts, and their raw reasoning in `content`.
fn joined_summary(item: &Value) -> String {
    let Some(parts) = item.get("summary").and_then(Value::as_array) else {
        return String::new();
    };
    let mut joined = String::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            joined.push_str(text);
        }
    }
    joined
}

fn error(object: &Value) -> ResponseError {
    ResponseError {
        code: optional_string(object, "code"),
        message: optional_string(object, "message").unwrap_or_default(),
        param: optional_string(object, "param"),
    }
}

/// The `response` object a progress or terminal event carries.
pub(crate) fn snapshot(response: &Value) -> Result<ResponseSnapshot, FrameError> {
    if !response.is_object() {
        return Err(FrameError::WrongType { field: "response", expected: "an object" });
    }
    let usage = match response.get("usage") {
        None | Some(Value::Null) => None,
        Some(usage) => Some(serde_json::from_value(usage.clone()).map_err(FrameError::UndecodableUsage)?),
    };
    let output = match response.get("output") {
        Some(Value::Array(items)) => items.iter().map(output_item).collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };
    Ok(ResponseSnapshot {
        id: optional_string(response, "id"),
        status: response.get("status").and_then(Value::as_str).and_then(ResponseStatus::from_str),
        usage,
        error: response.get("error").filter(|error| error.is_object()).map(error_of),
        incomplete_reason: response
            .get("incomplete_details")
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
            .and_then(IncompleteReason::from_str),
        service_tier: response
            .get("service_tier")
            .and_then(Value::as_str)
            .and_then(crate::values::ServiceTier::from_str),
        output,
    })
}

fn error_of(object: &Value) -> ResponseError {
    error(object)
}

/// A reasoning item in the form the next request needs, or `None` when it has no
/// replayable payload.
pub(crate) fn replayable(item: &ReasoningItem) -> Option<ReplayedReasoning> {
    item.encrypted_content
        .as_ref()
        .map(|content| ReplayedReasoning { id: item.id.clone(), encrypted_content: content.clone() })
}
