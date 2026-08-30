//! One streamed frame decoded, asserted through the public API alone.
//!
//! These live outside the crate rather than in a `mod tests`, and that placement
//! is itself an assertion: a consumer can decode every frame the wire sends
//! without reaching inside. A case that needed `pub(crate)` access would have
//! found a type the crate has not finished exposing.
//!
//! The frames are the shapes the wire really sends — including `sequence_number`
//! and `logprobs`, which mean nothing here and must be ignored rather than
//! refused, and `item_id`, which some gateways omit.

use openai::hosted::HostedToolPhase;
use openai::items::{FunctionArguments, OutputItem, ResponseError, ResponseSnapshot};
use openai::stream::{FrameError, PartBoundary, ProgressStage, StreamEvent, TextStream, data_payload};
use openai::values::{AssistantPhase, HostedTool, IncompleteReason, ResponseStatus};
use serde_json::json;

/// Only a `data:` line carries a payload, and the one optional space after
/// the colon is framing rather than data.
#[test]
fn only_data_lines_carry_a_payload() {
    assert_eq!(data_payload("data: {\"type\":\"x\"}"), Some("{\"type\":\"x\"}"));
    assert_eq!(data_payload("data:{\"type\":\"x\"}"), Some("{\"type\":\"x\"}"), "the space is optional");
    assert_eq!(data_payload("data:  two spaces"), Some(" two spaces"), "only one space is framing");
    assert_eq!(data_payload("event: response.completed"), None);
    assert_eq!(data_payload(": keep-alive comment"), None);
    assert_eq!(data_payload(""), None);
}

/// The captured shape of a text delta, decoded field by field.
#[test]
fn a_text_delta_decodes() {
    let event = StreamEvent::decode(
        r#"{"type":"response.output_text.delta","sequence_number":7,"item_id":"msg_1",
            "output_index":0,"content_index":0,"delta":"Hel","logprobs":[]}"#,
    )
    .unwrap();
    assert_eq!(
        event,
        StreamEvent::TextDelta { stream: TextStream::Output, output_index: 0, part_index: 0, delta: "Hel".to_owned() }
    );
    assert_eq!(event.kind(), "response.output_text.delta");
    assert!(!event.is_terminal());
}

#[test]
fn a_reasoning_summary_delta_decodes() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1", "output_index": 0, "summary_index": 1, "delta": "weighing",
        "sequence_number": 3
    }))
    .unwrap();
    assert_eq!(
        event,
        StreamEvent::TextDelta {
            stream: TextStream::ReasoningSummary,
            output_index: 0,
            part_index: 1,
            delta: "weighing".to_owned(),
        }
    );
}

/// An event type nobody has taught this crate about is a variant, not an    /// error: OpenAI may add one in any release.
#[test]
fn an_unknown_event_is_ignorable_rather_than_an_error() {
    let event = StreamEvent::decode(r#"{"type":"response.crystal_ball.delta","delta":"?"}"#).unwrap();
    assert_eq!(event, StreamEvent::Unmodeled { kind: "response.crystal_ball.delta".to_owned() });
    assert_eq!(event.kind(), "response.crystal_ball.delta");
    assert!(!event.is_terminal());
}

/// The events this release still leaves unmodeled read the same way, because
/// "well-formed, nothing to do" is one situation, not two.
///
/// Only the shell family is here now. Its payload is a structured command
/// list a caller running commands has to agree with exactly, and modeling it
/// thinly would be worse than saying plainly it is not modeled — the call's
/// own lifecycle still arrives through `OutputItemAdded` and    /// `OutputItemDone`.
#[test]
fn a_documented_but_unmodeled_event_is_also_ignorable() {
    for kind in [
        "response.shell_call_command.added",
        "response.shell_call_command.delta",
        "response.shell_call_command.done",
        "response.shell_call_output_content.delta",
        "response.shell_call_output_content.done",
    ] {
        let event = StreamEvent::from_json(&json!({"type": kind, "sequence_number": 1, "output_index": 0}))
            .unwrap_or_else(|error| panic!("{kind} should decode: {error}"));
        assert_eq!(event, StreamEvent::Unmodeled { kind: kind.to_owned() });
        assert_eq!(event.kind(), kind);
    }
}

#[test]
fn a_broken_frame_is_an_error() {
    assert!(matches!(StreamEvent::decode("not json at all"), Err(FrameError::NotJson(_))));
    assert!(matches!(StreamEvent::decode("[1,2]"), Err(FrameError::NotAnObject)));
    assert!(matches!(StreamEvent::decode("{}"), Err(FrameError::MissingField { field: "type" })));
    assert!(matches!(
        StreamEvent::from_json(&json!({"type": "response.output_text.delta", "output_index": 0})),
        Err(FrameError::MissingField { field: "content_index" })
    ));
    assert!(matches!(
        StreamEvent::from_json(&json!({
            "type": "response.output_text.delta", "output_index": -1, "content_index": 0, "delta": "x"
        })),
        Err(FrameError::WrongType { field: "output_index", .. })
    ));
}

#[test]
fn a_function_call_item_keeps_its_arguments_as_bytes() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.output_item.done", "output_index": 1, "sequence_number": 12,
        "item": {
            "type": "function_call", "id": "fc_1", "call_id": "call_abc", "name": "read_file",
            "arguments": "{\"path\":\"src/lib.rs\"}", "status": "completed"
        }
    }))
    .unwrap();
    let StreamEvent::OutputItemDone { output_index: 1, item: OutputItem::FunctionCall(call) } = event else {
        panic!("expected a finished function call");
    };
    assert_eq!(call.call_id, "call_abc");
    assert_eq!(call.name, "read_file");
    assert_eq!(call.arguments.as_str(), r#"{"path":"src/lib.rs"}"#);
    assert_eq!(call.arguments.decode().unwrap(), json!({"path": "src/lib.rs"}));
}

/// Malformed arguments stay readable and stay malformed. The bytes survive
/// so the caller can answer the model, and the failure is not silently an    /// empty argument set.
#[test]
fn malformed_arguments_fail_only_where_they_are_decoded() {
    let arguments = FunctionArguments::from_wire(r#"{"path": "src/lib.rs"#);
    assert_eq!(arguments.as_str(), r#"{"path": "src/lib.rs"#);
    let error = arguments.decode().unwrap_err();
    assert!(error.to_string().contains("EOF"), "{error}");
}

/// A function taking no arguments is streamed with them absent, and once    /// finished as the empty string. Both mean the empty argument set.
#[test]
fn absent_arguments_decode_to_the_empty_object() {
    assert_eq!(FunctionArguments::default().decode().unwrap(), json!({}));
    assert_eq!(FunctionArguments::from_wire("").decode().unwrap(), json!({}));
    assert_eq!(FunctionArguments::from_wire("  ").decode().unwrap(), json!({}));

    let event = StreamEvent::from_json(&json!({
        "type": "response.output_item.added", "output_index": 0,
        "item": {"type": "function_call", "call_id": "c1", "name": "finish"}
    }))
    .unwrap();
    let StreamEvent::OutputItemAdded { item: OutputItem::FunctionCall(call), .. } = event else {
        panic!("expected an announced function call");
    };
    assert_eq!(call.arguments, FunctionArguments::default());
}

/// A hosted tool's call item names its tool, and carries the rest of itself    /// for the fields that differ per tool.
#[test]
fn a_hosted_tool_call_item_names_its_tool() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.output_item.done", "output_index": 2,
        "item": {"type": "web_search_call", "id": "ws_1", "status": "completed", "queries": ["rust"]}
    }))
    .unwrap();
    let StreamEvent::OutputItemDone { item: OutputItem::HostedToolCall { tool, id, status, item }, .. } = event else {
        panic!("expected a hosted tool call");
    };
    assert_eq!(tool, HostedTool::WebSearch);
    assert_eq!(id.as_deref(), Some("ws_1"));
    assert_eq!(status.as_deref(), Some("completed"));
    assert_eq!(item["queries"], json!(["rust"]), "the undecoded fields survive");
}

/// An item kind belonging to no tool this crate knows is still not an error.
#[test]
fn an_output_item_kind_this_crate_ignores_is_not_an_error() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.output_item.done", "output_index": 2,
        "item": {"type": "crystal_ball_call", "id": "cb_1"}
    }))
    .unwrap();
    assert_eq!(
        event,
        StreamEvent::OutputItemDone {
            output_index: 2,
            item: OutputItem::Unmodeled { kind: "crystal_ball_call".to_owned() }
        }
    );
}

#[test]
fn a_message_item_carries_its_finished_text_and_phase() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.output_item.done", "output_index": 0,
        "item": {
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "phase": "final_answer",
            "content": [
                {"type": "output_text", "text": "Hello ", "annotations": []},
                {"type": "output_text", "text": "world", "annotations": []},
                {"type": "refusal", "refusal": "I cannot help with that"}
            ]
        }
    }))
    .unwrap();
    let StreamEvent::OutputItemDone { item: OutputItem::Message { text, phase, id, .. }, .. } = event else {
        panic!("expected a finished message");
    };
    assert_eq!(text, "Hello world", "output_text blocks join; a refusal is not text");
    assert_eq!(phase, Some(AssistantPhase::FinalAnswer));
    assert_eq!(id.as_deref(), Some("msg_1"));
}

#[test]
fn the_terminal_events_carry_usage_and_their_reason() {
    let completed = StreamEvent::from_json(&json!({
        "type": "response.completed", "sequence_number": 40,
        "response": {
            "id": "resp_1", "status": "completed",
            "usage": {
                "input_tokens": 15000,
                "input_tokens_details": {"cached_tokens": 12000, "cache_write_tokens": 3000},
                "output_tokens": 500, "output_tokens_details": {"reasoning_tokens": 400},
                "total_tokens": 15500
            }
        }
    }))
    .unwrap();
    let StreamEvent::Completed(snapshot) = completed else { panic!("expected completion") };
    assert!(StreamEvent::Completed(ResponseSnapshot::default()).is_terminal());
    assert_eq!(snapshot.id.as_deref(), Some("resp_1"));
    let usage = snapshot.usage.unwrap();
    assert_eq!(usage.input_tokens_details.cached_tokens, 12_000);
    assert_eq!(usage.input_tokens_details.cache_write_tokens, 3_000);
    assert_eq!(usage.cache_hit_rate(), Some(0.8));

    let incomplete = StreamEvent::from_json(&json!({
        "type": "response.incomplete",
        "response": {"id": "resp_2", "status": "incomplete",
                     "incomplete_details": {"reason": "max_output_tokens"}}
    }))
    .unwrap();
    let StreamEvent::Incomplete(snapshot) = incomplete else { panic!("expected an incomplete response") };
    assert_eq!(snapshot.incomplete_reason, Some(IncompleteReason::MaxOutputTokens));

    let failed = StreamEvent::from_json(&json!({
        "type": "response.failed",
        "response": {"id": "resp_3", "status": "failed",
                     "error": {"code": "server_error", "message": "upstream fell over"}}
    }))
    .unwrap();
    let StreamEvent::Failed(snapshot) = failed else { panic!("expected a failure") };
    let error = snapshot.error.unwrap();
    assert_eq!(error.code.as_deref(), Some("server_error"));
    assert_eq!(error.message, "upstream fell over");
}

/// An incomplete reason outside the documented pair reads as absent rather    /// than failing the frame, since the response itself is still usable.
#[test]
fn an_unrecognized_incomplete_reason_reads_as_absent() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.incomplete",
        "response": {"incomplete_details": {"reason": "some_new_reason"}}
    }))
    .unwrap();
    let StreamEvent::Incomplete(snapshot) = event else { panic!("expected an incomplete response") };
    assert_eq!(snapshot.incomplete_reason, None);
}

/// A `null` or absent usage is absent, and a usage object reporting only the
/// totals is those totals: an omitted breakdown is zero of that kind, which
/// is what a gateway that reports less than OpenAI does actually means. What
/// remains an error is an object whose counts contradict the schema, because
/// quietly reporting no cost would hide the one number this crate exists to    /// measure.
#[test]
fn usage_is_absent_or_as_much_of_it_as_was_reported() {
    let null = StreamEvent::from_json(&json!({"type": "response.completed", "response": {"usage": null}}));
    let StreamEvent::Completed(snapshot) = null.unwrap() else { panic!("expected completion") };
    assert_eq!(snapshot.usage, None);

    let missing = StreamEvent::from_json(&json!({"type": "response.completed", "response": {}}));
    let StreamEvent::Completed(snapshot) = missing.unwrap() else { panic!("expected completion") };
    assert_eq!(snapshot.usage, None);

    let totals_only = StreamEvent::from_json(&json!({
        "type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 2}}
    }));
    let StreamEvent::Completed(snapshot) = totals_only.unwrap() else { panic!("expected completion") };
    let usage = snapshot.usage.expect("two totals are two real numbers");
    assert_eq!(usage.input_tokens, 5);
    assert_eq!(usage.output_tokens, 2);
    assert_eq!(usage.input_tokens_details.cached_tokens, 0);

    let corrupt = StreamEvent::from_json(&json!({
        "type": "response.completed", "response": {"usage": {"input_tokens": "five"}}
    }));
    assert!(matches!(corrupt, Err(FrameError::UndecodableUsage(_))));
}

#[test]
fn the_bare_error_event_decodes() {
    let event = StreamEvent::decode(
        r#"{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":null,
            "sequence_number":2}"#,
    )
    .unwrap();
    assert_eq!(
        event,
        StreamEvent::Error(ResponseError {
            code: Some("rate_limit_exceeded".to_owned()),
            message: "slow down".to_owned(),
            param: None,
        })
    );
    assert!(event.is_terminal());
    assert_eq!(event.kind(), "error");
}

/// A terminal event may repeat the whole `output` array; it decodes with    /// they are read from there.
#[test]
fn every_text_stream_decodes_through_one_variant() {
    let delta = |kind: &str, index_field: &str| {
        StreamEvent::from_json(&json!({
            "type": kind, "item_id": "i", "output_index": 3, index_field: 2, "delta": "x", "sequence_number": 1
        }))
        .unwrap_or_else(|error| panic!("{kind}: {error}"))
    };
    for (kind, index_field, stream) in [
        ("response.output_text.delta", "content_index", TextStream::Output),
        ("response.refusal.delta", "content_index", TextStream::Refusal),
        ("response.reasoning_summary_text.delta", "summary_index", TextStream::ReasoningSummary),
        ("response.reasoning_text.delta", "content_index", TextStream::Reasoning),
    ] {
        assert_eq!(
            delta(kind, index_field),
            StreamEvent::TextDelta { stream, output_index: 3, part_index: 2, delta: "x".to_owned() }
        );
    }

    // A refusal states its whole text under `refusal`, not `text`.
    assert_eq!(
        StreamEvent::from_json(&json!({
            "type": "response.refusal.done", "output_index": 0, "content_index": 0, "refusal": "I cannot"
        }))
        .unwrap(),
        StreamEvent::TextDone {
            stream: TextStream::Refusal,
            output_index: 0,
            part_index: 0,
            text: "I cannot".to_owned()
        }
    );
}

/// Only the answer stream is answer text, and the accessor says so once so a    /// caller cannot decide it wrongly per call site.
#[test]
fn only_the_answer_stream_is_answer_text() {
    let of = |stream| StreamEvent::TextDelta { stream, output_index: 0, part_index: 0, delta: "x".to_owned() };
    assert_eq!(of(TextStream::Output).answer_text_delta(), Some("x"));
    for quiet in [TextStream::Refusal, TextStream::ReasoningSummary, TextStream::Reasoning] {
        assert_eq!(of(quiet).answer_text_delta(), None, "{quiet:?} is not answer text");
        assert!(!quiet.is_answer_text());
    }
    assert!(TextStream::Output.is_answer_text());
}

/// Eighteen lifecycle events decode into one variant carrying the pair, and    /// a tool this crate knows keeps its identity through the frame.
#[test]
fn a_hosted_tool_lifecycle_carries_its_tool_and_phase() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.web_search_call.searching", "output_index": 4, "item_id": "ws_9", "sequence_number": 2
    }))
    .unwrap();
    assert_eq!(
        event,
        StreamEvent::HostedToolLifecycle {
            tool: HostedTool::WebSearch,
            phase: HostedToolPhase::Searching,
            output_index: 4,
            item_id: Some("ws_9".to_owned()),
        }
    );
    assert!(!event.is_terminal(), "a tool running is not the end of the stream");
}

/// An `added` part is announced with an empty text, and the empty text is not    /// a broken frame.
#[test]
fn an_added_part_is_announced_empty() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.reasoning_summary_part.added", "item_id": "rs_1", "output_index": 0,
        "summary_index": 1, "part": {"type": "summary_text", "text": ""}, "sequence_number": 1
    }))
    .unwrap();
    assert_eq!(
        event,
        StreamEvent::Part {
            stream: TextStream::ReasoningSummary,
            boundary: PartBoundary::Added,
            output_index: 0,
            part_index: 1,
            text: String::new(),
        }
    );
}

/// A refusal is kept apart from the answer inside one message item, because    /// a refusal folded into the answer reads as the answer.
#[test]
fn a_message_keeps_its_refusal_apart_from_its_text() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.output_item.done", "output_index": 0,
        "item": {
            "type": "message", "id": "msg_1", "role": "assistant", "phase": "final_answer",
            "content": [
                {"type": "output_text", "text": "Here is what I can say. "},
                {"type": "refusal", "refusal": "I cannot help with the rest."}
            ]
        }
    }))
    .unwrap();
    let StreamEvent::OutputItemDone { item: OutputItem::Message { text, refusal, .. }, .. } = event else {
        panic!("expected a message");
    };
    assert_eq!(text, "Here is what I can say. ");
    assert_eq!(refusal, "I cannot help with the rest.");
}

/// A reasoning item's summary is read from `summary`, which is a different    /// array from the raw reasoning in `content`.
#[test]
fn a_reasoning_item_reads_its_summary_from_its_own_array() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.output_item.done", "output_index": 0,
        "item": {
            "type": "reasoning", "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "First I checked. "},
                        {"type": "summary_text", "text": "Then I answered."}],
            "content": []
        }
    }))
    .unwrap();
    let StreamEvent::OutputItemDone { item: OutputItem::Reasoning(item), .. } = event else {
        panic!("expected reasoning");
    };
    assert_eq!(item.summary, "First I checked. Then I answered.");
}

/// A progress event names which of the three stages arrived, and two of them
/// share one status — which is why the stage is its own value.
#[test]
fn a_progress_stage_survives_two_stages_sharing_one_status() {
    for stage in [ProgressStage::Created, ProgressStage::Queued, ProgressStage::InProgress] {
        let event = StreamEvent::from_json(&json!({"type": stage.as_str(), "response": {"status": "queued"}})).unwrap();
        assert_eq!(event.kind(), stage.as_str());
        let StreamEvent::ResponseProgress { stage: decoded, .. } = event else { panic!("expected progress") };
        assert_eq!(decoded, stage);
    }
    assert_eq!(ProgressStage::Created.status(), ResponseStatus::Queued);
    assert_eq!(ProgressStage::Queued.status(), ResponseStatus::Queued);
    assert_eq!(ProgressStage::InProgress.status(), ResponseStatus::InProgress);
}

/// A terminal snapshot's `output` array is decoded item by item, through the
/// same code as the item events.
#[test]
fn a_terminal_snapshot_decodes_its_output_array() {
    let event = StreamEvent::from_json(&json!({
        "type": "response.completed",
        "response": {"id": "resp_1", "output": [
            {"type": "message", "id": "msg_1", "role": "assistant",
             "content": [{"type": "output_text", "text": "done"}]},
            {"type": "function_call", "call_id": "c1", "name": "stop", "arguments": "{}"}
        ]}
    }))
    .unwrap();
    let StreamEvent::Completed(snapshot) = event else { panic!("expected completion") };
    assert_eq!(snapshot.output.len(), 2);
    assert!(matches!(&snapshot.output[0], OutputItem::Message { text, .. } if text == "done"));
    assert!(matches!(&snapshot.output[1], OutputItem::FunctionCall(call) if call.call_id == "c1"));
}
