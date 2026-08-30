//! Folding a sequence of events into one finished response, through the public
//! API alone.
//!
//! Outside the crate rather than in a `mod tests`, and that is itself an
//! assertion: accumulating a stream and reading the result needs nothing
//! `pub(crate)`. It also exercises the boundary the module exists for from the
//! side a consumer sees it: a `Settling` here cannot be read as an answer, and a
//! `Settled` cannot be assembled by hand.

use openai::hosted::HostedToolPhase;
use openai::items::{OutputItem, ResponseSnapshot};
use openai::settle::{PartDisagreement, SettleError, Settling};
use openai::settled::Outcome;
use openai::stream::{PartBoundary, PartKey, StreamEvent, TextStream};
use openai::values::{HostedTool, IncompleteReason, ResponseStatus};
use serde_json::json;

/// One answer-text delta, which is what most of these sequences are made of.
fn text_delta(output_index: u32, part_index: u32, delta: &str) -> StreamEvent {
    StreamEvent::TextDelta { stream: TextStream::Output, output_index, part_index, delta: delta.to_owned() }
}

/// A usage object with all four counts, so cache accounting is exercised rather
/// than defaulted past.
fn usage_frame() -> serde_json::Value {
    json!({
        "input_tokens": 1200,
        "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 100},
        "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 30},
        "total_tokens": 1240
    })
}

#[test]
fn deltas_accumulate_and_a_terminal_event_settles_them() {
    let mut settling = Settling::new();
    settling.consume(text_delta(0, 0, "Hel"));
    settling.consume(text_delta(0, 0, "lo, "));
    settling.consume(text_delta(0, 0, "world"));
    assert_eq!(settling.text_so_far(), "Hello, world");
    assert!(!settling.is_terminated());

    settling.consume(StreamEvent::Completed(ResponseSnapshot {
        id: Some("resp_1".to_owned()),
        status: Some(ResponseStatus::Completed),
        usage: Some(serde_json::from_value(usage_frame()).unwrap()),
        ..ResponseSnapshot::default()
    }));
    assert!(settling.is_terminated());

    let settled = settling.settle().unwrap();
    assert!(settled.is_completed());
    assert_eq!(settled.text, "Hello, world");
    assert_eq!(settled.id.as_deref(), Some("resp_1"));
    assert_eq!(settled.usage.unwrap().input_tokens_details.cached_tokens, 1_000);
    assert_eq!(settled.error(), None);
    assert_eq!(settled.events, 4);
}

/// The point of the whole module: without a terminal event there is no
/// `Settled`, and the text that did arrive is not handed over.
#[test]
fn a_truncated_stream_does_not_settle() {
    let mut settling = Settling::new();
    settling.consume(text_delta(0, 0, "half an ans"));
    assert_eq!(settling.text_so_far(), "half an ans", "readable as in-progress");
    assert!(!settling.is_terminated());

    let error = settling.settle().unwrap_err();
    let SettleError::Truncated { events, text_len } = error else { panic!("expected truncation") };
    assert_eq!(events, 1);
    assert_eq!(text_len, 11);
}

/// An empty stream is the degenerate truncation, and behaves the same.
#[test]
fn an_empty_stream_does_not_settle() {
    let error = Settling::new().settle().unwrap_err();
    assert!(matches!(error, SettleError::Truncated { events: 0, text_len: 0 }));
}

/// A full happy path from raw frames: reasoning, text, a function call, and
/// completion with usage.
#[test]
fn a_captured_happy_path_settles_into_text_and_one_function_call() {
    let frames = [
        r#"{"type":"response.created","sequence_number":0,"response":{"id":"resp_9","status":"in_progress"}}"#
            .to_owned(),
        r#"{"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_9"}}"#.to_owned(),
        r#"{"type":"response.output_item.added","sequence_number":2,"output_index":0,
            "item":{"type":"reasoning","id":"rs_1","summary":[]}}"#
            .to_owned(),
        r#"{"type":"response.reasoning_summary_part.added","sequence_number":3,"output_index":0,
            "summary_index":0,"item_id":"rs_1","part":{"type":"summary_text","text":""}}"#
            .to_owned(),
        r#"{"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_1",
            "output_index":0,"summary_index":0,"delta":"Reading the file"}"#
            .to_owned(),
        r#"{"type":"response.output_item.done","sequence_number":5,"output_index":0,
            "item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text",
            "text":"Reading the file"}],"encrypted_content":"gAAAAA","status":"completed"}}"#
            .to_owned(),
        r#"{"type":"response.output_item.added","sequence_number":6,"output_index":1,
            "item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress",
            "content":[]}}"#
            .to_owned(),
        r#"{"type":"response.content_part.added","sequence_number":7,"item_id":"msg_1",
            "output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}"#
            .to_owned(),
        r#"{"type":"response.output_text.delta","sequence_number":8,"item_id":"msg_1",
            "output_index":1,"content_index":0,"delta":"Reading "}"#
            .to_owned(),
        r#"{"type":"response.output_text.delta","sequence_number":9,"item_id":"msg_1",
            "output_index":1,"content_index":0,"delta":"it now."}"#
            .to_owned(),
        r#"{"type":"response.output_text.done","sequence_number":10,"item_id":"msg_1",
            "output_index":1,"content_index":0,"text":"Reading it now."}"#
            .to_owned(),
        r#"{"type":"response.output_item.done","sequence_number":11,"output_index":1,
            "item":{"type":"message","id":"msg_1","role":"assistant","status":"completed",
            "phase":"commentary","content":[{"type":"output_text","text":"Reading it now.",
            "annotations":[]}]}}"#
            .to_owned(),
        r#"{"type":"response.output_item.added","sequence_number":12,"output_index":2,
            "item":{"type":"function_call","id":"fc_1","call_id":"call_x","name":"read_file",
            "arguments":"","status":"in_progress"}}"#
            .to_owned(),
        r#"{"type":"response.function_call_arguments.delta","sequence_number":13,"item_id":"fc_1",
            "output_index":2,"delta":"{\"path\":"}"#
            .to_owned(),
        r#"{"type":"response.function_call_arguments.done","sequence_number":14,"item_id":"fc_1",
            "output_index":2,"arguments":"{\"path\":\"src/lib.rs\"}"}"#
            .to_owned(),
        r#"{"type":"response.output_item.done","sequence_number":15,"output_index":2,
            "item":{"type":"function_call","id":"fc_1","call_id":"call_x","name":"read_file",
            "arguments":"{\"path\":\"src/lib.rs\"}","status":"completed"}}"#
            .to_owned(),
        format!(
            r#"{{"type":"response.completed","sequence_number":16,"response":{{"id":"resp_9",
                "status":"completed","usage":{}}}}}"#,
            usage_frame()
        ),
    ];

    let mut settling = Settling::new();
    for frame in &frames {
        settling.consume_payload(frame).unwrap();
    }
    let settled = settling.settle().unwrap();

    assert_eq!(settled.outcome, Outcome::Completed);
    assert_eq!(settled.text, "Reading it now.");
    assert_eq!(settled.reasoning_summary, "Reading the file");
    assert_eq!(settled.items.len(), 3, "reasoning, message, function call");

    let calls: Vec<_> = settled.function_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id, "call_x");
    assert_eq!(calls[0].arguments.decode().unwrap(), json!({"path": "src/lib.rs"}));

    let reasoning: Vec<_> = settled.reasoning_items().collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0].replayable().unwrap().encrypted_content, "gAAAAA");

    let usage = settled.usage.unwrap();
    assert_eq!(usage.input_tokens_details.cached_tokens, 1_000);
    assert_eq!(usage.uncached_input_tokens(), 100);
}

#[test]
fn a_failed_stream_settles_as_failed_with_its_error() {
    let mut settling = Settling::new();
    settling
        .consume_payload(r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"parti"}"#)
        .unwrap();
    settling
        .consume_payload(
            r#"{"type":"response.failed","response":{"id":"resp_e","status":"failed",
                "error":{"code":"server_error","message":"upstream fell over"}}}"#,
        )
        .unwrap();

    let settled = settling.settle().unwrap();
    assert!(!settled.is_completed());
    let error = settled.error().unwrap();
    assert_eq!(error.code.as_deref(), Some("server_error"));
    assert_eq!(settled.text, "parti", "what arrived is kept; the outcome says it failed");
}

/// A `failed` event with no error object still settles, with a message that
/// says the object was missing rather than pretending there was no error.
#[test]
fn a_failure_without_an_error_object_still_names_itself_a_failure() {
    let mut settling = Settling::new();
    settling.consume_payload(r#"{"type":"response.failed","response":{"id":"r"}}"#).unwrap();
    let settled = settling.settle().unwrap();
    assert!(!settled.is_completed());
    assert_eq!(settled.error().unwrap().code, None);
    assert!(settled.error().unwrap().message.contains("without an error object"));
}

#[test]
fn an_incomplete_stream_settles_with_its_reason_and_its_partial_text() {
    let mut settling = Settling::new();
    settling
        .consume_payload(
            r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,
                "delta":"as far as I got"}"#,
        )
        .unwrap();
    settling
        .consume_payload(&format!(
            r#"{{"type":"response.incomplete","response":{{"id":"resp_i","status":"incomplete",
                "incomplete_details":{{"reason":"max_output_tokens"}},"usage":{}}}}}"#,
            usage_frame()
        ))
        .unwrap();

    let settled = settling.settle().unwrap();
    assert_eq!(settled.outcome, Outcome::Incomplete { reason: Some(IncompleteReason::MaxOutputTokens) });
    assert_eq!(settled.text, "as far as I got");
    assert!(settled.usage.is_some(), "an incomplete response still cost money");
    assert_eq!(settled.error(), None, "incomplete is not an error");
}

/// An event type this crate has never seen passes through the middle of a
/// stream without disturbing it. This is the server-side-release case.
#[test]
fn an_unknown_event_interleaved_changes_nothing() {
    let mut settling = Settling::new();
    settling
        .consume_payload(r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"be"}"#)
        .unwrap();
    let unknown =
        settling.consume_payload(r#"{"type":"response.telepathy.delta","output_index":0,"delta":"???"}"#).unwrap();
    assert_eq!(unknown, StreamEvent::Unmodeled { kind: "response.telepathy.delta".to_owned() });
    settling
        .consume_payload(r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"fore"}"#)
        .unwrap();
    settling.consume_payload(r#"{"type":"response.completed","response":{"id":"r"}}"#).unwrap();

    let settled = settling.settle().unwrap();
    assert!(settled.is_completed());
    assert_eq!(settled.text, "before", "the unknown event contributed nothing and broke nothing");
    assert_eq!(settled.events, 4, "it was still counted");
}

/// A truncated stream that had already produced a whole function call still
/// does not settle. Partial structure is no substitute for a terminal event.
#[test]
fn a_truncated_stream_with_finished_items_still_does_not_settle() {
    let mut settling = Settling::new();
    settling
        .consume_payload(
            r#"{"type":"response.output_item.done","output_index":0,
                "item":{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}}"#,
        )
        .unwrap();
    assert!(matches!(settling.settle(), Err(SettleError::Truncated { events: 1, text_len: 0 })));
}

/// The bare `error` event ends the stream as its own outcome, not as
/// `Failed`, because it carries no response object.
#[test]
fn a_bare_error_event_settles_as_errored() {
    let mut settling = Settling::new();
    settling
        .consume_payload(r#"{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":null}"#)
        .unwrap();
    let settled = settling.settle().unwrap();
    assert!(matches!(settled.outcome, Outcome::Errored { .. }));
    assert_eq!(settled.error().unwrap().code.as_deref(), Some("rate_limit_exceeded"));
    assert_eq!(settled.usage, None);
}

/// The first terminal event wins, so a duplicate cannot rewrite history.
#[test]
fn a_second_terminal_event_is_ignored() {
    let mut settling = Settling::new();
    settling.consume_payload(r#"{"type":"response.completed","response":{"id":"first"}}"#).unwrap();
    settling.consume_payload(r#"{"type":"response.failed","response":{"error":{"message":"too late"}}}"#).unwrap();
    let settled = settling.settle().unwrap();
    assert!(settled.is_completed());
    assert_eq!(settled.id.as_deref(), Some("first"));
}

/// The `done` form of an item replaces the announcement at the same index,
/// so arguments are the finished ones and the item is not duplicated.
#[test]
fn a_finished_item_replaces_its_announcement() {
    let mut settling = Settling::new();
    settling
        .consume_payload(
            r#"{"type":"response.output_item.added","output_index":0,
                "item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}"#,
        )
        .unwrap();
    settling
        .consume_payload(
            r#"{"type":"response.output_item.done","output_index":0,
                "item":{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"k\":1}"}}"#,
        )
        .unwrap();
    settling.consume_payload(r#"{"type":"response.completed","response":{}}"#).unwrap();

    let settled = settling.settle().unwrap();
    assert_eq!(settled.items.len(), 1, "one item, not two");
    assert_eq!(settled.function_calls().next().unwrap().arguments.decode().unwrap(), json!({"k": 1}));
}

/// Items settle in `output_index` order however the frames arrive.
#[test]
fn items_settle_in_output_order() {
    let mut settling = Settling::new();
    for (index, name) in [(2u32, "third"), (0, "first"), (1, "second")] {
        settling
            .consume_payload(&format!(
                r#"{{"type":"response.output_item.done","output_index":{index},
                    "item":{{"type":"function_call","call_id":"c{index}","name":"{name}",
                    "arguments":"{{}}"}}}}"#
            ))
            .unwrap();
    }
    settling.consume_payload(r#"{"type":"response.completed","response":{}}"#).unwrap();
    let settled = settling.settle().unwrap();
    let names: Vec<&str> = settled.function_calls().map(|call| call.name.as_str()).collect();
    assert_eq!(names, ["first", "second", "third"]);
}

/// Text from several items and parts joins in document order, not arrival
/// order, and a repeated part does not double.
#[test]
fn text_joins_in_document_order() {
    let mut settling = Settling::new();
    settling.consume(text_delta(1, 0, "second"));
    settling.consume(text_delta(0, 1, "part-two "));
    settling.consume(text_delta(0, 0, "part-one "));
    settling.consume(StreamEvent::Completed(ResponseSnapshot::default()));
    assert_eq!(settling.settle().unwrap().text, "part-one part-two second");
}

/// A broken frame surfaces as a frame error through the payload path, and
/// does not corrupt what has accumulated.
#[test]
fn a_broken_frame_does_not_disturb_the_accumulator() {
    let mut settling = Settling::new();
    settling
        .consume_payload(r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"kept"}"#)
        .unwrap();
    assert!(matches!(settling.consume_payload("{ not json"), Err(SettleError::Frame(_))));
    assert_eq!(settling.text_so_far(), "kept");
    assert_eq!(settling.event_count(), 1, "a frame that never decoded was never an event");
}

/// A terminal snapshot that repeats the whole `output` array is the
/// server's final word and supersedes the streamed items.
#[test]
fn a_terminal_output_array_supersedes_streamed_items() {
    let mut settling = Settling::new();
    settling
        .consume_payload(
            r#"{"type":"response.output_item.added","output_index":0,
                "item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}"#,
        )
        .unwrap();
    settling
        .consume_payload(
            r#"{"type":"response.completed","response":{"id":"r","output":[
                {"type":"function_call","call_id":"c1","name":"f","arguments":"{\"final\":true}"}]}}"#,
        )
        .unwrap();
    let settled = settling.settle().unwrap();
    assert_eq!(settled.items.len(), 1);
    assert_eq!(settled.function_calls().next().unwrap().arguments.decode().unwrap(), json!({"final": true}));
}

/// A message item is announced empty and filled by deltas, so the deltas at
/// its index are its text. Without that, prose said beside a function call
/// reads as an empty message and a caller iterating items loses it.
#[test]
fn a_streamed_message_item_carries_the_text_of_its_own_deltas() {
    let mut settling = Settling::new();
    for frame in [
        r#"{"type":"response.output_item.added","output_index":0,
            "item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}"#,
        r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"I will read it."}"#,
        r#"{"type":"response.output_item.added","output_index":1,
            "item":{"type":"function_call","call_id":"c1","name":"read","arguments":""}}"#,
        r#"{"type":"response.output_item.done","output_index":1,
            "item":{"type":"function_call","call_id":"c1","name":"read","arguments":"{\"path\":\"a\"}"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1"}}"#,
    ] {
        settling.consume_payload(frame).unwrap();
    }
    let settled = settling.settle().unwrap();

    assert_eq!(settled.text, "I will read it.");
    assert_eq!(settled.items.len(), 2);
    let OutputItem::Message { text, .. } = &settled.items[0] else {
        panic!("the first item is the message");
    };
    assert_eq!(text, "I will read it.", "the item carries the text its own deltas built");
    assert_eq!(settled.function_calls().count(), 1);
}

/// Text at an index nothing announced still describes a message there. A
/// provider that streams deltas and never repeats its `output` array is the
/// ordinary case, and a caller reading `items` must still find the answer.
#[test]
fn text_at_an_unannounced_index_is_still_a_message_item() {
    let mut settling = Settling::new();
    for frame in [
        r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"said"}"#,
        r#"{"type":"response.output_item.done","output_index":1,
            "item":{"type":"function_call","call_id":"c1","name":"read","arguments":"{}"}}"#,
        r#"{"type":"response.completed","response":{"id":"resp_1"}}"#,
    ] {
        settling.consume_payload(frame).unwrap();
    }
    let settled = settling.settle().unwrap();

    assert_eq!(settled.items.len(), 2, "{:?}", settled.items);
    let OutputItem::Message { text, .. } = &settled.items[0] else {
        panic!("the text became the item at its own index");
    };
    assert_eq!(text, "said");
    assert_eq!(settled.function_calls().count(), 1);
}

/// A `done` frame that agrees with the deltas adds nothing, and one that
/// disagrees is recorded rather than silently resolved.
///
/// This is the free check OpenAI's redundancy buys: every run of text arrives
/// twice, so a dropped, duplicated, or misordered delta is detectable without
/// any extra request. Overwriting quietly would throw that away.
#[test]
fn a_done_frame_checks_the_deltas_rather_than_replacing_them() {
    let done = |text: &str| StreamEvent::TextDone {
        stream: TextStream::Output,
        output_index: 0,
        part_index: 0,
        text: text.to_owned(),
    };

    let mut agreeing = Settling::new();
    agreeing.consume(text_delta(0, 0, "Hello, "));
    agreeing.consume(text_delta(0, 0, "world"));
    agreeing.consume(done("Hello, world"));
    agreeing.consume(StreamEvent::Completed(ResponseSnapshot::default()));
    let settled = agreeing.settle().unwrap();
    assert_eq!(settled.text, "Hello, world");
    assert_eq!(settled.part_disagreements, Vec::new(), "agreement is silent");

    // A delta the transport lost. The server's own statement wins, because it
    // is the better answer — and the loss is reported, because it is a bug.
    let mut lossy = Settling::new();
    lossy.consume(text_delta(0, 0, "Hello, "));
    lossy.consume(done("Hello, world"));
    lossy.consume(StreamEvent::Completed(ResponseSnapshot::default()));
    let settled = lossy.settle().unwrap();
    assert_eq!(settled.text, "Hello, world", "the reported whole wins");
    assert_eq!(
        settled.part_disagreements,
        vec![PartDisagreement {
            key: PartKey { output_index: 0, stream: TextStream::Output, part_index: 0 },
            accumulated: "Hello, ".to_owned(),
            reported: "Hello, world".to_owned(),
        }]
    );
}

/// A refusal never becomes answer text, at any stage of settling.
///
/// The whole reason the streams are separate values. A stream that refuses
/// and says nothing else settles with empty `text`, so a caller showing
/// `text` shows nothing rather than showing the refusal as an answer.
#[test]
fn a_refusal_never_becomes_answer_text() {
    let mut settling = Settling::new();
    settling.consume(StreamEvent::TextDelta {
        stream: TextStream::Refusal,
        output_index: 0,
        part_index: 0,
        delta: "I cannot help with that.".to_owned(),
    });
    settling.consume(StreamEvent::Completed(ResponseSnapshot::default()));
    let settled = settling.settle().unwrap();
    assert_eq!(settled.text, "", "a refusal is not an answer");
    assert_eq!(settled.refusal, "I cannot help with that.");

    // And the item at that index carries both fields, so a caller iterating
    // items sees the refusal where the model put it.
    let [OutputItem::Message { text, refusal, .. }] = &settled.items[..] else {
        panic!("expected one message, got {:?}", settled.items);
    };
    assert_eq!(text, "");
    assert_eq!(refusal, "I cannot help with that.");
}

/// Reasoning, in both of its streams, stays out of the answer.
#[test]
fn neither_reasoning_stream_becomes_answer_text() {
    let mut settling = Settling::new();
    for (stream, delta) in
        [(TextStream::ReasoningSummary, "Weighing the options. "), (TextStream::Reasoning, "The user asks about X.")]
    {
        settling.consume(StreamEvent::TextDelta { stream, output_index: 0, part_index: 0, delta: delta.to_owned() });
    }
    settling.consume(text_delta(1, 0, "The answer."));
    settling.consume(StreamEvent::Completed(ResponseSnapshot::default()));
    let settled = settling.settle().unwrap();
    assert_eq!(settled.text, "The answer.");
    assert_eq!(settled.reasoning_summary, "Weighing the options. ");
    assert_eq!(settled.reasoning, "The user asks about X.");
}

/// A hosted tool that ran is counted, because nothing else in a settled
/// response says a web search happened.
#[test]
fn a_settled_response_says_which_hosted_tools_ran() {
    let mut settling = Settling::new();
    for phase in [HostedToolPhase::InProgress, HostedToolPhase::Completed] {
        settling.consume(StreamEvent::HostedToolLifecycle {
            tool: HostedTool::WebSearch,
            phase,
            output_index: 0,
            item_id: Some("ws_1".to_owned()),
        });
    }
    settling.consume(text_delta(1, 0, "Found it."));
    settling.consume(StreamEvent::Completed(ResponseSnapshot::default()));
    let settled = settling.settle().unwrap();
    assert_eq!(settled.hosted_tool_events.get(&HostedTool::WebSearch), Some(&2));
    assert_eq!(settled.hosted_tool_events.get(&HostedTool::Mcp), None);
    assert_eq!(settled.text, "Found it.");
}

/// A part announced but never filled is a part that exists and is empty,
/// which is what keeps several summary paragraphs separate rather than
/// merging into one.
#[test]
fn an_announced_part_exists_before_any_delta_arrives() {
    let mut settling = Settling::new();
    for part_index in [0, 1] {
        settling.consume(StreamEvent::Part {
            stream: TextStream::ReasoningSummary,
            boundary: PartBoundary::Added,
            output_index: 0,
            part_index,
            text: String::new(),
        });
        settling.consume(StreamEvent::TextDelta {
            stream: TextStream::ReasoningSummary,
            output_index: 0,
            part_index,
            delta: format!("paragraph {part_index}. "),
        });
    }
    settling.consume(StreamEvent::Completed(ResponseSnapshot::default()));
    assert_eq!(settling.settle().unwrap().reasoning_summary, "paragraph 0. paragraph 1. ");
}
