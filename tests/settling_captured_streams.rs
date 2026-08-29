//! Whole captured SSE bodies, read the way a real consumer reads them.
//!
//! The unit tests in `stream` and `settle` check one frame and one fold at a
//! time. These drive complete response bodies — `event:` lines, `data:` lines,
//! blank separators and all — through nothing but the public API, so they also
//! prove the crate is usable without reaching inside it.

use openai::content::{FunctionCall, InputItem};
use openai::settle::{Outcome, SettleError, Settled, Settling};
use openai::stream::{OutputItem, data_payload};
use openai::values::IncompleteReason;
use serde_json::json;

/// Reads an SSE body exactly as a transport would: line by line, acting only on
/// `data:` lines and ignoring the framing around them.
fn read_body(body: &str) -> Result<Settled, SettleError> {
    let mut settling = Settling::new();
    for line in body.lines() {
        if let Some(payload) = data_payload(line) {
            settling.consume_payload(payload)?;
        }
    }
    settling.settle()
}

/// A GPT-5.6 turn that answers in text and then calls one function. Captured
/// frame shapes, including the `event:` lines a real body carries.
const HAPPY_PATH: &str = r#"event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_a1","object":"response","status":"in_progress","model":"gpt-5.6-sol"}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_a1","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_a","type":"reasoning","summary":[]}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","sequence_number":3,"item_id":"rs_a","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_a","output_index":0,"summary_index":0,"delta":"The user wants "}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","sequence_number":5,"item_id":"rs_a","output_index":0,"summary_index":0,"delta":"the file read."}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","sequence_number":6,"item_id":"rs_a","output_index":0,"summary_index":0,"text":"The user wants the file read."}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"id":"rs_a","type":"reasoning","summary":[{"type":"summary_text","text":"The user wants the file read."}],"encrypted_content":"gAAAAABo3Zk","status":"completed"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":8,"output_index":1,"item":{"id":"msg_a","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":9,"item_id":"msg_a","output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":10,"item_id":"msg_a","output_index":1,"content_index":0,"delta":"Let me ","logprobs":[]}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":11,"item_id":"msg_a","output_index":1,"content_index":0,"delta":"read that ","logprobs":[]}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":12,"item_id":"msg_a","output_index":1,"content_index":0,"delta":"file.","logprobs":[]}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":13,"item_id":"msg_a","output_index":1,"content_index":0,"text":"Let me read that file."}

event: response.content_part.done
data: {"type":"response.content_part.done","sequence_number":14,"item_id":"msg_a","output_index":1,"content_index":0,"part":{"type":"output_text","text":"Let me read that file.","annotations":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":15,"output_index":1,"item":{"id":"msg_a","type":"message","status":"completed","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"Let me read that file.","annotations":[]}]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":16,"output_index":2,"item":{"id":"fc_a","type":"function_call","status":"in_progress","arguments":"","call_id":"call_QT9","name":"read_file"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":17,"item_id":"fc_a","output_index":2,"delta":"{\"path\""}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":18,"item_id":"fc_a","output_index":2,"delta":":\"src/lib.rs\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":19,"item_id":"fc_a","output_index":2,"arguments":"{\"path\":\"src/lib.rs\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":20,"output_index":2,"item":{"id":"fc_a","type":"function_call","status":"completed","arguments":"{\"path\":\"src/lib.rs\"}","call_id":"call_QT9","name":"read_file"}}

event: response.completed
data: {"type":"response.completed","sequence_number":21,"response":{"id":"resp_a1","object":"response","status":"completed","model":"gpt-5.6-sol","usage":{"input_tokens":3021,"input_tokens_details":{"cached_tokens":2969,"cache_write_tokens":0},"output_tokens":142,"output_tokens_details":{"reasoning_tokens":96},"total_tokens":3163}}}
"#;

#[test]
fn a_happy_path_stream_settles_into_text_reasoning_and_one_call() {
    let settled = read_body(HAPPY_PATH).expect("a stream ending in response.completed settles");

    assert_eq!(settled.outcome, Outcome::Completed);
    assert!(settled.is_completed());
    assert_eq!(settled.id.as_deref(), Some("resp_a1"));
    assert_eq!(settled.text, "Let me read that file.");
    assert_eq!(settled.reasoning_summary, "The user wants the file read.");
    assert_eq!(settled.error(), None);

    // Three items, in output order: reasoning, message, function call.
    assert_eq!(settled.items.len(), 3);
    assert!(matches!(settled.items[0], OutputItem::Reasoning(_)));
    assert!(matches!(&settled.items[1], OutputItem::Message { text, .. } if text == "Let me read that file."));
    assert!(matches!(settled.items[2], OutputItem::FunctionCall(_)));

    // The one call, with its arguments decoded from the JSON string.
    let calls: Vec<_> = settled.function_calls().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id, "call_QT9");
    assert_eq!(calls[0].name, "read_file");
    assert_eq!(calls[0].arguments.decode().unwrap(), json!({"path": "src/lib.rs"}));

    // Usage carries the cache accounting, and the arithmetic in `usage` works
    // on it unchanged: 2,969 of 3,021 input tokens were served from cache.
    let usage = settled.usage.expect("response.completed reported usage");
    assert_eq!(usage.input_tokens, 3_021);
    assert_eq!(usage.input_tokens_details.cached_tokens, 2_969);
    assert_eq!(usage.input_tokens_details.cache_write_tokens, 0);
    assert_eq!(usage.uncached_input_tokens(), 52);
    assert_eq!(usage.output_tokens_details.reasoning_tokens, 96);
    assert!(usage.cache_hit_rate().unwrap() > 0.98);
}

/// What the settled stream is *for*: the next request's input items. Reasoning
/// replays via `encrypted_content`, and the call replays with the exact
/// argument bytes the model emitted, so the prefix stays byte-identical.
#[test]
fn a_settled_stream_feeds_the_next_turn_without_reserializing_arguments() {
    let settled = read_body(HAPPY_PATH).unwrap();

    let reasoning = settled.reasoning_items().next().unwrap().replayable().unwrap();
    assert_eq!(reasoning.id, "rs_a");
    assert_eq!(reasoning.encrypted_content, "gAAAAABo3Zk");
    let _: InputItem = InputItem::Reasoning(reasoning);

    let call = settled.function_calls().next().unwrap();
    let replayed = FunctionCall {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.as_str().to_owned(),
    };
    assert_eq!(replayed.arguments, r#"{"path":"src/lib.rs"}"#, "the model's own bytes, not a re-serialization");
    let _: InputItem = InputItem::FunctionCall(replayed);
}

/// Reading the same body incrementally shows the in-progress text growing while
/// the stream stays unsettled, which is the live-display case.
#[test]
fn text_is_readable_while_the_stream_is_still_unsettled() {
    let mut settling = Settling::new();
    let mut seen_partial = false;
    for line in HAPPY_PATH.lines() {
        if let Some(payload) = data_payload(line) {
            settling.consume_payload(payload).unwrap();
            if settling.text_so_far() == "Let me read that " {
                assert!(!settling.is_terminated(), "still streaming");
                seen_partial = true;
            }
        }
    }
    assert!(seen_partial, "the answer was observable mid-flight");
    assert!(settling.is_terminated());
    assert_eq!(settling.settle().unwrap().text, "Let me read that file.");
}

#[test]
fn a_failed_stream_settles_as_a_failure_carrying_its_error() {
    let body = r#"event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_f","status":"in_progress"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_f","output_index":0,"content_index":0,"delta":"Starting"}

event: response.failed
data: {"type":"response.failed","sequence_number":2,"response":{"id":"resp_f","object":"response","status":"failed","error":{"code":"server_error","message":"The model failed to generate a response."},"usage":null}}
"#;
    let settled = read_body(body).expect("a stream ending in response.failed settles");

    assert!(!settled.is_completed());
    assert!(matches!(settled.outcome, Outcome::Failed { .. }));
    let error = settled.error().expect("a failure carries an error");
    assert_eq!(error.code.as_deref(), Some("server_error"));
    assert_eq!(error.message, "The model failed to generate a response.");
    assert_eq!(settled.text, "Starting", "what arrived is kept; the outcome says it failed");
    assert_eq!(settled.usage, None, "a null usage is absent, not zero");
}

#[test]
fn an_incomplete_stream_settles_with_its_reason_and_what_it_produced() {
    let body = r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_i","output_index":0,"content_index":0,"delta":"Once upon a "}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":2,"item_id":"msg_i","output_index":0,"content_index":0,"delta":"time there"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"msg_i","type":"message","status":"incomplete","role":"assistant","content":[{"type":"output_text","text":"Once upon a time there","annotations":[]}]}}

event: response.incomplete
data: {"type":"response.incomplete","sequence_number":4,"response":{"id":"resp_i","object":"response","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"max_output_tokens":16,"usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":16,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":58}}}
"#;
    let settled = read_body(body).expect("a stream ending in response.incomplete settles");

    assert_eq!(settled.outcome, Outcome::Incomplete { reason: Some(IncompleteReason::MaxOutputTokens) });
    assert!(!settled.is_completed());
    assert_eq!(settled.error(), None, "stopping short is not an error");
    assert_eq!(settled.text, "Once upon a time there", "the truncated answer is still delivered");
    assert_eq!(settled.usage.unwrap().output_tokens, 16, "it still cost output tokens");
}

/// The server-side-release case: an event type this crate has never seen, in
/// the middle of an otherwise ordinary stream. It must change nothing.
#[test]
fn an_unknown_event_type_interleaved_does_not_break_the_stream() {
    let body = r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_u","output_index":0,"content_index":0,"delta":"before "}

event: response.holographic_projection.added
data: {"type":"response.holographic_projection.added","sequence_number":2,"output_index":0,"projection":{"dimensions":11,"fidelity":"ultra"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":3,"item_id":"msg_u","output_index":0,"content_index":0,"delta":"and after"}

event: response.completed
data: {"type":"response.completed","sequence_number":4,"response":{"id":"resp_u","status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":4,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":14}}}
"#;
    let settled = read_body(body).expect("an unknown event is not an error");

    assert!(settled.is_completed());
    assert_eq!(settled.text, "before and after", "the unknown event contributed nothing");
    assert_eq!(settled.events, 4, "and was still counted");
}

/// The whole point of separating the two states: a stream cut off mid-answer
/// yields no response at all, however complete its text happens to look.
#[test]
fn a_truncated_stream_never_settles() {
    let truncated = HAPPY_PATH
        .split_inclusive('\n')
        .take_while(|line| !line.contains("\"response.completed\""))
        .collect::<String>();

    // Everything but the terminal event arrived: full text, a finished call.
    let mut settling = Settling::new();
    for line in truncated.lines() {
        if let Some(payload) = data_payload(line) {
            settling.consume_payload(payload).unwrap();
        }
    }
    assert_eq!(settling.text_so_far(), "Let me read that file.", "the text looks complete");
    assert!(!settling.is_terminated(), "but nothing said the response was done");

    let error = settling.settle().expect_err("a stream without a terminal event must not settle");
    let SettleError::Truncated { events, text_len } = error else {
        panic!("expected truncation, got {error}");
    };
    assert_eq!(events, 21, "every frame before response.completed was consumed");
    assert_eq!(text_len, 22, "and the text it had is reported, not returned");
}
