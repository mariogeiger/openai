//! Function-tool bodies serialized by this crate and captured on 2026-09-09 UTC.
//!
//! Only `model` was rewritten to `gateway/gpt-6-astra` before POSTing to
//! `/v1/responses`. Request JSON records the
//! exact prompts. Captures contain no headers or credentials.

use openai::context::Context;
use openai::items::OutputItem;
use openai::model::{EffortLowToMax, Model};
use openai::prefix::PrefixSettings;
use openai::request::Request;
use openai::settle::{SettleError, Settling};
use openai::settled::Settled;
use openai::stream::data_payload;
use openai::tools::FunctionTool;
use serde_json::{Value, json};

fn settle(body: &str) -> Result<Settled, SettleError> {
    let mut settling = Settling::new();
    for line in body.lines() {
        if let Some(payload) = data_payload(line) {
            settling.consume_payload(payload)?;
        }
    }
    settling.settle()
}

fn request(body: &str) -> Value {
    serde_json::from_str(body).expect("a captured crate request is JSON")
}

#[test]
fn synchronous_function_roundtrip_keeps_call_shape() {
    let start = request(include_str!("data/captured_astra_sync-start.request.json"));
    assert_eq!(start["tools"][0]["async"], false);
    assert_eq!(start["tool_choice"], json!({"type": "function", "name": "get_number"}));
    assert_eq!(
        start["input"][0]["content"][0]["text"],
        "Call get_number with label sync. Do not answer before the tool result."
    );

    let called = settle(include_str!("data/captured_astra_sync-start.sse")).unwrap();
    let call = called.function_calls().next().expect("Astra called the forced tool");
    assert_eq!(call.name, "get_number");
    assert_eq!(call.arguments.decode().unwrap(), json!({"label": "sync"}));
    assert_eq!(call.asynchronous, None);

    let finish = request(include_str!("data/captured_astra_sync-finish.request.json"));
    assert_eq!(finish["input"][1]["call_id"], call.call_id);
    assert_eq!(finish["input"][2]["output"], r#"{"number":42}"#);
    assert_eq!(settle(include_str!("data/captured_astra_sync-finish.sse")).unwrap().text, "42");
}

#[test]
fn async_function_call_is_decoded_and_replayed() {
    let start = request(include_str!("data/captured_astra_async-start.request.json"));
    assert_eq!(start["tools"][0]["async"], true);
    assert_eq!(
        start["input"][0]["content"][0]["text"],
        "Start get_number with label async, then say ASYNC_STARTED without waiting for its result."
    );

    let called = settle(include_str!("data/captured_astra_async-start.sse")).unwrap();
    let call = called.function_calls().next().expect("Astra started the async tool");
    assert_eq!(call.arguments.decode().unwrap(), json!({"label": "async"}));
    assert_eq!(call.asynchronous, Some(true));
    assert!(matches!(called.items.first(), Some(OutputItem::FunctionCall(_))));
    assert!(matches!(called.items.get(1), Some(OutputItem::Message { text, .. }) if text == "ASYNC_STARTED"));

    let mut replay = Context::new(vec![FunctionTool::new("get_number", json!({"type": "object"})).with_async()]);
    replay.push_called_function(call);
    let replayed = serde_json::to_value(
        Request::new(&replay, PrefixSettings::new(Model::gpt_6_astra().with_effort(EffortLowToMax::Low))).unwrap(),
    )
    .unwrap();
    assert_eq!(replayed["input"][0]["async"], true);

    let finish = request(include_str!("data/captured_astra_async-finish.request.json"));
    assert_eq!(finish["input"][1]["async"], true);
    assert_eq!(finish["input"][2]["output"], r#"{"number":42}"#);
    assert_eq!(settle(include_str!("data/captured_astra_async-finish.sse")).unwrap().text, "ASYNC_STARTED");
}
