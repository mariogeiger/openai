//! Live GPT-6 Astra evidence captured on 2026-09-09 UTC.
//!
//! Every body was serialized by this crate, with only `model` rewritten by the
//! external runner to `gateway/gpt-6-astra`, and POSTed to
//! `/v1/responses`. The adjacent request JSON is
//! the original crate output, including the exact prompts. Captures contain no
//! request headers or credentials.

use openai::response::Response;
use openai::settle::{SettleError, Settling};
use openai::settled::Settled;
use openai::stream::data_payload;
use serde_json::{Value, json};

fn settle(body: &str) -> Result<Settled, SettleError> {
    let mut settling = Settling::new();
    for line in body.lines() {
        if let Some(payload) = data_payload(line).filter(|payload| *payload != "[DONE]") {
            settling.consume_payload(payload)?;
        }
    }
    settling.settle()
}

fn request(body: &str) -> Value {
    serde_json::from_str(body).expect("a captured crate request is JSON")
}

#[test]
fn minimal_astra_text_stream_settles() {
    let asked = request(include_str!("data/captured_astra_minimal-stream.request.json"));
    assert_eq!(asked["model"], "gpt-6-astra");
    assert_eq!(asked["input"][0]["content"][0]["text"], "Reply with exactly ASTRA_OK.");
    assert_eq!(asked["reasoning"]["effort"], "low");

    let settled = settle(include_str!("data/captured_astra_minimal-stream.sse")).unwrap();
    assert!(settled.is_completed());
    assert_eq!(settled.text, "ASTRA_OK");
    assert_eq!(settled.usage.unwrap().total_tokens, 20);
}

#[test]
fn configuration_update_changes_history_not_request_effort() {
    let asked = request(include_str!("data/captured_astra_configuration-update.request.json"));
    assert_eq!(asked["reasoning"]["effort"], "low");
    assert_eq!(asked["input"][2], json!({"type": "configuration_update", "reasoning": {"effort": "high"}}));
    assert_eq!(asked["input"][3]["content"][0]["text"], "Analyze 2+2, then reply with exactly UPDATED_OK.");
    let settled = settle(include_str!("data/captured_astra_configuration-update.sse")).unwrap();
    assert_eq!(settled.text, "UPDATED_OK");
    assert_eq!(settled.usage.unwrap().output_tokens_details.reasoning_tokens, 22);
}

#[test]
fn configuration_update_preserves_the_cached_prefix() {
    let first_request = request(include_str!("data/captured_astra_cache-start.request.json"));
    let next_request = request(include_str!("data/captured_astra_cache-updated.request.json"));
    assert_eq!(first_request["input"][0], next_request["input"][0]);
    assert_eq!(
        first_request["input"][0]["content"].as_array().unwrap().last().unwrap()["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
    assert_eq!(next_request["reasoning"]["effort"], "low");
    assert_eq!(next_request["input"][3], json!({"type": "configuration_update", "reasoning": {"effort": "high"}}));

    let first = settle(include_str!("data/captured_astra_cache-start.sse")).unwrap();
    let next = settle(include_str!("data/captured_astra_cache-updated.sse")).unwrap();
    let first_usage = first.usage.unwrap();
    let next_usage = next.usage.unwrap();
    assert_eq!(first_usage.input_tokens, 3_137);
    assert_eq!(first_usage.input_tokens_details.cached_tokens, 0);
    assert_eq!(first_usage.input_tokens_details.cache_write_tokens, 3_134);
    assert_eq!(next_usage.input_tokens, 3_156);
    assert_eq!(next_usage.input_tokens_details.cached_tokens, 3_134);
    assert_eq!(next_usage.input_tokens_details.cache_write_tokens, 19);
    assert_eq!(next.text, "CACHE_OK");
}

#[test]
fn buffered_astra_responses_decode_at_low_max_and_pro() {
    let cases = [
        (
            include_str!("data/captured_astra_buffered.request.json"),
            include_str!("data/captured_astra_buffered.response.json"),
            "low",
            None,
            "ASTRA_OK",
        ),
        (
            include_str!("data/captured_astra_max-effort.request.json"),
            include_str!("data/captured_astra_max-effort.response.json"),
            "max",
            None,
            "ASTRA_OK",
        ),
        (
            include_str!("data/captured_astra_pro-mode.request.json"),
            include_str!("data/captured_astra_pro-mode.response.json"),
            "low",
            Some("pro"),
            "ASTRA_OK",
        ),
    ];
    for (asked, answered, effort, mode, text) in cases {
        let asked = request(asked);
        assert_eq!(asked["model"], "gpt-6-astra");
        assert_eq!(asked["stream"], false);
        assert_eq!(asked["reasoning"]["effort"], effort);
        match mode {
            Some(mode) => assert_eq!(asked["reasoning"]["mode"], mode),
            None => assert!(asked["reasoning"].get("mode").is_none()),
        }
        let response = Response::decode(answered).unwrap();
        assert_eq!(response.model.as_deref(), Some("gateway/gpt-6-astra"));
        assert_eq!(response.text(), text);
    }
}
