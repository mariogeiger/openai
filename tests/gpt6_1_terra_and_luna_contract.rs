//! GPT-6.1 Terra and GPT-6 Luna: facts from their OpenAI model pages, read on
//! 2026-11-18, and what the same day's live runs showed. Both take GPT-6 Sol's
//! parameter set, so each is a tier of the one type that set has.
//!
//! The captured streams were serialized by this crate, with only `model`
//! rewritten by the external runner to `gateway/<model>`, and POSTed to
//! `/v1/responses`. The adjacent request JSON is the crate's own output. They
//! contain no request headers or credentials.

use openai::content::InputBlock;
use openai::context::{BreakpointSlot, Context};
use openai::model::{CacheOptions, EffortNoneToMax, GptNoneToMaxTier, Model, ModelId, YearMonth};
use openai::prefix::PrefixSettings;
use openai::request::Request;
use openai::settle::{SettleError, Settling};
use openai::settled::Settled;
use openai::stream::data_payload;
use openai::values::{CacheMode, CacheTtl, ReasoningEffort};
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

#[test]
fn documented_identity_limits_and_prices() {
    // (model, wire id, knowledge cutoff, input, cached read, cache write, output)
    // in nanodollars per token, from each page's text-token table.
    for (id, wire, cutoff, price) in [
        (ModelId::Gpt6_1Terra, "gpt-6.1-terra", YearMonth { year: 2026, month: 6 }, [1_000, 50, 1_250, 5_000]),
        (ModelId::Gpt6Luna, "gpt-6-luna", YearMonth { year: 2026, month: 5 }, [100, 10, 125, 500]),
    ] {
        assert_eq!(id.api_id(), wire);
        assert_eq!(id.default_effort(), Some(ReasoningEffort::Medium));
        assert_eq!(id.context_window_tokens(), 1_050_000);
        assert_eq!(id.max_input_tokens(), 922_000);
        assert_eq!(id.max_output_tokens(), 128_000);
        assert_eq!(id.knowledge_cutoff(), cutoff);
        assert!(id.supports_explicit_cache_breakpoints());
        assert_eq!(id.min_cacheable_prefix_tokens(), 1_024);
        let pricing = id.pricing();
        assert_eq!(
            [
                pricing.input_nanodollars_per_token,
                pricing.cached_input_nanodollars_per_token,
                pricing.cache_write_nanodollars_per_token,
                pricing.output_nanodollars_per_token
            ],
            price
        );
        assert_eq!(pricing.cache_write_nanodollars_per_token * 4, pricing.input_nanodollars_per_token * 5, "1.25x");
    }
    assert_eq!(ModelId::Gpt6_1Terra.pricing().cached_input_nanodollars_per_token * 20, 1_000, "5% of input");
    assert_eq!(ModelId::Gpt6Luna.pricing().cached_input_nanodollars_per_token * 10, 100, "10% of input");
}

#[test]
fn the_constructors_select_their_tier() {
    assert_eq!(Model::gpt_6_1_terra().tier, GptNoneToMaxTier::Terra6_1);
    assert_eq!(Model::gpt_6_luna().tier, GptNoneToMaxTier::Luna6);
    assert_eq!(Model::from(Model::gpt_6_1_terra()).id(), ModelId::Gpt6_1Terra);
    assert_eq!(Model::from(Model::gpt_6_luna()).api_id(), "gpt-6-luna");
}

/// `none` through `max`, as both pages say, with nothing sent until chosen.
#[test]
fn defaults_leave_effort_unset_and_all_six_efforts_serialize() {
    let mut context = Context::new(vec![]);
    context.push_user(vec![InputBlock::text("Reply with exactly: alive")]);
    for (model, wire) in [(Model::gpt_6_1_terra(), "gpt-6.1-terra"), (Model::gpt_6_luna(), "gpt-6-luna")] {
        let prefix = PrefixSettings::new(model);
        assert_eq!(prefix.effort(), None);
        assert_eq!(
            serde_json::to_value(Request::new(&context, prefix).unwrap()).unwrap(),
            json!({
                "model": wire,
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Reply with exactly: alive"}]}],
                "parallel_tool_calls": true,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "reasoning": {"context": "auto"},
                "tool_choice": "auto",
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                "stream": false,
                "store": true,
                "background": false
            })
        );
        for (effort, wire) in [
            (EffortNoneToMax::None, "none"),
            (EffortNoneToMax::Low, "low"),
            (EffortNoneToMax::Medium, "medium"),
            (EffortNoneToMax::High, "high"),
            (EffortNoneToMax::Xhigh, "xhigh"),
            (EffortNoneToMax::Max, "max"),
        ] {
            let model = model.with_effort(effort);
            let body = serde_json::to_value(Request::new(&context, PrefixSettings::new(model)).unwrap()).unwrap();
            assert_eq!(body["reasoning"]["effort"], wire);
            assert_eq!(model.without_effort().effort, None);
        }
    }
}

/// Measured on 2026-11-18: an anchored breakpoint under `explicit` mode was
/// written (6,595 tokens) and read back whole by both models.
#[test]
fn explicit_breakpoints_and_the_twelve_hour_ttl_are_preserved() {
    let mut context = Context::new(vec![]);
    context.push_user(vec![InputBlock::text("Cache this prefix.")]);
    context.anchor_breakpoint(BreakpointSlot::S0).unwrap();
    for model in [Model::gpt_6_1_terra(), Model::gpt_6_luna()] {
        let model = model
            .with_explicit_cache_only()
            .with_caching(CacheOptions { mode: CacheMode::Explicit, ttl: CacheTtl::TwelveHours });
        let request =
            Request::new(&context, PrefixSettings::new(model)).unwrap().with_max_output_tokens(128_000).unwrap();
        let body = serde_json::to_value(request).unwrap();
        assert_eq!(body["prompt_cache_options"], json!({"mode": "explicit", "ttl": "12h"}));
        assert!(body.get("prompt_cache_retention").is_none());
        assert!(Request::new(&context, PrefixSettings::new(model)).unwrap().with_max_output_tokens(128_001).is_err());
    }
}

fn request(body: &str) -> Value {
    serde_json::from_str(body).expect("a captured crate request is JSON")
}

#[test]
fn a_captured_terra_stream_at_max_effort_settles() {
    let asked = request(include_str!("data/captured_gpt6_1_terra_max-effort-stream.request.json"));
    assert_eq!(asked["model"], "gpt-6.1-terra");
    assert_eq!(asked["reasoning"]["effort"], "max");
    assert_eq!(asked["prompt_cache_options"], json!({"mode": "implicit", "ttl": "30m"}));

    let settled = settle(include_str!("data/captured_gpt6_1_terra_max-effort-stream.sse")).unwrap();
    assert!(settled.is_completed());
    assert_eq!(settled.text, "blue");
    assert!(settled.part_disagreements.is_empty());
    assert_eq!(settled.reasoning_items().count(), 1, "the reasoning item arrives to be replayed");
    let usage = settled.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens, usage.total_tokens), (14, 29, 43));
    assert_eq!(usage.output_tokens_details.reasoning_tokens, 22);
    assert_eq!(usage.input_tokens_details.cache_write_tokens_details.ttl_12h_tokens, 0);
}

#[test]
fn a_captured_luna_stream_at_high_effort_settles() {
    let asked = request(include_str!("data/captured_gpt6_luna_high-effort-stream.request.json"));
    assert_eq!(asked["model"], "gpt-6-luna");
    assert_eq!(asked["reasoning"]["effort"], "high");

    let settled = settle(include_str!("data/captured_gpt6_luna_high-effort-stream.sse")).unwrap();
    assert!(settled.is_completed());
    assert_eq!(settled.text, "blue");
    assert!(settled.part_disagreements.is_empty());
    let usage = settled.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens, usage.total_tokens), (14, 17, 31));
    assert_eq!(usage.output_tokens_details.reasoning_tokens, 10);
}
