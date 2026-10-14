//! GPT-6.1 Sol facts and request fields from its OpenAI model page and the
//! GPT-6 guide, read on 2026-10-14. It takes Astra's parameter set, so it is a
//! tier of the one type that set has.

use openai::content::InputBlock;
use openai::context::{BreakpointSlot, Context};
use openai::model::{EffortLowToMax, Gpt6LowToMaxTier, Model, ModelId, YearMonth};
use openai::prefix::PrefixSettings;
use openai::request::{Request, RequestError};
use openai::values::{CacheMode, ReasoningContext, ReasoningEffort, ReasoningMode};
use serde_json::json;

#[test]
fn documented_identity_limits_and_prices() {
    let model = Model::from(Model::gpt_6_1_sol());
    let id = model.id();
    assert_eq!(id, ModelId::Gpt6_1Sol);
    assert_eq!(model.api_id(), "gpt-6.1-sol");
    assert_eq!(id.default_effort(), Some(ReasoningEffort::Medium));
    assert_eq!(id.context_window_tokens(), 1_050_000);
    assert_eq!(id.max_output_tokens(), 128_000);
    assert_eq!(id.knowledge_cutoff(), YearMonth { year: 2026, month: 4 });
    assert!(id.supports_explicit_cache_breakpoints());
    assert_eq!(id.min_cacheable_prefix_tokens(), 1_024);
    let price = id.pricing();
    assert_eq!(price.input_nanodollars_per_token, 2_000);
    assert_eq!(price.cached_input_nanodollars_per_token, 100, "5% of input");
    assert_eq!(price.cache_write_nanodollars_per_token, 2_500, "1.25x input");
    assert_eq!(price.output_nanodollars_per_token, 10_000);
}

#[test]
fn defaults_leave_reasoning_effort_unset() {
    let mut context = Context::new(vec![]);
    context.push_user(vec![InputBlock::text("Reply with exactly: alive")]);
    let prefix = PrefixSettings::new(Model::gpt_6_1_sol());
    assert_eq!(prefix.effort(), None);
    let request = Request::new(&context, prefix).unwrap();
    assert_eq!(
        serde_json::to_value(request).unwrap(),
        json!({
            "model": "gpt-6.1-sol",
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
}

/// `low` through `max`, and no `none`: the type has no such variant.
#[test]
fn all_five_efforts_serialize_without_rounding() {
    let context = Context::new(vec![]);
    for (effort, wire) in [
        (EffortLowToMax::Low, "low"),
        (EffortLowToMax::Medium, "medium"),
        (EffortLowToMax::High, "high"),
        (EffortLowToMax::Xhigh, "xhigh"),
        (EffortLowToMax::Max, "max"),
    ] {
        let model = Model::gpt_6_1_sol().with_effort(effort);
        assert_eq!(model.tier, Gpt6LowToMaxTier::Sol6_1);
        let request = Request::new(&context, PrefixSettings::new(model)).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap()["reasoning"]["effort"], wire);
        assert_eq!(model.without_effort().effort, None);
    }
}

#[test]
fn cache_controls_modes_and_output_bounds_are_preserved() {
    let mut context = Context::new(vec![]);
    context.push_user(vec![InputBlock::text("Cache this prefix.")]);
    context.anchor_breakpoint(BreakpointSlot::S0).unwrap();
    let model = Model::gpt_6_1_sol()
        .with_mode(ReasoningMode::Pro)
        .with_reasoning_context(ReasoningContext::Auto)
        .with_explicit_cache_only();
    let request = Request::new(&context, PrefixSettings::new(model)).unwrap().with_max_output_tokens(128_000).unwrap();
    let body = serde_json::to_value(request).unwrap();
    assert_eq!(body["reasoning"]["mode"], "pro");
    assert_eq!(body["prompt_cache_options"], json!({"mode": "explicit", "ttl": "30m"}));
    assert!(body.get("prompt_cache_retention").is_none());
    assert_eq!(model.caching.mode, CacheMode::Explicit);
    assert!(Request::new(&context, PrefixSettings::new(model)).unwrap().with_max_output_tokens(128_001).is_err());
}

/// Sharing Astra's parameter set does not extend Astra's `configuration_update`
/// support, which the crate has verified on Astra alone.
#[test]
fn configuration_updates_stay_with_astra() {
    let mut context = Context::new(vec![]);
    context.push_user(vec![InputBlock::text("first")]);
    context.push_configuration_update(EffortLowToMax::High);
    context.push_user(vec![InputBlock::text("second")]);
    assert!(Request::new(&context, PrefixSettings::new(Model::gpt_6_astra())).is_ok());
    assert!(matches!(
        Request::new(&context, PrefixSettings::new(Model::gpt_6_1_sol())),
        Err(RequestError::ConfigurationUpdateRequiresAstra { model: "gpt-6.1-sol" })
    ));
}
