//! The `usage` object, and exact cost arithmetic over it.
//!
//! This is the one response shape the crate deserializes, because it is the one
//! that tells you whether the caching design in the rest of the crate is
//! working. `cached_tokens` and `cache_write_tokens` are the measurement; every
//! other type here exists to make a request that moves them the right way.
//!
//! The three input classes partition `input_tokens`: read from cache, written to
//! cache, or neither. Cost is therefore a dot product of three token counts with
//! three per-token rates, computed in integer nanodollars so no rounding error
//! can creep into a bill.

use serde::Deserialize;

use crate::model::Pricing;

/// The breakdown of input tokens by how the cache treated them.
///
/// The two reported counts do not overlap, and together they do not have to
/// exhaust `input_tokens`: what is left over is ordinary uncached input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct InputTokensDetails {
    /// Tokens served from the cache, billed at the cached rate.
    ///
    /// Zero after any prefix change, which is exactly how a broken cache
    /// announces itself: no error, just a bill.
    #[serde(default)]
    pub cached_tokens: u32,
    /// Tokens written to the cache. On GPT-5.6 these carry a 1.25× surcharge;
    /// earlier generations do not charge for writes at all.
    #[serde(default)]
    pub cache_write_tokens: u32,
}

/// The breakdown of output tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct OutputTokensDetails {
    /// Tokens spent reasoning. Invisible in the answer, billed as output, and
    /// counted against `max_output_tokens`.
    #[serde(default)]
    pub reasoning_tokens: u32,
}

/// What one response cost, in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Usage {
    /// Every input token, cached and uncached alike.
    pub input_tokens: u32,
    /// How the cache treated them.
    pub input_tokens_details: InputTokensDetails,
    /// Every generated token, reasoning included.
    pub output_tokens: u32,
    /// How many of those were reasoning.
    pub output_tokens_details: OutputTokensDetails,
    /// Input plus output.
    pub total_tokens: u32,
}

impl Usage {
    /// Input tokens billed at the ordinary rate: neither read from nor written
    /// to the cache.
    ///
    /// Saturating, not wrapping: if OpenAI's counts ever fail to add up, a zero
    /// is a visibly wrong cost, while a wrapped `u32` is a plausible one.
    pub fn uncached_input_tokens(&self) -> u32 {
        self.input_tokens
            .saturating_sub(self.input_tokens_details.cached_tokens)
            .saturating_sub(self.input_tokens_details.cache_write_tokens)
    }

    /// The share of input served from cache, in `[0.0, 1.0]`.
    ///
    /// The number to watch: the design in this crate is working when it is high
    /// and stays high. `None` for a response with no input tokens, because the
    /// ratio is undefined rather than zero.
    pub fn cache_hit_rate(&self) -> Option<f64> {
        (self.input_tokens > 0)
            .then(|| f64::from(self.input_tokens_details.cached_tokens) / f64::from(self.input_tokens))
    }

    /// Exact cost in nanodollars: one billionth of a dollar.
    ///
    /// Integer arithmetic throughout, because every published rate and every
    /// cache multiplier lands on a whole nanodollar — so this is the true cost,
    /// not a floating-point approximation of it. Convert to dollars only for
    /// display.
    ///
    /// Excludes what [`Pricing`] excludes: the long-context surcharge above
    /// 272K input tokens, batch and flex discounts, and regional uplift.
    pub fn cost_nanodollars(&self, pricing: Pricing) -> u64 {
        u64::from(self.uncached_input_tokens()) * pricing.input_nanodollars_per_token
            + u64::from(self.input_tokens_details.cached_tokens) * pricing.cached_input_nanodollars_per_token
            + u64::from(self.input_tokens_details.cache_write_tokens) * pricing.cache_write_nanodollars_per_token
            + u64::from(self.output_tokens) * pricing.output_nanodollars_per_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;

    /// The shape from OpenAI's own prompt-caching guide: request two of the
    /// worked example, 12,000 tokens read and 3,000 written.
    fn documented_usage() -> Usage {
        serde_json::from_str(
            r#"{
                "input_tokens": 15000,
                "input_tokens_details": {"cached_tokens": 12000, "cache_write_tokens": 3000},
                "output_tokens": 500,
                "output_tokens_details": {"reasoning_tokens": 400},
                "total_tokens": 15500
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn the_documented_usage_shape_parses() {
        let usage = documented_usage();
        assert_eq!(usage.input_tokens, 15_000);
        assert_eq!(usage.input_tokens_details.cached_tokens, 12_000);
        assert_eq!(usage.input_tokens_details.cache_write_tokens, 3_000);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 400);
        assert_eq!(usage.total_tokens, 15_500);
    }

    /// The three input classes partition `input_tokens` exactly.
    #[test]
    fn the_input_classes_partition_the_input() {
        let usage = documented_usage();
        assert_eq!(usage.uncached_input_tokens(), 0);
        assert_eq!(
            usage.uncached_input_tokens()
                + usage.input_tokens_details.cached_tokens
                + usage.input_tokens_details.cache_write_tokens,
            usage.input_tokens
        );
    }

    /// A missing detail field means zero, so an older response body still parses
    /// rather than failing where the number simply was not reported.
    #[test]
    fn absent_detail_fields_read_as_zero() {
        let usage: Usage = serde_json::from_str(
            r#"{
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 10,
                "output_tokens_details": {},
                "total_tokens": 110
            }"#,
        )
        .unwrap();
        assert_eq!(usage.input_tokens_details.cache_write_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
        assert_eq!(usage.uncached_input_tokens(), 100);
    }

    #[test]
    fn the_hit_rate_is_undefined_without_input() {
        let usage = documented_usage();
        assert_eq!(usage.cache_hit_rate(), Some(0.8));

        let empty = Usage {
            input_tokens: 0,
            input_tokens_details: InputTokensDetails { cached_tokens: 0, cache_write_tokens: 0 },
            output_tokens: 0,
            output_tokens_details: OutputTokensDetails { reasoning_tokens: 0 },
            total_tokens: 0,
        };
        assert_eq!(empty.cache_hit_rate(), None);
    }

    /// Exact, checked by hand: 12,000 read at 400 nd + 3,000 written at 5,000 nd
    /// + 500 output at 20,000 nd = 4,800,000 + 15,000,000 + 10,000,000.
    #[test]
    fn cost_is_exact_integer_arithmetic() {
        assert_eq!(documented_usage().cost_nanodollars(ModelId::Gpt5_6Sol.pricing()), 29_800_000);
    }

    /// The guide's claim, reproduced as arithmetic: writing a prefix once and
    /// reading it nine times costs 2.15× the ordinary rate, against 10× with no
    /// caching. Integer nanodollars make this an equality, not an approximation.
    #[test]
    fn one_write_and_nine_reads_cost_2_15x() {
        let pricing = ModelId::Gpt5_6Sol.pricing();
        let prefix = 10_000u32;
        let usage = |cached, written| Usage {
            input_tokens: prefix,
            input_tokens_details: InputTokensDetails { cached_tokens: cached, cache_write_tokens: written },
            output_tokens: 0,
            output_tokens_details: OutputTokensDetails { reasoning_tokens: 0 },
            total_tokens: prefix,
        };

        let write = usage(0, prefix).cost_nanodollars(pricing);
        let read = usage(prefix, 0).cost_nanodollars(pricing);
        let uncached = usage(0, 0).cost_nanodollars(pricing);

        assert_eq!(write * 4, uncached * 5, "a write costs 1.25x");
        assert_eq!(read * 10, uncached, "a read costs 0.1x");
        // 1.25 + 9 x 0.1 = 2.15, against 10 uncached: 43 x uncached = 20 x total.
        assert_eq!((write + 9 * read) * 20, uncached * 43);
    }

    /// GPT-5.5 Pro grants no cached discount, so a cache hit saves nothing there
    /// — the arithmetic must say so rather than quietly applying a tenth.
    #[test]
    fn a_model_without_a_cached_discount_bills_reads_in_full() {
        let pricing = ModelId::Gpt5_5Pro.pricing();
        let cached = Usage {
            input_tokens: 1_000,
            input_tokens_details: InputTokensDetails { cached_tokens: 1_000, cache_write_tokens: 0 },
            output_tokens: 0,
            output_tokens_details: OutputTokensDetails { reasoning_tokens: 0 },
            total_tokens: 1_000,
        };
        let uncached =
            Usage { input_tokens_details: InputTokensDetails { cached_tokens: 0, cache_write_tokens: 0 }, ..cached };
        assert_eq!(cached.cost_nanodollars(pricing), uncached.cost_nanodollars(pricing));
    }

    /// If the reported counts ever fail to add up, the leftover saturates to
    /// zero instead of wrapping into a plausible-looking huge number.
    #[test]
    fn inconsistent_counts_saturate_rather_than_wrap() {
        let usage = Usage {
            input_tokens: 10,
            input_tokens_details: InputTokensDetails { cached_tokens: 50, cache_write_tokens: 50 },
            output_tokens: 0,
            output_tokens_details: OutputTokensDetails { reasoning_tokens: 0 },
            total_tokens: 10,
        };
        assert_eq!(usage.uncached_input_tokens(), 0);
    }
}
