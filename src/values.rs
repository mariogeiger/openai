//! Wire vocabulary: enums whose variants are exactly the strings the API
//! accepts, plus the lookup tables that translate them.
//!
//! Keeping the vocabulary in one file means a string literal appears once in
//! the crate. A typo is then a compile error somewhere rather than a 400
//! everywhere.

/// Declares an enum whose variants map one-to-one onto API strings.
///
/// The `roundtrip` form additionally generates `from_str`, for the enums that
/// also appear in responses. Both directions are pure `match` on a primitive.
macro_rules! api_enum {
    (@base $(#[$outer:meta])* $name:ident { $($(#[$inner:meta])* $variant:ident => $s:literal),* $(,)? }) => {
        $(#[$outer])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $($(#[$inner])* $variant),* }
        impl $name {
            /// The exact string this variant serializes to.
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $s),* }
            }
        }
        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
            }
        }
    };
    (roundtrip $(#[$outer:meta])* $name:ident { $($(#[$inner:meta])* $variant:ident => $s:literal),* $(,)? }) => {
        api_enum! { @base $(#[$outer])* $name { $($(#[$inner])* $variant => $s),* } }
        impl $name {
            /// The documented inverse of [`Self::as_str`]. `None` for any string
            /// the API does not define, so a newly added value is visible as
            /// unknown rather than silently mapped onto an existing variant.
            #[allow(clippy::should_implement_trait)]
            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $($s => Some($name::$variant),)*
                    _ => None,
                }
            }
        }
    };
    ($(#[$outer:meta])* $name:ident { $($(#[$inner:meta])* $variant:ident => $s:literal),* $(,)? }) => {
        api_enum! { @base $(#[$outer])* $name { $($(#[$inner])* $variant => $s),* } }
    };
}

pub(crate) use api_enum;

api_enum! {
    /// Who an input message speaks as. `Developer` and `User` outrank each
    /// other in that order; `Assistant` replays what the model said before.
    ///
    /// `system` exists on the wire as a legacy alias for `developer` and is
    /// absent here: two spellings of one role would be two different prefixes
    /// for the same meaning, which is exactly the silent cache miss this crate
    /// is built to prevent.
    Role {
        /// Instructions from the application. Outranks `user`.
        Developer => "developer",
        /// Input from the person using the application.
        User => "user",
        /// A previous model turn, replayed as context.
        Assistant => "assistant",
    }
}

api_enum! { roundtrip
    /// Whether an assistant message was intermediate commentary or the final
    /// answer. OpenAI asks that the value be preserved and resent on every
    /// replayed assistant message; dropping it degrades tool-heavy flows.
    ///
    /// Round-trips, because it travels in both directions: it is sent on a
    /// replayed assistant message and read back off a streamed `message`
    /// output item.
    AssistantPhase {
        /// An interim update, such as a preamble before a tool call.
        Commentary => "commentary",
        /// The completed answer for that turn.
        FinalAnswer => "final_answer",
    }
}

api_enum! {
    /// How much of an image the model is asked to look at. Higher detail costs
    /// more input tokens and, because images sit inside the hashed prefix,
    /// changing it invalidates the cache from that image onward.
    ImageDetail {
        /// Let OpenAI choose. On GPT-5.6+ this renders at high quality.
        Auto => "auto",
        /// Cheaper, coarser rendering.
        Low => "low",
        /// Finer rendering, more input tokens.
        High => "high",
        /// The image exactly as supplied, unresized.
        Original => "original",
    }
}

api_enum! {
    /// How verbose the answer should be. Part of the hashed prefix: OpenAI
    /// turns it into instructions the model reads.
    Verbosity {
        /// Terser answers.
        Low => "low",
        /// The API default.
        Medium => "medium",
        /// Fuller answers.
        High => "high",
    }
}

api_enum! {
    /// Which earlier reasoning the model may render into this turn. Part of the
    /// hashed prefix, because it changes what the model is shown.
    ReasoningContext {
        /// Reasoning from the active turn only.
        CurrentTurn => "current_turn",
        /// Reasoning from earlier turns too. The GPT-5.6 family's default.
        AllTurns => "all_turns",
    }
}

api_enum! {
    /// How much compute the model spends before answering, on GPT-5.6 models.
    /// Earlier generations reject `Max`, which is why effort is carried per
    /// model rather than shared — see [`crate::model`].
    ReasoningEffort {
        /// No reasoning tokens at all.
        None => "none",
        /// Modest reasoning.
        Low => "low",
        /// The GPT-5.6 default.
        Medium => "medium",
        /// Deeper reasoning.
        High => "high",
        /// Long runs.
        Xhigh => "xhigh",
        /// The most reasoning available. GPT-5.6 and later only.
        Max => "max",
    }
}

api_enum! {
    /// Standard or Pro execution, independent of effort. GPT-5.6 only.
    ReasoningMode {
        /// The default execution path.
        Standard => "standard",
        /// More model work per request, billed at the same token rates.
        Pro => "pro",
    }
}

api_enum! {
    /// How much of the model's own reasoning to summarize back to the caller.
    /// Summaries are output, not input, so this does not affect the prefix.
    ReasoningSummary {
        /// The most detailed summarizer the model offers.
        Auto => "auto",
        /// A short summary.
        Concise => "concise",
        /// A long summary.
        Detailed => "detailed",
    }
}

api_enum! {
    /// `prompt_cache_options.mode`, on GPT-5.6 and later.
    ///
    /// The choice is not cosmetic: it decides how many of the four cache-write
    /// slots your own breakpoints may use. See
    /// [`CacheWriteBudget`](crate::request::CacheWriteBudget).
    CacheMode {
        /// OpenAI adds a breakpoint at the end of the latest eligible message,
        /// consuming one write slot and leaving three for explicit ones.
        Implicit => "implicit",
        /// No automatic breakpoint: all four slots are yours. With no explicit
        /// breakpoint at all, the request uses no caching whatsoever.
        Explicit => "explicit",
    }
}

api_enum! {
    /// `prompt_cache_options.ttl`, the minimum lifetime of every breakpoint the
    /// request writes. GPT-5.6 and later accept exactly one value, so this
    /// enum has exactly one variant — the type states the constraint instead of
    /// inviting a string the API would refuse.
    CacheTtl {
        /// Thirty minutes after the latest write or reuse; also the default.
        ThirtyMinutes => "30m",
    }
}

api_enum! {
    /// `prompt_cache_retention`, the *maximum* retention policy on models older
    /// than GPT-5.6. A different field from [`CacheTtl`], on different models,
    /// which is why the two live on different model types and can never be
    /// sent together.
    CacheRetention {
        /// GPU-local only: roughly 5–10 minutes idle, up to an hour. The
        /// default for organizations with Zero Data Retention.
        InMemory => "in_memory",
        /// Extended retention: typically ~30 minutes, up to 24 hours.
        TwentyFourHours => "24h",
    }
}

api_enum! { roundtrip
    /// Why a response stopped short of a complete answer.
    IncompleteReason {
        /// Generation hit `max_output_tokens` or the context window.
        MaxOutputTokens => "max_output_tokens",
        /// A safety filter interrupted the response.
        ContentFilter => "content_filter",
    }
}

api_enum! { roundtrip
    /// The `error.type` OpenAI returns on a failed request. Useful for deciding
    /// whether to retry: `RateLimit` and `ServerError` are transient, the rest
    /// are the caller's own bug.
    ErrorType {
        /// The request itself was malformed or refused.
        InvalidRequest => "invalid_request_error",
        /// The credential was missing, wrong, or revoked.
        Authentication => "authentication_error",
        /// The credential is valid but lacks access.
        Permission => "permission_error",
        /// The named object does not exist.
        NotFound => "not_found_error",
        /// A rate or spend limit was hit. Transient.
        RateLimit => "rate_limit_error",
        /// A failure on OpenAI's side. Transient.
        ServerError => "server_error",
    }
}

impl ErrorType {
    /// The documented HTTP-status-to-error mapping, as a pure lookup. `None`
    /// for a status OpenAI does not document, so an unexpected code stays
    /// visibly unexpected.
    pub fn from_status(status: u16) -> Option<Self> {
        Some(match status {
            400 | 422 => Self::InvalidRequest,
            401 => Self::Authentication,
            403 => Self::Permission,
            404 => Self::NotFound,
            429 => Self::RateLimit,
            500..=599 => Self::ServerError,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_matches_the_wire() {
        assert_eq!(Role::Developer.as_str(), "developer");
        assert_eq!(CacheMode::Explicit.as_str(), "explicit");
        assert_eq!(CacheTtl::ThirtyMinutes.as_str(), "30m");
        assert_eq!(CacheRetention::TwentyFourHours.as_str(), "24h");
        assert_eq!(ReasoningEffort::Max.as_str(), "max");
        assert_eq!(AssistantPhase::FinalAnswer.as_str(), "final_answer");
    }

    #[test]
    fn serializing_an_enum_yields_a_bare_string() {
        assert_eq!(serde_json::to_value(Verbosity::Low).unwrap(), serde_json::json!("low"));
    }

    /// `from_str` must invert `as_str` exactly, and refuse anything else.
    #[test]
    fn response_vocabulary_roundtrips() {
        for e in [
            ErrorType::InvalidRequest,
            ErrorType::Authentication,
            ErrorType::Permission,
            ErrorType::NotFound,
            ErrorType::RateLimit,
            ErrorType::ServerError,
        ] {
            assert_eq!(ErrorType::from_str(e.as_str()), Some(e));
        }
        for r in [IncompleteReason::MaxOutputTokens, IncompleteReason::ContentFilter] {
            assert_eq!(IncompleteReason::from_str(r.as_str()), Some(r));
        }
        assert_eq!(ErrorType::from_str("not_a_real_error"), None);
        assert_eq!(IncompleteReason::from_str("nonsense"), None);
    }

    #[test]
    fn statuses_map_to_error_types() {
        assert_eq!(ErrorType::from_status(400), Some(ErrorType::InvalidRequest));
        assert_eq!(ErrorType::from_status(422), Some(ErrorType::InvalidRequest));
        assert_eq!(ErrorType::from_status(429), Some(ErrorType::RateLimit));
        assert_eq!(ErrorType::from_status(503), Some(ErrorType::ServerError));
        assert_eq!(ErrorType::from_status(200), None);
        assert_eq!(ErrorType::from_status(302), None);
    }
}
