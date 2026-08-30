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
        // `Ord` as well as `Eq`: these are keys as often as they are values —
        // a count per tool, a set of includes — and a total order that follows
        // declaration order is both free and stable.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// Which of the two roles that speak the *input* vocabulary is speaking.
    /// `Developer` outranks `User`.
    ///
    /// `assistant` is absent, and that absence is the fix for a 400 the crate
    /// used to make expressible: OpenAI accepts `input_text` and `input_image`
    /// under these two roles and only `output_text` and `refusal` under
    /// `assistant`. So the assistant role is not a third variant here — it is
    /// [`Message::Assistant`](crate::content::Message::Assistant), which
    /// carries the other vocabulary.
    ///
    /// `system` exists on the wire as a legacy alias for `developer` and is
    /// absent too: two spellings of one role would be two different prefixes
    /// for the same meaning, which is exactly the silent cache miss this crate
    /// is built to prevent.
    InputRole {
        /// Instructions from the application. Outranks `user`.
        Developer => "developer",
        /// Input from the person using the application.
        User => "user",
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
        /// Let the model decide. The API's documented default: "If omitted or
        /// set to `auto`, the model determines the context mode." So `auto` is
        /// the spelling of the default, and the crate emits it rather than
        /// choosing `all_turns` on the caller's behalf — the GPT-5.6 family
        /// resolves `auto` to `all_turns` and earlier models to `current_turn`,
        /// which is a fact about the model, not about this request.
        Auto => "auto",
        /// Reasoning from the active turn only.
        CurrentTurn => "current_turn",
        /// Reasoning from earlier turns too.
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

api_enum! {
    /// One entry of `include`: extra output the response omits unless asked for.
    ///
    /// Every variant names data the model produced anyway. The field decides
    /// whether it comes back, and one of these is load-bearing rather than
    /// diagnostic: without [`Self::ReasoningEncryptedContent`] a stateless
    /// caller has no reasoning item to replay, so a reasoning model loses its
    /// own train of thought between a tool call and its result.
    ///
    /// `include` is not part of the hashed prefix — it selects what the
    /// *response* carries, not what the model reads — so it lives on the
    /// request and may vary per call.
    Include {
        /// The reasoning item's replayable opaque payload. Required to replay
        /// reasoning with `store: false` or under Zero Data Retention; see
        /// [`Context::push_reasoning`](crate::context::Context::push_reasoning),
        /// which is what consumes it.
        ReasoningEncryptedContent => "reasoning.encrypted_content",
        /// The file-search tool call's search results.
        FileSearchCallResults => "file_search_call.results",
        /// The web-search tool call's results.
        WebSearchCallResults => "web_search_call.results",
        /// The sources behind the web-search tool call's action.
        WebSearchCallActionSources => "web_search_call.action.sources",
        /// Image URLs from an input message, echoed back.
        MessageInputImageUrl => "message.input_image.image_url",
        /// Image URLs from a computer-call output.
        ComputerCallOutputImageUrl => "computer_call_output.output.image_url",
        /// What the code interpreter's Python execution printed.
        CodeInterpreterCallOutputs => "code_interpreter_call.outputs",
        /// Per-token log probabilities on assistant messages.
        ///
        /// Reachable but not useful on the models this crate carries: they are
        /// all reasoning models, and `top_logprobs` — which decides how many
        /// alternatives each position reports — is refused by every one of them
        /// with *"logprobs are not supported with reasoning models."* The entry
        /// exists because the API accepts it and because a non-reasoning model
        /// added later would need it.
        MessageOutputTextLogprobs => "message.output_text.logprobs",
    }
}

api_enum! { roundtrip
    /// `service_tier`: which processing pool serves the request.
    ///
    /// A cost and latency choice, not a content one, so it is outside the
    /// hashed prefix. Roundtrips because the response echoes the tier that
    /// *actually* served the request, which may differ from the one asked for —
    /// `fast` is answered as `priority`.
    ServiceTier {
        /// Whatever the project is configured for. The documented default.
        Auto => "auto",
        /// Standard pricing and performance.
        Default => "default",
        /// Flex processing: cheaper, slower, and it may be queued.
        Flex => "flex",
        /// Scale processing.
        Scale => "scale",
        /// Priority processing. Also what a `fast` request is reported as.
        Priority => "priority",
        /// Fast mode, reported back as [`Self::Priority`].
        Fast => "fast",
        /// Ultrafast processing, access-controlled and limited to some models.
        Ultrafast => "ultrafast",
    }
}

api_enum! {
    /// `truncation`: what to do when the input exceeds the context window.
    ///
    /// Note which one is documented as the default. Dropping items from the
    /// front is a prefix change by definition, so `Auto` and prompt caching are
    /// in tension: the first turn that truncates rewrites the prefix.
    Truncation {
        /// Drop items from the beginning of the conversation to fit.
        Auto => "auto",
        /// Fail with a 400 instead. The documented default.
        Disabled => "disabled",
    }
}

api_enum! { roundtrip
    /// `status` on a response object: where generation got to.
    ///
    /// Six values, and only three of them ever arrive on a streamed terminal
    /// event. `Queued` and `InProgress` name a response still being generated —
    /// reachable through background mode or a retrieval — and `Cancelled` names
    /// one a caller stopped. Roundtrips because it is read, never sent.
    ResponseStatus {
        /// The model answered.
        Completed => "completed",
        /// Generation failed; `error` says why.
        Failed => "failed",
        /// Still generating.
        InProgress => "in_progress",
        /// A caller cancelled it.
        Cancelled => "cancelled",
        /// Accepted and waiting for capacity. Reached under `background` or a
        /// queued service tier.
        Queued => "queued",
        /// Stopped short; `incomplete_details.reason` says why.
        Incomplete => "incomplete",
    }
}

api_enum! { roundtrip
    /// Which built-in tool a hosted-tool streaming event belongs to.
    ///
    /// The Responses API streams 26 events for hosted tools, and they are one
    /// family name crossed with a handful of lifecycle phases rather than 26
    /// unrelated shapes. Naming the family separately from the phase — see
    /// [`HostedToolPhase`](crate::hosted::HostedToolPhase) — is what collapses
    /// them into one variant a consumer can match once, and it is what makes a
    /// tool OpenAI adds next year one more variant here instead of a decoder
    /// change.
    ///
    /// Every one of these tools runs on OpenAI's side. This crate does not build
    /// the tool definitions that turn them on — see `SOUL.md` — but it decodes
    /// their events, because a consumer that enabled one through a hand-written
    /// tool array still has to read what comes back.
    HostedTool {
        /// `file_search`, over vector stores.
        FileSearch => "file_search",
        /// `web_search`.
        WebSearch => "web_search",
        /// `code_interpreter`, running Python in a container.
        CodeInterpreter => "code_interpreter",
        /// `image_generation`.
        ImageGeneration => "image_generation",
        /// An MCP server's tool call.
        Mcp => "mcp",
        /// An MCP server's tool listing, which precedes any call to it.
        McpListTools => "mcp_list_tools",
        /// `shell`, running commands.
        Shell => "shell",
        /// `local_shell`, whose commands the caller runs.
        LocalShell => "local_shell",
        /// `computer_use_preview`.
        Computer => "computer",
        /// `apply_patch`, editing files.
        ApplyPatch => "apply_patch",
        /// A custom tool, whose input is free text rather than JSON.
        Custom => "custom",
        /// `tool_search`, which finds other tools.
        ToolSearch => "tool_search",
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

// ── Metadata ─────────────────────────────────────────────────────────────────

/// `metadata`: the caller's own key-value pairs, echoed back on the response.
///
/// Three limits the reference states, and this type is the reason none of them
/// can be exceeded silently: at most 16 pairs, keys at most 64 characters,
/// values at most 512. The API answers a 400 to any of the three, so the
/// checking constructor is the only way in and there are no public fields to
/// assign around it.
///
/// Lengths are counted in characters, not bytes, because that is the unit the
/// reference names.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Metadata(std::collections::BTreeMap<String, String>);

/// Which documented metadata limit a rejected map broke.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    /// More than the 16 pairs the API accepts.
    TooManyPairs {
        /// How many were offered.
        pairs: usize,
    },
    /// A key longer than 64 characters.
    KeyTooLong {
        /// The offending key.
        key: String,
        /// Its length in characters.
        characters: usize,
    },
    /// A value longer than 512 characters.
    ValueTooLong {
        /// The key whose value was too long.
        key: String,
        /// The value's length in characters.
        characters: usize,
    },
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataError::TooManyPairs { pairs } => {
                write!(f, "metadata holds {pairs} pairs; the API accepts at most {}", Metadata::MAX_PAIRS)
            }
            MetadataError::KeyTooLong { key, characters } => write!(
                f,
                "metadata key `{key}` is {characters} characters; the API accepts at most {}",
                Metadata::MAX_KEY_CHARACTERS
            ),
            MetadataError::ValueTooLong { key, characters } => write!(
                f,
                "the metadata value at `{key}` is {characters} characters; the API accepts at most {}",
                Metadata::MAX_VALUE_CHARACTERS
            ),
        }
    }
}

impl std::error::Error for MetadataError {}

impl Metadata {
    /// The most pairs the API accepts.
    pub const MAX_PAIRS: usize = 16;
    /// The longest key the API accepts, in characters.
    pub const MAX_KEY_CHARACTERS: usize = 64;
    /// The longest value the API accepts, in characters.
    pub const MAX_VALUE_CHARACTERS: usize = 512;

    /// Every pair at once, checked against all three documented limits.
    ///
    /// Whole-map rather than one-pair-at-a-time because the pair count is a
    /// property of the map: a builder that accepted a seventeenth pair and
    /// failed later would have already let the invalid value exist.
    pub fn new(pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>) -> Result<Self, MetadataError> {
        let mut map = std::collections::BTreeMap::new();
        for (key, value) in pairs {
            let (key, value) = (key.into(), value.into());
            let characters = key.chars().count();
            if characters > Self::MAX_KEY_CHARACTERS {
                return Err(MetadataError::KeyTooLong { key, characters });
            }
            let characters = value.chars().count();
            if characters > Self::MAX_VALUE_CHARACTERS {
                return Err(MetadataError::ValueTooLong { key, characters });
            }
            map.insert(key, value);
        }
        if map.len() > Self::MAX_PAIRS {
            return Err(MetadataError::TooManyPairs { pairs: map.len() });
        }
        Ok(Self(map))
    }

    /// The pairs, in key order. Sorted because the map is: two metadata values
    /// built from the same pairs in different orders are the same value, and
    /// serialize to the same bytes.
    pub fn pairs(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(key, value)| (key.as_str(), value.as_str()))
    }

    /// How many pairs it holds.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether it holds none, in which case the field is omitted rather than
    /// sent as `{}`.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl serde::Serialize for Metadata {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_matches_the_wire() {
        assert_eq!(InputRole::Developer.as_str(), "developer");
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

    /// Each of the three documented metadata limits refuses, naming which one.
    #[test]
    fn metadata_refuses_what_the_api_refuses() {
        let seventeen: Vec<(String, String)> = (0..17).map(|n| (format!("key{n}"), "v".to_owned())).collect();
        assert_eq!(Metadata::new(seventeen), Err(MetadataError::TooManyPairs { pairs: 17 }));

        let long_key = "k".repeat(65);
        assert_eq!(
            Metadata::new([(long_key.clone(), "v")]),
            Err(MetadataError::KeyTooLong { key: long_key, characters: 65 })
        );

        assert_eq!(
            Metadata::new([("k", "v".repeat(513))]),
            Err(MetadataError::ValueTooLong { key: "k".to_owned(), characters: 513 })
        );

        // The boundaries themselves are accepted: the limits are inclusive.
        assert!(Metadata::new([("k".repeat(64), "v".repeat(512))]).is_ok());
        let sixteen: Vec<(String, String)> = (0..16).map(|n| (format!("key{n}"), "v".to_owned())).collect();
        assert_eq!(Metadata::new(sixteen).unwrap().len(), 16);
    }

    /// Lengths count characters, not bytes, because that is the unit the
    /// reference names — so a 64-character key of multi-byte characters is legal
    /// even though it is 256 bytes.
    #[test]
    fn metadata_limits_count_characters_rather_than_bytes() {
        let key = "\u{1f600}".repeat(64);
        assert_eq!(key.len(), 256, "the key really is longer in bytes");
        assert!(Metadata::new([(key, "v")]).is_ok());
    }

    /// Two maps built from the same pairs in different orders are one value and
    /// serialize identically. Metadata is not part of the hashed prefix, but a
    /// type whose bytes depend on insertion order invites the opposite habit.
    #[test]
    fn metadata_serializes_in_key_order() {
        let forwards = Metadata::new([("a", "1"), ("b", "2")]).unwrap();
        let backwards = Metadata::new([("b", "2"), ("a", "1")]).unwrap();
        assert_eq!(forwards, backwards);
        assert_eq!(serde_json::to_string(&forwards).unwrap(), r#"{"a":"1","b":"2"}"#);
        assert_eq!(forwards.pairs().collect::<Vec<_>>(), vec![("a", "1"), ("b", "2")]);
    }

    /// A tier the response reports must read back as the variant that named it.
    #[test]
    fn the_service_tier_the_response_reports_reads_back() {
        for tier in [ServiceTier::Auto, ServiceTier::Default, ServiceTier::Flex, ServiceTier::Ultrafast] {
            assert_eq!(ServiceTier::from_str(tier.as_str()), Some(tier));
        }
        // Live: a request asking for `fast` is served and reported as
        // `priority`, so both strings must decode.
        assert_eq!(ServiceTier::from_str("fast"), Some(ServiceTier::Fast));
        assert_eq!(ServiceTier::from_str("priority"), Some(ServiceTier::Priority));
        assert_eq!(ServiceTier::from_str("turbo"), None);
    }
}
