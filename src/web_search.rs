//! The hosted web search tool: how a request offers it, and the call a search
//! leaves in the response for the next request to replay.
//!
//! OpenAI runs the search itself, inside the response. The model's output then
//! holds a `web_search_call` item beside its message, and a stateless caller
//! sends that item back in `input` on the next turn so the conversation it
//! replays is the one the model had. The call is therefore one type in both
//! directions: [`WebSearchCall::from_item`] reads it off a response, and its
//! [`Serialize`] impl writes it into a request.
//!
//! Each `search` action is billed on top of tokens; opening a page or finding
//! text within one is not a separate search.

use serde::Serialize;
use serde_json::Value;

use crate::values::{SearchContextSize, WebSearchCallStatus};

// ── Declaring the tool ───────────────────────────────────────────────────────

/// Where the user roughly is, so results can be localized.
///
/// Every field is optional on the wire, and the empty location is meaningful:
/// the API documents that an omitted `user_location` falls back to the United
/// States, and that `{"type": "approximate"}` alone is how to avoid that
/// fallback. So [`Self::unknown`] exists and is not the same as sending none.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct UserLocation {
    #[serde(rename = "type")]
    kind: ApproximateLocation,
    /// Free text, e.g. `San Francisco`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// The two-letter ISO country code, e.g. `US`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Free text, e.g. `California`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// The IANA time zone, e.g. `America/Los_Angeles`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

/// The only location `type` the API documents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ApproximateLocation {
    #[default]
    Approximate,
}

impl UserLocation {
    /// A location that states nothing, which stops the United States fallback.
    pub fn unknown() -> Self {
        Self::default()
    }
}

/// The web search tool, offered to the model beside any function tools.
///
/// It joins the `tools` array, so adding or changing it rewrites the start of
/// the cached prefix exactly as changing a function tool does.
///
/// Two fields have documented defaults and are therefore always sent:
/// live access is on and the context size is `medium` unless the caller says
/// otherwise. The other two have documented behavior for absence — every
/// domain, and the United States fallback — so absence stays expressible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSearchTool {
    /// Whether search may fetch live content. `false` runs it against cached
    /// content only.
    pub external_web_access: bool,
    /// Only these domains and their subdomains, written without a scheme.
    /// `None` allows every domain.
    pub allowed_domains: Option<Vec<String>>,
    /// How much of the context window results may fill.
    pub search_context_size: SearchContextSize,
    /// Where to localize results for. `None` falls back to the United States.
    pub user_location: Option<UserLocation>,
}

impl Default for WebSearchTool {
    /// The API's documented defaults: live access, `medium` context, every
    /// domain, no location.
    fn default() -> Self {
        Self {
            external_web_access: true,
            allowed_domains: None,
            search_context_size: SearchContextSize::Medium,
            user_location: None,
        }
    }
}

impl WebSearchTool {
    /// The tool with the API's documented defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Serialize)]
struct FiltersWire<'a> {
    allowed_domains: &'a [String],
}

#[derive(Serialize)]
struct WebSearchToolWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    external_web_access: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    filters: Option<FiltersWire<'a>>,
    search_context_size: SearchContextSize,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_location: Option<&'a UserLocation>,
}

impl Serialize for WebSearchTool {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        WebSearchToolWire {
            kind: "web_search",
            external_web_access: self.external_web_access,
            filters: self.allowed_domains.as_deref().map(|allowed_domains| FiltersWire { allowed_domains }),
            search_context_size: self.search_context_size,
            user_location: self.user_location.as_ref(),
        }
        .serialize(s)
    }
}

// ── The call a search leaves behind ──────────────────────────────────────────

/// One source a search consulted.
///
/// Returned only when the request includes
/// [`Include::WebSearchCallActionSources`](crate::values::Include::WebSearchCallActionSources),
/// and often longer than the list of citations, because it names everything
/// the model read rather than what it quoted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSearchSource {
    /// A web page.
    Url(String),
    /// A source of another `type` — the guide names real-time feeds such as
    /// `oai-weather` — kept whole so that replay sends it back unchanged.
    Unmodeled(Value),
}

impl Serialize for WebSearchSource {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct UrlWire<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            url: &'a str,
        }
        match self {
            Self::Url(url) => UrlWire { kind: "url", url }.serialize(s),
            Self::Unmodeled(source) => source.serialize(s),
        }
    }
}

impl WebSearchSource {
    fn decode(source: &Value) -> Self {
        match (source.get("type").and_then(Value::as_str), source.get("url").and_then(Value::as_str)) {
            (Some("url"), Some(url)) if source.as_object().is_some_and(|fields| fields.len() == 2) => {
                Self::Url(url.to_owned())
            }
            _ => Self::Unmodeled(source.clone()),
        }
    }
}

/// What the model did with the web in one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSearchAction {
    /// A search: the one action billed as a search.
    Search {
        /// The query, where the API reported a single one.
        query: Option<String>,
        /// The queries, where the API reported several. Absent and empty are
        /// kept apart because they replay as different bytes.
        queries: Option<Vec<String>>,
        /// What the search consulted, when the request asked for it.
        sources: Option<Vec<WebSearchSource>>,
    },
    /// Opening one page from the results.
    OpenPage {
        /// The page, where the API reported it.
        url: Option<String>,
    },
    /// Looking for text within a page already loaded.
    FindInPage {
        /// What was looked for.
        pattern: String,
        /// The page searched.
        url: String,
    },
}

impl WebSearchAction {
    fn decode(action: &Value) -> Option<Self> {
        let text = |field: &str| action.get(field).and_then(Value::as_str).map(str::to_owned);
        Some(match action.get("type")?.as_str()? {
            "search" => Self::Search {
                query: text("query"),
                queries: match action.get("queries") {
                    None | Some(Value::Null) => None,
                    Some(queries) => {
                        Some(queries.as_array()?.iter().map(|q| q.as_str().map(str::to_owned)).collect::<Option<_>>()?)
                    }
                },
                sources: match action.get("sources") {
                    None | Some(Value::Null) => None,
                    Some(sources) => Some(sources.as_array()?.iter().map(WebSearchSource::decode).collect()),
                },
            },
            "open_page" => Self::OpenPage { url: text("url") },
            "find_in_page" => Self::FindInPage { pattern: text("pattern")?, url: text("url")? },
            _ => return None,
        })
    }

    /// Every query this action searched, single or several.
    pub fn searched_queries(&self) -> Vec<&str> {
        match self {
            Self::Search { query, queries, .. } => {
                let mut all: Vec<&str> = queries.iter().flatten().map(String::as_str).collect();
                if let Some(query) = query
                    && !all.contains(&query.as_str())
                {
                    all.insert(0, query);
                }
                all
            }
            Self::OpenPage { .. } | Self::FindInPage { .. } => Vec::new(),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WebSearchActionWire<'a> {
    Search {
        #[serde(skip_serializing_if = "Option::is_none")]
        query: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        queries: Option<&'a [String]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        sources: Option<&'a [WebSearchSource]>,
    },
    OpenPage {
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<&'a str>,
    },
    FindInPage {
        pattern: &'a str,
        url: &'a str,
    },
}

impl Serialize for WebSearchAction {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Search { query, queries, sources } => WebSearchActionWire::Search {
                query: query.as_deref(),
                queries: queries.as_deref(),
                sources: sources.as_deref(),
            },
            Self::OpenPage { url } => WebSearchActionWire::OpenPage { url: url.as_deref() },
            Self::FindInPage { pattern, url } => WebSearchActionWire::FindInPage { pattern, url },
        }
        .serialize(s)
    }
}

/// One `web_search_call` item, as a response returns it and a request replays it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSearchCall {
    /// The item's identifier.
    pub id: String,
    /// Where the call stands.
    pub status: WebSearchCallStatus,
    /// What the model did.
    pub action: WebSearchAction,
}

#[derive(Serialize)]
struct WebSearchCallWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    id: &'a str,
    status: WebSearchCallStatus,
    action: &'a WebSearchAction,
}

impl Serialize for WebSearchCall {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        WebSearchCallWire { kind: "web_search_call", id: &self.id, status: self.status, action: &self.action }
            .serialize(s)
    }
}

impl WebSearchCall {
    /// The call a `web_search_call` output item holds, as
    /// [`OutputItem::HostedToolCall`](crate::items::OutputItem::HostedToolCall)
    /// carries it.
    ///
    /// `None` for an item this crate cannot replay faithfully: another type, a
    /// status or action outside the documented vocabulary, or a field of the
    /// wrong shape. Replaying a guess would send the model a conversation it
    /// did not have.
    pub fn from_item(item: &Value) -> Option<Self> {
        if item.get("type")?.as_str()? != "web_search_call" {
            return None;
        }
        Some(Self {
            id: item.get("id")?.as_str()?.to_owned(),
            status: WebSearchCallStatus::from_str(item.get("status")?.as_str()?)?,
            action: WebSearchAction::decode(item.get("action")?)?,
        })
    }

    /// Whether this call is billed as a search.
    pub fn is_search(&self) -> bool {
        matches!(self.action, WebSearchAction::Search { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_default_tool_sends_its_documented_defaults() {
        assert_eq!(
            serde_json::to_value(WebSearchTool::new()).unwrap(),
            json!({"type": "web_search", "external_web_access": true, "search_context_size": "medium"})
        );
    }

    #[test]
    fn filters_and_an_unknown_location_take_their_documented_shapes() {
        let tool = WebSearchTool {
            allowed_domains: Some(vec!["pubmed.ncbi.nlm.nih.gov".to_owned()]),
            user_location: Some(UserLocation::unknown()),
            search_context_size: SearchContextSize::Low,
            ..WebSearchTool::new()
        };
        let wire = serde_json::to_value(tool).unwrap();
        assert_eq!(wire["filters"], json!({"allowed_domains": ["pubmed.ncbi.nlm.nih.gov"]}));
        assert_eq!(wire["user_location"], json!({"type": "approximate"}), "the fallback-stopping empty location");
        assert_eq!(wire["search_context_size"], "low");
    }

    /// The documented output item decodes and serializes back to itself, which
    /// is what replay needs.
    #[test]
    fn a_documented_search_call_round_trips() {
        let item = json!({
            "type": "web_search_call", "id": "ws_67c9fa0502748190b7dd390736892e100be649c1a5ff9609",
            "status": "completed",
            "action": {"type": "search", "query": "positive news story today",
                       "sources": [{"type": "url", "url": "https://example.com/a"},
                                   {"type": "api", "name": "oai-weather"}]}
        });
        let call = WebSearchCall::from_item(&item).unwrap();
        assert!(call.is_search());
        assert_eq!(call.action.searched_queries(), vec!["positive news story today"]);
        let WebSearchAction::Search { sources: Some(sources), .. } = &call.action else { panic!("expected sources") };
        assert_eq!(sources[0], WebSearchSource::Url("https://example.com/a".to_owned()));
        assert!(matches!(sources[1], WebSearchSource::Unmodeled(_)));
        assert_eq!(serde_json::to_value(&call).unwrap(), item);
    }

    #[test]
    fn page_actions_are_not_searches_and_round_trip() {
        for action in [
            json!({"type": "open_page", "url": "https://example.com"}),
            json!({"type": "find_in_page", "pattern": "born", "url": "https://example.com"}),
        ] {
            let item = json!({"type": "web_search_call", "id": "ws_1", "status": "completed", "action": action});
            let call = WebSearchCall::from_item(&item).unwrap();
            assert!(!call.is_search());
            assert!(call.action.searched_queries().is_empty());
            assert_eq!(serde_json::to_value(&call).unwrap(), item);
        }
    }

    #[test]
    fn an_item_outside_the_vocabulary_is_not_replayed() {
        let base = json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
                          "action": {"type": "search", "queries": ["a", "b"]}});
        assert_eq!(WebSearchCall::from_item(&base).unwrap().action.searched_queries(), vec!["a", "b"]);
        let mut novel_status = base.clone();
        novel_status["status"] = json!("teleporting");
        assert_eq!(WebSearchCall::from_item(&novel_status), None);
        let mut novel_action = base.clone();
        novel_action["action"] = json!({"type": "summon"});
        assert_eq!(WebSearchCall::from_item(&novel_action), None);
        assert_eq!(WebSearchCall::from_item(&json!({"type": "file_search_call"})), None);
    }
}
