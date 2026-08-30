//! Hosted-tool streaming events, as one family crossed with one phase.
//!
//! OpenAI streams 26 events for its built-in tools —
//! `response.web_search_call.in_progress`, `response.mcp_call.completed`,
//! `response.code_interpreter_call_code.delta`, and so on. Written out as 26
//! variants they look like 26 unrelated shapes, and a consumer that wants to
//! show "a tool is running" has to match all of them.
//!
//! They are not 26 shapes. Every one of them is a
//! [`HostedTool`] crossed with a
//! [`HostedToolPhase`], and every one carries the same two fields: which output
//! item it belongs to and that item's identifier. So the crate decodes the
//! *product* — one variant, two enums — and a consumer matches the pair it cares
//! about. The multiplication is the point: twelve tools times six phases is
//! seventy-two possible events from one variant, and the eighteen OpenAI has not
//! shipped yet cost nothing.
//!
//! The phases that carry a payload are the exception, and they carry it because
//! there is something to accumulate: code being written, arguments being
//! streamed, a partial image. Those phases hold their delta.

use crate::values::HostedTool;

/// Where a hosted-tool call has got to.
///
/// The lifecycle every hosted tool shares. Not every tool sends every phase —
/// only `file_search` and `web_search` send [`Self::Searching`], only
/// `code_interpreter` sends [`Self::Interpreting`] — but the phases mean the
/// same thing wherever they appear, which is what makes one enum right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HostedToolPhase {
    /// The call has started.
    InProgress,
    /// It is searching. `file_search` and `web_search` only.
    Searching,
    /// It is running code. `code_interpreter` only.
    Interpreting,
    /// It is producing an image. `image_generation` only.
    Generating,
    /// It finished.
    Completed,
    /// It failed. MCP calls and listings only; other tools report failure
    /// through the item's own status.
    Failed,
}

impl HostedToolPhase {
    /// The phase's own wire word, the part after the last `.` of the event type.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Searching => "searching",
            Self::Interpreting => "interpreting",
            Self::Generating => "generating",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// The `(tool, phase)` pair a hosted-tool lifecycle event names, or `None` when
/// the event type is not one.
///
/// The inverse of [`lifecycle_event_type`], and a pure `match` on the string, so
/// the whole family costs one comparison chain rather than one branch per tool.
pub(crate) fn lifecycle_of(kind: &str) -> Option<(HostedTool, HostedToolPhase)> {
    use HostedTool::{CodeInterpreter, FileSearch, ImageGeneration, Mcp, McpListTools, WebSearch};
    use HostedToolPhase::{Completed, Failed, Generating, InProgress, Interpreting, Searching};
    Some(match kind {
        "response.file_search_call.in_progress" => (FileSearch, InProgress),
        "response.file_search_call.searching" => (FileSearch, Searching),
        "response.file_search_call.completed" => (FileSearch, Completed),
        "response.web_search_call.in_progress" => (WebSearch, InProgress),
        "response.web_search_call.searching" => (WebSearch, Searching),
        "response.web_search_call.completed" => (WebSearch, Completed),
        "response.code_interpreter_call.in_progress" => (CodeInterpreter, InProgress),
        "response.code_interpreter_call.interpreting" => (CodeInterpreter, Interpreting),
        "response.code_interpreter_call.completed" => (CodeInterpreter, Completed),
        "response.image_generation_call.in_progress" => (ImageGeneration, InProgress),
        "response.image_generation_call.generating" => (ImageGeneration, Generating),
        "response.image_generation_call.completed" => (ImageGeneration, Completed),
        "response.mcp_call.in_progress" => (Mcp, InProgress),
        "response.mcp_call.completed" => (Mcp, Completed),
        "response.mcp_call.failed" => (Mcp, Failed),
        "response.mcp_list_tools.in_progress" => (McpListTools, InProgress),
        "response.mcp_list_tools.completed" => (McpListTools, Completed),
        "response.mcp_list_tools.failed" => (McpListTools, Failed),
        _ => return None,
    })
}

/// The event type a `(tool, phase)` pair names, or `None` for a pair OpenAI does
/// not send.
///
/// The inverse of [`lifecycle_of`], and the reason
/// [`StreamEvent::kind`](crate::stream::StreamEvent::kind) can return a real
/// wire string for a decoded lifecycle event rather than a reconstruction that
/// might not match. `None` is honest: `(WebSearch, Interpreting)` is not an
/// event, and no string should be invented for it.
pub(crate) fn lifecycle_event_type(tool: HostedTool, phase: HostedToolPhase) -> Option<&'static str> {
    use HostedTool::{CodeInterpreter, FileSearch, ImageGeneration, Mcp, McpListTools, WebSearch};
    use HostedToolPhase::{Completed, Failed, Generating, InProgress, Interpreting, Searching};
    Some(match (tool, phase) {
        (FileSearch, InProgress) => "response.file_search_call.in_progress",
        (FileSearch, Searching) => "response.file_search_call.searching",
        (FileSearch, Completed) => "response.file_search_call.completed",
        (WebSearch, InProgress) => "response.web_search_call.in_progress",
        (WebSearch, Searching) => "response.web_search_call.searching",
        (WebSearch, Completed) => "response.web_search_call.completed",
        (CodeInterpreter, InProgress) => "response.code_interpreter_call.in_progress",
        (CodeInterpreter, Interpreting) => "response.code_interpreter_call.interpreting",
        (CodeInterpreter, Completed) => "response.code_interpreter_call.completed",
        (ImageGeneration, InProgress) => "response.image_generation_call.in_progress",
        (ImageGeneration, Generating) => "response.image_generation_call.generating",
        (ImageGeneration, Completed) => "response.image_generation_call.completed",
        (Mcp, InProgress) => "response.mcp_call.in_progress",
        (Mcp, Completed) => "response.mcp_call.completed",
        (Mcp, Failed) => "response.mcp_call.failed",
        (McpListTools, InProgress) => "response.mcp_list_tools.in_progress",
        (McpListTools, Completed) => "response.mcp_list_tools.completed",
        (McpListTools, Failed) => "response.mcp_list_tools.failed",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every lifecycle event OpenAI documents decodes to a pair, and every pair
    /// names back the exact string it came from.
    ///
    /// This is the test the factoring rests on: a round trip through both
    /// directions proves the product is a faithful renaming rather than a lossy
    /// summary. A tool or phase added on one side alone fails here.
    #[test]
    fn every_documented_lifecycle_event_roundtrips() {
        const DOCUMENTED: [&str; 18] = [
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call.completed",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.image_generation_call.completed",
            "response.mcp_call.in_progress",
            "response.mcp_call.completed",
            "response.mcp_call.failed",
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
            "response.mcp_list_tools.failed",
        ];
        for kind in DOCUMENTED {
            let (tool, phase) = lifecycle_of(kind).unwrap_or_else(|| panic!("{kind} decodes"));
            assert_eq!(lifecycle_event_type(tool, phase), Some(kind));
        }
    }

    /// A pair OpenAI does not send names nothing, rather than a plausible string
    /// that would decode to a different event or to none.
    #[test]
    fn a_pair_the_api_does_not_send_names_nothing() {
        assert_eq!(lifecycle_event_type(HostedTool::WebSearch, HostedToolPhase::Interpreting), None);
        assert_eq!(lifecycle_event_type(HostedTool::FileSearch, HostedToolPhase::Failed), None);
        assert_eq!(lifecycle_of("response.web_search_call.interpreting"), None);
        assert_eq!(lifecycle_of("response.output_text.delta"), None);
    }

    /// The phase word is the tail of its own event types.
    #[test]
    fn a_phase_names_the_tail_of_its_event() {
        for phase in [
            HostedToolPhase::InProgress,
            HostedToolPhase::Searching,
            HostedToolPhase::Completed,
            HostedToolPhase::Failed,
        ] {
            let kind = lifecycle_event_type(HostedTool::FileSearch, phase)
                .or_else(|| lifecycle_event_type(HostedTool::Mcp, phase))
                .expect("one of the two sends it");
            assert!(kind.ends_with(phase.as_str()), "{kind} does not end with {}", phase.as_str());
        }
    }
}
