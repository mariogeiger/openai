//! Every streaming event the reference documents, decoded.
//!
//! The data is `tests/data/documented_stream_events.jsonl`: one line per event,
//! taken from the *Example* block of each section of
//! <https://developers.openai.com/api/reference/resources/responses/streaming-events>,
//! transcribed on 2026-08-30 when the page documented 58 event types. Only one
//! line was altered: the page's `response.function_call_arguments.delta` example
//! is missing a comma after its `delta` value and is not JSON, so it is repaired
//! to the shape every other delta event has.
//!
//! # Why against the reference rather than against the crate
//!
//! A test that builds a frame the way the crate expects it and then decodes it
//! agrees with whatever the crate currently does. That is how an `input_text`
//! defect once survived a full suite. These frames come from OpenAI's own page,
//! so the assertion fails when the crate and the reference disagree rather than
//! when the crate changes.
//!
//! # What it asserts
//!
//! Three things, in order of what each would cost if it broke.
//!
//! 1. **Every documented event decodes.** Not one of the 58 may be a
//!    [`FrameError`]: a documented frame that fails to decode is a broken
//!    decoder, not a new event.
//! 2. **The events this release models are modeled.** A count, so a variant
//!    silently degrading into `Unmodeled` — which no other test would catch,
//!    because ignoring is legal — fails here.
//! 3. **`kind()` round-trips.** A modeled event must name back the exact wire
//!    string it came from, which proves the factoring into
//!    `(family, phase)` pairs is a faithful renaming rather than a lossy one.

use std::collections::BTreeSet;

use openai::stream::{FrameError, StreamEvent};

/// The frames, one per documented event type.
fn documented_frames() -> Vec<(String, serde_json::Value)> {
    include_str!("data/documented_stream_events.jsonl")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let frame: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|error| panic!("fixture line is not JSON: {error}\n{line}"));
            let kind = frame["type"].as_str().expect("every frame names its type").to_owned();
            (kind, frame)
        })
        .collect()
}

/// The five events this release deliberately leaves unmodeled.
///
/// The shell family, whose payload is a structured command list a caller running
/// commands has to agree with exactly. Modeling it thinly would be worse than
/// saying plainly that it is not modeled; the call's own lifecycle still arrives
/// through `response.output_item.added` and `.done`.
const DELIBERATELY_UNMODELED: [&str; 5] = [
    "response.shell_call_command.added",
    "response.shell_call_command.delta",
    "response.shell_call_command.done",
    "response.shell_call_output_content.delta",
    "response.shell_call_output_content.done",
];

/// The reference documented 58 event types when this fixture was taken. Stated
/// so a fixture that loses a line fails rather than passing more easily.
const DOCUMENTED_EVENT_TYPES: usize = 58;

#[test]
fn the_fixture_holds_every_documented_event_once() {
    let frames = documented_frames();
    assert_eq!(frames.len(), DOCUMENTED_EVENT_TYPES);
    let distinct: BTreeSet<&str> = frames.iter().map(|(kind, _)| kind.as_str()).collect();
    assert_eq!(distinct.len(), DOCUMENTED_EVENT_TYPES, "a type appears twice");
}

/// Not one documented frame may fail to decode.
///
/// This is the assertion that matters most: an event OpenAI documents and this
/// crate refuses is a consumer's stream dying on a frame the server was entitled
/// to send.
#[test]
fn every_documented_event_decodes() {
    let mut broken = Vec::new();
    for (kind, frame) in documented_frames() {
        if let Err(error) = StreamEvent::from_json(&frame) {
            broken.push(format!("{kind}: {error}"));
        }
    }
    assert!(broken.is_empty(), "documented frames failed to decode:\n{}", broken.join("\n"));
}

/// Every event except the shell family is modeled, and the shell family is not.
///
/// Both directions, because each catches something the other cannot: the first
/// catches a variant that quietly became `Unmodeled`, and the second catches an
/// exclusion this file claims but the decoder no longer makes.
#[test]
fn exactly_the_shell_family_is_unmodeled() {
    let mut unmodeled = BTreeSet::new();
    for (kind, frame) in documented_frames() {
        let event = StreamEvent::from_json(&frame).unwrap_or_else(|error| panic!("{kind}: {error}"));
        if matches!(event, StreamEvent::Unmodeled { .. }) {
            unmodeled.insert(kind);
        }
    }
    let expected: BTreeSet<String> = DELIBERATELY_UNMODELED.iter().map(|kind| (*kind).to_owned()).collect();
    assert_eq!(unmodeled, expected);
    assert_eq!(
        DOCUMENTED_EVENT_TYPES - unmodeled.len(),
        53,
        "53 of the 58 documented events are modeled; update this count deliberately"
    );
}

/// A modeled event names back the exact string it was decoded from.
///
/// The events are factored — one `TextDelta` variant for four text streams, one
/// `HostedToolLifecycle` for eighteen lifecycle events — and this is what proves
/// the factoring loses nothing. A pair that named the wrong string, or a
/// plausible string OpenAI does not send, fails here.
#[test]
fn a_modeled_event_names_its_own_wire_type() {
    for (kind, frame) in documented_frames() {
        let event = StreamEvent::from_json(&frame).unwrap_or_else(|error| panic!("{kind}: {error}"));
        assert_eq!(event.kind(), kind.as_str(), "{kind} names itself differently");
    }
}

/// A field the wire always sends is required, and a field it may omit is not.
///
/// Both halves were learned the same way: a hand-written fixture omitted
/// `output_index`, which the wire always sends, and requiring `item_id` would
/// fail frames from gateways that omit it. So this asserts the asymmetry rather
/// than assuming it.
#[test]
fn output_index_is_required_and_item_id_is_not() {
    let without_index = serde_json::json!({
        "type": "response.output_text.delta", "content_index": 0, "delta": "hi", "item_id": "msg_1"
    });
    assert!(matches!(StreamEvent::from_json(&without_index), Err(FrameError::MissingField { field: "output_index" })));

    let without_item_id =
        serde_json::json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "hi"});
    assert!(StreamEvent::from_json(&without_item_id).is_ok(), "item_id is optional on the wire");
}
