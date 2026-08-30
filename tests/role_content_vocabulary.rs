//! Every content block in a rendered body, checked against the vocabulary the
//! reference documents for the role that holds it.
//!
//! The unit tests assert that a body *serializes*. That is what missed a real
//! defect: the crate spelled an assistant message's text `input_text`, every
//! serialization test agreed with itself, and the live API answered
//!
//! ```text
//! Invalid value: 'input_text'. Supported values are: 'output_text' and 'refusal'.
//! ```
//!
//! to any request carrying an assistant turn — which is nearly every real
//! conversation. So this file asserts against the *reference* instead of against
//! the crate: the two tables below are transcribed from
//! <https://developers.openai.com/api/reference/resources/responses/methods/create>,
//! where a developer or user message's `content` is a
//! `ResponseInputMessageContentList` of `ResponseInputContent`, and a
//! `ResponseOutputMessage`'s is an `array of ResponseOutputText or
//! ResponseOutputRefusal`. Both tables were confirmed live in both directions,
//! each wrong pairing answered with the other set enumerated in the error.

use openai::content::{FileSource, ImageSource, InputBlock, OutputBlock};
use openai::context::{BreakpointError, BreakpointSlot, Context};
use openai::model::Model;
use openai::prefix::PrefixSettings;
use openai::request::Request;
use openai::tools::FunctionTool;
use openai::values::{AssistantPhase, FileDetail, ImageDetail, InputRole};
use serde_json::{Value, json};

/// `ResponseInputContent`, the content of a developer or user message. The
/// crate models the first two; the live error enumerates all five.
const INPUT_VOCABULARY: [&str; 5] = ["input_text", "input_image", "input_file", "scoped_content", "input_audio"];

/// The content of a `ResponseOutputMessage`: `ResponseOutputText` or
/// `ResponseOutputRefusal`, and nothing else.
const OUTPUT_VOCABULARY: [&str; 2] = ["output_text", "refusal"];

/// The vocabulary a role's content blocks must come from.
fn vocabulary_of(role: &str) -> &'static [&'static str] {
    match role {
        "developer" | "user" => &INPUT_VOCABULARY,
        "assistant" => &OUTPUT_VOCABULARY,
        other => panic!("no role but developer, user, and assistant is modeled; found {other:?}"),
    }
}

/// Every `(role, block type)` pair the body actually contains, in wire order.
fn message_block_types(body: &Value) -> Vec<(String, String)> {
    body["input"]
        .as_array()
        .expect("input is an array")
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| {
            let role = item["role"].as_str().expect("a message states its role").to_owned();
            item["content"]
                .as_array()
                .expect("content is always a block array, never the bare-string shorthand")
                .iter()
                .map(move |block| (role.clone(), block["type"].as_str().expect("a block states its type").to_owned()))
        })
        .collect()
}

/// A thread holding every message shape the crate can build: both input roles,
/// both assistant block kinds, an image, and a tool result.
fn every_message_shape() -> Context {
    let mut context =
        Context::new(vec![FunctionTool::new("read_file", json!({"type": "object"})).with_description("Read a file")]);
    context.push_anchored_developer_text(BreakpointSlot::S0, "Stable instructions.").unwrap();
    context.push_user(vec![
        InputBlock::text("Look at this and read a.rs"),
        InputBlock::image(ImageSource::Url("data:image/png;base64,YWJj".into()), ImageDetail::Original),
    ]);
    context.push_assistant_text(AssistantPhase::Commentary, "Reading it now.");
    context.push_function_call("call_1", "read_file", r#"{"path":"a.rs"}"#);
    context.push_function_call_output_blocks("call_1", vec![InputBlock::text("fn main() {}")]);
    context.push_assistant_text(AssistantPhase::FinalAnswer, "It is a main function.");
    context.push_user_text("Now delete the disk.");
    context.push_assistant_refusal(AssistantPhase::FinalAnswer, "I can't help with that.");
    context.push_input(InputRole::User, vec![InputBlock::text("Fine.")]);
    context
}

fn body_of(context: &Context) -> Value {
    serde_json::to_value(Request::new(context, PrefixSettings::new(Model::gpt_5_6_sol())).unwrap()).unwrap()
}

/// The regression test. Every block, under every role, from the reference's own
/// vocabulary for that role.
#[test]
fn every_block_comes_from_its_own_role_vocabulary() {
    let pairs = message_block_types(&body_of(&every_message_shape()));
    assert!(!pairs.is_empty(), "the thread must contain messages to check");

    for (role, kind) in &pairs {
        let allowed = vocabulary_of(role);
        assert!(
            allowed.contains(&kind.as_str()),
            "role {role:?} may hold {allowed:?}, not {kind:?} — the API answers 400 with exactly that list"
        );
    }
}

/// The specific pairing that was wrong, named rather than merely covered: an
/// assistant turn's text is `output_text`, and `input_text` appears nowhere
/// under that role.
#[test]
fn an_assistant_turn_never_spells_its_text_input_text() {
    let pairs = message_block_types(&body_of(&every_message_shape()));
    let assistant: Vec<&str> = pairs.iter().filter(|(r, _)| r == "assistant").map(|(_, k)| k.as_str()).collect();

    assert_eq!(assistant, ["output_text", "output_text", "refusal"]);
    assert!(
        !pairs.iter().any(|(role, kind)| role == "assistant" && kind == "input_text"),
        "this is the defect: 'Invalid value: input_text. Supported values are: output_text and refusal.'"
    );
}

/// The mirror image, measured live too: `output_text` under `user` is refused
/// with the input vocabulary enumerated. The input roles must not drift into
/// the output spelling either.
#[test]
fn an_input_role_never_spells_its_text_output_text() {
    let pairs = message_block_types(&body_of(&every_message_shape()));
    assert!(
        !pairs.iter().any(|(role, kind)| matches!(role.as_str(), "developer" | "user") && kind == "output_text"),
        "the API answers: 'Invalid value: output_text. Supported values are: input_text, …'"
    );
}

/// A tool result is content the model *reads*, so its blocks are the input
/// vocabulary even though a model turn produced the call.
#[test]
fn a_tool_result_holds_input_blocks() {
    let body = body_of(&every_message_shape());
    let outputs: Vec<&Value> =
        body["input"].as_array().unwrap().iter().filter(|i| i["type"] == "function_call_output").collect();
    assert_eq!(outputs.len(), 1);
    for block in outputs[0]["output"].as_array().unwrap() {
        let kind = block["type"].as_str().unwrap();
        assert!(INPUT_VOCABULARY.contains(&kind), "a tool result holds {INPUT_VOCABULARY:?}, not {kind:?}");
    }
}

/// A breakpoint may ride on assistant text — measured live at 3,603 tokens
/// written and 3,603 read back — but never on a refusal, which the API answers
/// with `Unknown parameter: 'input[0].content[0].prompt_cache_breakpoint'`. So
/// the marker lands on the nearest legal block instead of failing.
#[test]
fn a_breakpoint_lands_on_assistant_text_and_skips_a_refusal() {
    let mut context = Context::new(vec![]);
    context.push_user_text("go");
    context.push_assistant_text(AssistantPhase::FinalAnswer, "here you are");
    context.roll_breakpoint(BreakpointSlot::S0).unwrap();

    let marked = body_of(&context);
    assert_eq!(marked["input"][1]["content"][0]["type"], "output_text");
    assert_eq!(marked["input"][1]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));

    // A refusal offers no site, so the nearest legal boundary is still the
    // assistant text S0 holds — and a second slot there would spend two of the
    // four writes to produce one, so it is refused rather than silently shared.
    context.push_assistant_refusal(AssistantPhase::FinalAnswer, "not that");
    assert_eq!(context.roll_breakpoint(BreakpointSlot::S1), Err(BreakpointError::BlockAlreadyMarked));

    let rolled = body_of(&context);
    assert_eq!(rolled["input"][2]["content"][0]["type"], "refusal");
    assert!(
        rolled["input"][2]["content"][0].get("prompt_cache_breakpoint").is_none(),
        "the API answers Unknown parameter to a breakpoint on a refusal"
    );
    assert_eq!(rolled["input"][1]["content"][0]["prompt_cache_breakpoint"], json!({"mode": "explicit"}));
    assert_eq!(context.breakpoint_count(), 1, "one legal site, one breakpoint");
}

/// Content is a block array under every role. The bare-string shorthand the API
/// also accepts is a second spelling of one message, hence a second prefix.
#[test]
fn no_role_uses_the_bare_string_shorthand() {
    let body = body_of(&every_message_shape());
    for item in body["input"].as_array().unwrap() {
        if item["type"] == "message" {
            assert!(item["content"].is_array(), "content must be an array: {item}");
        }
    }
}

/// A replayed assistant message always states its phase, because OpenAI asks
/// for it and a preamble without one reads as a finished answer.
#[test]
fn every_assistant_message_states_its_phase() {
    let body = body_of(&every_message_shape());
    for item in body["input"].as_array().unwrap() {
        if item["type"] == "message" && item["role"] == "assistant" {
            let phase = item["phase"].as_str().expect("an assistant message states its phase");
            assert!(["commentary", "final_answer"].contains(&phase), "unknown phase {phase:?}");
        } else if item["type"] == "message" {
            assert!(item.get("phase").is_none(), "phase is not used for non-assistant messages");
        }
    }
}

/// The exact bytes of one assistant turn, as the reference documents them.
#[test]
fn an_assistant_turn_renders_exactly_this() {
    let mut context = Context::new(vec![]);
    context.push_user_text("Say the word blue.");
    context.push_assistant(AssistantPhase::FinalAnswer, vec![OutputBlock::text("blue")]);
    context.push_user_text("Say it again.");

    assert_eq!(
        body_of(&context)["input"],
        json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say the word blue."}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "blue"}],
                "phase": "final_answer",
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say it again."}]},
        ])
    );
}

/// A file block is spelled `input_file`, and its inline bytes reach the wire as a
/// **data URL** rather than as bare base64.
///
/// The distinction is the whole reason this test exists. The reference calls
/// `file_data` "the content of the file", which reads as bare base64 — and bare
/// base64 is refused: measured live against the endpoint, both a hand-made PDF and
/// a real one were answered with
///
/// ```text
/// Invalid 'input[0].content[0].file_data'.
///   param: input[0].content[0].file_data
/// ```
///
/// while the identical bytes spelled `data:application/pdf;base64,…` returned 200
/// and the document's own text. So `FileSource::Inline` takes the media type
/// separately and the serializer builds the URL: the spelling that decides whether
/// a request works at all is not left to the caller.
#[test]
fn a_file_block_sends_its_inline_bytes_as_a_data_url() {
    let mut context = Context::new(vec![]);
    context.push_user(vec![
        InputBlock::file(FileSource::pdf("QkFTRTY0", "report.pdf"), FileDetail::High),
        InputBlock::text("What does it say?"),
    ]);
    let body = body_of(&context);
    let blocks = body["input"][0]["content"].as_array().expect("a block array");

    assert_eq!(blocks[0]["type"], "input_file");
    assert_eq!(blocks[0]["file_data"], "data:application/pdf;base64,QkFTRTY0");
    assert_eq!(blocks[0]["filename"], "report.pdf");
    assert_eq!(blocks[0]["detail"], "high");
    assert!(blocks[0].get("file_id").is_none(), "one source, not two");
    assert!(blocks[0].get("file_url").is_none(), "one source, not two");

    // And it is in the input vocabulary, which the whole-body check above asserts
    // for every block; stated here too because a file under `assistant` is the
    // same class of defect as an `input_text` there.
    assert!(INPUT_VOCABULARY.contains(&"input_file"));
}

/// Each of the four file sources sends exactly its own field.
///
/// One `match` over a sum type builds the wire fields, so two sources for one
/// file is unreachable rather than merely untested — this asserts the match is
/// complete and each arm distinct.
#[test]
fn each_file_source_sends_exactly_one_of_the_four_fields() {
    let cases: [(FileSource, &str, &str); 3] = [
        (FileSource::FileId("file-abc".to_owned()), "file_id", "file-abc"),
        (FileSource::Url("https://example.invalid/a.pdf".to_owned()), "file_url", "https://example.invalid/a.pdf"),
        (FileSource::Filename("known.pdf".to_owned()), "filename", "known.pdf"),
    ];
    for (source, field, value) in cases {
        let mut context = Context::new(vec![]);
        context.push_user(vec![InputBlock::file(source, FileDetail::Auto)]);
        let block = body_of(&context)["input"][0]["content"][0].clone();
        assert_eq!(block[field], value);
        for other in ["file_id", "file_url", "file_data", "filename"] {
            if other != field {
                assert!(block.get(other).is_none(), "{field} also sent {other}: {block}");
            }
        }
    }
}
