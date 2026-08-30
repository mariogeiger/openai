//! Prints the body of a two-turn conversation, for posting to a live endpoint.
//!
//! Not a test: it takes no credential and reaches no network. It exists so the
//! bytes a live check sends are the crate's own bytes rather than a hand-written
//! imitation of them.

use openai::context::Context;
use openai::model::Model;
use openai::prefix::PrefixSettings;
use openai::request::Request;
use openai::values::AssistantPhase;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::new(vec![]);
    context.push_user_text("Say the word blue.");
    context.push_assistant_text(AssistantPhase::FinalAnswer, "blue");
    context.push_user_text("Say it again.");

    let request =
        Request::new(&context, PrefixSettings::new(Model::gpt_5_6_sol()))?.without_streaming().without_storage();
    println!("{}", serde_json::to_string(&request)?);
    Ok(())
}
