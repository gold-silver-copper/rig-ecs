//! ECS lifecycle-observer migration of the live Groq request-hook regression.

use anyhow::Result;
use rig::{
    client::{CompletionClient, ProviderClient},
    providers::groq,
};

use super::AGENT_MODEL;
use crate::support::{RequestLifecycleProbe, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    let agent = groq::Client::from_env()?
        .agent(AGENT_MODEL)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();
    let probe = RequestLifecycleProbe::new("abc123");
    probe.install(&agent)?;
    let response = agent.prompt("Entertain me!").await?;
    assert_nonempty_response(&response);
    probe.assert_observed("Entertain me!");
    Ok(())
}
