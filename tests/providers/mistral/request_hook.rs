//! ECS lifecycle-observer migration of the live Mistral request-hook regression.

use anyhow::Result;
use rig::{
    client::{CompletionClient, ProviderClient},
    providers::mistral,
};

use super::DEFAULT_MODEL;
use crate::support::{RequestLifecycleProbe, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    let agent = mistral::Client::from_env()?
        .agent(DEFAULT_MODEL)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();
    let probe = RequestLifecycleProbe::new("abc123");
    probe.install(&agent)?;
    let response = agent.prompt("Entertain me!").await?;
    assert_nonempty_response(&response);
    probe.assert_observed("Entertain me!");
    Ok(())
}
