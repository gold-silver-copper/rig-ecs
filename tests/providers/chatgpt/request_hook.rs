//! ECS lifecycle-observer migration of the live ChatGPT request-hook regression.

use anyhow::Result;
use rig::client::CompletionClient;

use crate::{
    chatgpt::{LIVE_MODEL, live_client},
    support::{RequestLifecycleProbe, assert_nonempty_response},
};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();
    let probe = RequestLifecycleProbe::new("abc123");
    probe.install(&agent)?;
    let response = agent.prompt("Entertain me!").await?;
    assert_nonempty_response(&response);
    probe.assert_observed("Entertain me!");
    Ok(())
}
