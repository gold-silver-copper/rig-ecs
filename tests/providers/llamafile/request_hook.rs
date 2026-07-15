//! ECS lifecycle-observer migration of the live llamafile request-hook regression.

use anyhow::Result;
use rig::client::CompletionClient;

use super::support;
use crate::support::{RequestLifecycleProbe, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }
    let agent = support::client()
        .agent(support::model_name())
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();
    let probe = RequestLifecycleProbe::new("abc123");
    probe.install(&agent)?;
    let response = agent.prompt("Entertain me!").await?;
    assert_nonempty_response(&response);
    probe.assert_observed("Entertain me!");
    Ok(())
}
