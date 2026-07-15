//! ECS lifecycle-observer migration of the xAI request-hook regression.

use anyhow::Result;
use rig::{client::CompletionClient, providers::xai};

use super::support::with_xai_cassette_result;
use crate::support::{RequestLifecycleProbe, assert_nonempty_response};

#[tokio::test]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    with_xai_cassette_result(
        "request_hook/request_hook_records_prompt_and_response",
        |client| async move {
            let agent = client
                .agent(xai::GROK_3_MINI)
                .preamble("You are a comedian here to entertain the user using humour and jokes.")
                .build();
            let probe = RequestLifecycleProbe::new("abc123");
            probe.install(&agent)?;
            let response = agent.prompt("Entertain me!").await?;
            assert_nonempty_response(&response);
            probe.assert_observed("Entertain me!");
            Ok(())
        },
    )
    .await
}
