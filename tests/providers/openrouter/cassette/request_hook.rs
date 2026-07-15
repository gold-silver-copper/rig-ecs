//! ECS lifecycle-observer migration of the OpenRouter request-hook regression.

use anyhow::Result;
use rig::client::CompletionClient;

use super::super::{DEFAULT_MODEL, support::with_openrouter_cassette_result};
use crate::support::{RequestLifecycleProbe, assert_nonempty_response};

#[tokio::test]
async fn request_hook_records_prompt_and_response() -> Result<()> {
    with_openrouter_cassette_result(
        "request_hook/request_hook_records_prompt_and_response",
        |client| async move {
            let agent = client
                .agent(DEFAULT_MODEL)
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
