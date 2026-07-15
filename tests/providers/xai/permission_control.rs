//! ECS policy migration of the xAI permission-control regression.

use anyhow::Result;
use rig::{
    agent::stream_to_stdout, client::CompletionClient, providers::xai, streaming::StreamingPrompt,
};

use super::support::with_xai_cassette_result;
use crate::support::{
    PermissionControlProbe, PermissionFile, ReadFileHead, ReadFileTail, assert_nonempty_response,
};

const PROMPT: &str = "Use the available tools to read test.txt now. Do not ask any follow-up questions; just read the file and report its content.";
const PREAMBLE: &str = "You are a helpful assistant that can read files using different methods.";

async fn run(client: xai::Client, streaming: bool) -> Result<()> {
    let file = PermissionFile::new("xai", if streaming { "stream" } else { "prompt" })?;
    let agent = client
        .agent(xai::GROK_4)
        .preamble(PREAMBLE)
        .tool(ReadFileHead(file.path().to_path_buf()))
        .tool(ReadFileTail(file.path().to_path_buf()))
        .build();
    let probe = PermissionControlProbe::default();
    probe.install(&agent)?;
    if streaming {
        let mut stream = agent.stream_prompt(PROMPT).max_turns(5).await;
        assert_nonempty_response(stream_to_stdout(&mut stream).await?.output());
    } else {
        assert_nonempty_response(&agent.prompt(PROMPT).max_turns(5).await?);
    }
    probe.assert_completed();
    Ok(())
}

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_xai_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| run(client, false),
    )
    .await
}

#[tokio::test]
async fn permission_control_streaming_example() -> Result<()> {
    with_xai_cassette_result(
        "permission_control/permission_control_streaming_example",
        |client| run(client, true),
    )
    .await
}
