//! ECS policy migration of the Copilot permission-control regression.

use anyhow::Result;
use rig::{agent::stream_to_stdout, client::CompletionClient, streaming::StreamingPrompt};

use crate::{
    copilot::{LIVE_LIGHT_MODEL, live_client, with_copilot_cassette_result},
    support::{
        PermissionControlProbe, PermissionFile, ReadFileHead, ReadFileTail,
        assert_nonempty_response,
    },
};

const PROMPT: &str = "Use the available tools to read test.txt now. Do not ask any follow-up questions; just read the file and report its content.";
const PREAMBLE: &str = "You are a helpful assistant that can read files using different methods.";

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_copilot_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let file = PermissionFile::new("copilot", "blocking")?;
            let agent = client
                .agent(LIVE_LIGHT_MODEL)
                .preamble(PREAMBLE)
                .tool(ReadFileHead(file.path().to_path_buf()))
                .tool(ReadFileTail(file.path().to_path_buf()))
                .build();
            let probe = PermissionControlProbe::default();
            probe.install(&agent)?;
            assert_nonempty_response(&agent.prompt(PROMPT).max_turns(5).await?);
            probe.assert_completed();
            Ok(())
        },
    )
    .await
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn permission_control_streaming_example() -> Result<()> {
    let file = PermissionFile::new("copilot", "streaming")?;
    let agent = live_client()
        .agent(LIVE_LIGHT_MODEL)
        .preamble(PREAMBLE)
        .tool(ReadFileHead(file.path().to_path_buf()))
        .tool(ReadFileTail(file.path().to_path_buf()))
        .build();
    let probe = PermissionControlProbe::default();
    probe.install(&agent)?;
    let mut stream = agent.stream_prompt(PROMPT).max_turns(5).await;
    assert_nonempty_response(stream_to_stdout(&mut stream).await?.output());
    probe.assert_completed();
    Ok(())
}
