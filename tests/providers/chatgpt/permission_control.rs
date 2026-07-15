//! ECS policy migration of the live ChatGPT permission-control regression.

use anyhow::Result;
use rig::{agent::stream_to_stdout, client::CompletionClient, streaming::StreamingPrompt};

use crate::{
    chatgpt::{LIVE_MODEL, live_client},
    support::{
        PermissionControlProbe, PermissionFile, ReadFileHead, ReadFileTail,
        assert_nonempty_response,
    },
};

const PROMPT: &str = "Use the available tools to read test.txt now. Do not ask any follow-up questions; just read the file and report its content.";

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn permission_control_prompt_example() -> Result<()> {
    let file = PermissionFile::new("chatgpt", "prompt")?;
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble("You are a helpful assistant that can read files using different methods.")
        .tool(ReadFileHead(file.path().to_path_buf()))
        .tool(ReadFileTail(file.path().to_path_buf()))
        .build();
    let probe = PermissionControlProbe::default();
    probe.install(&agent)?;
    assert_nonempty_response(&agent.prompt(PROMPT).max_turns(5).await?);
    probe.assert_completed();
    Ok(())
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn permission_control_streaming_example() -> Result<()> {
    let file = PermissionFile::new("chatgpt", "stream")?;
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble("You are a helpful assistant that can read files using different methods.")
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
