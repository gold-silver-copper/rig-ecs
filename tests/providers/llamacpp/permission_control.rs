//! ECS policy migration of the live llama.cpp permission-control regression.

use anyhow::Result;
use rig::{agent::stream_to_stdout, client::CompletionClient, streaming::StreamingPrompt};

use super::support;
use crate::support::{
    PermissionControlProbe, PermissionFile, ReadFileHead, ReadFileTail, assert_nonempty_response,
};

const PROMPT: &str = "Use the available tools to read test.txt now. If read_file_head is unavailable, use read_file_tail and report the content.";

async fn run(streaming: bool) -> Result<()> {
    let file = PermissionFile::new("llamacpp", if streaming { "stream" } else { "prompt" })?;
    let agent = support::completions_client()
        .agent(support::model_name())
        .preamble("You are a helpful assistant that can read files using different methods.")
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
#[ignore = "requires a local llama.cpp server"]
async fn permission_control_prompt_example() -> Result<()> {
    run(false).await
}

#[tokio::test]
#[ignore = "requires a local llama.cpp server"]
async fn permission_control_streaming_example() -> Result<()> {
    run(true).await
}
