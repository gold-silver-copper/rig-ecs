//! ECS policy migration of the former permission-control hook regression.

use anyhow::Result;
use rig::{
    agent::stream_to_stdout, client::CompletionClient, providers, streaming::StreamingPrompt,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};

use super::super::support::with_openai_cassette_result;
use crate::support::{PermissionControlProbe, assert_nonempty_response};

const TEST_CONTENT: &str = "hello world\n";

struct FileCleanup(PathBuf);

impl FileCleanup {
    fn new(test_name: &str) -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "rig-openai-permission-{test_name}-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, TEST_CONTENT)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for FileCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Deserialize)]
struct ReadFileArgs {}

#[derive(Debug, thiserror::Error)]
#[error("file operation failed")]
struct FileError;

macro_rules! file_tool {
    ($name:ident, $tool_name:literal, $command:literal) => {
        #[derive(Deserialize, Serialize)]
        struct $name(PathBuf);

        impl Tool for $name {
            const NAME: &'static str = $tool_name;
            type Error = FileError;
            type Args = ReadFileArgs;
            type Output = String;

            fn description(&self) -> String {
                if $command == "head" {
                    "Read the first line of test.txt using the head command".to_owned()
                } else {
                    "Read the last line of test.txt using the tail command".to_owned()
                }
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object", "properties": {}})
            }

            async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
                let output = std::process::Command::new($command)
                    .arg("-1")
                    .arg(&self.0)
                    .output()
                    .map_err(|_| FileError)?;
                if !output.status.success() {
                    return Err(FileError);
                }
                Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
            }
        }
    };
}

file_tool!(ReadFileHead, "read_file_head", "head");
file_tool!(ReadFileTail, "read_file_tail", "tail");

fn build_agent(
    client: &providers::openai::Client,
    path: &Path,
) -> rig::agent::Agent<impl rig::completion::CompletionModel + use<>> {
    client
        .agent(providers::openai::GPT_4O_MINI)
        .preamble("You are a helpful assistant that can read files using different methods.")
        .tool(ReadFileHead(path.to_path_buf()))
        .tool(ReadFileTail(path.to_path_buf()))
        .build()
}

const PROMPT: &str = "Use the available tools to read test.txt now. Do not ask any follow-up questions; just read the file and report its content.";

#[tokio::test]
async fn permission_control_prompt_example() -> Result<()> {
    with_openai_cassette_result(
        "permission_control/permission_control_prompt_example",
        |client| async move {
            let cleanup = FileCleanup::new("blocking")?;
            let agent = build_agent(&client, cleanup.path());
            let probe = PermissionControlProbe::default();
            probe.install(&agent)?;
            let response = agent.prompt(PROMPT).max_turns(5).await?;
            assert_nonempty_response(&response);
            probe.assert_completed();
            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn permission_control_streaming_example() -> Result<()> {
    with_openai_cassette_result(
        "permission_control/permission_control_streaming_example",
        |client| async move {
            let cleanup = FileCleanup::new("streaming")?;
            let agent = build_agent(&client, cleanup.path());
            let probe = PermissionControlProbe::default();
            probe.install(&agent)?;
            let mut stream = agent.stream_prompt(PROMPT).max_turns(5).await;
            let response = stream_to_stdout(&mut stream).await?;
            assert_nonempty_response(response.output());
            probe.assert_completed();
            Ok(())
        },
    )
    .await
}
