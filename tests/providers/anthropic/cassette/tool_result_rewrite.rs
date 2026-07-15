//! ECS tool-result policy migration of the redaction regression.

use super::super::support::with_anthropic_cassette;
use crate::support::{collect_stream_final_response, install_policy};
use rig::{
    client::CompletionClient, providers::anthropic, runtime::PolicyRule,
    streaming::StreamingPrompt, tool::Tool,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};

const PROMPT: &str =
    "Look up user u-42 with the get_user_record tool and tell me their account status.";
const SECRET: &str = "123-45-6789";
const REDACTED: &str = "name=Alice; ssn=[REDACTED]; status=active";
#[derive(Deserialize)]
struct LookupArgs {
    #[allow(dead_code)]
    user_id: String,
}
#[derive(Debug, thiserror::Error)]
#[error("lookup error")]
struct LookupError;
#[derive(Clone, Default)]
struct GetUserRecord {
    raw: Arc<Mutex<Vec<String>>>,
}
impl Tool for GetUserRecord {
    const NAME: &'static str = "get_user_record";
    type Error = LookupError;
    type Args = LookupArgs;
    type Output = String;
    fn description(&self) -> String {
        "Look up a user record by id.".to_owned()
    }
    fn parameters(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"user_id":{"type":"string","description":"The user id, e.g. 'u-42'"}},"required":["user_id"]})
    }
    async fn call(&self, _args: Self::Args) -> Result<String, LookupError> {
        let value = format!("name=Alice; ssn={SECRET}; status=active");
        self.raw.lock().unwrap().push(value.clone());
        Ok(value)
    }
}
async fn run(client: anthropic::Client, streaming: bool, tool: GetUserRecord) {
    let agent=client.agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble("You are a support agent. Use the get_user_record tool to look up a user, then report their account status.")
            .tool(tool).build();
    install_policy(
        &agent,
        "redact-ssn",
        0,
        1,
        PolicyRule::RewriteToolResult {
            tool: Some("get_user_record".to_owned()),
            presentation: REDACTED.to_owned(),
        },
    )
    .unwrap();
    let answer = if streaming {
        let mut stream = agent.stream_prompt(PROMPT).max_turns(5).await;
        collect_stream_final_response(&mut stream).await.unwrap()
    } else {
        agent.prompt(PROMPT).max_turns(5).await.unwrap()
    };
    assert!(!answer.is_empty());
    assert!(!answer.contains(SECRET));
}
fn produced_secret(tool: &GetUserRecord) -> bool {
    tool.raw
        .lock()
        .unwrap()
        .iter()
        .any(|value| value.contains(SECRET))
}
#[tokio::test]
async fn tool_result_redacted_by_hook_blocking() {
    let tool = GetUserRecord::default();
    let probe = tool.clone();
    with_anthropic_cassette(
        "tool_result_rewrite/tool_result_redacted_by_hook_blocking",
        move |client| run(client, false, tool),
    )
    .await;
    assert!(produced_secret(&probe));
}
#[tokio::test]
async fn tool_result_redacted_by_hook_streaming() {
    let tool = GetUserRecord::default();
    let probe = tool.clone();
    with_anthropic_cassette(
        "tool_result_rewrite/tool_result_redacted_by_hook_streaming",
        move |client| run(client, true, tool),
    )
    .await;
    assert!(produced_secret(&probe));
}
