//! ECS tool-call policy migration of the argument-rewrite regression.

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
    "What is the weather in Tokyo right now? Use the get_weather tool to find out.";
#[derive(Clone, Debug, PartialEq)]
struct ObservedCall {
    location: String,
    units: Option<String>,
}
#[derive(Deserialize)]
struct WeatherArgs {
    location: String,
    #[serde(default)]
    units: Option<String>,
}
#[derive(Debug, thiserror::Error)]
#[error("weather error")]
struct WeatherError;
#[derive(Clone, Default)]
struct GetWeather {
    calls: Arc<Mutex<Vec<ObservedCall>>>,
}
impl Tool for GetWeather {
    const NAME: &'static str = "get_weather";
    type Error = WeatherError;
    type Args = WeatherArgs;
    type Output = String;
    fn description(&self) -> String {
        "Get the current weather for a location.".to_owned()
    }
    fn parameters(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"location":{"type":"string","description":"City to get the weather for, e.g. 'Tokyo'"}},"required":["location"]})
    }
    async fn call(&self, args: Self::Args) -> Result<String, WeatherError> {
        self.calls.lock().unwrap().push(ObservedCall {
            location: args.location.clone(),
            units: args.units.clone(),
        });
        Ok(format!(
            "It is 18 degrees Celsius and sunny in {}.",
            args.location
        ))
    }
}

async fn run(client: anthropic::Client, streaming: bool, tool: GetWeather) {
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble("You are a weather assistant. Always use the get_weather tool to answer.")
        .tool(tool)
        .build();
    install_policy(
        &agent,
        "pin-celsius",
        0,
        1,
        PolicyRule::RewriteToolArguments {
            tool: Some("get_weather".to_owned()),
            arguments: json!({"location":"Tokyo","units":"celsius"}),
        },
    )
    .unwrap();
    if streaming {
        let mut stream = agent.stream_prompt(PROMPT).max_turns(5).await;
        assert!(
            !collect_stream_final_response(&mut stream)
                .await
                .unwrap()
                .is_empty()
        );
    } else {
        assert!(!agent.prompt(PROMPT).max_turns(5).await.unwrap().is_empty());
    }
}

fn assert_injected(tool: &GetWeather) {
    let calls = tool.calls.lock().unwrap();
    assert!(!calls.is_empty());
    assert!(
        calls
            .iter()
            .all(|call| call.units.as_deref() == Some("celsius") && call.location == "Tokyo")
    );
}
#[tokio::test]
async fn tool_call_args_rewritten_by_hook_blocking() {
    let tool = GetWeather::default();
    let probe = tool.clone();
    with_anthropic_cassette(
        "tool_call_rewrite_args/tool_call_args_rewritten_by_hook_blocking",
        move |client| run(client, false, tool),
    )
    .await;
    assert_injected(&probe);
}
#[tokio::test]
async fn tool_call_args_rewritten_by_hook_streaming() {
    let tool = GetWeather::default();
    let probe = tool.clone();
    with_anthropic_cassette(
        "tool_call_rewrite_args/tool_call_args_rewritten_by_hook_streaming",
        move |client| run(client, true, tool),
    )
    .await;
    assert_injected(&probe);
}
