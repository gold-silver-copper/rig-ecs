//! ECS request-policy migration of the per-turn override regression.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rig::{
    bevy_ecs::{
        observer::{Observer, On},
        prelude::Commands,
    },
    client::CompletionClient,
    providers::anthropic,
    runtime::{
        CompletionRequestPrepared, ModelToolChoice, PolicyPoint, PolicyRule, PolicyStatus,
        RequestPatch, RequestPolicyDecision, RequestPolicyInvocation,
    },
    streaming::StreamingPrompt,
    tool::Tool,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;
use crate::support::{collect_stream_final_response, install_policy};

const PROMPT: &str = "I'm planning a trip to Paris. Use a tool to help me prepare.";
const PREAMBLE: &str =
    "You are a travel assistant. Use the available tools to gather information before answering.";

#[derive(Deserialize)]
struct WeatherArgs {
    location: String,
}
#[derive(Deserialize)]
struct TimeArgs {
    #[allow(dead_code)]
    timezone: String,
}
#[derive(Debug, thiserror::Error)]
#[error("tool error")]
struct ToolErr;

#[derive(Clone, Default)]
struct GetWeather {
    calls: Arc<AtomicUsize>,
}

impl Tool for GetWeather {
    const NAME: &'static str = "get_weather";
    type Error = ToolErr;
    type Args = WeatherArgs;
    type Output = String;
    fn description(&self) -> String {
        "Get the current weather for a location.".to_owned()
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"location":{"type":"string","description":"City name"}},"required":["location"]})
    }
    async fn call(&self, args: Self::Args) -> Result<String, ToolErr> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "It is 18 degrees Celsius and clear in {}.",
            args.location
        ))
    }
}

#[derive(Clone, Default)]
struct GetTime;
impl Tool for GetTime {
    const NAME: &'static str = "get_time";
    type Error = ToolErr;
    type Args = TimeArgs;
    type Output = String;
    fn description(&self) -> String {
        "Get the current time in a timezone.".to_owned()
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"timezone":{"type":"string","description":"IANA timezone"}},"required":["timezone"]})
    }
    async fn call(&self, _args: Self::Args) -> Result<String, ToolErr> {
        Ok("12:00".to_owned())
    }
}

fn patch_first_turn(mut event: On<RequestPolicyInvocation>) {
    event.decision = Some(RequestPolicyDecision::Patch(
        RequestPatch::new()
            .active_tools([GetWeather::NAME])
            .tool_choice(ModelToolChoice::Required),
    ));
}

#[derive(Deserialize)]
struct Interaction {
    when: RecordedRequest,
}
#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

fn assert_first_request_was_overridden(scenario: &str) {
    let contents =
        std::fs::read_to_string(crate::cassettes::cassette_path("anthropic", scenario)).unwrap();
    let body = serde_yaml::Deserializer::from_str(&contents)
        .filter_map(|doc| Interaction::deserialize(doc).ok())
        .find_map(|item| item.when.body)
        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        .unwrap();
    let tools = body["tools"].as_array().unwrap();
    assert!(tools.iter().any(|tool| tool["name"] == GetWeather::NAME));
    assert!(!tools.iter().any(|tool| tool["name"] == GetTime::NAME));
    assert_ne!(body["tool_choice"]["type"], "auto");
}

async fn run(client: anthropic::Client, streaming: bool, weather: GetWeather) {
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(PREAMBLE)
        .tool(weather)
        .tool(GetTime)
        .build();
    let policy = install_policy(
        &agent,
        "first-turn-weather",
        0,
        1,
        PolicyRule::Custom(PolicyPoint::Request),
    )
    .unwrap();
    agent
        .with_runtime_mut(|runtime| {
            runtime
                .world_mut()
                .entity_mut(policy)
                .observe(patch_first_turn);
            runtime.world_mut().spawn(Observer::new(
                move |_event: On<CompletionRequestPrepared>, mut commands: Commands| {
                    commands.entity(policy).insert(PolicyStatus::Retired);
                },
            ));
        })
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

#[tokio::test]
async fn request_overridden_by_hook_blocking() {
    let weather = GetWeather::default();
    let probe = weather.clone();
    let scenario = "request_override/request_overridden_by_hook_blocking";
    with_anthropic_cassette(
        "request_override/request_overridden_by_hook_blocking",
        move |client| run(client, false, weather),
    )
    .await;
    assert!(probe.calls.load(Ordering::SeqCst) >= 1);
    assert_first_request_was_overridden(scenario);
}

#[tokio::test]
async fn request_overridden_by_hook_streaming() {
    let weather = GetWeather::default();
    let probe = weather.clone();
    let scenario = "request_override/request_overridden_by_hook_streaming";
    with_anthropic_cassette(
        "request_override/request_overridden_by_hook_streaming",
        move |client| run(client, true, weather),
    )
    .await;
    assert!(probe.calls.load(Ordering::SeqCst) >= 1);
    assert_first_request_was_overridden(scenario);
}
