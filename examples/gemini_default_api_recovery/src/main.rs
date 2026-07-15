//! Live streaming canary for Gemini's legacy `default_api` tool-name emission.
//!
//! Requires `GEMINI_API_KEY`. The ECS invalid-call policy snapshots the
//! `JavaScript` capability and repairs `default_api` before execution. A
//! separate observation-only listener records the provider emission without
//! participating in steering.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use std::{
    env,
    sync::{Arc, Mutex},
};

use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, PromptResponse, StreamingResult};
use rig::bevy_ecs::{observer::On, prelude::World};
use rig::client::{CompletionClient, ProviderClient};
use rig::message::ToolResultContent;
use rig::providers::gemini::{
    self,
    completion::gemini_api_types::{AdditionalParameters, GenerationConfig, ThinkingConfig},
};
use rig::runtime::{InvalidToolCallDetected, Policy, PolicyRule, RunState, StableId, TenantId};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt};
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

const MODEL: &str = "gemini-3.1-pro-preview";
const ATTEMPTS_ENV: &str = "RIG_GEMINI_DEFAULT_API_CANARY_ATTEMPTS";
const MARKER: &str = "workspace-canary-marker";

const PREAMBLE: &str = r#"
You are a workspace assistant. The JavaScript execution runtime exposes an
async Workspace.listCollection(id, depth) method. Legacy clients may refer to
that same runtime as default_api. Use the runtime when asked to inspect data,
then summarize the returned collection.
"#;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct JavaScriptProgram {
    title: String,
    description: String,
    code: String,
}

#[repr(transparent)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecutorResponse(Result<serde_json::Value, String>);

#[derive(Debug, thiserror::Error)]
#[error("JavaScript execution failed")]
struct JavaScriptToolError;

#[derive(Clone)]
struct JavaScript;

impl Tool for JavaScript {
    const NAME: &'static str = "JavaScript";
    type Error = JavaScriptToolError;
    type Args = JavaScriptProgram;
    type Output = ExecutorResponse;

    fn description(&self) -> String {
        "Execute workspace JavaScript; legacy agents may call this default_api".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        schema_for!(JavaScriptProgram).to_value()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(ExecutorResponse(Ok(json!({
            "id": "collection-canary-id",
            "title": "Canary Collection",
            "records": [{"id": "record-canary", "title": MARKER}],
            "receivedCode": args.code
        }))))
    }
}

#[derive(Default)]
struct Observation {
    invalid_names: Arc<Mutex<Vec<String>>>,
    tool_calls: Vec<String>,
    tool_results: Vec<String>,
    streamed_text: String,
    final_response: Option<PromptResponse>,
}

fn install_repair(
    world: &mut World,
    agent: rig::runtime::AgentHandle,
    invalid_names: Arc<Mutex<Vec<String>>>,
) -> anyhow::Result<()> {
    world.add_observer(move |event: On<InvalidToolCallDetected>| {
        if let Ok(mut names) = invalid_names.lock() {
            names.push(event.invalid.call.name.clone());
        }
    });
    // Spawn through Runtime below; this helper only installs observation.
    let _ = agent;
    Ok(())
}

fn additional_params() -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::to_value(AdditionalParameters {
        generation_config: Some(GenerationConfig {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_budget: Some(16_384),
                thinking_level: None,
            }),
            ..Default::default()
        }),
        additional_params: None,
    })?)
}

fn attempts() -> usize {
    env::var(ATTEMPTS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|attempts| *attempts > 0)
        .unwrap_or(2)
}

fn prompt(attempt: usize) -> String {
    format!(
        "Inspect collection `collection-canary-id`. Intentionally call the execution runtime by its legacy structured tool name `default_api`. Execute `async function inspect() {{ return await Workspace.listCollection(\"collection-canary-id\", 1); }} inspect();`, then answer in one sentence containing `{MARKER}`. Attempt {attempt}."
    )
}

fn demonstrate_invalid_retry() -> anyhow::Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.spawn_policy(
        ecs_demo::id("retry-legacy-tool")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::RetryInvalidTool {
                tool: Some("missing_workspace_api".to_owned()),
                feedback: "Use an advertised tool name on the next model call".to_owned(),
            },
        },
        agent,
    )?;
    let pending = runtime
        .handle()
        .prompt(agent, "Demonstrate invalid retry")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_with_tool(&runtime, &model, "missing_workspace_api")?;
    let retry_model = ecs_demo::next_effect(&mut runtime)?;
    let feedback = retry_model
        .model_input()
        .and_then(|input| input.tool_results.first())
        .ok_or_else(|| anyhow::anyhow!("invalid retry did not reach the next model operation"))?;
    println!("invalid-call retry feedback: {}", feedback.presentation);
    ecs_demo::complete_text(&runtime, &retry_model, "retry recovered")?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .ok_or_else(|| anyhow::anyhow!("retry prompt was not admitted"))?;
    anyhow::ensure!(
        matches!(runtime.observe_run(run)?, Some(RunState::Completed(_))),
        "invalid retry did not complete"
    );
    Ok(())
}

async fn consume(
    mut stream: StreamingResult<gemini::streaming::StreamingCompletionResponse>,
    invalid_names: Arc<Mutex<Vec<String>>>,
) -> anyhow::Result<Observation> {
    let mut observation = Observation {
        invalid_names,
        ..Observation::default()
    };
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                observation.streamed_text.push_str(&text.text);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            }) => {
                observation.tool_calls.push(tool_call.function.name);
            }
            MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            }) => match tool_result.content.first() {
                ToolResultContent::Json { value } => {
                    observation.tool_results.push(value.to_string());
                }
                ToolResultContent::Text(text) => {
                    observation.tool_results.push(text.text.clone());
                }
                ToolResultContent::Image(_) => {}
            },
            MultiTurnStreamItem::FinalResponse(response) => {
                observation.final_response = Some(response);
                break;
            }
            _ => {}
        }
    }
    Ok(observation)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    demonstrate_invalid_retry()?;
    let mut repaired = false;
    for attempt in 1..=attempts() {
        let agent = gemini::Client::from_env()?
            .agent(MODEL)
            .name(format!("default-api-canary-{attempt}"))
            .preamble(PREAMBLE)
            .additional_params(additional_params()?)
            .tool(JavaScript)
            .default_max_turns(8)
            .temperature(0.0)
            .build();
        let invalid_names = Arc::new(Mutex::new(Vec::new()));
        let observed_names = Arc::clone(&invalid_names);
        agent.with_runtime_mut(|runtime| {
            install_repair(runtime.world_mut(), agent.handle(), observed_names)?;
            runtime.spawn_policy(
                StableId::new("repair-default-api")?,
                TenantId::new("local")?,
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::RepairInvalidTool {
                        from: Some("default_api".to_owned()),
                        to: JavaScript::NAME.to_owned(),
                    },
                },
                agent.handle(),
            )?;
            Ok::<_, anyhow::Error>(())
        })??;

        let stream = agent
            .stream_prompt(prompt(attempt))
            .history(Vec::<rig::message::Message>::new())
            .await;
        let observation = consume(stream, invalid_names).await?;
        let invalid = observation
            .invalid_names
            .lock()
            .map(|names| names.clone())
            .unwrap_or_default();
        repaired |= invalid.iter().any(|name| name == "default_api");
        anyhow::ensure!(
            observation
                .tool_calls
                .iter()
                .all(|name| name != "default_api"),
            "repaired name leaked to the public tool-call stream"
        );
        anyhow::ensure!(
            observation
                .tool_calls
                .iter()
                .any(|name| name == JavaScript::NAME),
            "Gemini never called the repaired JavaScript capability"
        );
        anyhow::ensure!(
            observation
                .tool_results
                .iter()
                .all(|result| !result.contains("ToolNotFoundError: default_api")),
            "legacy name reached execution"
        );
        let final_response = observation
            .final_response
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("stream ended without a final response"))?;
        anyhow::ensure!(
            final_response.output().contains(MARKER),
            "final response omitted the tool marker"
        );
        println!(
            "attempt {attempt}: invalid={invalid:?}, tools={:?}, final={}",
            observation.tool_calls,
            final_response.output()
        );
    }
    anyhow::ensure!(
        repaired,
        "no attempt emitted default_api; increase {ATTEMPTS_ENV}"
    );
    Ok(())
}
