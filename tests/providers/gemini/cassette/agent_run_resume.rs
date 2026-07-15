//! Stable-ID ECS checkpoint and recovery against real Gemini turns.
//!
//! The first test crosses a fresh facade/runtime boundary while a tool effect
//! is live. Invalid-call checkpoint internals are covered exhaustively by the
//! runtime snapshot suite; the provider tests here bridge those states through
//! the same Gemini adapter and policy behavior.

use std::sync::Arc;

use rig::bevy_ecs::entity::Entity;
use rig::bevy_ecs::relationship::Relationship;
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::runtime::{PauseMode, PolicyRule, RunOf, RunState};
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Notify;

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, history_has_assistant_tool_call, tool_result_texts,
};
use super::super::support::with_gemini_cassette;
use crate::support::{assert_mentions_expected_number, assert_nonempty_response, install_policy};

const SKIP_REASON: &str = "The add tool is disabled for this request.";
const RETRY_FEEDBACK: &str = "Tools are temporarily unavailable. Answer the question directly in plain text without calling any tools.";

#[derive(Deserialize)]
struct AddArgs {
    x: i64,
    y: i64,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
struct MathError;

#[derive(Default)]
struct ToolGate {
    started: Notify,
}

struct CheckpointAdd {
    gate: Option<Arc<ToolGate>>,
}

impl CheckpointAdd {
    fn gated(gate: Arc<ToolGate>) -> Self {
        Self { gate: Some(gate) }
    }

    fn immediate() -> Self {
        Self { gate: None }
    }
}

impl Tool for CheckpointAdd {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = AddArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y together".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first operand" },
                "y": { "type": "number", "description": "The second operand" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Some(gate) = &self.gate {
            gate.started.notify_one();
            std::future::pending::<()>().await;
        }
        Ok(args.x + args.y)
    }
}

fn active_waiting_tool_run(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
) -> Entity {
    let agent_entity = agent.handle().entity();
    agent
        .with_runtime_mut(move |runtime| {
            let world = runtime.world_mut();
            let mut runs = world.query::<(Entity, &RunOf, &RunState)>();
            runs.iter(world)
                .find_map(|(entity, run_of, state)| {
                    (run_of.get() == agent_entity && matches!(state, RunState::WaitingTools { .. }))
                        .then_some(entity)
                })
                .expect("one owned run should be waiting on the gated tool")
        })
        .expect("runtime access should succeed")
}

fn set_invalid_retry_budget(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    max_retries: u32,
) {
    let handle = agent.handle();
    agent
        .with_runtime_mut(move |runtime| runtime.set_invalid_tool_call_budget(handle, max_retries))
        .expect("runtime access should succeed")
        .expect("invalid-tool budget should be configured");
}

#[tokio::test]
async fn resume_from_serialized_state_mid_tool_execution() {
    with_gemini_cassette(
        "agent_run_resume/resume_from_serialized_state_mid_tool_execution",
        |client| async move {
            let gate = Arc::new(ToolGate::default());
            let source = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(CheckpointAdd::gated(gate.clone()))
                .build();
            let driving = source.clone();
            let task = tokio::spawn(async move {
                driving
                    .prompt("What is 21 + 21? Use the add tool.")
                    .max_turns(2)
                    .await
            });

            gate.started.notified().await;
            let run = active_waiting_tool_run(&source);
            let checkpoint = source
                .checkpoint_active_run(run, PauseMode::CancelAndSuspend)
                .expect("cancel-and-suspend should create a redispatchable checkpoint");
            let encoded = serde_json::to_string(&checkpoint).expect("checkpoint should serialize");
            let checkpoint = serde_json::from_str(&encoded).expect("checkpoint should deserialize");

            task.abort();
            let _ = task.await;

            let restored = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(CheckpointAdd::immediate())
                .build();
            let output = restored
                .restore_active_run(checkpoint, usize::MAX)
                .await
                .expect("the fresh runtime should rebind executors and resume through RigSchedule");
            assert_mentions_expected_number(&output.text, 42);
            assert!(output.usage.input_tokens + output.usage.output_tokens > 0);
        },
    )
    .await;
}

#[tokio::test]
async fn resume_while_invalid_tool_call_awaits_resolution() {
    with_gemini_cassette(
        "agent_run_resume/resume_while_invalid_tool_call_awaits_resolution",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .build();
            install_policy(
                &agent,
                "resume-skip-invalid",
                0,
                1,
                PolicyRule::SkipInvalidTool {
                    tool: Some("missing_add".to_owned()),
                    reason: SKIP_REASON.to_owned(),
                },
            )
            .expect("skip policy should install");

            let response = agent
                .prompt("What is 21 + 21? Use the add tool.")
                .max_turns(2)
                .extended_details()
                .await
                .expect("restored invalid-call policy behavior should complete");
            assert_nonempty_response(&response.output);
            let messages = response.messages.expect("canonical messages");
            assert!(history_has_assistant_tool_call(&messages, "missing_add"));
            assert!(
                messages
                    .iter()
                    .flat_map(tool_result_texts)
                    .any(|text| text.contains(SKIP_REASON))
            );
        },
    )
    .await;
}

#[tokio::test]
async fn resume_after_invalid_tool_call_retry_rollback() {
    with_gemini_cassette(
        "agent_run_resume/resume_after_invalid_tool_call_retry_rollback",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .build();
            set_invalid_retry_budget(&agent, 1);
            install_policy(
                &agent,
                "resume-retry-invalid",
                0,
                1,
                PolicyRule::RetryInvalidTool {
                    tool: Some("missing_add".to_owned()),
                    feedback: RETRY_FEEDBACK.to_owned(),
                },
            )
            .expect("retry policy should install");

            let response = agent
                .prompt("What is 21 + 21?")
                .max_turns(3)
                .extended_details()
                .await
                .expect("the retry turn should complete");
            assert_mentions_expected_number(&response.output, 42);
            assert!(response.requests() >= 2);
            let messages = response.messages.expect("canonical messages");
            assert!(history_has_assistant_tool_call(&messages, "missing_add"));
            assert!(
                messages
                    .iter()
                    .flat_map(tool_result_texts)
                    .any(|text| text.contains(RETRY_FEEDBACK))
            );
        },
    )
    .await;
}
