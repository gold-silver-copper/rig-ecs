//! Invalid tool-call recovery through ECS policy entities against Gemini
//! cassette turns: fail-fast, repair, skip, retry-budget exhaustion, and an
//! invalid repair target.

use std::sync::{Arc, Mutex};

use rig::bevy_ecs::prelude::On;
use rig::client::CompletionClient;
use rig::completion::PromptError;
use rig::message::ToolChoice;
use rig::providers::gemini;
use rig::runtime::{
    InvalidToolCallDetected, ModelToolChoice, PendingInvalidToolCall, PolicyRule, ToolCallPrepared,
};

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, Subtract, Sum, assistant_tool_call_names,
    history_has_assistant_tool_call, tool_result_texts,
};
use super::super::support::with_gemini_cassette;
use crate::support::{assert_mentions_expected_number, assert_nonempty_response, install_policy};

const SKIP_REASON: &str = "The add tool is disabled for this request.";

#[derive(Clone, Default)]
struct InvalidCallProbe(Arc<Mutex<Vec<PendingInvalidToolCall>>>);

impl InvalidCallProbe {
    fn install(&self, agent: &rig::agent::Agent<gemini::completion::CompletionModel>) {
        let observed = self.clone();
        agent
            .with_runtime_mut(move |runtime| {
                runtime
                    .world_mut()
                    .add_observer(move |event: On<InvalidToolCallDetected>| {
                        observed
                            .0
                            .lock()
                            .expect("invalid-call observations")
                            .push(event.invalid.clone());
                    });
            })
            .expect("invalid-call observer should install");
    }

    fn calls(&self) -> Vec<PendingInvalidToolCall> {
        self.0.lock().expect("invalid-call observations").clone()
    }
}

#[derive(Clone, Default)]
struct PreparedToolProbe(Arc<Mutex<Vec<String>>>);

impl PreparedToolProbe {
    fn install(&self, agent: &rig::agent::Agent<gemini::completion::CompletionModel>) {
        let observed = self.clone();
        agent
            .with_runtime_mut(move |runtime| {
                runtime
                    .world_mut()
                    .add_observer(move |event: On<ToolCallPrepared>| {
                        observed
                            .0
                            .lock()
                            .expect("prepared tool observations")
                            .push(event.call.decision.name.clone());
                    });
            })
            .expect("prepared-tool observer should install");
    }

    fn names(&self) -> Vec<String> {
        self.0.lock().expect("prepared tool observations").clone()
    }
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
async fn fail_resolution_returns_unknown_tool_call() {
    let probe = InvalidCallProbe::default();
    let observed = probe.clone();
    with_gemini_cassette(
        "agent_run_recovery/fail_resolution_returns_unknown_tool_call",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            probe.install(&agent);

            let error = agent
                .prompt("What is 21 + 21? Use the add tool.")
                .await
                .expect_err("an unhandled unknown tool must fail closed");
            let PromptError::UnknownToolCall { tool_name, .. } = error else {
                panic!("expected UnknownToolCall, got {error:?}");
            };
            assert_eq!(tool_name, "missing_add");

            let calls = observed.calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].call.name, "missing_add");
            assert_eq!(
                calls[0]
                    .available_tools
                    .iter()
                    .map(|tool| tool.name.as_str())
                    .collect::<Vec<_>>(),
                ["add"]
            );
            assert_eq!(calls[0].tool_choice, Some(ModelToolChoice::Required));
        },
    )
    .await;
}

#[tokio::test]
async fn repair_renames_tool_call_and_executes_it() {
    let probe = PreparedToolProbe::default();
    let observed = probe.clone();
    with_gemini_cassette(
        "agent_run_recovery/repair_renames_tool_call_and_executes_it",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Sum)
                .build();
            probe.install(&agent);
            install_policy(
                &agent,
                "repair-add-typo",
                0,
                1,
                PolicyRule::RepairInvalidTool {
                    from: Some("missing_add".to_owned()),
                    to: "sum".to_owned(),
                },
            )
            .expect("repair policy should install");

            let response = agent
                .prompt("Use the add tool to compute 2 + 3, then state the result.")
                .max_turns(3)
                .extended_details()
                .await
                .expect("repair to an accepted tool should complete");
            assert_mentions_expected_number(&response.output, 5);

            let messages = response.messages.expect("canonical messages");
            let recorded = messages
                .iter()
                .flat_map(assistant_tool_call_names)
                .collect::<Vec<_>>();
            assert!(
                recorded.iter().any(|name| name == "missing_add"),
                "provider-significant emitted content stays intact: {recorded:?}"
            );
            assert_eq!(observed.names(), ["sum"]);
        },
    )
    .await;
}

#[tokio::test]
async fn skip_suppresses_every_call_in_the_turn() {
    with_gemini_cassette(
        "agent_run_recovery/skip_suppresses_every_call_in_the_turn",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            install_policy(
                &agent,
                "skip-missing-add",
                0,
                1,
                PolicyRule::SkipInvalidTool {
                    tool: Some("missing_add".to_owned()),
                    reason: SKIP_REASON.to_owned(),
                },
            )
            .expect("skip policy should install");

            let response = agent
                .prompt(
                    "Compute 3 + 5 and 10 - 4. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.",
                )
                .max_turns(3)
                .extended_details()
                .await
                .expect("the skipped invalid call should become synthetic feedback");
            assert_nonempty_response(&response.output);
            let messages = response.messages.expect("canonical messages");
            let presentations = messages
                .iter()
                .flat_map(tool_result_texts)
                .collect::<Vec<_>>();
            assert!(
                presentations.iter().any(|text| text.contains(SKIP_REASON)),
                "{presentations:?}"
            );
            assert!(
                history_has_assistant_tool_call(&messages, "missing_add"),
                "the provider-emitted invalid call remains auditable"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn retry_with_exhausted_budget_fails_with_unknown_tool_call() {
    with_gemini_cassette(
        "agent_run_recovery/retry_with_exhausted_budget_fails_with_unknown_tool_call",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            set_invalid_retry_budget(&agent, 0);
            install_policy(
                &agent,
                "retry-missing-add",
                0,
                1,
                PolicyRule::RetryInvalidTool {
                    tool: Some("missing_add".to_owned()),
                    feedback: "Try a different tool.".to_owned(),
                },
            )
            .expect("retry policy should install");

            let error = agent
                .prompt("What is 21 + 21? Use the add tool.")
                .await
                .expect_err("retry without budget must fail");
            let PromptError::UnknownToolCall { tool_name, .. } = error else {
                panic!("expected UnknownToolCall, got {error:?}");
            };
            assert_eq!(tool_name, "missing_add");
        },
    )
    .await;
}

#[tokio::test]
async fn repair_to_disallowed_name_fails_with_unknown_tool_call() {
    with_gemini_cassette(
        "agent_run_recovery/repair_to_disallowed_name_fails_with_unknown_tool_call",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            install_policy(
                &agent,
                "bad-repair-target",
                0,
                1,
                PolicyRule::RepairInvalidTool {
                    from: Some("missing_add".to_owned()),
                    to: "multiply".to_owned(),
                },
            )
            .expect("repair policy should install");

            let error = agent
                .prompt("What is 21 + 21? Use the add tool.")
                .await
                .expect_err("repair outside the immutable snapshot must fail");
            let PromptError::UnknownToolCall { tool_name, .. } = error else {
                panic!("expected UnknownToolCall, got {error:?}");
            };
            assert_eq!(tool_name, "multiply");
        },
    )
    .await;
}
