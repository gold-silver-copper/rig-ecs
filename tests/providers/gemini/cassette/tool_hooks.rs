//! ECS policy dispatch on the tool execution path: skip-with-reason,
//! stop-before-dispatch, and observation of every call/result pair.

use rig::bevy_ecs::prelude::{In, On, Query};
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::runtime::{
    PolicyPoint, PolicyResponderId, PolicyRule, ToolCallPolicyDecision, ToolCallPolicyInvocation,
    ToolCallPrepared, ToolEffectInput, ToolResultPresentationFinalized,
};
use rig::tool::Tool;

use super::super::agent_run_support::tool_result_texts;
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, FORCE_TOOLS_PREAMBLE, ToolEventRecorder};
use crate::support::{assert_nonempty_response, install_policy};

const SKIP_REASON: &str = "the add tool is down for maintenance; report exactly that to the user";
const TERMINATE_REASON: &str = "tool execution vetoed by policy hook";

fn stop_add(In(event): In<ToolCallPolicyInvocation>) -> Option<ToolCallPolicyDecision> {
    Some(if event.call.decision.name == CountingAdd::NAME {
        ToolCallPolicyDecision::Stop(TERMINATE_REASON.to_owned())
    } else {
        ToolCallPolicyDecision::Run
    })
}

#[tokio::test]
async fn on_tool_call_skip_returns_reason_without_executing() {
    let add = CountingAdd::default();
    let counter = add.counter.clone();

    with_gemini_cassette(
        "tool_hooks/on_tool_call_skip_returns_reason_without_executing",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            install_policy(
                &agent,
                "skip-add",
                0,
                1,
                PolicyRule::SkipToolCall {
                    tool: Some(CountingAdd::NAME.to_owned()),
                    reason: SKIP_REASON.to_owned(),
                },
            )
            .expect("skip policy should install");

            let response = agent
                .prompt("What is 19 + 23?")
                .max_turns(3)
                .extended_details()
                .await
                .expect("a skipped tool call should not fail the run");

            assert_eq!(counter.count(), 0, "the skipped tool should never execute");

            let messages = response
                .messages
                .expect("extended details should carry the run's messages");
            let texts: Vec<String> = messages.iter().flat_map(tool_result_texts).collect();
            assert_eq!(
                texts,
                vec![SKIP_REASON.to_string()],
                "the skip reason should be the synthetic tool result"
            );
            assert_nonempty_response(&response.output);
        },
    )
    .await;
}

#[tokio::test]
async fn on_tool_call_terminate_cancels_run() {
    let add = CountingAdd::default();
    let counter = add.counter.clone();

    with_gemini_cassette(
        "tool_hooks/on_tool_call_terminate_cancels_run",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            let policy = install_policy(
                &agent,
                "stop-add",
                0,
                1,
                PolicyRule::Custom(PolicyPoint::ToolCall),
            )
            .expect("stop policy should install");
            agent
                .with_runtime_mut(|runtime| {
                    runtime
                        .register_tool_call_policy_responder(
                            policy,
                            PolicyResponderId::new("stop-add").unwrap(),
                            stop_add,
                        )
                        .expect("tool-call responder should register");
                })
                .expect("policy observer should install");

            let error = agent
                .prompt("What is 19 + 23?")
                .max_turns(3)
                .extended_details()
                .await
                .expect_err("a stopping policy should fail the run");

            assert_eq!(counter.count(), 0, "the vetoed tool should never execute");
            assert!(
                error.to_string().contains("stop-add"),
                "policy denial should identify the stopping policy: {error:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn hooks_observe_every_tool_call_and_result() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_for_test = recorder.clone();

    with_gemini_cassette(
        "tool_hooks/hooks_observe_every_tool_call_and_result",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            let call_recorder = recorder.clone();
            let result_recorder = recorder.clone();
            agent
                .with_runtime_mut(move |runtime| {
                    runtime
                        .world_mut()
                        .add_observer(move |event: On<ToolCallPrepared>| {
                            call_recorder
                                .calls
                                .lock()
                                .expect("calls lock should not be poisoned")
                                .push((
                                    event.call.decision.name.clone(),
                                    event.call.arguments.to_string(),
                                ));
                        });
                    runtime.world_mut().add_observer(
                        move |event: On<ToolResultPresentationFinalized>,
                              inputs: Query<&ToolEffectInput>| {
                            let arguments = result_recorder
                                .calls
                                .lock()
                                .expect("calls lock should not be poisoned")
                                .last()
                                .map_or_else(String::new, |(_, arguments)| arguments.clone());
                            let input = inputs.get(event.operation).expect(
                                "finalized tool operation should retain its immutable input",
                            );
                            result_recorder
                                .results
                                .lock()
                                .expect("results lock should not be poisoned")
                                .push((
                                    input.decision.name.clone(),
                                    arguments,
                                    event.presentation.clone(),
                                ));
                        },
                    );
                })
                .expect("observers should install");

            let response = agent
                .prompt("Use the add tool to calculate 19 + 23, then report the result.")
                .max_turns(3)
                .await
                .expect("recorded tool prompt should succeed");

            let calls = recorder_for_test.recorded_calls();
            assert_eq!(
                calls.len(),
                1,
                "exactly one tool call should be observed: {calls:?}"
            );
            let (call_name, call_args) = &calls[0];
            assert_eq!(call_name, CountingAdd::NAME);
            let args: serde_json::Value =
                serde_json::from_str(call_args).expect("observed args should be JSON");
            assert_eq!(args, serde_json::json!({ "x": 19, "y": 23 }));

            let results = recorder_for_test.recorded_results();
            assert_eq!(
                results.len(),
                1,
                "exactly one tool result should be observed"
            );
            let (result_name, result_args, result_output) = &results[0];
            assert_eq!(result_name, CountingAdd::NAME);
            assert_eq!(
                result_args, call_args,
                "result hook should see the same args"
            );
            assert_eq!(
                result_output, "<typed tool output: 1 parts>",
                "observe-only telemetry should summarize typed tool output"
            );

            assert!(
                response.contains("42"),
                "final answer should report 42: {response:?}"
            );
        },
    )
    .await;
}
