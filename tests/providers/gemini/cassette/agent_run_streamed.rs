//! Streaming parity coverage through the ECS facade: multi-turn execution,
//! invalid-call fail/repair/skip policies, model-call budgets, and a
//! tool-call policy stop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingError};
use rig::bevy_ecs::prelude::On;
use rig::client::CompletionClient;
use rig::completion::PromptError;
use rig::message::ToolChoice;
use rig::providers::gemini;
use rig::runtime::{
    InvalidToolCallDetected, ModelTurnFinished, PendingInvalidToolCall, PolicyPoint, PolicyRule,
    ToolCallPolicyDecision, ToolCallPolicyInvocation, ToolCallPrepared,
};
use rig::streaming::StreamingPrompt;

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, Subtract, Sum, assert_canonical_assistant_order,
    history_has_assistant_tool_call, is_tool_result_user_message, tool_result_texts,
};
use super::super::support::with_gemini_cassette;
use crate::support::{assert_mentions_expected_number, assert_nonempty_response, install_policy};

const SKIP_REASON: &str = "The add tool is disabled for this request.";
const STOP_REASON: &str = "cancelled by test policy";

#[derive(Clone, Default)]
struct StreamProbe {
    invalid: Arc<Mutex<Vec<PendingInvalidToolCall>>>,
    prepared_tools: Arc<Mutex<Vec<String>>>,
    model_turns: Arc<AtomicUsize>,
}

impl StreamProbe {
    fn install(&self, agent: &rig::agent::Agent<gemini::completion::CompletionModel>) {
        let invalid = self.clone();
        let prepared = self.clone();
        let turns = self.clone();
        agent
            .with_runtime_mut(move |runtime| {
                runtime
                    .world_mut()
                    .add_observer(move |event: On<InvalidToolCallDetected>| {
                        invalid
                            .invalid
                            .lock()
                            .expect("invalid stream observations")
                            .push(event.invalid.clone());
                    });
                runtime
                    .world_mut()
                    .add_observer(move |event: On<ToolCallPrepared>| {
                        prepared
                            .prepared_tools
                            .lock()
                            .expect("prepared stream tools")
                            .push(event.call.decision.name.clone());
                    });
                runtime
                    .world_mut()
                    .add_observer(move |_event: On<ModelTurnFinished>| {
                        turns.model_turns.fetch_add(1, Ordering::SeqCst);
                    });
            })
            .expect("stream observers should install");
    }

    fn invalid_calls(&self) -> Vec<PendingInvalidToolCall> {
        self.invalid
            .lock()
            .expect("invalid stream observations")
            .clone()
    }

    fn prepared_tools(&self) -> Vec<String> {
        self.prepared_tools
            .lock()
            .expect("prepared stream tools")
            .clone()
    }
}

async fn collect_response<R>(
    stream: &mut rig::agent::StreamingResult<R>,
) -> Result<rig::agent::PromptResponse, StreamingError> {
    let mut response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(final_response) = item? {
            response = Some(final_response);
        }
    }
    Ok(response.expect("stream should produce one final response"))
}

#[tokio::test]
async fn streamed_hand_driven_multi_turn_run_completes() {
    let probe = StreamProbe::default();
    let observed = probe.clone();
    with_gemini_cassette(
        "agent_run_streamed/streamed_hand_driven_multi_turn_run_completes",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            probe.install(&agent);

            let mut stream = agent
                .stream_prompt(
                    "Use the tools to compute (7 + 4) - 2: first compute 7 + 4 with the add tool, then subtract 2 from that result with the subtract tool, then state the final result.",
                )
                .max_turns(5)
                .await;
            let response = collect_response(&mut stream)
                .await
                .expect("streamed multi-turn run should complete");

            assert_mentions_expected_number(&response.output, 9);
            assert!(response.usage.total_tokens > 0);
            assert!(response.completion_calls.len() >= 2);
            assert!(observed.model_turns.load(Ordering::SeqCst) >= 2);
            let messages = response.messages.expect("canonical streamed messages");
            assert!(history_has_assistant_tool_call(&messages, "add"));
            assert!(history_has_assistant_tool_call(&messages, "subtract"));
            assert_canonical_assistant_order(&messages);
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_invalid_tool_call_fails_fast_mid_stream() {
    let probe = StreamProbe::default();
    let observed = probe.clone();
    with_gemini_cassette(
        "agent_run_streamed/streamed_invalid_tool_call_fails_fast_mid_stream",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            probe.install(&agent);

            let mut stream = agent
                .stream_prompt("What is 21 + 21? Use the add tool.")
                .max_turns(2)
                .await;
            let error = collect_response(&mut stream)
                .await
                .expect_err("an unhandled invalid streamed call must fail closed");
            let StreamingError::Prompt(error) = error else {
                panic!("expected a prompt error, got {error:?}");
            };
            let PromptError::UnknownToolCall { tool_name, .. } = *error else {
                panic!("expected UnknownToolCall");
            };
            assert_eq!(tool_name, "missing_add");
            let calls = observed.invalid_calls();
            assert_eq!(calls.len(), 1);
            assert!(calls[0].streaming_origin);
            assert_eq!(calls[0].call.name, "missing_add");
            assert_eq!(calls[0].available_tools[0].name, "add");
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_repair_continues_the_same_stream() {
    let probe = StreamProbe::default();
    let observed = probe.clone();
    with_gemini_cassette(
        "agent_run_streamed/streamed_repair_continues_the_same_stream",
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
                "stream-repair-add",
                0,
                1,
                PolicyRule::RepairInvalidTool {
                    from: Some("missing_add".to_owned()),
                    to: "sum".to_owned(),
                },
            )
            .expect("stream repair policy should install");

            let mut stream = agent
                .stream_prompt("Use the add tool to compute 2 + 3, then state the result.")
                .max_turns(3)
                .await;
            let response = collect_response(&mut stream)
                .await
                .expect("repair should continue the shared streaming progression path");
            assert_mentions_expected_number(&response.output, 5);
            assert_eq!(observed.prepared_tools(), ["sum"]);
            assert!(observed.invalid_calls()[0].streaming_origin);
            let messages = response.messages.expect("canonical streamed messages");
            assert!(history_has_assistant_tool_call(&messages, "missing_add"));
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_skip_abandons_the_turn_and_recovers() {
    let probe = StreamProbe::default();
    let observed = probe.clone();
    with_gemini_cassette(
        "agent_run_streamed/streamed_skip_abandons_the_turn_and_recovers",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .build();
            probe.install(&agent);
            install_policy(
                &agent,
                "stream-skip-add",
                0,
                1,
                PolicyRule::SkipInvalidTool {
                    tool: Some("missing_add".to_owned()),
                    reason: SKIP_REASON.to_owned(),
                },
            )
            .expect("stream skip policy should install");

            let mut stream = agent
                .stream_prompt("What is 21 + 21? Use the add tool.")
                .max_turns(3)
                .await;
            let response = collect_response(&mut stream)
                .await
                .expect("synthetic skip feedback should let streaming continue");
            assert_nonempty_response(&response.output);
            assert!(observed.invalid_calls()[0].streaming_origin);
            assert!(observed.prepared_tools().is_empty());
            let messages = response.messages.expect("canonical streamed messages");
            let results = messages
                .iter()
                .flat_map(tool_result_texts)
                .collect::<Vec<_>>();
            assert!(results.iter().any(|result| result.contains(SKIP_REASON)));
        },
    )
    .await;
}

#[tokio::test]
async fn builtin_streaming_max_turns_error_carries_pending_message() {
    with_gemini_cassette(
        "agent_run_streamed/builtin_streaming_max_turns_error_carries_pending_message",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            let mut stream = agent
                .stream_prompt("What is 21 + 21? Use the add tool.")
                .max_turns(2)
                .await;

            let error = collect_response(&mut stream)
                .await
                .expect_err("required tools should exhaust the model budget");
            let StreamingError::Prompt(error) = error else {
                panic!("expected a prompt error, got {error:?}");
            };
            let PromptError::MaxTurnsError {
                max_turns,
                chat_history,
                prompt,
            } = *error
            else {
                panic!("expected MaxTurnsError");
            };
            assert_eq!(max_turns, 2);
            assert!(is_tool_result_user_message(&prompt));
            assert!(history_has_assistant_tool_call(&chat_history, "add"));
        },
    )
    .await;
}

#[tokio::test]
async fn builtin_streaming_cancellation_history_includes_assistant_turn() {
    let probe = StreamProbe::default();
    let observed = probe.clone();
    with_gemini_cassette(
        "agent_run_streamed/builtin_streaming_cancellation_history_includes_assistant_turn",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            probe.install(&agent);
            let policy = install_policy(
                &agent,
                "stream-stop-add",
                0,
                1,
                PolicyRule::Custom(PolicyPoint::ToolCall),
            )
            .expect("stream stop policy should install");
            agent
                .with_runtime_mut(move |runtime| {
                    runtime.world_mut().entity_mut(policy).observe(
                        |mut event: On<ToolCallPolicyInvocation>| {
                            event.decision =
                                Some(ToolCallPolicyDecision::Stop(STOP_REASON.to_owned()));
                        },
                    );
                })
                .expect("stream stop observer should install");

            let mut stream = agent
                .stream_prompt("What is 21 + 21? Use the add tool.")
                .max_turns(2)
                .await;
            let error = collect_response(&mut stream)
                .await
                .expect_err("the stopping policy must prevent a final response");
            assert!(error.to_string().contains("stream-stop-add"), "{error:?}");
            assert_eq!(observed.model_turns.load(Ordering::SeqCst), 1);
            assert!(observed.prepared_tools().is_empty());
        },
    )
    .await;
}
