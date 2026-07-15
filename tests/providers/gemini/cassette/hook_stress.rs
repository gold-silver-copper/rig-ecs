//! Long, multi-turn ECS policy and observer stress workflows recorded against Gemini.
//!
//! These regressions preserve the former hook suite's externally observable behavior while
//! exercising ordered policy entities, immediate lifecycle observers, and extension-owned typed
//! components instead of a hook stack or untyped scratchpad.

use std::collections::BTreeMap;

use futures::StreamExt;
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::runtime::{PolicyRule, RequestPatch, RetrievedDocument};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt};
use rig::tool::Tool;
use rig::{agent::MultiTurnStreamItem, agent::StreamingError};

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract, ToolEventRecorder};
use super::hook_stress_context::{EventTap, TallyReader, install_tally_observers, install_tap};
use super::hook_stress_tools::{install_arg_patch, install_recorder};
use crate::support::{assert_nonempty_response, install_policy};

const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself. Perform the steps in order, using the result of each step as an input to the next. Once you have the final tool result, reply with the final numeric answer in plain text.";

#[tokio::test]
async fn lifecycle_and_scratchpad_thread_across_multi_turn_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = EventTap::default();
    let reader = TallyReader::default();
    let recorder_probe = recorder.clone();
    let reader_probe = reader.clone();

    with_gemini_cassette(
        "hook_stress/lifecycle_and_scratchpad_thread_across_multi_turn_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            install_tap(&agent, recorder, false);
            install_tally_observers(&agent, EventTap::default(), reader);

            let response = agent
                .prompt(
                    "First add 10 and 5 with the add tool. Then subtract 3 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .await
                .expect("dependent multi-turn tool run should succeed");

            assert_nonempty_response(&response);
            assert_eq!(recorder_probe.distinct_run_ids(), 1);
            assert_eq!(recorder_probe.is_streaming(), Some(false));
            assert_eq!(recorder_probe.agent_name().as_deref(), Some("stress-agent"));

            let max_turn = recorder_probe
                .breadcrumbs()
                .iter()
                .map(|breadcrumb| breadcrumb.turn)
                .max()
                .unwrap_or_default();
            assert!(max_turn >= 2, "the workflow should span multiple turns");

            let tool_calls = recorder_probe.count("ToolCall");
            assert_eq!(tool_calls, recorder_probe.count("ToolResult"));
            assert_eq!(tool_calls, add_calls.count() + subtract_calls.count());
            assert!(add_calls.count() >= 1 && subtract_calls.count() >= 1);

            let tallies = reader_probe.tallies();
            assert!(!tallies.is_empty());
            assert!(tallies.windows(2).all(|window| window[0] <= window[1]));
            assert_eq!(tallies.last().copied(), Some(tool_calls));
        },
    )
    .await;
}

const VAULT_FACT: &str = "Operational note: the vault access code is CINNABAR-42.";

#[tokio::test]
async fn request_patch_injects_context_and_narrows_active_tools_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();

    with_gemini_cassette(
        "hook_stress/request_patch_injects_context_and_narrows_active_tools_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a helpful assistant. Use a tool for any arithmetic. Consult the \
                     provided context for any facts you are asked about.",
                )
                .tool(add)
                .tool(subtract)
                .build();
            install_policy(
                &agent,
                "vault-add-only",
                0,
                1,
                PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .extra_context([RetrievedDocument {
                            id: "vault-note".to_owned(),
                            text: VAULT_FACT.to_owned(),
                            metadata: Default::default(),
                        }])
                        .active_tools(["add"])
                        .temperature(0.0),
                ),
            )
            .expect("request policy should install");

            let response = agent
                .prompt(
                    "Two things: (1) tell me the vault access code, and (2) use a tool to compute \
                     41 + 1.",
                )
                .max_turns(5)
                .await
                .expect("context-injecting, tool-narrowing run should succeed");

            assert!(response.contains("CINNABAR-42"));
            assert_eq!(subtract_calls.count(), 0);
            assert!(add_calls.count() >= 1);
        },
    )
    .await;
}

const REDACTION_MARKER: &str = "REDACTED-SUM-ZK7";

#[tokio::test]
async fn chained_arg_rewrite_then_result_redaction_blocking() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress/chained_arg_rewrite_then_result_redaction_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the add tool for the addition. \
                     After the tool result is available, report the exact tool result text \
                     verbatim as your final answer.",
                )
                .temperature(0.0)
                .tool(add)
                .build();

            install_arg_patch(&agent, "force-seven", 0, "x", serde_json::json!(7));
            install_arg_patch(&agent, "force-eight", 1, "y", serde_json::json!(8));
            install_recorder(&agent, recorder);
            install_policy(
                &agent,
                "redact-sum",
                2,
                1,
                PolicyRule::RewriteToolResult {
                    tool: Some(CountingAdd::NAME.to_owned()),
                    presentation: REDACTION_MARKER.to_owned(),
                },
            )
            .expect("redaction policy should install");

            let response = agent
                .prompt("Use the add tool to add 2 and 2, then report the exact tool result.")
                .max_turns(4)
                .await
                .expect("chained rewrite and redaction should succeed");

            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&calls[0].1).expect("JSON arguments"),
                serde_json::json!({"x": 7, "y": 8})
            );
            let results = recorder_probe.recorded_results();
            assert_eq!(results.len(), 1);
            assert_eq!(
                results[0].2, REDACTION_MARKER,
                "finalized observers must receive only policy-approved presentation"
            );
            assert!(response.contains(REDACTION_MARKER));
            assert!(!response.contains("15"));
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_lifecycle_ordering_and_context_streaming_flag() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = EventTap::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress/streaming_lifecycle_ordering_and_context_streaming_flag",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();
            install_tap(&agent, recorder, true);

            let mut stream = agent
                .stream_prompt(
                    "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .await;
            let mut events = Vec::new();
            let mut final_text = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::ToolCall { .. },
                    )) => events.push("tool_call"),
                    Ok(MultiTurnStreamItem::ToolExecutionCommitted { .. }) => {
                        events.push("tool_execution_committed");
                    }
                    Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                        ..
                    })) => events.push("tool_result"),
                    Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                        events.push("final_response");
                        final_text = Some(response.output().to_owned());
                    }
                    Ok(_) => {}
                    Err(StreamingError::Prompt(error)) => panic!("stream errored: {error:?}"),
                    Err(error) => panic!("stream errored: {error:?}"),
                }
            }
            assert_nonempty_response(final_text.as_deref().expect("final response"));
            let position = |tag| events.iter().position(|event| *event == tag).expect(tag);
            assert!(position("tool_call") < position("tool_execution_committed"));
            assert!(position("tool_execution_committed") <= position("tool_result"));
            assert!(position("tool_result") < position("final_response"));
            assert_eq!(recorder_probe.is_streaming(), Some(true));
            assert_eq!(recorder_probe.distinct_run_ids(), 1);
            assert_eq!(recorder_probe.agent_name().as_deref(), Some("stress-agent"));
            assert!(recorder_probe.count("ModelTurnFinished") >= 2);
            assert!(add_calls.count() >= 1 && subtract_calls.count() >= 1);
        },
    )
    .await;
}

#[tokio::test]
async fn multi_tool_workflow_pairs_calls_and_results_per_turn_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let recorder = EventTap::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress/multi_tool_workflow_pairs_calls_and_results_per_turn_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the provided tools for every \
                     arithmetic operation. These two computations are independent — you may request \
                     them together. Once you have both results, report both numbers.",
                )
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();
            install_tap(&agent, recorder, false);

            let response = agent
                .prompt(
                    "Independently compute 12 + 8 using the add tool and 30 - 7 using the subtract \
                     tool, then report both results.",
                )
                .max_turns(5)
                .await
                .expect("independent multi-tool run should succeed");
            assert_nonempty_response(&response);
            assert!(add_calls.count() >= 1 && subtract_calls.count() >= 1);

            let mut per_turn: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
            for breadcrumb in recorder_probe.breadcrumbs() {
                let counts = per_turn.entry(breadcrumb.turn).or_default();
                match breadcrumb.tag {
                    "ToolCall" => counts.0 += 1,
                    "ToolResult" => counts.1 += 1,
                    _ => {}
                }
            }
            assert!(per_turn.values().all(|(calls, results)| calls == results));
            assert_eq!(
                recorder_probe.count("ToolCall"),
                add_calls.count() + subtract_calls.count()
            );
        },
    )
    .await;
}

const SUBTRACT_SKIP_REASON: &str =
    "the subtract tool is offline; treat its result as unavailable and continue";

#[tokio::test]
async fn skip_in_multi_tool_workflow_leaves_tool_unexecuted_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();

    with_gemini_cassette(
        "hook_stress/skip_in_multi_tool_workflow_leaves_tool_unexecuted_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the provided tools for every \
                     arithmetic operation. If a tool reports it is unavailable, acknowledge that in \
                     your answer and still report any results you do have.",
                )
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();
            install_policy(
                &agent,
                "skip-subtract",
                0,
                1,
                PolicyRule::SkipToolCall {
                    tool: Some(CountingSubtract::NAME.to_owned()),
                    reason: SUBTRACT_SKIP_REASON.to_owned(),
                },
            )
            .expect("skip policy should install");

            let response = agent
                .prompt(
                    "Use the add tool to compute 14 + 6, and use the subtract tool to compute \
                     40 - 9. Report what you can.",
                )
                .max_turns(5)
                .await
                .expect("a skipped tool must not fail the run");
            assert_nonempty_response(&response);
            assert_eq!(subtract_calls.count(), 0);
            assert!(add_calls.count() >= 1);
        },
    )
    .await;
}
