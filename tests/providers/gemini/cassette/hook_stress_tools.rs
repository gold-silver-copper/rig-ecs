//! ECS policy stress suite: the tool-execution lifecycle — chained
//! argument rewrites, chained result rewrites (redact / wrap / truncate),
//! stopping from a tool result (post-execution), and model-driven recovery
//! from a tool error. Recorded against real Gemini.

use rig::bevy_ecs::prelude::On;
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::runtime::{
    PolicyPoint, PolicyRule, ToolCallPolicyDecision, ToolCallPolicyInvocation, ToolCallPrepared,
    ToolResultPolicyDecision, ToolResultPolicyInvocation, ToolResultPresentationFinalized,
};
use rig::tool::Tool;
use serde_json::json;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CodewordLookup, CountingAdd, MottoTool, ToolEventRecorder};
use crate::support::{assert_nonempty_response, install_policy};

fn install_arg_patch(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    id: &str,
    order: u32,
    key: &'static str,
    value: serde_json::Value,
) {
    let policy = install_policy(
        agent,
        id,
        order,
        1,
        PolicyRule::Custom(PolicyPoint::ToolCall),
    )
    .expect("argument policy should install");
    agent
        .with_runtime_mut(move |runtime| {
            runtime.world_mut().entity_mut(policy).observe(
                move |mut event: On<ToolCallPolicyInvocation>| {
                    if event.call.decision.name != CountingAdd::NAME {
                        event.decision = Some(ToolCallPolicyDecision::Run);
                        return;
                    }
                    let mut arguments = event.call.arguments.clone();
                    arguments
                        .as_object_mut()
                        .expect("add arguments should be an object")
                        .insert(key.to_owned(), value.clone());
                    event.decision = Some(ToolCallPolicyDecision::Rewrite(arguments));
                },
            );
        })
        .expect("argument observer should install");
}

fn install_result_transform(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    id: &str,
    order: u32,
    tool: &'static str,
    transform: impl Fn(&str) -> String + Send + Sync + 'static,
) {
    let policy = install_policy(
        agent,
        id,
        order,
        1,
        PolicyRule::Custom(PolicyPoint::ToolResult),
    )
    .expect("result policy should install");
    agent
        .with_runtime_mut(move |runtime| {
            runtime.world_mut().entity_mut(policy).observe(
                move |mut event: On<ToolResultPolicyInvocation>| {
                    event.decision = Some(if event.result.name == tool {
                        ToolResultPolicyDecision::Rewrite(transform(&event.result.presentation))
                    } else {
                        ToolResultPolicyDecision::Keep
                    });
                },
            );
        })
        .expect("result observer should install");
}

fn install_recorder(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    recorder: ToolEventRecorder,
) {
    let call_recorder = recorder.clone();
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
            runtime
                .world_mut()
                .add_observer(move |event: On<ToolResultPresentationFinalized>| {
                    let arguments = recorder
                        .calls
                        .lock()
                        .expect("calls lock should not be poisoned")
                        .last()
                        .map_or_else(String::new, |(_, arguments)| arguments.clone());
                    recorder
                        .results
                        .lock()
                        .expect("results lock should not be poisoned")
                        .push((
                            event.raw.name.clone(),
                            arguments,
                            event.presentation.clone(),
                        ));
                });
        })
        .expect("recorder observers should install");
}

// ---------------------------------------------------------------------------
// Argument rewriting.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn arg_rewrite_sets_one_key_preserving_rest_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress_tools/arg_rewrite_sets_one_key_preserving_rest_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant. Use the add tool for the addition.")
                .temperature(0.0)
                .tool(add)
                .build();

            install_arg_patch(&agent, "force-x", 0, "x", json!(100));
            install_recorder(&agent, recorder);

            let response = agent
                .prompt("Use the add tool to add 3 and 4, then report the tool's result.")
                .max_turns(4)
                .await
                .expect("single-key arg rewrite run should succeed");

            assert_nonempty_response(&response);
            assert!(add_calls.count() >= 1, "the tool should execute");
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1, "one add call, saw {calls:?}");
            let observed: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            assert_eq!(observed["x"], json!(100), "x must be the rewritten value");
            assert!(
                observed.get("y").is_some(),
                "the model's y argument must be preserved: {observed}"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn two_arg_rewrites_chain_blocking() {
    let add = CountingAdd::default();
    let recorder = ToolEventRecorder::default();
    let recorder_probe = recorder.clone();

    with_gemini_cassette(
        "hook_stress_tools/two_arg_rewrites_chain_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant. Use the add tool for the addition.")
                .temperature(0.0)
                .tool(add)
                .build();

            install_arg_patch(&agent, "force-x", 0, "x", json!(7));
            install_arg_patch(&agent, "force-y", 1, "y", json!(8));
            install_recorder(&agent, recorder);

            let response = agent
                .prompt("Use the add tool to add 1 and 1, then report the tool's result.")
                .max_turns(4)
                .await
                .expect("chained arg rewrite run should succeed");

            assert_nonempty_response(&response);
            let calls = recorder_probe.recorded_calls();
            assert_eq!(calls.len(), 1);
            let observed: serde_json::Value =
                serde_json::from_str(&calls[0].1).expect("observed args are JSON");
            assert_eq!(
                observed,
                json!({ "x": 7, "y": 8 }),
                "both chained rewrites must compose"
            );
            let results = recorder_probe.recorded_results();
            assert_eq!(
                results[0].2, "15",
                "the tool executed against the composed args"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Result rewriting.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_result_rewrites_chain_redact_then_wrap_blocking() {
    let add = CountingAdd::default();

    with_gemini_cassette(
        "hook_stress_tools/two_result_rewrites_chain_redact_then_wrap_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. Use the add tool, then report the exact tool \
                     result text verbatim.",
                )
                .temperature(0.0)
                .tool(add)
                .build();

            install_policy(
                &agent,
                "redact-result",
                0,
                1,
                PolicyRule::RewriteToolResult {
                    tool: Some("add".to_owned()),
                    presentation: "SECRET".to_owned(),
                },
            )
            .expect("redaction policy should install");
            install_result_transform(&agent, "wrap-result", 1, "add", |value| {
                format!("[{value}]")
            });

            let response = agent
                .prompt("Use the add tool to add 2 and 2, then report the exact tool result.")
                .max_turns(4)
                .await
                .expect("chained result rewrite run should succeed");

            assert!(
                response.contains("[SECRET]"),
                "both chained result rewrites must compose (redact then wrap): {response:?}"
            );
            assert!(
                !response.contains('4'),
                "the raw tool result must not reach the model: {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn result_truncation_reaches_model_blocking() {
    with_gemini_cassette(
        "hook_stress_tools/result_truncation_reaches_model_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "Call the fetch_motto tool, then report the exact tool result text verbatim.",
                )
                .temperature(0.0)
                .tool(MottoTool)
                .build();

            install_result_transform(&agent, "truncate-result", 0, "fetch_motto", |value| {
                value.chars().take(6).collect()
            });

            // The motto is "steady hands\ncalm waters"; truncate to its first 6
            // chars ("steady") before the model sees it.
            let response = agent
                .prompt("Call fetch_motto and report exactly what it returns.")
                .max_turns(4)
                .await
                .expect("result truncation run should succeed");

            assert!(
                response.contains("steady"),
                "the truncated result prefix must reach the model: {response:?}"
            );
            assert!(
                !response.contains("waters"),
                "the truncated-off suffix must not reach the model: {response:?}"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Terminate from a ToolResult (post-execution).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn terminate_from_tool_result_cancels_after_execution_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();

    with_gemini_cassette(
        "hook_stress_tools/terminate_from_tool_result_cancels_after_execution_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant. Use the add tool for the addition.")
                .temperature(0.0)
                .tool(add)
                .build();

            install_policy(
                &agent,
                "veto-add-result",
                0,
                1,
                PolicyRule::StopToolResult {
                    tool: Some("add".to_owned()),
                    reason: "result vetoed by policy hook".to_owned(),
                },
            )
            .expect("result stop policy should install");

            let error = agent
                .prompt("Use the add tool to add 21 and 21, then report the result.")
                .max_turns(4)
                .await
                .expect_err("a tool-result stop should fail the run");

            // The tool executed before the terminate fired.
            assert!(
                add_calls.count() >= 1,
                "the tool body must have run before the ToolResult terminate"
            );
            assert!(
                error.to_string().contains("veto-add-result"),
                "policy denial should identify the stopping policy: {error:?}"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Model-driven recovery from a tool error.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_error_guidance_drives_model_retry_blocking() {
    let lookup = CodewordLookup::default();
    let lookup_calls = lookup.counter.clone();

    with_gemini_cassette(
        "hook_stress_tools/tool_error_guidance_drives_model_retry_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You look up team codewords with the lookup_codeword tool. If the tool returns \
                     an error with guidance, follow that guidance and try again, then report the \
                     codeword you obtain.",
                )
                .temperature(0.0)
                .tool(lookup)
                .build();

            // The first (red) lookup errors with corrective guidance pointing at
            // the blue team; the model should retry and obtain the blue codeword.
            let response = agent
                .prompt("Look up the codeword for the red team.")
                .max_turns(5)
                .await
                .expect("tool-error recovery run should succeed");

            assert!(
                lookup_calls.count() >= 2,
                "the model should retry the lookup after the error guidance, saw {} call(s)",
                lookup_calls.count()
            );
            assert!(
                response.to_ascii_lowercase().contains("azure-falcon"),
                "the model should report the recovered codeword: {response:?}"
            );
        },
    )
    .await;
}
