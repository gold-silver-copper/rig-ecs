//! ECS policy stress suite: streaming lifecycle and blocking-vs-streaming
//! parity — text delta / stream finish / model turn events on the
//! streaming surface, result redaction reaching the final response,
//! `active_tools` narrowing and `Skip` on the streaming driver, and the same
//! workflow producing the same answer on both surfaces. Recorded against real
//! Gemini.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::bevy_ecs::prelude::On;
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::runtime::{
    ModelTurnFinished, PolicyRule, RequestPatch, StreamResponseFinished, TextDeltaObserved,
    ToolCallPrepared,
};
use rig::streaming::StreamingPrompt;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract};
use crate::support::{
    assert_mentions_expected_number, assert_nonempty_response, collect_stream_final_response,
    install_policy,
};

const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself. Perform the steps in order, using the result of each step as an input to the next. Once you have the final tool result, reply with the final numeric answer in plain text.";

#[derive(Clone, Default)]
struct StreamingTap {
    text_deltas: Arc<AtomicUsize>,
    stream_finishes: Arc<AtomicUsize>,
    model_turns: Arc<AtomicUsize>,
    tool_calls: Arc<AtomicUsize>,
}

impl StreamingTap {
    fn count(&self, event: &str) -> usize {
        match event {
            "TextDelta" => self.text_deltas.load(Ordering::SeqCst),
            "StreamResponseFinish" => self.stream_finishes.load(Ordering::SeqCst),
            "ModelTurnFinished" => self.model_turns.load(Ordering::SeqCst),
            "ToolCall" => self.tool_calls.load(Ordering::SeqCst),
            _ => 0,
        }
    }
}

fn install_streaming_tap(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    tap: StreamingTap,
) {
    agent
        .with_runtime_mut(move |runtime| {
            let text = tap.clone();
            runtime
                .world_mut()
                .add_observer(move |_event: On<TextDeltaObserved>| {
                    text.text_deltas.fetch_add(1, Ordering::SeqCst);
                });
            let finish = tap.clone();
            runtime
                .world_mut()
                .add_observer(move |_event: On<StreamResponseFinished>| {
                    finish.stream_finishes.fetch_add(1, Ordering::SeqCst);
                });
            let turns = tap.clone();
            runtime
                .world_mut()
                .add_observer(move |_event: On<ModelTurnFinished>| {
                    turns.model_turns.fetch_add(1, Ordering::SeqCst);
                });
            runtime
                .world_mut()
                .add_observer(move |_event: On<ToolCallPrepared>| {
                    tap.tool_calls.fetch_add(1, Ordering::SeqCst);
                });
        })
        .expect("streaming observers should install");
}

#[tokio::test]
async fn streaming_text_only_emits_text_deltas_and_stream_finish() {
    let tap = StreamingTap::default();
    let probe = tap.clone();

    with_gemini_cassette(
        "hook_stress_streaming/streaming_text_only_emits_text_deltas_and_stream_finish",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a concise assistant. Answer directly in plain text.")
                .temperature(0.0)
                .build();

            install_streaming_tap(&agent, tap);

            let mut stream = agent
                .stream_prompt("In one short sentence, describe the color of a clear daytime sky.")
                .max_turns(2)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert_nonempty_response(&final_text);

            assert!(
                probe.count("TextDelta") >= 1,
                "a streamed text turn must emit TextDelta events"
            );
            assert!(
                probe.count("StreamResponseFinish") >= 1,
                "a streamed text turn must emit StreamResponseFinish"
            );
            assert!(
                probe.count("ModelTurnFinished") >= 1,
                "ModelTurnFinished must fire on the streaming surface"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn streaming_tool_turns_fire_model_turn_finished() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let tap = StreamingTap::default();
    let probe = tap.clone();

    with_gemini_cassette(
        "hook_stress_streaming/streaming_tool_turns_fire_model_turn_finished",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            install_streaming_tap(&agent, tap);

            let mut stream = agent
                .stream_prompt(
                    "First add 40 and 2 with the add tool. Then subtract 10 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert_nonempty_response(&final_text);

            assert!(
                probe.count("ToolCall") >= 1,
                "the streamed run should call tools"
            );
            assert!(
                probe.count("ModelTurnFinished") >= 2,
                "ModelTurnFinished must fire once per accepted turn on the streaming surface, \
                 including tool turns"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_result_redaction_reaches_final_response() {
    let add = CountingAdd::default();

    with_gemini_cassette(
        "hook_stress_streaming/streaming_result_redaction_reaches_final_response",
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
                "stream-redaction",
                0,
                1,
                PolicyRule::RewriteToolResult {
                    tool: Some("add".to_owned()),
                    presentation: "STREAM-REDACTED-Q3".into(),
                },
            )
            .expect("redaction policy should install");

            let mut stream = agent
                .stream_prompt(
                    "Use the add tool to add 5 and 5, then report the exact tool result.",
                )
                .max_turns(4)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert!(
                final_text.contains("STREAM-REDACTED-Q3"),
                "the redacted result must reach the streamed final response: {final_text:?}"
            );
            assert!(
                !final_text.contains("10"),
                "the raw tool result must not reach the model: {final_text:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_active_tools_narrowing_filters_a_tool() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();

    with_gemini_cassette(
        "hook_stress_streaming/streaming_active_tools_narrowing_filters_a_tool",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. Use a provided tool for any arithmetic you \
                     can; if a tool is unavailable, say so and continue.",
                )
                .tool(add)
                .tool(subtract)
                .build();

            install_policy(
                &agent,
                "stream-add-only",
                0,
                1,
                PolicyRule::PatchRequest(
                    RequestPatch::new().active_tools(["add"]).temperature(0.0),
                ),
            )
            .expect("tool narrowing policy should install");

            let mut stream = agent
                .stream_prompt("Compute 12 + 8, then compute 30 - 7. Report whichever you can.")
                .max_turns(5)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert_nonempty_response(&final_text);
            assert!(
                add_calls.count() >= 1,
                "add stays advertised and should run"
            );
            assert_eq!(
                subtract_calls.count(),
                0,
                "subtract is filtered out of active_tools on the streaming surface too"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_skip_leaves_tool_unexecuted() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let subtract_calls = subtract.counter.clone();

    with_gemini_cassette(
        "hook_stress_streaming/streaming_skip_leaves_tool_unexecuted",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. You MUST use the provided tools. If a tool \
                     reports it is unavailable, acknowledge that and report any results you have.",
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
                    tool: Some("subtract".to_owned()),
                    reason: "the subtract tool is offline; continue without it".to_owned(),
                },
            )
            .expect("skip policy should install");

            let mut stream = agent
                .stream_prompt("Add 14 and 6, and subtract 9 from 40. Report what you can.")
                .max_turns(5)
                .await;

            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert_nonempty_response(&final_text);
            assert_eq!(
                subtract_calls.count(),
                0,
                "a skipped tool must never execute on the streaming surface"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn blocking_and_streaming_produce_same_final_answer() {
    const PROMPT: &str = "First add 10 and 5 with the add tool. Then subtract 3 from that sum with \
         the subtract tool. Report the final number.";
    const EXPECTED: i32 = 12;

    // Blocking surface.
    let add_b = CountingAdd::default();
    let sub_b = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress_streaming/parity_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add_b)
                .tool(sub_b)
                .build();
            let response = agent
                .prompt(PROMPT)
                .max_turns(6)
                .await
                .expect("blocking parity run should succeed");
            assert_mentions_expected_number(&response, EXPECTED);
        },
    )
    .await;

    // Streaming surface — same workflow, same expected answer.
    let add_s = CountingAdd::default();
    let sub_s = CountingSubtract::default();
    with_gemini_cassette(
        "hook_stress_streaming/parity_streaming",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add_s)
                .tool(sub_s)
                .build();
            let mut stream = agent.stream_prompt(PROMPT).max_turns(6).await;
            let final_text = collect_stream_final_response(&mut stream)
                .await
                .expect("a final response");
            assert_mentions_expected_number(&final_text, EXPECTED);
        },
    )
    .await;
}
