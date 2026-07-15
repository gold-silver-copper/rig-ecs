//! ECS observation stress suite: run identity, a typed agent component
//! shared across observers and turns, and extension composition (multiple taps,
//! observe-only both-fire, request patch
//! accumulation, `active_tools` intersection). Recorded against real Gemini.
//!
//! Assertions are loose for model-shaped values and exact only for
//! rig-synthesized values (see `tools_support`'s note).

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use rig::bevy_ecs;
use rig::bevy_ecs::prelude::{Component, On, Query};
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::runtime::{
    Agent as RuntimeAgent, CompletionRequestPrepared, ModelTurnFinished, PolicyRule, RequestPatch,
    RetrievedDocument, ToolCallPrepared, ToolResultPresentationFinalized,
};

use super::super::support::with_gemini_cassette;
use super::super::tools_support::{CountingAdd, CountingSubtract, EmbedMultiply};
use crate::support::{assert_nonempty_response, install_policy};

const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself. Perform the steps in order, using the result of each step as an input to the next. Once you have the final tool result, reply with the final numeric answer in plain text.";

#[derive(Clone, Debug, PartialEq, Eq)]
struct Breadcrumb {
    tag: &'static str,
    turn: usize,
}

#[derive(Clone, Default)]
struct EventTap {
    breadcrumbs: Arc<Mutex<Vec<Breadcrumb>>>,
    run_ids: Arc<Mutex<BTreeSet<u64>>>,
    streaming: Arc<Mutex<Option<bool>>>,
    agent_name: Arc<Mutex<Option<String>>>,
    call_ids: Arc<Mutex<Vec<String>>>,
    result_ids: Arc<Mutex<Vec<String>>>,
}

impl EventTap {
    fn record(&self, run: rig::bevy_ecs::entity::Entity, tag: &'static str, turn: usize) {
        self.run_ids.lock().expect("run ids").insert(run.to_bits());
        self.breadcrumbs
            .lock()
            .expect("breadcrumbs")
            .push(Breadcrumb { tag, turn });
    }

    fn distinct_run_ids(&self) -> usize {
        self.run_ids.lock().expect("run ids").len()
    }

    fn is_streaming(&self) -> Option<bool> {
        *self.streaming.lock().expect("streaming")
    }

    fn agent_name(&self) -> Option<String> {
        self.agent_name.lock().expect("agent name").clone()
    }

    fn count(&self, tag: &str) -> usize {
        self.breadcrumbs
            .lock()
            .expect("breadcrumbs")
            .iter()
            .filter(|breadcrumb| breadcrumb.tag == tag)
            .count()
    }

    fn distinct_turns(&self) -> Vec<usize> {
        self.breadcrumbs
            .lock()
            .expect("breadcrumbs")
            .iter()
            .map(|breadcrumb| breadcrumb.turn)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn call_ids(&self) -> Vec<String> {
        self.call_ids.lock().expect("call ids").clone()
    }

    fn result_ids(&self) -> Vec<String> {
        self.result_ids.lock().expect("result ids").clone()
    }
}

fn install_tap(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    tap: EventTap,
    streaming: bool,
) {
    let agent_entity = agent.handle().entity();
    agent
        .with_runtime_mut(move |runtime| {
            *tap.streaming.lock().expect("streaming") = Some(streaming);
            *tap.agent_name.lock().expect("agent name") = runtime
                .world()
                .get::<RuntimeAgent>(agent_entity)
                .and_then(|agent| agent.name.clone());

            let request_tap = tap.clone();
            runtime
                .world_mut()
                .add_observer(move |event: On<CompletionRequestPrepared>| {
                    request_tap.record(event.run, "CompletionCall", 1);
                });
            let call_tap = tap.clone();
            runtime
                .world_mut()
                .add_observer(move |event: On<ToolCallPrepared>| {
                    call_tap.record(event.run, "ToolCall", 1);
                    call_tap
                        .call_ids
                        .lock()
                        .expect("call ids")
                        .push(event.call.call_id.clone());
                });
            let result_tap = tap.clone();
            runtime
                .world_mut()
                .add_observer(move |event: On<ToolResultPresentationFinalized>| {
                    result_tap.record(event.run, "ToolResult", 1);
                    result_tap
                        .result_ids
                        .lock()
                        .expect("result ids")
                        .push(event.raw.call_id.clone());
                });
            runtime
                .world_mut()
                .add_observer(move |event: On<ModelTurnFinished>| {
                    tap.record(event.run, "ModelTurnFinished", event.turn as usize + 1);
                });
        })
        .expect("tap observers should install");
}

#[derive(Component, Default)]
struct ToolCallTally(usize);

#[derive(Clone, Default)]
struct TallyReader(Arc<Mutex<Vec<usize>>>);

impl TallyReader {
    fn tallies(&self) -> Vec<usize> {
        self.0.lock().expect("tallies").clone()
    }
}

fn install_tally_observers(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    tap: EventTap,
    reader: TallyReader,
) {
    let agent_entity = agent.handle().entity();
    agent
        .with_runtime_mut(move |runtime| {
            runtime
                .world_mut()
                .entity_mut(agent_entity)
                .insert(ToolCallTally::default());
            runtime.world_mut().add_observer(
                move |event: On<ToolCallPrepared>, mut tallies: Query<&mut ToolCallTally>| {
                    tap.record(event.run, "ToolCall", 1);
                    if let Ok(mut tally) = tallies.get_mut(agent_entity) {
                        tally.0 += 1;
                    }
                },
            );
            runtime.world_mut().add_observer(
                move |_event: On<ModelTurnFinished>, tallies: Query<&ToolCallTally>| {
                    if let Ok(tally) = tallies.get(agent_entity) {
                        reader.0.lock().expect("tallies").push(tally.0);
                    }
                },
            );
        })
        .expect("typed tally observers should install");
}

fn fact(id: &str, text: &str) -> RetrievedDocument {
    RetrievedDocument {
        id: id.to_owned(),
        text: text.to_owned(),
        metadata: Default::default(),
    }
}

// ---------------------------------------------------------------------------
// HookContext identity + turn advancement.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hook_context_identity_stable_and_turn_advances_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let tap = EventTap::default();
    let probe = tap.clone();

    with_gemini_cassette(
        "hook_stress_context/hook_context_identity_stable_and_turn_advances_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            install_tap(&agent, tap, false);

            let response = agent
                .prompt(
                    "First add 9 and 6 with the add tool. Then subtract 4 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .await
                .expect("dependent chain should succeed");

            assert_nonempty_response(&response);
            assert_eq!(probe.distinct_run_ids(), 1, "run_id must be stable");
            assert_eq!(probe.is_streaming(), Some(false));
            assert_eq!(probe.agent_name().as_deref(), Some("stress-agent"));

            let turns = probe.distinct_turns();
            assert_eq!(
                turns.first().copied(),
                Some(1),
                "the first observed turn must be turn 1, saw {turns:?}"
            );
            assert!(
                turns.len() >= 2 && *turns.last().expect("turn") >= 2,
                "a dependent chain must reach >= 2 turns, saw {turns:?}"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn agent_name_absent_when_unconfigured_blocking() {
    let add = CountingAdd::default();
    let tap = EventTap::default();
    let probe = tap.clone();

    with_gemini_cassette(
        "hook_stress_context/agent_name_absent_when_unconfigured_blocking",
        |client| async move {
            // No `.name(..)` on the builder.
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            install_tap(&agent, tap, false);

            let response = agent
                .prompt("Use the add tool to add 3 and 4, then report the result.")
                .max_turns(4)
                .await
                .expect("run should succeed");

            assert_nonempty_response(&response);
            assert_eq!(
                probe.agent_name(),
                None,
                "agent_name() must be None when the agent has no configured name"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Scratchpad: shared across two hooks and growing across turns.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scratchpad_tally_grows_across_turns_and_is_read_by_second_hook_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let tap = EventTap::default();
    let reader = TallyReader::default();
    let tap_probe = tap.clone();
    let reader_probe = reader.clone();

    with_gemini_cassette(
        "hook_stress_context/scratchpad_tally_grows_across_turns_and_is_read_by_second_hook_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            install_tally_observers(&agent, tap, reader);

            let response = agent
                .prompt(
                    "First add 30 and 12 with the add tool. Then subtract 5 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .await
                .expect("dependent chain should succeed");

            assert_nonempty_response(&response);
            let total_calls = add_calls.count() + subtract_calls.count();
            let tallies = reader_probe.tallies();
            assert!(
                tallies.len() >= 2,
                "a multi-turn run should fire ModelTurnFinished at least twice, saw {tallies:?}"
            );
            assert!(
                tallies.windows(2).all(|w| w[0] <= w[1]),
                "the shared scratchpad tally must be non-decreasing, saw {tallies:?}"
            );
            assert!(
                tallies.first() < tallies.last(),
                "the tally must grow across turns (cross-turn shared state), saw {tallies:?}"
            );
            assert_eq!(
                *tallies.last().expect("a tally"),
                total_calls,
                "the final tally must equal the total ToolCall count"
            );
            assert_eq!(
                tap_probe.count("ToolCall"),
                total_calls,
                "observed ToolCall events must equal real executions"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Tool-call correlation: internal_call_id pairs ToolCall with ToolResult.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn internal_call_id_correlates_tool_call_and_result_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let tap = EventTap::default();
    let probe = tap.clone();

    with_gemini_cassette(
        "hook_stress_context/internal_call_id_correlates_tool_call_and_result_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .tool(subtract)
                .build();

            install_tap(&agent, tap, false);

            let response = agent
                .prompt(
                    "First add 7 and 7 with the add tool. Then subtract 2 from that sum with the \
                     subtract tool. Report the final number.",
                )
                .max_turns(6)
                .await
                .expect("dependent chain should succeed");

            assert_nonempty_response(&response);
            let call_ids = probe.call_ids();
            let result_ids = probe.result_ids();
            assert!(!call_ids.is_empty(), "the run should make tool calls");
            assert_eq!(
                call_ids, result_ids,
                "each ToolResult must carry the same internal_call_id as its ToolCall, in order"
            );
            assert!(
                call_ids.iter().all(|id| !id.is_empty()),
                "internal_call_ids must be non-empty"
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// HookStack composition: multiple observe-only hooks, add_hook append.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_observe_only_hooks_both_observe_the_run_blocking() {
    let add = CountingAdd::default();
    let first = EventTap::default();
    let second = EventTap::default();
    let first_probe = first.clone();
    let second_probe = second.clone();

    with_gemini_cassette(
        "hook_stress_context/two_observe_only_hooks_both_observe_the_run_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            install_tap(&agent, first, false);
            install_tap(&agent, second, false);

            let response = agent
                .prompt("Use the add tool to add 8 and 8, then report the result.")
                .max_turns(4)
                .await
                .expect("run should succeed");

            assert_nonempty_response(&response);
            // Both observe-only hooks see the same run — neither short-circuits.
            for tag in [
                "CompletionCall",
                "ToolCall",
                "ToolResult",
                "ModelTurnFinished",
            ] {
                assert_eq!(
                    first_probe.count(tag),
                    second_probe.count(tag),
                    "both hooks must observe the same number of {tag} events"
                );
            }
            assert!(first_probe.count("ToolCall") >= 1);
        },
    )
    .await;
}

#[tokio::test]
async fn add_hook_appends_across_builder_and_request_blocking() {
    let add = CountingAdd::default();
    let builder_hook = EventTap::default();
    let request_hook = EventTap::default();
    let builder_probe = builder_hook.clone();
    let request_probe = request_hook.clone();

    with_gemini_cassette(
        "hook_stress_context/add_hook_appends_across_builder_and_request_blocking",
        |client| async move {
            // One hook on the agent builder, one on the request: both must fire
            // (the request-level add_hook appends to the agent-default stack).
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(CHAIN_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .build();

            install_tap(&agent, builder_hook, false);
            install_tap(&agent, request_hook, false);

            let response = agent
                .prompt("Use the add tool to add 5 and 6, then report the result.")
                .max_turns(4)
                .await
                .expect("run should succeed");

            assert_nonempty_response(&response);
            assert!(
                builder_probe.count("CompletionCall") >= 1,
                "the agent-default (builder) hook must still fire"
            );
            assert!(
                request_probe.count("CompletionCall") >= 1,
                "the request-level hook must fire too"
            );
            assert_eq!(
                builder_probe.count("ToolCall"),
                request_probe.count("ToolCall"),
                "both stacked hooks observe the same tool calls"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn completion_call_patches_accumulate_from_two_hooks_blocking() {
    let add = CountingAdd::default();

    with_gemini_cassette(
        "hook_stress_context/completion_call_patches_accumulate_from_two_hooks_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a helpful assistant. Consult the provided context for any facts you \
                     are asked about; use a tool for arithmetic.",
                )
                .tool(add)
                .build();

            install_policy(
                &agent,
                "harbor-context",
                0,
                1,
                PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .extra_context([fact("harbor", "The harbor code is ALPHA-11.")])
                        .temperature(0.0),
                ),
            )
            .expect("first context policy should install");
            install_policy(
                &agent,
                "orchard-context",
                1,
                1,
                PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .extra_context([fact("orchard", "The orchard code is BETA-22.")]),
                ),
            )
            .expect("second context policy should install");

            // Two independent hooks each inject a different fact via extra_context.
            // Patches must accumulate (append), so BOTH facts reach the model.
            let response = agent
                .prompt("Tell me both the harbor code and the orchard code.")
                .max_turns(4)
                .await
                .expect("accumulated context run should succeed");

            assert!(
                response.contains("ALPHA-11"),
                "the first hook's injected fact must reach the model: {response:?}"
            );
            assert!(
                response.contains("BETA-22"),
                "the second hook's injected fact must reach the model too (patches accumulate): \
                 {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn two_hooks_narrow_active_tools_to_intersection_blocking() {
    let add = CountingAdd::default();
    let subtract = CountingSubtract::default();
    let multiply = EmbedMultiply::default();
    let add_calls = add.counter.clone();
    let subtract_calls = subtract.counter.clone();
    let multiply_calls = multiply.counter.clone();

    with_gemini_cassette(
        "hook_stress_context/two_hooks_narrow_active_tools_to_intersection_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble(
                    "You are a calculator assistant. Use a provided tool for any arithmetic you \
                     can. If a needed tool is unavailable, say so and move on.",
                )
                .tool(add)
                .tool(subtract)
                .tool(multiply)
                .build();

            install_policy(
                &agent,
                "add-or-subtract",
                0,
                1,
                PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .active_tools(["add", "subtract"])
                        .temperature(0.0),
                ),
            )
            .expect("first narrowing policy should install");
            install_policy(
                &agent,
                "add-or-multiply",
                1,
                1,
                PolicyRule::PatchRequest(RequestPatch::new().active_tools(["add", "multiply"])),
            )
            .expect("second narrowing policy should install");

            // Two narrowing hooks: {add, subtract} ∩ {add, multiply} == {add}.
            let response = agent
                .prompt(
                    "Compute 6 + 2, then 10 - 3, then 4 * 5. Report whichever results you can \
                     obtain.",
                )
                .max_turns(5)
                .await
                .expect("intersected-tools run should succeed");

            assert_nonempty_response(&response);
            // Only `add` is in the intersection, so only it can execute.
            assert!(
                add_calls.count() >= 1,
                "add is in the intersection and should run"
            );
            assert_eq!(
                subtract_calls.count(),
                0,
                "subtract is outside the intersection and must never execute"
            );
            assert_eq!(
                multiply_calls.count(),
                0,
                "multiply is outside the intersection and must never execute"
            );
        },
    )
    .await;
}
