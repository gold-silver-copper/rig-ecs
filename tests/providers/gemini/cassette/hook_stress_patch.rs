//! ECS request-policy stress suite: `RequestPatch` steering —
//! instruction override, `tool_choice`, per-turn `history` replacement, and
//! multi-field patches. Recorded against real Gemini; each patch effect is
//! proven by a downstream-observable change (the model can't echo settings).

use rig::bevy_ecs::prelude::{Commands, On};
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::runtime::{
    CompletionRequestPrepared, ModelToolChoice, PolicyRule, PolicyStatus, RequestPatch,
    RetrievedDocument, TranscriptEntry,
};

use super::super::support::with_gemini_cassette;
use super::super::tools_support::CountingAdd;
use crate::support::{assert_nonempty_response, install_policy};

const CODEWORD: &str = "ZULU-99";

fn install_request_patch(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    id: &str,
    patch: RequestPatch,
) {
    install_policy(agent, id, 0, 1, PolicyRule::PatchRequest(patch))
        .expect("request policy should install");
}

fn install_first_request_patch(
    agent: &rig::agent::Agent<gemini::completion::CompletionModel>,
    id: &str,
    patch: RequestPatch,
) {
    let policy = install_policy(agent, id, 0, 1, PolicyRule::PatchRequest(patch))
        .expect("request policy should install");
    agent
        .with_runtime_mut(move |runtime| {
            runtime.world_mut().add_observer(
                move |_event: On<CompletionRequestPrepared>, mut commands: Commands| {
                    commands.entity(policy).insert(PolicyStatus::Retired);
                },
            );
        })
        .expect("retirement observer should install");
}

#[tokio::test]
async fn preamble_override_forces_codeword_blocking() {
    with_gemini_cassette(
        "hook_stress_patch/preamble_override_forces_codeword_blocking",
        |client| async move {
            // The agent's own preamble says nothing about a codeword.
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a terse assistant.")
                .build();

            install_request_patch(
                &agent,
                "override-instructions",
                RequestPatch::new()
                    .instructions(format!(
                        "You are a terse assistant. End every reply with the exact token \
                         {CODEWORD} on its own, verbatim."
                    ))
                    .temperature(0.0),
            );

            // A hook overrides the preamble for this turn to require a codeword
            // suffix — a behavior change only the injected preamble can cause.
            let response = agent
                .prompt("Greet me in one short sentence.")
                .max_turns(2)
                .await
                .expect("preamble-override run should succeed");

            assert!(
                response.contains(CODEWORD),
                "the overridden preamble must change behavior; answer: {response:?}"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn tool_choice_required_forces_a_tool_call_blocking() {
    let add = CountingAdd::default();
    let add_calls = add.counter.clone();

    with_gemini_cassette(
        "hook_stress_patch/tool_choice_required_forces_a_tool_call_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a calculator assistant.")
                .tool(add)
                .build();

            install_first_request_patch(
                &agent,
                "require-first-tool",
                RequestPatch::new()
                    .tool_choice(ModelToolChoice::Required)
                    .temperature(0.0),
            );

            // Force tool_choice = Required on the FIRST turn only, so the model
            // must call the tool up front. (Forcing it every turn would force a
            // tool call on every turn and loop until max_turns — a real footgun
            // this stress test surfaced.)
            let response = agent
                .prompt("Use the add tool to compute 12 plus 30, then report the number.")
                .max_turns(4)
                .await
                .expect("tool_choice=Required run should succeed");

            assert_nonempty_response(&response);
            assert!(
                add_calls.count() >= 1,
                "tool_choice=Required (via RequestPatch) must force a tool call"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn history_replacement_injects_prior_fact_blocking() {
    with_gemini_cassette(
        "hook_stress_patch/history_replacement_injects_prior_fact_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a helpful assistant. Use the conversation so far to answer.")
                .build();

            install_request_patch(
                &agent,
                "replace-history",
                RequestPatch::new()
                    .history([TranscriptEntry::User(
                        "For this session, the passphrase is OMEGA-7. Acknowledge and remember it."
                            .to_owned(),
                    )])
                    .temperature(0.0),
            );

            // A hook replaces the messages sent this turn with a synthetic prior
            // exchange that establishes a fact — the model answers from it.
            let response = agent
                .prompt("What is the passphrase?")
                .max_turns(2)
                .await
                .expect("history-replacement run should succeed");

            assert!(
                response.contains("OMEGA-7"),
                "the per-turn history view must reach the model; answer: {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn multi_field_patch_applies_preamble_and_context_blocking() {
    with_gemini_cassette(
        "hook_stress_patch/multi_field_patch_applies_preamble_and_context_blocking",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .name("stress-agent")
                .preamble("You are a terse assistant.")
                .build();

            install_request_patch(
                &agent,
                "instructions-and-context",
                RequestPatch::new()
                    .instructions(format!(
                        "You are a terse assistant. End every reply with the exact token \
                         {CODEWORD}."
                    ))
                    .extra_context([RetrievedDocument {
                        id: "depot".to_owned(),
                        text: "The depot code is GAMMA-33.".to_owned(),
                        metadata: Default::default(),
                    }])
                    .temperature(0.0),
            );

            // One patch sets BOTH the preamble and an extra_context document; both
            // fields must take effect.
            let response = agent
                .prompt("What is the depot code? Keep it short.")
                .max_turns(2)
                .await
                .expect("multi-field patch run should succeed");

            assert!(
                response.contains("GAMMA-33"),
                "the patch's extra_context must reach the model; answer: {response:?}"
            );
            assert!(
                response.contains(CODEWORD),
                "the patch's preamble override must also take effect; answer: {response:?}"
            );
        },
    )
    .await;
}
