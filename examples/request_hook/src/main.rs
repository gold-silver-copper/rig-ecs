//! Compose request steering and observation with ECS policies and observers.
//!
//! Two ordered policy entities contribute independent per-operation patches:
//! one appends context and one changes sampling. Observe-only systems log the
//! resolved run/agent identity and update typed run-scoped state. Observer
//! registration order has no steering semantics.

use anyhow::Result;
use rig::bevy_ecs::{
    lifecycle::Add,
    observer::On,
    prelude::{Commands, Component, Query},
};
use rig::{
    client::{CompletionClient, ProviderClient},
    providers::openai,
    runtime::{
        CompletionRequestPrepared, CompletionResponseApplied, Policy, PolicyRule, RequestPatch,
        RetrievedDocument, RigOperationContext, RunRecord, StableId, TenantId,
    },
};

#[derive(Component, Default)]
struct CompletionCallCount(u32);

fn initialize_run_state(event: On<Add, RunRecord>, mut commands: Commands) {
    commands
        .entity(event.entity)
        .insert(CompletionCallCount::default());
}

fn observe_request(
    event: On<CompletionRequestPrepared>,
    context: RigOperationContext<'_, '_>,
    mut counters: Query<&mut CompletionCallCount>,
) {
    let Some(context) = context.resolve(event.operation) else {
        return;
    };
    let call = counters.get_mut(event.run).map_or(0, |mut counter| {
        counter.0 = counter.0.saturating_add(1);
        counter.0
    });
    println!(
        "request run={} turn={} streaming=false agent={:?} policies={}",
        context.run_id.as_str(),
        call,
        context.agent_name,
        event.policies.len()
    );
}

fn observe_response(event: On<CompletionResponseApplied>) {
    println!(
        "response run={:?} text_bytes={}",
        event.run,
        event.effective.text.len()
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .name("request-policy-example")
        .preamble("You are a comedian. Use supplied context and keep the joke family-friendly.")
        .build();

    agent.with_runtime_mut(|runtime| {
        runtime.world_mut().add_observer(initialize_run_state);
        runtime.world_mut().add_observer(observe_request);
        runtime.world_mut().add_observer(observe_response);

        runtime.spawn_policy(
            StableId::new("append-comedy-context")?,
            TenantId::new("local")?,
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::PatchRequest(RequestPatch::new().extra_context([
                    RetrievedDocument {
                        id: "house-style".to_owned(),
                        text: "House rule: the punchline must mention a polite penguin.".to_owned(),
                        metadata: Default::default(),
                    },
                ])),
            },
            agent.handle(),
        )?;
        runtime.spawn_policy(
            StableId::new("lower-sampling-temperature")?,
            TenantId::new("local")?,
            Policy {
                order: 1,
                revision: 1,
                rule: PolicyRule::PatchRequest(RequestPatch::new().temperature(0.2)),
            },
            agent.handle(),
        )?;
        Ok::<_, anyhow::Error>(())
    })??;

    let response = agent.prompt("Tell me a short joke.").await?;
    println!("\nFinal response:\n{response}");
    Ok(())
}
