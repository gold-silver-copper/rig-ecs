//! Multiple agents and independently controlled runs in one ECS world.
//!
//! A general assistant delegates Spanish translation to a translator agent.
//! The child is freeze-paused after its completion reaches ingress; while it is
//! paused, an unrelated run on the same assistant completes. Resuming the child
//! commits its result and releases the parent without any lockstep execution.

use anyhow::{Context, Result};
use rig::runtime::{
    Agent, EffectCompletion, EffectOutput, ModelCapability, ModelEffectOutput, PauseMode,
    RunControl, RunRecord, RunState, Runtime, RuntimeConfig, StableId, TenantId, TranscriptEntry,
    Usage, WaitingForChildren,
};

fn id(value: &str) -> Result<StableId> {
    Ok(StableId::new(value)?)
}

fn complete(runtime: &Runtime, request: &rig::runtime::EffectRequest, text: &str) -> Result<()> {
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: text.to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })?;
    Ok(())
}

fn main() -> Result<()> {
    let tenant = TenantId::new("multi-agent-example")?;
    let mut runtime = Runtime::new(RuntimeConfig::default())?;
    let model = runtime.spawn_model(
        id("shared-model")?,
        tenant.clone(),
        ModelCapability {
            provider: "example".to_owned(),
            model: "deterministic".to_owned(),
            revision: 1,
            retired: false,
        },
    )?;
    let assistant = runtime.spawn_agent(
        id("assistant")?,
        tenant.clone(),
        Agent {
            name: Some("general assistant".to_owned()),
            instructions: "Delegate non-English input to the translator".to_owned(),
            ..Agent::default()
        },
        model,
    )?;
    let translator = runtime.spawn_agent(
        id("translator")?,
        tenant,
        Agent {
            name: Some("translator".to_owned()),
            instructions: "Translate input to English".to_owned(),
            ..Agent::default()
        },
        model,
    )?;

    let parent_pending = runtime.handle().prompt(assistant, "¿Cómo estás?")?;
    runtime.run_until_stalled()?;
    let parent_request = runtime
        .effects()
        .try_recv()?
        .context("expected the assistant model effect")?;
    let parent = runtime
        .resolve_run(&parent_pending)
        .context("assistant prompt was not admitted")?;
    let child_pending =
        runtime
            .handle()
            .spawn_child_run(parent, translator, "Translate: ¿Cómo estás?")?;
    runtime.run_until_stalled()?;
    let child_request = runtime
        .effects()
        .try_recv()?
        .context("expected the translator model effect")?;
    let child = runtime
        .resolve_run(&child_pending)
        .context("translator child was not admitted")?;
    runtime
        .handle()
        .pause_with_mode(child, PauseMode::FreezeAfterIngress)?;
    complete(&runtime, &child_request, "How are you?")?;
    complete(
        &runtime,
        &parent_request,
        "Translated: How are you? Final response: I am well.",
    )?;
    runtime.run_until_stalled()?;
    anyhow::ensure!(
        runtime.world().get::<RunControl>(child.entity())
            == Some(&RunControl::Paused(PauseMode::FreezeAfterIngress)),
        "translator did not freeze after validated ingress"
    );
    anyhow::ensure!(
        runtime
            .world()
            .get::<WaitingForChildren>(parent.entity())
            .is_some(),
        "parent advanced while its child was paused"
    );

    let sibling_pending = runtime
        .handle()
        .prompt(assistant, "Answer this unrelated English request")?;
    runtime.run_until_stalled()?;
    let sibling_request = runtime
        .effects()
        .try_recv()?
        .context("expected the unrelated sibling effect")?;
    complete(&runtime, &sibling_request, "unrelated run completed")?;
    runtime.run_until_stalled()?;
    let sibling = runtime
        .resolve_run(&sibling_pending)
        .context("sibling prompt was not admitted")?;
    println!(
        "sibling while translator paused: {:?}",
        runtime.observe_run(sibling)?
    );
    anyhow::ensure!(
        matches!(
            runtime.world().get::<RunState>(sibling.entity()),
            Some(RunState::Completed(_))
        ),
        "paused child blocked an unrelated sibling"
    );

    runtime.handle().resume(child)?;
    runtime.run_until_stalled()?;
    let translated = runtime
        .world()
        .get::<RunRecord>(parent.entity())
        .context("parent run record disappeared")?
        .transcript
        .iter()
        .find_map(|entry| match entry {
            TranscriptEntry::ChildResult { result, .. } => result.as_ref().ok(),
            _ => None,
        })
        .context("parent did not receive the translator result")?;
    println!("translator result: {translated}");
    println!("parent terminal: {:?}", runtime.observe_run(parent)?);
    Ok(())
}
