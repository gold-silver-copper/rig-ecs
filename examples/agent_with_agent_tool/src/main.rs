//! Delegate active parent work to a dynamically spawned child agent.
//!
//! Agent-as-tool composition is represented directly as run topology: the host
//! creates a child agent while the parent is active, spawns two related child
//! runs, and lets all three progress through one schedule. Child results commit
//! by creation ordinal rather than completion order. A second scenario shows
//! parent cancellation propagating to an active child.

use anyhow::{Context, Result};
use rig::runtime::{
    Agent, EffectCompletion, EffectOutput, ModelCapability, ModelEffectOutput, ParentRun,
    RunRecord, RunState, Runtime, RuntimeConfig, StableId, TenantId, TranscriptEntry, Usage,
    WaitingForChildren,
};

fn id(value: &str) -> Result<StableId> {
    Ok(StableId::new(value)?)
}

fn tenant() -> Result<TenantId> {
    Ok(TenantId::new("delegation-example")?)
}

fn runtime() -> Result<(Runtime, rig::runtime::AgentHandle)> {
    let mut runtime = Runtime::new(RuntimeConfig::default())?;
    let model = runtime.spawn_model(
        id("model")?,
        tenant()?,
        ModelCapability {
            provider: "example".to_owned(),
            model: "deterministic".to_owned(),
            revision: 1,
            retired: false,
        },
    )?;
    let parent = runtime.spawn_agent(
        id("parent-agent")?,
        tenant()?,
        Agent {
            name: Some("orchestrator".to_owned()),
            instructions: "Delegate arithmetic subtasks".to_owned(),
            ..Agent::default()
        },
        model,
    )?;
    Ok((runtime, parent))
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

fn ordered_delegation() -> Result<()> {
    println!("=== dynamic child-agent delegation ===");
    let (mut runtime, parent_agent) = runtime()?;
    let parent_pending = runtime
        .handle()
        .prompt(parent_agent, "Compute two independent arithmetic subtasks")?;
    runtime.run_until_stalled()?;
    let parent_request = runtime
        .effects()
        .try_recv()?
        .context("expected the parent model effect")?;
    let parent = runtime
        .resolve_run(&parent_pending)
        .context("the parent prompt should resolve to a run")?;

    let pending_child_agent = runtime.handle().spawn_agent(
        id("calculator-agent")?,
        tenant()?,
        Agent {
            name: Some("calculator".to_owned()),
            instructions: "Solve one arithmetic subtask".to_owned(),
            ..Agent::default()
        },
        id("model")?,
    )?;
    runtime.run_until_stalled()?;
    let child_agent = pending_child_agent
        .try_resolve()?
        .context("dynamic agent command was not reconciled")??;
    let first_pending =
        runtime
            .handle()
            .spawn_child_run(parent, child_agent, "First subtask: calculate 2 + 5")?;
    let second_pending =
        runtime
            .handle()
            .spawn_child_run(parent, child_agent, "Second subtask: calculate 9 - 4")?;
    runtime.run_until_stalled()?;
    let first = runtime
        .resolve_run(&first_pending)
        .context("first child was not admitted")?;
    let second = runtime
        .resolve_run(&second_pending)
        .context("second child was not admitted")?;
    anyhow::ensure!(
        runtime.world().get::<ParentRun>(first.entity()) == Some(&ParentRun(parent.entity()))
            && runtime.world().get::<ParentRun>(second.entity())
                == Some(&ParentRun(parent.entity())),
        "child relationships were not established"
    );
    anyhow::ensure!(
        runtime
            .world()
            .get::<WaitingForChildren>(parent.entity())
            .is_some(),
        "parent did not enter WaitingForChildren"
    );
    let first_operation = match runtime.world().get::<RunState>(first.entity()) {
        Some(RunState::WaitingModel { operation }) => *operation,
        state => anyhow::bail!("first child is not waiting on a model: {state:?}"),
    };
    let second_operation = match runtime.world().get::<RunState>(second.entity()) {
        Some(RunState::WaitingModel { operation }) => *operation,
        state => anyhow::bail!("second child is not waiting on a model: {state:?}"),
    };
    let requests = [
        runtime
            .effects()
            .try_recv()?
            .context("missing child request")?,
        runtime
            .effects()
            .try_recv()?
            .context("missing child request")?,
    ];
    let request_for = |operation| {
        requests
            .iter()
            .find(|request| request.operation == operation)
            .context("child request did not match its run operation")
    };

    // The parent result is retained until every child has committed.
    complete(
        &runtime,
        &parent_request,
        "parent synthesized the child results",
    )?;
    // Complete the second child first to prove external timing is not semantic order.
    complete(&runtime, request_for(second_operation)?, "9 - 4 = 5")?;
    runtime.run_until_stalled()?;
    anyhow::ensure!(
        runtime
            .world()
            .get::<WaitingForChildren>(parent.entity())
            .is_some(),
        "second-child completion must not release the parent early"
    );
    complete(&runtime, request_for(first_operation)?, "2 + 5 = 7")?;
    runtime.run_until_stalled()?;

    let child_results = runtime
        .world()
        .get::<RunRecord>(parent.entity())
        .context("parent record disappeared")?
        .transcript
        .iter()
        .filter_map(|entry| match entry {
            TranscriptEntry::ChildResult { result, .. } => result.as_ref().ok().cloned(),
            _ => None,
        })
        .collect::<Vec<_>>();
    println!("committed child results: {child_results:?}");
    anyhow::ensure!(
        child_results == ["2 + 5 = 7", "9 - 4 = 5"],
        "child results were not committed in creation order"
    );
    println!("parent terminal: {:?}\n", runtime.observe_run(parent)?);
    Ok(())
}

fn cancellation_propagation() -> Result<()> {
    println!("=== parent cancellation propagation ===");
    let (mut runtime, parent_agent) = runtime()?;
    let parent_pending = runtime.handle().prompt(parent_agent, "delegate and wait")?;
    runtime.run_until_stalled()?;
    let _parent_effect = runtime
        .effects()
        .try_recv()?
        .context("missing parent effect")?;
    let parent = runtime
        .resolve_run(&parent_pending)
        .context("missing parent run")?;
    let child_pending =
        runtime
            .handle()
            .spawn_child_run(parent, parent_agent, "long-running child")?;
    runtime.run_until_stalled()?;
    let _child_effect = runtime
        .effects()
        .try_recv()?
        .context("missing child effect")?;
    let child = runtime
        .resolve_run(&child_pending)
        .context("missing child run")?;
    runtime.handle().cancel(parent)?;
    runtime.run_until_stalled()?;
    anyhow::ensure!(
        runtime.observe_run(child)? == Some(RunState::Cancelled),
        "active child did not inherit parent cancellation"
    );
    println!(
        "parent: {:?}, child: {:?}",
        runtime.observe_run(parent)?,
        runtime.observe_run(child)?
    );
    Ok(())
}

fn main() -> Result<()> {
    ordered_delegation()?;
    cancellation_propagation()
}
