//! Two complementary ways to drive the ECS agent lifecycle.
//!
//! Part 1 advances the public schedule one pass at a time, suspends a live tool
//! operation, serializes the stable-ID run graph, and resumes it in a fresh
//! world. Part 2 uses `run_until_stalled` while a normal observe-only ECS
//! listener reports prepared tool calls.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::{Context, Result};
use rig::bevy_ecs::observer::On;
use rig::runtime::{
    ActiveRunSnapshot, EffectRequest, PauseMode, RunState, Runtime, RuntimeConfig, ToolCallPrepared,
};

fn step_until_effect(runtime: &mut Runtime, label: &str) -> Result<EffectRequest> {
    for pass in 1..=32 {
        runtime.update();
        let state = {
            let world = runtime.world_mut();
            let mut runs = world.query::<&RunState>();
            runs.iter(world).next().cloned()
        };
        println!("{label} pass {pass}: {state:?}");
        if let Some(effect) = runtime.effects().try_recv()? {
            return Ok(effect);
        }
    }
    anyhow::bail!("{label} made no externally visible progress")
}

fn drive_to_terminal(runtime: &mut Runtime, run: rig::runtime::RunHandle) -> Result<RunState> {
    for _ in 0..32 {
        runtime.update();
        if let Some(state @ (RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled)) =
            runtime.observe_run(run)?
        {
            return Ok(state);
        }
    }
    anyhow::bail!("run did not reach a terminal phase")
}

fn hand_driven_checkpoint() -> Result<()> {
    println!("=== Part 1: hand-driven schedule and checkpoint ===");
    let (mut source, agent) = ecs_demo::runtime(true)?;
    let domain = source.snapshot()?;
    let pending = source.handle().prompt(agent, "Use lookup, then answer")?;

    let model = step_until_effect(&mut source, "model")?;
    ecs_demo::complete_with_tool(&source, &model, "lookup")?;
    let live_tool = step_until_effect(&mut source, "tool")?;
    let source_run = source
        .resolve_run(&pending)
        .context("the stepped prompt should resolve to a run")?;
    source
        .handle()
        .pause_with_mode(source_run, PauseMode::CancelAndSuspend)?;
    source.run_until_stalled()?;
    let checkpoint = source.snapshot_active_run(source_run)?;
    let encoded = serde_json::to_string_pretty(&checkpoint)?;
    let checkpoint: ActiveRunSnapshot = serde_json::from_str(&encoded)?;
    println!(
        "serialized {} bytes with tool generation {} suspended",
        encoded.len(),
        live_tool.generation
    );

    let mut restored = Runtime::new(RuntimeConfig::default())?;
    restored.restore(domain)?;
    let runs = restored.restore_active_run(checkpoint)?;
    let restored_entity = runs
        .0
        .get(pending.stable_id())
        .copied()
        .context("restored graph omitted the root run")?;
    let restored_run = restored.run_handle(restored_entity)?;
    restored.handle().resume(restored_run)?;
    let restored_tool = step_until_effect(&mut restored, "restored tool")?;
    anyhow::ensure!(
        restored_tool.generation > live_tool.generation,
        "restoration must redispatch with a fresh generation"
    );
    ecs_demo::complete_tool(&restored, &restored_tool, "restored lookup result")?;
    let final_model = step_until_effect(&mut restored, "final model")?;
    ecs_demo::complete_text(&restored, &final_model, "completed after restoration")?;
    println!(
        "terminal: {:?}\n",
        drive_to_terminal(&mut restored, restored_run)?
    );
    Ok(())
}

fn observe_prepared_tool(event: On<ToolCallPrepared>) {
    println!(
        "[observer] run={:?} tool={} index={}",
        event.run, event.call.decision.name, event.call.index
    );
}

fn schedule_driven_observation() -> Result<()> {
    println!("=== Part 2: run-until-stalled with observation ===");
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.world_mut().add_observer(observe_prepared_tool);
    let pending = runtime.handle().prompt(agent, "Use lookup normally")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_with_tool(&runtime, &model, "lookup")?;
    let tool = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_tool(&runtime, &tool, "ordinary lookup result")?;
    let final_model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_text(&runtime, &final_model, "normal schedule completed")?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .context("the observed prompt should resolve to a run")?;
    println!("terminal: {:?}", runtime.observe_run(run)?);
    Ok(())
}

fn drain_pause_with_active_sibling() -> Result<()> {
    println!("=== Part 3: drain pause beside an active sibling ===");
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    let paused_pending = runtime
        .handle()
        .prompt(agent, "pause before tool dispatch")?;
    let sibling_pending = runtime.handle().prompt(agent, "complete independently")?;
    runtime.run_until_stalled()?;
    let paused = runtime
        .resolve_run(&paused_pending)
        .context("paused run was not admitted")?;
    let sibling = runtime
        .resolve_run(&sibling_pending)
        .context("sibling run was not admitted")?;
    let paused_operation = match runtime.world().get::<RunState>(paused.entity()) {
        Some(RunState::WaitingModel { operation }) => *operation,
        state => anyhow::bail!("paused run is not waiting on a model: {state:?}"),
    };
    let sibling_operation = match runtime.world().get::<RunState>(sibling.entity()) {
        Some(RunState::WaitingModel { operation }) => *operation,
        state => anyhow::bail!("sibling run is not waiting on a model: {state:?}"),
    };
    let requests = [
        runtime
            .effects()
            .try_recv()?
            .context("missing model request")?,
        runtime
            .effects()
            .try_recv()?
            .context("missing model request")?,
    ];
    let paused_request = requests
        .iter()
        .find(|request| request.operation == paused_operation)
        .context("missing paused request")?;
    let sibling_request = requests
        .iter()
        .find(|request| request.operation == sibling_operation)
        .context("missing sibling request")?;
    runtime.handle().pause(paused)?;
    ecs_demo::complete_with_tool(&runtime, paused_request, "lookup")?;
    ecs_demo::complete_text(&runtime, sibling_request, "sibling completed while paused")?;
    runtime.run_until_stalled()?;
    anyhow::ensure!(
        matches!(
            runtime.world().get::<RunState>(sibling.entity()),
            Some(RunState::Completed(_))
        ),
        "drain-paused run blocked its sibling"
    );
    anyhow::ensure!(
        runtime.effects().try_recv()?.is_none(),
        "drain pause dispatched new tool work"
    );
    runtime.handle().resume(paused)?;
    let tool = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_tool(&runtime, &tool, "resumed tool result")?;
    let final_model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_text(&runtime, &final_model, "drain-paused run completed")?;
    runtime.run_until_stalled()?;
    println!(
        "paused: {:?}, sibling: {:?}",
        runtime.observe_run(paused)?,
        runtime.observe_run(sibling)?
    );
    Ok(())
}

fn main() -> Result<()> {
    hand_driven_checkpoint()?;
    schedule_driven_observation()?;
    drain_pause_with_active_sibling()
}
