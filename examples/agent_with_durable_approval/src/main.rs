//! Checkpoint an approval wait and resume it in a fresh ECS runtime.
//!
//! The checkpoint contains stable IDs, the ordered policy cursor, and the
//! pending approval decision. Runtime-local entities and effect channels are
//! rebuilt from the separately persisted domain snapshot.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::{Context, Result};
use rig::runtime::{
    ActiveRunSnapshot, EffectCompletion, EffectOutput, PauseMode, Policy,
    PolicyApprovalEffectOutput, PolicyPoint, PolicyRule, RunState, Runtime, RuntimeConfig,
};

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(false)?;
    runtime.spawn_policy(
        ecs_demo::id("durable-approval")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 4,
            rule: PolicyRule::RequireApproval {
                point: PolicyPoint::Request,
                prompt: "Approve model request?".to_owned(),
            },
        },
        agent,
    )?;
    let domain = runtime.snapshot()?;
    let pending = runtime.handle().prompt(agent, "A durable request")?;
    let original_approval = ecs_demo::next_effect(&mut runtime)?;
    let run = runtime
        .resolve_run(&pending)
        .context("the prompt should resolve to a run entity")?;
    runtime
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)?;
    runtime.run_until_stalled()?;

    let encoded = serde_json::to_string_pretty(&runtime.snapshot_active_run(run)?)?;
    let checkpoint: ActiveRunSnapshot = serde_json::from_str(&encoded)?;
    println!(
        "checkpointed {} bytes while approval generation {} was live",
        encoded.len(),
        original_approval.generation
    );

    let mut restored = Runtime::new(RuntimeConfig::default())?;
    restored.restore(domain)?;
    let restored_runs = restored.restore_active_run(checkpoint)?;
    let restored_entity = restored_runs
        .0
        .get(pending.stable_id())
        .copied()
        .context("the restored graph should contain the root run")?;
    let restored_run = restored.run_handle(restored_entity)?;
    restored.handle().resume(restored_run)?;

    let approval = ecs_demo::next_effect(&mut restored)?;
    let approval_input = approval
        .policy_approval_input()
        .context("restoration should redispatch the pending approval")?;
    println!(
        "restored approval policy={} revision={}",
        approval_input.policy_id.as_str(),
        approval_input.revision
    );
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: None,
            })),
        })?;
    let model = ecs_demo::next_effect(&mut restored)?;
    ecs_demo::complete_text(&restored, &model, "approved after restoration")?;
    restored.run_until_stalled()?;
    match restored.observe_run(restored_run)? {
        Some(RunState::Completed(output)) => {
            println!("restored run completed: {}", output.text);
        }
        state => anyhow::bail!("restored run did not complete: {state:?}"),
    }
    Ok(())
}
