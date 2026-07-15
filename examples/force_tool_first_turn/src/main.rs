//! # Forcing a tool on the first turn: a request-policy footgun and its fix
//!
//! Request patches are deliberately per-model-operation and non-sticky. An
//! unconditional policy is nevertheless evaluated again on every operation,
//! so it re-forces a tool until the run exhausts its model-call budget. The
//! corrected entity-targeted policy inspects typed run state and contributes
//! `Required` only before the first committed turn.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::{Context, Result};
use rig::bevy_ecs::{observer::On, prelude::Query};
use rig::runtime::{
    CanonicalError, ModelToolChoice, Policy, PolicyPoint, PolicyRule, RequestPatch,
    RequestPolicyDecision, RequestPolicyInvocation, RunRecord, RunState,
};

fn force_first_turn(mut event: On<RequestPolicyInvocation>, runs: Query<&RunRecord>) {
    let first_turn = runs
        .get(event.run)
        .is_ok_and(|record| record.next_turn == 0);
    event.decision = Some(if first_turn {
        RequestPolicyDecision::Patch(RequestPatch::new().tool_choice(ModelToolChoice::Required))
    } else {
        RequestPolicyDecision::Continue
    });
}

fn complete_tool_turn(runtime: &mut rig::runtime::Runtime) -> Result<ModelToolChoice> {
    let model = ecs_demo::next_effect(runtime)?;
    let choice = model
        .model_input()
        .context("expected a model effect")?
        .tool_choice
        .clone()
        .context("expected the policy to set tool choice")?;
    ecs_demo::complete_with_tool(runtime, &model, "lookup")?;
    let tool = ecs_demo::next_effect(runtime)?;
    ecs_demo::complete_tool(runtime, &tool, "lookup result")?;
    Ok(choice)
}

fn demonstrate_footgun() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.spawn_policy(
        ecs_demo::id("force-every-turn")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::PatchRequest(
                RequestPatch::new().tool_choice(ModelToolChoice::Required),
            ),
        },
        agent,
    )?;
    let pending = runtime.handle().prompt_configured(
        agent,
        "Look something up",
        std::iter::empty(),
        None,
        Some(2),
        None,
    )?;

    println!("=== unconditional policy (the footgun) ===");
    println!("turn 1: {:?}", complete_tool_turn(&mut runtime)?);
    println!("turn 2: {:?}", complete_tool_turn(&mut runtime)?);
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .context("the footgun prompt should resolve to a run")?;
    match runtime.observe_run(run)? {
        Some(RunState::Failed(CanonicalError::ModelCallBudget { limit: 2 })) => {
            println!("budget exhausted because Required was reapplied\n");
        }
        state => anyhow::bail!("unexpected footgun outcome: {state:?}"),
    }
    Ok(())
}

fn demonstrate_fix() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    let policy = runtime.spawn_policy(
        ecs_demo::id("force-first-turn")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::Custom(PolicyPoint::Request),
        },
        agent,
    )?;
    runtime
        .world_mut()
        .entity_mut(policy)
        .observe(force_first_turn);
    let pending = runtime.handle().prompt(agent, "Look something up")?;

    println!("=== run-state-gated policy (the fix) ===");
    println!("turn 1: {:?}", complete_tool_turn(&mut runtime)?);
    let final_model = ecs_demo::next_effect(&mut runtime)?;
    let final_input = final_model
        .model_input()
        .context("expected a second model effect")?;
    println!("turn 2: {:?} (agent baseline)", final_input.tool_choice);
    ecs_demo::complete_text(&runtime, &final_model, "finished after one tool call")?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .context("the fixed prompt should resolve to a run")?;
    match runtime.observe_run(run)? {
        Some(RunState::Completed(output)) => println!("final answer: {}", output.text),
        state => anyhow::bail!("unexpected fixed outcome: {state:?}"),
    }
    Ok(())
}

fn main() -> Result<()> {
    demonstrate_footgun()?;
    demonstrate_fix()
}
