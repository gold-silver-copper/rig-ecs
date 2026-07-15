//! Show that approval state is durable ECS state rather than a suspended callback.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::Result;
use rig::runtime::{
    EffectCompletion, EffectOutput, Policy, PolicyApprovalEffectOutput, PolicyPoint, PolicyRule,
    RequestPolicyEvaluation, RequestPolicyEvaluationPhase,
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
    runtime.handle().prompt(agent, "A durable request")?;
    let approval = ecs_demo::next_effect(&mut runtime)?;
    let waiting = runtime
        .world_mut()
        .query::<&RequestPolicyEvaluation>()
        .iter(runtime.world())
        .any(|evaluation| {
            matches!(
                evaluation.phase,
                RequestPolicyEvaluationPhase::WaitingApproval { .. }
            )
        });
    println!("world contains waiting approval: {waiting}");
    runtime
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
    let model = ecs_demo::next_effect(&mut runtime)?;
    println!("resumed model operation: {:?}", model.operation);
    Ok(())
}
