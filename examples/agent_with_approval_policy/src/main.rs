//! Gate a tool call with an asynchronous ECS approval policy.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::Result;
use rig::runtime::{
    EffectCompletion, EffectOutput, Policy, PolicyApprovalEffectOutput, PolicyPoint, PolicyRule,
};

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.spawn_policy(
        ecs_demo::id("approval")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::RequireApproval {
                point: PolicyPoint::ToolCall,
                prompt: "Allow lookup?".to_owned(),
            },
        },
        agent,
    )?;

    runtime.handle().prompt(agent, "Use lookup")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_with_tool(&runtime, &model, "lookup")?;
    let approval = ecs_demo::next_effect(&mut runtime)?;
    println!("approval request: {:?}", approval.policy_approval_input());
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: Some("approved by example operator".to_owned()),
            })),
        })?;
    let tool = ecs_demo::next_effect(&mut runtime)?;
    println!(
        "approved tool: {}",
        tool.tool_input().unwrap().decision.name
    );
    Ok(())
}
