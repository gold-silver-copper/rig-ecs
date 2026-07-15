//! Recover a provider-emitted invalid tool name through ECS policy resolution.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::Result;
use rig::runtime::{Policy, PolicyRule};

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.spawn_policy(
        ecs_demo::id("repair-default-api-name")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::RepairInvalidTool {
                from: Some("default_api".to_owned()),
                to: "lookup".to_owned(),
            },
        },
        agent,
    )?;
    runtime.handle().prompt(agent, "Use the default API")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_with_tool(&runtime, &model, "default_api")?;
    let repaired = ecs_demo::next_effect(&mut runtime)?;
    println!(
        "repaired to: {}",
        repaired.tool_input().unwrap().decision.name
    );
    Ok(())
}
