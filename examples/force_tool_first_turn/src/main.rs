//! Force a tool on one request with a non-sticky ECS request patch.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::{Context, Result};
use rig::runtime::{ModelToolChoice, Policy, PolicyRule, RequestPatch};

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.spawn_policy(
        ecs_demo::id("force-tool")?,
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
    runtime.handle().prompt(agent, "Look something up")?;
    let request = ecs_demo::next_effect(&mut runtime)?;
    let model_input = request
        .model_input()
        .context("expected the policy-patched model request")?;
    println!("effective tool choice: {:?}", model_input.tool_choice);
    Ok(())
}
