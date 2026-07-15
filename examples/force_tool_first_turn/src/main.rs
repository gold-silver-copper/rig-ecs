//! Force a tool on one request with a non-sticky ECS request patch.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::Result;
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
    println!(
        "effective tool choice: {:?}",
        request.model_input().unwrap().tool_choice
    );
    Ok(())
}
