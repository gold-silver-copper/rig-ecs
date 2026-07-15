//! Manually service model and tool effects outside the authoritative ECS world.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::Result;

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.handle().prompt(agent, "Run lookup manually")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_with_tool(&runtime, &model, "lookup")?;
    let tool = ecs_demo::next_effect(&mut runtime)?;
    println!("manual arguments: {}", tool.tool_input().unwrap().arguments);
    ecs_demo::complete_tool(&runtime, &tool, "manual result")?;
    let next_model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_text(&runtime, &next_model, "done")?;
    runtime.run_until_stalled()?;
    Ok(())
}
