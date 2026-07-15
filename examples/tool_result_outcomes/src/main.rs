//! Rewrite tool presentation while retaining immutable raw audit data.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::{Context, Result};
use rig::runtime::{Policy, PolicyRule};

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(true)?;
    runtime.spawn_policy(
        ecs_demo::id("redact-result")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::RewriteToolResult {
                tool: Some("lookup".to_owned()),
                presentation: "[redacted]".to_owned(),
            },
        },
        agent,
    )?;
    runtime.handle().prompt(agent, "Use lookup")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_with_tool(&runtime, &model, "lookup")?;
    let tool = ecs_demo::next_effect(&mut runtime)?;
    ecs_demo::complete_tool(&runtime, &tool, "unredacted")?;
    let next_model = ecs_demo::next_effect(&mut runtime)?;
    let result = next_model
        .model_input()
        .context("expected a follow-up model effect")?
        .tool_results
        .first()
        .context("expected the completed tool result in the follow-up request")?;
    println!("raw={}, presentation={}", result.raw, result.presentation);
    Ok(())
}
