//! Manually host a complete multi-round model/tool loop around ECS.
//!
//! Nothing executes tools automatically. The host receives immutable model and
//! tool effects, executes two first-round calls locally, deliberately returns
//! their completions in reverse order, feeds the runtime's ordered batch back
//! to the next model operation, then performs a dependent final calculation.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::{Context, Result};
use rig::runtime::{
    EffectCompletion, EffectOutput, ModelEffectOutput, ModelToolCall, RunState, ToolCapability,
    ToolEffectInput, ToolEffectOutput, ToolGrant, Usage,
};

fn install_tool(
    runtime: &mut rig::runtime::Runtime,
    agent: rig::runtime::AgentHandle,
    name: &str,
    order: u32,
) -> Result<()> {
    let tool = runtime.spawn_tool(
        ecs_demo::id(&format!("{name}-tool"))?,
        ecs_demo::tenant()?,
        ToolCapability {
            name: name.to_owned(),
            description: format!("Example {name} operation"),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"]
            }),
            order,
            revision: 1,
            retired: false,
        },
    )?;
    runtime.grant_tool(
        ecs_demo::id(&format!("{name}-grant"))?,
        ecs_demo::tenant()?,
        ToolGrant {
            order,
            enabled: true,
        },
        agent,
        tool,
    )?;
    Ok(())
}

fn complete_model(
    runtime: &rig::runtime::Runtime,
    request: &rig::runtime::EffectRequest,
    calls: Vec<ModelToolCall>,
    text: &str,
) -> Result<()> {
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: text.to_owned(),
                usage: Usage::default(),
                tool_calls: calls,
            })),
        })?;
    Ok(())
}

fn call(id: &str, name: &str, x: i64, y: i64) -> ModelToolCall {
    ModelToolCall {
        id: id.to_owned(),
        provider_result_id: id.to_owned(),
        provider_call_id: None,
        name: name.to_owned(),
        arguments: serde_json::json!({"x": x, "y": y}),
    }
}

fn execute(input: &ToolEffectInput) -> Result<ToolEffectOutput> {
    let x = input
        .arguments
        .get("x")
        .and_then(serde_json::Value::as_i64)
        .context("tool arguments omitted integer x")?;
    let y = input
        .arguments
        .get("y")
        .and_then(serde_json::Value::as_i64)
        .context("tool arguments omitted integer y")?;
    let value = match input.decision.name.as_str() {
        "add" => x + y,
        "subtract" => x - y,
        name => anyhow::bail!("manual host has no executor for `{name}`"),
    };
    println!("  {}({}, {}) -> {value}", input.decision.name, x, y);
    Ok(ToolEffectOutput {
        call_id: input.call_id.clone(),
        provider_result_id: input.provider_result_id.clone(),
        provider_call_id: input.provider_call_id.clone(),
        name: input.decision.name.clone(),
        raw: serde_json::json!(value),
        presentation: value.to_string(),
        failure: None,
    })
}

fn complete_tool(
    runtime: &rig::runtime::Runtime,
    request: &rig::runtime::EffectRequest,
) -> Result<()> {
    let output = execute(request.tool_input().context("expected a tool effect")?)?;
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Tool(output)),
        })?;
    Ok(())
}

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(false)?;
    install_tool(&mut runtime, agent, "add", 0)?;
    install_tool(&mut runtime, agent, "subtract", 1)?;
    let pending = runtime.handle().prompt(
        agent,
        "Calculate (20 - 5) + (8 - 3), using tools for every intermediate step",
    )?;

    let first_model = ecs_demo::next_effect(&mut runtime)?;
    complete_model(
        &runtime,
        &first_model,
        vec![
            call("left", "subtract", 20, 5),
            call("right", "subtract", 8, 3),
        ],
        "",
    )?;
    let left = ecs_demo::next_effect(&mut runtime)?;
    let right = runtime
        .effects()
        .try_recv()?
        .context("expected the parallel sibling tool effect")?;
    println!("round 1: model requested two calls");
    // External work may settle in any order; ECS commits the logical batch by index.
    complete_tool(&runtime, &right)?;
    complete_tool(&runtime, &left)?;

    let second_model = ecs_demo::next_effect(&mut runtime)?;
    let first_results = &second_model
        .model_input()
        .context("expected the second model request")?
        .tool_results;
    println!(
        "model received ordered results: {:?}",
        first_results
            .iter()
            .map(|result| (&result.call_id, &result.presentation))
            .collect::<Vec<_>>()
    );
    complete_model(
        &runtime,
        &second_model,
        vec![call("total", "add", 15, 5)],
        "",
    )?;
    let total = ecs_demo::next_effect(&mut runtime)?;
    println!("round 2: model requested the dependent add call");
    complete_tool(&runtime, &total)?;

    let final_model = ecs_demo::next_effect(&mut runtime)?;
    let total_result = final_model
        .model_input()
        .and_then(|input| input.tool_results.first())
        .context("expected the final tool result")?;
    complete_model(
        &runtime,
        &final_model,
        Vec::new(),
        &format!("The final answer is {}.", total_result.presentation),
    )?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .context("the prompt should resolve to a run")?;
    match runtime.observe_run(run)? {
        Some(RunState::Completed(output)) => println!("\n{}", output.text),
        state => anyhow::bail!("manual loop ended unexpectedly: {state:?}"),
    }
    Ok(())
}
