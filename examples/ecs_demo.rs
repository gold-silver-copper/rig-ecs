//! Small deterministic effect host shared by ECS runtime examples.

#![allow(dead_code)]

use anyhow::{Context, Result};
use rig::runtime::{
    Agent, AgentHandle, EffectCompletion, EffectOutput, EffectRequest, ModelCapability,
    ModelEffectOutput, ModelToolCall, Runtime, RuntimeConfig, StableId, TenantId, ToolCapability,
    ToolEffectOutput, ToolGrant, Usage,
};

pub fn id(value: &str) -> Result<StableId> {
    StableId::new(value).context("stable identity")
}

pub fn tenant() -> Result<TenantId> {
    TenantId::new("example").context("tenant identity")
}

pub fn runtime(with_tool: bool) -> Result<(Runtime, AgentHandle)> {
    let mut runtime = Runtime::new(RuntimeConfig::default())?;
    let model = runtime.spawn_model(
        id("model")?,
        tenant()?,
        ModelCapability {
            provider: "example".to_owned(),
            model: "deterministic".to_owned(),
            revision: 1,
            retired: false,
        },
    )?;
    let agent = runtime.spawn_agent(
        id("agent")?,
        tenant()?,
        Agent {
            instructions: "Demonstrate the ECS lifecycle.".to_owned(),
            ..Agent::default()
        },
        model,
    )?;
    if with_tool {
        let tool = runtime.spawn_tool(
            id("tool")?,
            tenant()?,
            ToolCapability {
                name: "lookup".to_owned(),
                description: "Deterministic example lookup".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                order: 0,
                revision: 1,
                retired: false,
            },
        )?;
        runtime.grant_tool(
            id("tool-grant")?,
            tenant()?,
            ToolGrant {
                order: 0,
                enabled: true,
            },
            agent,
            tool,
        )?;
    }
    Ok((runtime, agent))
}

pub fn next_effect(runtime: &mut Runtime) -> Result<EffectRequest> {
    runtime.run_until_stalled()?;
    runtime
        .effects()
        .try_recv()?
        .context("expected an ECS effect")
}

pub fn complete_text(runtime: &Runtime, request: &EffectRequest, text: &str) -> Result<()> {
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
                tool_calls: Vec::new(),
            })),
        })?;
    Ok(())
}

pub fn complete_with_tool(runtime: &Runtime, request: &EffectRequest, name: &str) -> Result<()> {
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call-1".to_owned(),
                    provider_result_id: "call-1".to_owned(),
                    provider_call_id: None,
                    name: name.to_owned(),
                    arguments: serde_json::json!({"query": "example"}),
                }],
            })),
        })?;
    Ok(())
}

pub fn complete_tool(runtime: &Runtime, request: &EffectRequest, presentation: &str) -> Result<()> {
    let input = request.tool_input().context("expected a tool effect")?;
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: input.call_id.clone(),
                provider_result_id: input.provider_result_id.clone(),
                provider_call_id: input.provider_call_id.clone(),
                name: input.decision.name.clone(),
                raw: serde_json::json!({"value": 42}).into(),
                presentation: presentation.into(),
                failure: None,
            })),
        })?;
    Ok(())
}
