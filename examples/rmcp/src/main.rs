//! Reconcile a dynamically discovered MCP-style tool through ECS discovery effects.

use anyhow::{Context, Result};
use rig::runtime::{
    DiscoveredTool, DiscoveryEffectOutput, EffectCompletion, EffectOutput, Runtime, RuntimeConfig,
    StableId, TenantId,
};

fn main() -> Result<()> {
    let mut runtime = Runtime::new(RuntimeConfig::default())?;
    let source = runtime.spawn_discovery_source(
        StableId::new("mcp-source")?,
        TenantId::new("example")?,
        "mcp",
    )?;
    runtime.handle().refresh_discovery(source)?;
    runtime.run_until_stalled()?;
    let request = runtime.effects().try_recv()?.context("discovery effect")?;
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Discovery(DiscoveryEffectOutput {
                tools: vec![DiscoveredTool {
                    key: "lookup".to_owned(),
                    name: "lookup".to_owned(),
                    description: "Discovered from MCP".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                }],
            })),
        })?;
    runtime.run_until_stalled()?;
    println!("MCP discovery reconciled into the ECS world");
    Ok(())
}
