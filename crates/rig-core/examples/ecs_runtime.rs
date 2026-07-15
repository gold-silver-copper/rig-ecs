//! Minimal deterministic use of Rig's standalone ECS runtime.

use rig_core::runtime::{
    Agent, EffectCompletion, EffectOutput, ModelCapability, ModelEffectOutput, RunState, Runtime,
    RuntimeConfig, StableId, TenantId, Usage,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = Runtime::new(RuntimeConfig::default())?;
    let tenant = TenantId::new("example")?;
    let model = runtime.spawn_model(
        StableId::new("model")?,
        tenant.clone(),
        ModelCapability {
            provider: "example".into(),
            model: "deterministic".into(),
            revision: 1,
            retired: false,
        },
    )?;
    let agent = runtime.spawn_agent(
        StableId::new("agent")?,
        tenant,
        Agent {
            instructions: "Answer concisely.".into(),
            ..Agent::default()
        },
        model,
    )?;

    let pending = runtime.handle().prompt(agent, "Hello")?;
    runtime.run_until_stalled()?;
    let effect = runtime
        .effects()
        .try_recv()?
        .ok_or("model effect was not dispatched")?;
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: effect.operation,
            generation: effect.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "Hello from ECS.".into(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })?;
    runtime.run_until_stalled()?;

    let run = runtime
        .resolve_run(&pending)
        .ok_or("prompt was not ingested")?;
    let Some(RunState::Completed(output)) = runtime.observe_run(run)? else {
        return Err("run did not complete".into());
    };
    println!("{}", output.text);
    Ok(())
}
