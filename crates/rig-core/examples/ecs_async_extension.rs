//! Add an asynchronous operation kind without changing Rig's core effect enums.

use rig_core::runtime::{
    Agent, EcsEffect, ExtensionEffectResult, ModelCapability, Runtime, RuntimeConfig, StableId,
    TenantId, dispatch_extension_effect, settle_extension_effect, spawn_extension_effect,
};

#[derive(Clone, Debug)]
struct ModerationInput {
    text: String,
}

#[derive(Debug)]
struct ModerationOutput {
    allowed: bool,
}

struct Moderation;

impl EcsEffect for Moderation {
    type Input = ModerationInput;
    type Output = ModerationOutput;
    type Error = String;

    const KIND: &'static str = "example.moderation.v1";
}

async fn moderate(input: ModerationInput) -> Result<ModerationOutput, String> {
    Ok(ModerationOutput {
        allowed: !input.text.contains("blocked"),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = Runtime::new(RuntimeConfig::default())?;
    let tenant = TenantId::new("example")?;
    let model = runtime.spawn_model(
        StableId::new("model")?,
        tenant.clone(),
        ModelCapability {
            provider: "example".to_owned(),
            model: "deterministic".to_owned(),
            revision: 1,
            retired: false,
        },
    )?;
    let agent = runtime.spawn_agent(StableId::new("agent")?, tenant, Agent::default(), model)?;
    let pending = runtime.handle().prompt(agent, "screen this input")?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .ok_or("prompt was not ingested")?;

    let operation = spawn_extension_effect::<Moderation>(
        runtime.world_mut(),
        run.entity(),
        StableId::new("moderation-operation")?,
        ModerationInput {
            text: "screen this input".to_owned(),
        },
    )?;
    let (generation, input) =
        dispatch_extension_effect::<Moderation>(runtime.world_mut(), operation)?;
    let result = futures::executor::block_on(moderate(input));
    settle_extension_effect::<Moderation>(runtime.world_mut(), operation, generation, result)?;

    let result = runtime
        .world()
        .get::<ExtensionEffectResult<Moderation>>(operation)
        .ok_or("typed result was not committed")?;
    println!(
        "allowed: {}",
        result.0.as_ref().map_err(String::as_str)?.allowed
    );
    Ok(())
}
