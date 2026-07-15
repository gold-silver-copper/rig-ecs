//! Step a run by driving the shared ECS schedule one pass at a time.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::Result;
use rig::runtime::{EffectCompletion, EffectOutput, ModelEffectOutput, RunState, Usage};

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(false)?;
    let pending = runtime.handle().prompt(agent, "Show schedule stepping")?;

    runtime.update();
    let run = runtime
        .resolve_run(&pending)
        .expect("the prompt was ingested");
    println!(
        "after one pass: {:?}",
        runtime.world().get::<RunState>(run.entity())
    );

    let request = runtime.effects().try_recv()?.expect("model effect");
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "stepped to completion".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })?;
    while !matches!(
        runtime.observe_run(run)?,
        Some(RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled)
    ) {
        runtime.update();
    }
    println!("terminal: {:?}", runtime.observe_run(run)?);
    Ok(())
}
