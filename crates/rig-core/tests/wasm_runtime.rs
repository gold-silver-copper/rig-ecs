//! Browser-executed smoke coverage for the local ECS runtime path.

#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use std::time::Duration;

use rig_core::runtime::{
    Agent, EffectCompletion, EffectOutput, ModelCapability, ModelEffectOutput, RunState, Runtime,
    RuntimeConfig, StableId, TenantId, Usage,
};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

#[wasm_bindgen_test(async)]
async fn browser_local_future_drives_the_same_effect_schedule() {
    // This timer is backed by the browser event loop. It catches accidental
    // reintroduction of thread-only timer or executor assumptions.
    futures_timer::Delay::new(Duration::from_millis(1)).await;

    let mut runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let tenant = TenantId::new("browser").unwrap();
    let model = runtime
        .spawn_model(
            id("browser-model"),
            tenant.clone(),
            ModelCapability {
                provider: "local".to_owned(),
                model: "fixture".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .unwrap();
    let agent = runtime
        .spawn_agent(id("browser-agent"), tenant, Agent::default(), model)
        .unwrap();
    let pending = runtime.handle().prompt(agent, "hello from wasm").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "browser-local".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();
    assert!(matches!(
        runtime.observe_run(run).unwrap(),
        Some(RunState::Completed(output)) if output.text == "browser-local"
    ));
}
