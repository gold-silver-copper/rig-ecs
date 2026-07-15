//! Operational smoke benchmarks for critical ECS runtime shapes.
//!
//! Every fallible value below is benchmark fixture setup or an invariant under
//! measurement; aborting the benchmark is more useful than measuring recovery.

#![allow(clippy::expect_used)]

use std::time::{Duration, Instant};

use rig_core::runtime::{
    Agent, AgentHandle, EffectCompletion, EffectDelta, EffectDeltaKind, EffectOutput,
    ModelCapability, ModelEffectOutput, ModelToolCall, Runtime, RuntimeConfig, StableId, TenantId,
    ToolCapability, ToolEffectOutput, ToolGrant, Usage,
};

const ITERATIONS: usize = 100;

fn id(value: impl Into<String>) -> StableId {
    StableId::new(value).expect("benchmark identities are non-empty")
}

fn setup() -> (Runtime, AgentHandle) {
    let mut runtime = Runtime::new(RuntimeConfig {
        command_capacity: 4096,
        effect_capacity: 4096,
        completion_capacity: 4096,
        subscriber_capacity: 4096,
        ..RuntimeConfig::default()
    })
    .expect("runtime installs");
    let tenant = TenantId::new("bench").expect("tenant is valid");
    let model = runtime
        .spawn_model(
            id("model"),
            tenant.clone(),
            ModelCapability {
                provider: "bench".to_owned(),
                model: "fake".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .expect("model spawns");
    let agent = runtime
        .spawn_agent(id("agent"), tenant, Agent::default(), model)
        .expect("agent spawns");
    (runtime, agent)
}

fn complete_model(runtime: &Runtime, request: &rig_core::runtime::EffectRequest) {
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "ok".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .expect("completion queue has capacity");
}

fn one_shot() -> Duration {
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let (mut runtime, agent) = setup();
        runtime.handle().prompt(agent, "hello").expect("submit");
        runtime.run_until_stalled().expect("prepare");
        let request = runtime
            .effects()
            .try_recv()
            .expect("queue")
            .expect("effect");
        complete_model(&runtime, &request);
        runtime.run_until_stalled().expect("commit");
    }
    started.elapsed()
}

fn concurrent_runs() -> Duration {
    let (mut runtime, agent) = setup();
    let started = Instant::now();
    for index in 0..ITERATIONS {
        runtime
            .handle()
            .prompt(agent, format!("prompt {index}"))
            .expect("submit");
    }
    runtime.run_until_stalled().expect("prepare all");
    while let Some(request) = runtime.effects().try_recv().expect("queue") {
        complete_model(&runtime, &request);
    }
    runtime.run_until_stalled().expect("commit all");
    started.elapsed()
}

fn large_capability_set() -> Duration {
    let (mut runtime, agent) = setup();
    let tenant = TenantId::new("bench").expect("tenant");
    for index in 0..1_000 {
        let tool = runtime
            .spawn_tool(
                id(format!("tool-{index}")),
                tenant.clone(),
                ToolCapability {
                    name: format!("tool_{index}"),
                    description: "benchmark tool".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: index,
                    revision: 1,
                    retired: false,
                },
            )
            .expect("tool");
        runtime
            .grant_tool(
                id(format!("grant-{index}")),
                tenant.clone(),
                ToolGrant {
                    order: index,
                    enabled: true,
                },
                agent,
                tool,
            )
            .expect("grant");
    }
    let started = Instant::now();
    runtime
        .handle()
        .prompt(agent, "select tools")
        .expect("submit");
    runtime.run_until_stalled().expect("prepare");
    let request = runtime
        .effects()
        .try_recv()
        .expect("queue")
        .expect("effect");
    assert_eq!(request.model_input().expect("model").tools.len(), 1_000);
    started.elapsed()
}

fn streaming() -> Duration {
    let (mut runtime, agent) = setup();
    let (_, stream) = runtime
        .handle()
        .prompt_stream(agent, "stream")
        .expect("submit");
    runtime.run_until_stalled().expect("prepare");
    let request = runtime
        .effects()
        .try_recv()
        .expect("queue")
        .expect("effect");
    let started = Instant::now();
    let deltas = runtime.effects().delta_sender();
    for sequence in 0..100 {
        deltas
            .try_send(EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence,
                provider_correlation: None,
                kind: EffectDeltaKind::Text("x".to_owned()),
            })
            .expect("delta");
    }
    complete_model(&runtime, &request);
    runtime.run_until_stalled().expect("apply stream");
    let mut received = 0;
    while stream.try_recv().expect("stream").is_some() {
        received += 1;
    }
    assert_eq!(received, 101);
    started.elapsed()
}

fn parallel_tool_batch() -> Duration {
    let (mut runtime, agent) = setup();
    let tenant = TenantId::new("bench").expect("tenant");
    for index in 0..16 {
        let tool = runtime
            .spawn_tool(
                id(format!("batch-tool-{index}")),
                tenant.clone(),
                ToolCapability {
                    name: format!("batch_tool_{index}"),
                    description: "batch tool".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: index,
                    revision: 1,
                    retired: false,
                },
            )
            .expect("tool");
        runtime
            .grant_tool(
                id(format!("batch-grant-{index}")),
                tenant.clone(),
                ToolGrant {
                    order: index,
                    enabled: true,
                },
                agent,
                tool,
            )
            .expect("grant");
    }
    runtime.handle().prompt(agent, "batch").expect("submit");
    runtime.run_until_stalled().expect("prepare model");
    let model = runtime.effects().try_recv().expect("queue").expect("model");
    let calls = (0..16)
        .map(|index| ModelToolCall {
            id: format!("call-{index}"),
            provider_result_id: format!("call-{index}"),
            provider_call_id: None,
            name: format!("batch_tool_{index}"),
            arguments: serde_json::json!({}),
        })
        .collect();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: calls,
            })),
        })
        .expect("model completion");
    runtime.run_until_stalled().expect("dispatch tools");
    let mut requests = Vec::new();
    while let Some(request) = runtime.effects().try_recv().expect("queue") {
        requests.push(request);
    }
    let started = Instant::now();
    for request in requests.into_iter().rev() {
        let input = request.tool_input().expect("tool");
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
                    raw: serde_json::json!(input.index),
                    presentation: input.index.to_string(),
                    failure: None,
                })),
            })
            .expect("tool completion");
    }
    runtime.run_until_stalled().expect("atomic batch commit");
    started.elapsed()
}

fn report(name: &str, duration: Duration, operations: usize) {
    println!(
        "{name}: {:.1} ns/op ({operations} operations)",
        duration.as_nanos() as f64 / operations as f64
    );
}

fn main() {
    report("one-shot", one_shot(), ITERATIONS);
    report("concurrent-runs", concurrent_runs(), ITERATIONS);
    report("large-capability-set", large_capability_set(), 1);
    report("streaming-deltas", streaming(), 100);
    report("parallel-tool-batch", parallel_tool_batch(), 16);
}
