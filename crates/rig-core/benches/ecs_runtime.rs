//! Operational smoke benchmarks for critical ECS runtime shapes.
//!
//! Every fallible value below is benchmark fixture setup or an invariant under
//! measurement; aborting the benchmark is more useful than measuring recovery.

#![allow(clippy::expect_used)]

use std::time::{Duration, Instant};

use rig_core::bevy_ecs::{prelude::Entity, relationship::Relationship};
use rig_core::runtime::{
    Agent, AgentHandle, EffectCompletion, EffectDelta, EffectDeltaKind, EffectOutput,
    ModelCapability, ModelEffectOutput, ModelToolCall, OperationOf, PauseMode, Policy,
    PolicyApprovalEffectOutput, PolicyPoint, PolicyRule, Runtime, RuntimeConfig, StableId,
    TenantId, ToolCapability, ToolEffectOutput, ToolGrant, Usage,
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
        max_effect_dispatches_per_pass: 4096,
        per_run_effect_limit: 4096,
        per_agent_effect_limit: 4096,
        per_tenant_effect_limit: 4096,
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

fn spawn_peer_agent(runtime: &mut Runtime, index: usize) -> AgentHandle {
    let model = {
        let world = runtime.world_mut();
        let mut models =
            world.query_filtered::<Entity, rig_core::bevy_ecs::prelude::With<ModelCapability>>();
        models.iter(world).next().expect("benchmark model exists")
    };
    runtime
        .spawn_agent(
            id(format!("agent-{index}")),
            TenantId::new("bench").expect("tenant"),
            Agent::default(),
            model,
        )
        .expect("peer agent spawns")
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

fn concurrent_runs_across_agents() -> Duration {
    let (mut runtime, first) = setup();
    let mut agents = vec![first];
    agents.extend((1..8).map(|index| spawn_peer_agent(&mut runtime, index)));
    let started = Instant::now();
    for (index, agent) in agents.iter().copied().cycle().take(ITERATIONS).enumerate() {
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

fn pause_resume_alongside_active() -> Duration {
    let (mut runtime, agent) = setup();
    let frozen_pending = runtime.handle().prompt(agent, "frozen").expect("submit");
    let active_pending = runtime.handle().prompt(agent, "active").expect("submit");
    runtime.run_until_stalled().expect("dispatch both");
    let frozen = runtime.resolve_run(&frozen_pending).expect("frozen run");
    let active = runtime.resolve_run(&active_pending).expect("active run");
    runtime
        .handle()
        .pause_with_mode(frozen, PauseMode::FreezeAfterIngress)
        .expect("pause");
    runtime.run_until_stalled().expect("apply pause");
    let requests = [
        runtime
            .effects()
            .try_recv()
            .expect("queue")
            .expect("request"),
        runtime
            .effects()
            .try_recv()
            .expect("queue")
            .expect("request"),
    ];
    let started = Instant::now();
    for request in &requests {
        complete_model(&runtime, request);
    }
    runtime.run_until_stalled().expect("active commits");
    assert!(matches!(
        runtime
            .world()
            .get::<rig_core::runtime::RunState>(active.entity()),
        Some(rig_core::runtime::RunState::Completed(_))
    ));
    assert!(matches!(
        runtime
            .world()
            .get::<rig_core::runtime::RunState>(frozen.entity()),
        Some(rig_core::runtime::RunState::WaitingModel { .. })
    ));
    runtime.handle().resume(frozen).expect("resume");
    runtime.run_until_stalled().expect("frozen commits");
    started.elapsed()
}

fn dynamic_child_run() -> Duration {
    let (mut runtime, agent) = setup();
    let parent_pending = runtime.handle().prompt(agent, "parent").expect("parent");
    runtime.run_until_stalled().expect("parent dispatch");
    let parent_request = runtime
        .effects()
        .try_recv()
        .expect("queue")
        .expect("parent request");
    let parent = runtime.resolve_run(&parent_pending).expect("parent run");
    let started = Instant::now();
    let child_pending = runtime
        .handle()
        .spawn_child_run(parent, agent, "child")
        .expect("child");
    runtime.run_until_stalled().expect("child dispatch");
    let child_request = runtime
        .effects()
        .try_recv()
        .expect("queue")
        .expect("child request");
    let child = runtime.resolve_run(&child_pending).expect("child run");
    assert_eq!(
        runtime
            .world()
            .get::<OperationOf>(child_request.operation)
            .expect("child owner")
            .get(),
        child.entity()
    );
    complete_model(&runtime, &child_request);
    runtime.run_until_stalled().expect("child commit");
    complete_model(&runtime, &parent_request);
    runtime.run_until_stalled().expect("parent commit");
    started.elapsed()
}

fn policy_heavy_run() -> Duration {
    let (mut runtime, agent) = setup();
    let tenant = TenantId::new("bench").expect("tenant");
    for index in 0..100 {
        runtime
            .spawn_policy(
                id(format!("policy-{index}")),
                tenant.clone(),
                Policy {
                    order: index,
                    revision: 1,
                    rule: PolicyRule::Allow,
                },
                agent,
            )
            .expect("policy");
    }
    let started = Instant::now();
    runtime
        .handle()
        .prompt(agent, "policy-heavy")
        .expect("submit");
    runtime.run_until_stalled().expect("evaluate policies");
    let request = runtime
        .effects()
        .try_recv()
        .expect("queue")
        .expect("request");
    complete_model(&runtime, &request);
    runtime.run_until_stalled().expect("commit");
    started.elapsed()
}

fn one_policy_one_run() -> Duration {
    let (mut runtime, agent) = setup();
    runtime
        .spawn_policy(
            id("single-policy"),
            TenantId::new("bench").expect("tenant"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Allow,
            },
            agent,
        )
        .expect("policy");
    let started = Instant::now();
    runtime
        .handle()
        .prompt(agent, "one policy")
        .expect("submit");
    runtime.run_until_stalled().expect("evaluate");
    let request = runtime
        .effects()
        .try_recv()
        .expect("queue")
        .expect("request");
    complete_model(&runtime, &request);
    runtime.run_until_stalled().expect("commit");
    started.elapsed()
}

fn one_policy_many_runs() -> Duration {
    const RUNS: usize = 1_000;
    let (mut runtime, agent) = setup();
    runtime
        .spawn_policy(
            id("shared-policy"),
            TenantId::new("bench").expect("tenant"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Allow,
            },
            agent,
        )
        .expect("policy");
    for index in 0..RUNS {
        runtime
            .handle()
            .prompt(agent, format!("shared {index}"))
            .expect("submit");
    }
    let started = Instant::now();
    runtime.run_until_stalled().expect("evaluate all");
    let mut requests = Vec::with_capacity(RUNS);
    while let Some(request) = runtime.effects().try_recv().expect("queue") {
        requests.push(request);
    }
    assert_eq!(requests.len(), RUNS);
    for request in &requests {
        complete_model(&runtime, request);
    }
    runtime.run_until_stalled().expect("commit all");
    started.elapsed()
}

fn asynchronous_approval() -> Duration {
    let (mut runtime, agent) = setup();
    runtime
        .spawn_policy(
            id("approval"),
            TenantId::new("bench").expect("tenant"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::Request,
                    prompt: "approve".to_owned(),
                },
            },
            agent,
        )
        .expect("policy");
    let started = Instant::now();
    runtime.handle().prompt(agent, "approval").expect("submit");
    runtime.run_until_stalled().expect("approval dispatch");
    let approval = runtime
        .effects()
        .try_recv()
        .expect("queue")
        .expect("approval");
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: None,
            })),
        })
        .expect("approval completion");
    runtime.run_until_stalled().expect("model dispatch");
    let model = runtime.effects().try_recv().expect("queue").expect("model");
    complete_model(&runtime, &model);
    runtime.run_until_stalled().expect("commit");
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

fn streaming_with_policy() -> Duration {
    let (mut runtime, agent) = setup();
    runtime
        .spawn_policy(
            id("stream-policy"),
            TenantId::new("bench").expect("tenant"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::StopTextDeltaContains {
                    needle: "never".to_owned(),
                    reason: "benchmark".to_owned(),
                },
            },
            agent,
        )
        .expect("policy");
    let (_, stream) = runtime
        .handle()
        .prompt_stream(agent, "stream policy")
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
    while stream.try_recv().expect("stream").is_some() {}
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
    runtime
        .spawn_policy(
            id("batch-policy"),
            tenant,
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RewriteToolArguments {
                    tool: None,
                    arguments: serde_json::json!({}),
                },
            },
            agent,
        )
        .expect("batch policy");
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
                    raw: serde_json::json!(input.index).into(),
                    presentation: input.index.to_string().into(),
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
    report("concurrent-runs-one-agent", concurrent_runs(), ITERATIONS);
    report(
        "concurrent-runs-eight-agents",
        concurrent_runs_across_agents(),
        ITERATIONS,
    );
    report(
        "pause-resume-alongside-active",
        pause_resume_alongside_active(),
        2,
    );
    report("dynamic-child-run", dynamic_child_run(), 2);
    report("policy-heavy-run", policy_heavy_run(), 100);
    report("one-policy-one-run", one_policy_one_run(), 1);
    report("one-policy-1000-runs", one_policy_many_runs(), 1_000);
    report("asynchronous-approval", asynchronous_approval(), 1);
    report("large-capability-set", large_capability_set(), 1);
    report("streaming-deltas", streaming(), 100);
    report("streaming-deltas-with-policy", streaming_with_policy(), 100);
    report("parallel-tool-batch-with-policy", parallel_tool_batch(), 16);
}
