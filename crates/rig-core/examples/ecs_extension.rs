//! Install Rig into an existing Bevy world and extend its schedule.

use rig_core::{
    bevy_ecs::{
        prelude::{Added, Entity, On, Query, World},
        schedule::{IntoScheduleConfigs, Schedules},
    },
    runtime::{
        Agent, EffectCompletion, EffectOutput, LifecycleTelemetry, LifecycleTelemetryBundle,
        ModelCapability, ModelEffectInput, ModelEffectOutput, RequestPatch,
        RequestPatchPolicyBundle, ResolvedOperationContext, RigExtension, RigOperationContext,
        RigSchedule, RigSet, RunCompleted, RunState, RuntimeConfig, StableId, TenantId, Usage,
        install_runtime,
    },
};

fn audit_prepared_requests(
    operations: Query<Entity, Added<ModelEffectInput>>,
    context: RigOperationContext<'_, '_>,
) {
    for operation in &operations {
        if let Some(ResolvedOperationContext {
            run_id, agent_name, ..
        }) = context.resolve(operation)
        {
            println!("prepared request for {run_id:?} on {agent_name:?}");
        }
    }
}

fn observe_completion(event: On<RunCompleted>) {
    println!("run {:?} completed", event.run);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = World::new();
    let installed = install_runtime(&mut world, RuntimeConfig::default())?;
    let mut schedules = world.resource_mut::<Schedules>();
    let schedule = schedules
        .get_mut(RigSchedule)
        .ok_or("Rig schedule was not installed")?;
    schedule.add_systems(audit_prepared_requests.in_set(RigSet::FinalizeRequest));
    let tenant = TenantId::new("example")?;
    let model = installed.spawn_model(
        &mut world,
        StableId::new("model")?,
        tenant.clone(),
        ModelCapability {
            provider: "host".to_owned(),
            model: "embedded".to_owned(),
            revision: 1,
            retired: false,
        },
    )?;
    let agent = installed.spawn_agent(
        &mut world,
        StableId::new("agent")?,
        tenant.clone(),
        Agent {
            name: Some("embedded-agent".to_owned()),
            ..Agent::default()
        },
        model,
    )?;
    world
        .entity_mut(agent.entity())
        .insert(LifecycleTelemetryBundle::default());
    RequestPatchPolicyBundle::new(
        StableId::new("embedded-request-patch")?,
        tenant,
        agent.entity(),
        0,
        1,
        RequestPatch::new().instructions("Answer as an embedded ECS extension."),
    )
    .install(&mut world)?;
    let pending = installed.handle.prompt(agent, "hello from the host")?;
    let mut request = None;
    for _ in 0..4 {
        world.run_schedule(RigSchedule);
        if let Some(effect) = installed.effects.try_recv()? {
            request = Some(effect);
            break;
        }
    }
    let request = request.ok_or("model effect was not dispatched")?;
    let run = installed
        .resolve_run(&world, &pending)?
        .ok_or("run was not ingested")?;
    world.entity_mut(run.entity()).observe(observe_completion);
    installed
        .effects
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "hello from Rig".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })?;
    for _ in 0..4 {
        world.run_schedule(RigSchedule);
    }
    let state = installed
        .observe_run(&mut world, run)?
        .ok_or("run disappeared before observation")?;
    if !matches!(state, RunState::Completed(_)) {
        return Err("embedded run did not complete".into());
    }
    let telemetry = world
        .get::<LifecycleTelemetry>(agent.entity())
        .ok_or("telemetry bundle disappeared")?;
    println!("prepared {} request(s)", telemetry.requests_prepared);
    Ok(())
}
