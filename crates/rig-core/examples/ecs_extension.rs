//! Install Rig into an existing Bevy world and extend its schedule.

use rig_core::{
    bevy_ecs::{
        component::Component,
        prelude::{Commands, Entity, Query, With, Without, World},
        schedule::{IntoScheduleConfigs, Schedules},
    },
    runtime::{
        Agent, EffectCompletion, EffectOutput, ModelCapability, ModelEffectOutput, RigSchedule,
        RigSet, RunState, RuntimeConfig, StableId, TenantId, Usage, install_runtime,
    },
};

#[derive(Component)]
struct Audited;

fn audit_new_agents(
    mut commands: Commands,
    agents: Query<Entity, (With<Agent>, Without<Audited>)>,
) {
    for entity in &agents {
        println!("agent {entity:?} is ready for audit");
        commands.entity(entity).insert(Audited);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = World::new();
    let installed = install_runtime(&mut world, RuntimeConfig::default())?;
    let mut schedules = world.resource_mut::<Schedules>();
    let schedule = schedules
        .get_mut(RigSchedule)
        .ok_or("Rig schedule was not installed")?;
    schedule.add_systems(audit_new_agents.in_set(RigSet::Publish));
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
        tenant,
        Agent::default(),
        model,
    )?;
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
    world.run_schedule(RigSchedule);
    let run = installed
        .resolve_run(&world, &pending)?
        .ok_or("run was not ingested")?;
    let state = installed
        .observe_run(&mut world, run)?
        .ok_or("run disappeared before observation")?;
    if !matches!(state, RunState::Completed(_)) {
        return Err("embedded run did not complete".into());
    }
    Ok(())
}
