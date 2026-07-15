//! Migration coverage for replacing host-only `ToolContext` with typed ECS state.

#![allow(clippy::unwrap_used)]

use rig_core::{
    bevy_ecs::prelude::Component,
    runtime::{Agent, ModelCapability, Runtime, RuntimeConfig, StableId, TenantId},
    tool::Tool,
};
use rig_derive::rig_tool;

mod domain {
    #[derive(serde::Deserialize, rig_core::schemars::JsonSchema)]
    pub struct ToolContext {
        pub label: String,
    }
}

#[rig_tool(description = "An application type named ToolContext remains a model argument")]
fn domain_context_is_an_ordinary_argument(
    context: domain::ToolContext,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(context.label)
}

#[derive(Component, Debug, PartialEq, Eq)]
struct RunOffset(i32);

#[tokio::test]
async fn same_named_domain_type_remains_a_model_argument() {
    let definition = rig_core::tool::tool_definition(&DomainContextIsAnOrdinaryArgument);
    assert!(
        definition
            .parameters
            .get("properties")
            .and_then(|properties| properties.get("context"))
            .is_some_and(serde_json::Value::is_object)
    );
    let output = DomainContextIsAnOrdinaryArgument
        .call(DomainContextIsAnOrdinaryArgumentParameters {
            context: domain::ToolContext {
                label: "domain".to_owned(),
            },
        })
        .await
        .unwrap();
    assert_eq!(output, serde_json::json!("domain"));
}

#[test]
fn extension_state_is_an_ordinary_typed_run_component() {
    let mut runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let tenant = TenantId::new("derive-test").unwrap();
    let model = runtime
        .spawn_model(
            StableId::new("model").unwrap(),
            tenant.clone(),
            ModelCapability {
                provider: "test".to_owned(),
                model: "test".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .unwrap();
    let agent = runtime
        .spawn_agent(
            StableId::new("agent").unwrap(),
            tenant,
            Agent::default(),
            model,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "typed state").unwrap();
    runtime.run_until_stalled().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();
    runtime
        .world_mut()
        .entity_mut(run.entity())
        .insert(RunOffset(4));
    assert_eq!(
        runtime.world().get::<RunOffset>(run.entity()),
        Some(&RunOffset(4))
    );
}

#[test]
fn obsolete_runtime_context_parameters_are_rejected() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/tool_context/*.rs");
}
