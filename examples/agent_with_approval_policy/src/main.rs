//! Policy-based, non-interactive tool approval with ECS-native state.
//!
//! A typed component stores the operator's rules on an addressable policy
//! entity. Its targeted responder runs once per call in deterministic policy
//! order: read-only search is allowed, transfers up to a limit are allowed,
//! and everything else fails closed as model-visible skipped-call feedback.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use rig::bevy_ecs::{observer::On, prelude::Component};
use rig::runtime::{
    EffectCompletion, EffectOutput, ModelEffectOutput, ModelToolCall, Policy, PolicyPoint,
    PolicyRule, RunState, ToolCallPolicyDecision, ToolCallPolicyInvocation, ToolCapability,
    ToolGrant, Usage,
};

const SEARCH_WEB: &str = "search_web";
const TRANSFER_FUNDS: &str = "transfer_funds";

#[derive(Component)]
struct ApprovalRules {
    auto_approve: BTreeSet<String>,
    max_auto_transfer: u64,
}

fn enforce_approval_rules(
    mut event: On<ToolCallPolicyInvocation>,
    rules: rig::bevy_ecs::prelude::Query<&ApprovalRules>,
) {
    let Ok(rules) = rules.get(event.policy) else {
        return;
    };
    let name = event.call.decision.name.as_str();
    event.decision = Some(if rules.auto_approve.contains(name) {
        ToolCallPolicyDecision::Run
    } else if name == TRANSFER_FUNDS {
        match event
            .call
            .arguments
            .get("amount")
            .and_then(|value| value.as_u64())
        {
            Some(amount) if amount <= rules.max_auto_transfer => ToolCallPolicyDecision::Run,
            Some(amount) => ToolCallPolicyDecision::Skip(format!(
                "denied by policy: transfers over ${} require human approval; ${amount} exceeds the limit",
                rules.max_auto_transfer
            )),
            None => ToolCallPolicyDecision::Skip(
                "denied by policy: could not read the transfer amount".to_owned(),
            ),
        }
    } else {
        ToolCallPolicyDecision::Skip(format!(
            "denied by policy: `{name}` is not on the approved tool list"
        ))
    });
}

fn install_tool(
    runtime: &mut rig::runtime::Runtime,
    agent: rig::runtime::AgentHandle,
    id: &str,
    name: &str,
    order: u32,
) -> Result<()> {
    let tool = runtime.spawn_tool(
        ecs_demo::id(id)?,
        ecs_demo::tenant()?,
        ToolCapability {
            name: name.to_owned(),
            description: format!("Example {name} capability"),
            parameters: serde_json::json!({"type": "object"}),
            order,
            revision: 1,
            retired: false,
        },
    )?;
    runtime.grant_tool(
        ecs_demo::id(&format!("{id}-grant"))?,
        ecs_demo::tenant()?,
        ToolGrant {
            order,
            enabled: true,
        },
        agent,
        tool,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(false)?;
    install_tool(&mut runtime, agent, "search-tool", SEARCH_WEB, 0)?;
    install_tool(&mut runtime, agent, "transfer-tool", TRANSFER_FUNDS, 1)?;
    let policy = runtime.spawn_policy(
        ecs_demo::id("approval-rules")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::Custom(PolicyPoint::ToolCall),
        },
        agent,
    )?;
    runtime
        .world_mut()
        .entity_mut(policy)
        .insert(ApprovalRules {
            auto_approve: BTreeSet::from([SEARCH_WEB.to_owned()]),
            max_auto_transfer: 1_000,
        })
        .observe(enforce_approval_rules);

    let pending = runtime
        .handle()
        .prompt(agent, "Research an amount, then transfer $5000")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
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
                tool_calls: vec![
                    ModelToolCall {
                        id: "search-call".to_owned(),
                        provider_result_id: "search-call".to_owned(),
                        provider_call_id: None,
                        name: SEARCH_WEB.to_owned(),
                        arguments: serde_json::json!({"query": "recommended amount"}),
                    },
                    ModelToolCall {
                        id: "transfer-call".to_owned(),
                        provider_result_id: "transfer-call".to_owned(),
                        provider_call_id: None,
                        name: TRANSFER_FUNDS.to_owned(),
                        arguments: serde_json::json!({"to": "B-2", "amount": 5_000}),
                    },
                ],
            })),
        })?;

    // Only the approved search call crosses the external effect boundary.
    let search = ecs_demo::next_effect(&mut runtime)?;
    let search_input = search
        .tool_input()
        .context("expected the approved search effect")?;
    println!("dispatched approved tool: {}", search_input.decision.name);
    ecs_demo::complete_tool(&runtime, &search, "recommended amount: $1000")?;

    let follow_up = ecs_demo::next_effect(&mut runtime)?;
    let results = &follow_up
        .model_input()
        .context("expected a follow-up model request")?
        .tool_results;
    let search_result = results.first().context("missing search result")?;
    let transfer_result = results.get(1).context("missing skipped transfer result")?;
    println!("{}: {}", search_result.name, search_result.presentation);
    println!("{}: {}", transfer_result.name, transfer_result.presentation);
    ecs_demo::complete_text(
        &runtime,
        &follow_up,
        "The transfer was not executed because it exceeded the policy limit.",
    )?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .context("the prompt should resolve to a run")?;
    match runtime.observe_run(run)? {
        Some(RunState::Completed(output)) => println!("final response: {}", output.text),
        state => anyhow::bail!("unexpected run outcome: {state:?}"),
    }
    Ok(())
}
