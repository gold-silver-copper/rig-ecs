//! Classify tool failures as immutable facts and apply ordered ECS policy.
//!
//! The first targeted result policy records failure metadata in a typed
//! run-scoped component. The second policy reads the record for the same call
//! ID: disk `EIO` is fatal, while `ENETUNREACH` remains model-visible,
//! retryable feedback. Raw data, operator diagnostics, and presentation stay
//! separate throughout.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use anyhow::{Context, Result};
use rig::bevy_ecs::{
    lifecycle::Add,
    observer::On,
    prelude::{Commands, Component, In, Query},
};
use rig::runtime::{
    CanonicalError, EffectCompletion, EffectOutput, ModelEffectOutput, ModelToolCall, Policy,
    PolicyPoint, PolicyResponderId, PolicyRule, RunRecord, RunState, ToolCapability,
    ToolEffectFailure, ToolEffectOutput, ToolGrant, ToolResultPolicyDecision,
    ToolResultPolicyInvocation, Usage,
};
use rig::tool::ToolErrorKind;

const TOOL_NAME: &str = "system_probe";

#[derive(Clone, Copy)]
enum Mode {
    Fatal,
    Recoverable,
}

#[derive(Clone)]
struct FailureRecord {
    call_id: String,
    kind: ToolErrorKind,
    retryable: Option<bool>,
    refusal: bool,
    message: String,
}

#[derive(Component, Default)]
struct FailureLedger(Vec<FailureRecord>);

fn initialize_ledger(event: On<Add, RunRecord>, mut commands: Commands) {
    commands
        .entity(event.entity)
        .insert(FailureLedger::default());
}

fn record_failure(
    In(event): In<ToolResultPolicyInvocation>,
    mut ledgers: Query<&mut FailureLedger>,
) -> Option<ToolResultPolicyDecision> {
    if let Some(failure) = &event.result.failure
        && let Ok(mut ledger) = ledgers.get_mut(event.run)
    {
        ledger.0.push(FailureRecord {
            call_id: event.result.call_id.clone(),
            kind: failure.kind,
            retryable: failure.retryable,
            refusal: failure.refusal,
            message: failure.message.clone(),
        });
    }
    Some(ToolResultPolicyDecision::Keep)
}

fn redact_model_presentation(
    In(event): In<ToolResultPolicyInvocation>,
) -> Option<ToolResultPolicyDecision> {
    Some(if event.result.failure.is_some() {
        ToolResultPolicyDecision::Rewrite("system probe unavailable".into())
    } else {
        ToolResultPolicyDecision::Keep
    })
}

fn stop_fatal_failure(
    In(event): In<ToolResultPolicyInvocation>,
    ledgers: Query<&FailureLedger>,
) -> Option<ToolResultPolicyDecision> {
    let record = ledgers.get(event.run).ok().and_then(|ledger| {
        ledger
            .0
            .iter()
            .find(|record| record.call_id == event.result.call_id)
    });
    Some(match record {
        Some(record) if record.kind == ToolErrorKind::Other => {
            ToolResultPolicyDecision::Stop(format!("fatal disk I/O failure ({})", record.message))
        }
        Some(record) => {
            println!(
                "recorded call={} kind={} retryable={:?} refusal={}",
                record.call_id, record.kind, record.retryable, record.refusal
            );
            ToolResultPolicyDecision::Keep
        }
        None => ToolResultPolicyDecision::Keep,
    })
}

fn parse_mode() -> Option<Mode> {
    match std::env::args().nth(1).as_deref() {
        Some("fatal") => Some(Mode::Fatal),
        Some("recoverable") => Some(Mode::Recoverable),
        _ => None,
    }
}

fn install_probe(
    runtime: &mut rig::runtime::Runtime,
    agent: rig::runtime::AgentHandle,
) -> Result<()> {
    let tool = runtime.spawn_tool(
        ecs_demo::id("system-probe-tool")?,
        ecs_demo::tenant()?,
        ToolCapability {
            name: TOOL_NAME.to_owned(),
            description: "Probe a disk or network operation".to_owned(),
            parameters: serde_json::json!({"type":"object"}),
            order: 0,
            revision: 1,
            retired: false,
        },
    )?;
    runtime.grant_tool(
        ecs_demo::id("system-probe-grant")?,
        ecs_demo::tenant()?,
        ToolGrant {
            order: 0,
            enabled: true,
        },
        agent,
        tool,
    )?;
    Ok(())
}

fn install_policies(
    runtime: &mut rig::runtime::Runtime,
    agent: rig::runtime::AgentHandle,
) -> Result<()> {
    runtime.world_mut().add_observer(initialize_ledger);
    let recorder = runtime.spawn_policy(
        ecs_demo::id("record-failure")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::Custom(PolicyPoint::ToolResult),
        },
        agent,
    )?;
    runtime.register_tool_result_policy_responder(
        recorder,
        PolicyResponderId::new("record-failure")?,
        record_failure,
    )?;
    let redaction = runtime.spawn_policy(
        ecs_demo::id("redact-failure")?,
        ecs_demo::tenant()?,
        Policy {
            order: 1,
            revision: 1,
            rule: PolicyRule::Custom(PolicyPoint::ToolResult),
        },
        agent,
    )?;
    runtime.register_tool_result_policy_responder(
        redaction,
        PolicyResponderId::new("redact-failure")?,
        redact_model_presentation,
    )?;
    let fatal = runtime.spawn_policy(
        ecs_demo::id("stop-fatal-failure")?,
        ecs_demo::tenant()?,
        Policy {
            order: 2,
            revision: 1,
            rule: PolicyRule::Custom(PolicyPoint::ToolResult),
        },
        agent,
    )?;
    runtime.register_tool_result_policy_responder(
        fatal,
        PolicyResponderId::new("stop-fatal-failure")?,
        stop_fatal_failure,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let Some(mode) = parse_mode() else {
        println!("Usage: tool_result_outcomes <fatal|recoverable>");
        return Ok(());
    };
    let (mut runtime, agent) = ecs_demo::runtime(false)?;
    install_probe(&mut runtime, agent)?;
    install_policies(&mut runtime, agent)?;
    let pending = runtime.handle().prompt(agent, "Run the system probe")?;
    let model = ecs_demo::next_effect(&mut runtime)?;
    let operation = match mode {
        Mode::Fatal => "read_disk",
        Mode::Recoverable => "connect_network",
    };
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
                tool_calls: vec![ModelToolCall {
                    id: "probe-call".to_owned(),
                    provider_result_id: "probe-call".to_owned(),
                    provider_call_id: None,
                    name: TOOL_NAME.to_owned(),
                    arguments: serde_json::json!({"operation": operation}),
                }],
            })),
        })?;
    let tool = ecs_demo::next_effect(&mut runtime)?;
    let input = tool
        .tool_input()
        .context("expected the probe tool effect")?;
    let failure = match mode {
        Mode::Fatal => ToolEffectFailure {
            message: "EIO while reading /var/data".to_owned(),
            retryable: Some(false),
            kind: ToolErrorKind::Other,
            refusal: false,
        },
        Mode::Recoverable => ToolEffectFailure {
            message: "ENETUNREACH while connecting".to_owned(),
            retryable: Some(true),
            kind: ToolErrorKind::Network,
            refusal: false,
        },
    };
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: input.call_id.clone(),
                provider_result_id: input.provider_result_id.clone(),
                provider_call_id: input.provider_call_id.clone(),
                name: input.decision.name.clone(),
                raw: serde_json::json!({"error": failure.kind.as_str()}).into(),
                presentation: format!("probe failed: {}", failure.kind).into(),
                failure: Some(failure),
            })),
        })?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .context("the prompt should resolve to a run")?;

    match mode {
        Mode::Fatal => match runtime.observe_run(run)? {
            Some(RunState::Failed(CanonicalError::PolicyTerminated { termination })) => {
                println!("fatal result stopped by {termination}");
            }
            state => anyhow::bail!("unexpected fatal outcome: {state:?}"),
        },
        Mode::Recoverable => {
            let follow_up = runtime
                .effects()
                .try_recv()?
                .context("recoverable feedback should reach the model")?;
            let result = follow_up
                .model_input()
                .and_then(|input| input.tool_results.first())
                .context("follow-up request should retain the tool result")?;
            println!(
                "model sees `{}` while raw audit data remains {}",
                result.presentation, result.raw
            );
            ecs_demo::complete_text(&runtime, &follow_up, "reported recoverable failure")?;
            runtime.run_until_stalled()?;
            match runtime.observe_run(run)? {
                Some(RunState::Completed(output)) => println!("completed: {}", output.text),
                state => anyhow::bail!("unexpected recoverable outcome: {state:?}"),
            }
        }
    }
    Ok(())
}
