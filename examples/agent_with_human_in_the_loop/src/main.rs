//! Interactive human-in-the-loop control for every ECS tool call.
//!
//! An ordered approval policy first suspends the call on an external operation.
//! The host reads stdin while holding no ECS borrow, stores the answer as a
//! typed policy component, and approves resumption. The next targeted policy
//! maps that durable decision to run, skip, argument rewrite, or stop.

#[path = "../../ecs_demo.rs"]
mod ecs_demo;

use std::io::Write;

use anyhow::{Context, Result};
use rig::bevy_ecs::prelude::{Component, In, Query};
use rig::runtime::{
    EffectCompletion, EffectOutput, ModelEffectOutput, ModelToolCall, Policy,
    PolicyApprovalEffectOutput, PolicyPoint, PolicyResponderId, PolicyRule, RunState,
    ToolCallPolicyDecision, ToolCallPolicyEvaluation, ToolCallPolicyEvaluationPhase,
    ToolCallPolicyInvocation, ToolCapability, ToolEffectInput, ToolGrant, Usage,
};

const SEND_EMAIL: &str = "send_email";
const DELETE_FILE: &str = "delete_file";

#[derive(Component, Clone)]
enum ReviewerDecision {
    Run,
    Rewrite(serde_json::Value),
    Skip(String),
    Stop(String),
}

fn apply_reviewer_decision(
    In(event): In<ToolCallPolicyInvocation>,
    decisions: Query<&ReviewerDecision>,
) -> Option<ToolCallPolicyDecision> {
    Some(match decisions.get(event.policy) {
        Ok(ReviewerDecision::Run) => ToolCallPolicyDecision::Run,
        Ok(ReviewerDecision::Rewrite(arguments)) => {
            ToolCallPolicyDecision::Rewrite(arguments.clone())
        }
        Ok(ReviewerDecision::Skip(reason)) => ToolCallPolicyDecision::Skip(reason.clone()),
        Ok(ReviewerDecision::Stop(reason)) => ToolCallPolicyDecision::Stop(reason.clone()),
        Err(_) => ToolCallPolicyDecision::Stop(
            "reviewer decision was missing; refusing to execute".to_owned(),
        ),
    })
}

fn ask(prompt: &str) -> Option<String> {
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line.trim().to_owned()),
    }
}

fn review(input: &ToolEffectInput) -> ReviewerDecision {
    println!("\nThe agent wants to run `{}`", input.decision.name);
    println!("arguments: {}", input.arguments);
    let Some(choice) = ask("[a]pprove / [d]eny / [e]dit arguments / a[b]ort? ") else {
        return ReviewerDecision::Stop("no reviewer input was available".to_owned());
    };
    match choice.to_ascii_lowercase().as_str() {
        "a" | "approve" => ReviewerDecision::Run,
        "d" | "deny" | "n" | "no" => ReviewerDecision::Skip(
            ask("reason shown to the model: ")
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "denied by the human reviewer".to_owned()),
        ),
        "e" | "edit" => ask("replacement JSON arguments: ")
            .and_then(|value| serde_json::from_str(&value).ok())
            .map_or_else(
                || ReviewerDecision::Skip("reviewer supplied invalid JSON".to_owned()),
                ReviewerDecision::Rewrite,
            ),
        "b" | "abort" | "q" | "quit" => {
            ReviewerDecision::Stop("run aborted by the human reviewer".to_owned())
        }
        other => ReviewerDecision::Skip(format!("denied: unrecognized reviewer input `{other}`")),
    }
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
            description: format!("Side-effecting {name} operation"),
            parameters: serde_json::json!({"type":"object"}),
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

fn complete_call(
    runtime: &rig::runtime::Runtime,
    model: &rig::runtime::EffectRequest,
    id: &str,
    name: &str,
    arguments: serde_json::Value,
) -> Result<()> {
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
                    id: id.to_owned(),
                    provider_result_id: id.to_owned(),
                    provider_call_id: None,
                    name: name.to_owned(),
                    arguments,
                }],
            })),
        })?;
    Ok(())
}

fn service_review(
    runtime: &mut rig::runtime::Runtime,
    decision_policy: rig::bevy_ecs::prelude::Entity,
) -> Result<Option<rig::runtime::EffectRequest>> {
    let approval = ecs_demo::next_effect(runtime)?;
    let call = {
        let world = runtime.world_mut();
        let mut evaluations = world.query::<&ToolCallPolicyEvaluation>();
        evaluations
            .iter(world)
            .find_map(|evaluation| match evaluation.phase {
                ToolCallPolicyEvaluationPhase::WaitingApproval { operation, .. }
                    if operation == approval.operation =>
                {
                    Some(evaluation.effective.clone())
                }
                _ => None,
            })
            .context("approval operation was not tied to a tool-call evaluation")?
    };
    let decision = review(&call);
    runtime
        .world_mut()
        .entity_mut(decision_policy)
        .insert(decision);
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: Some("reviewer decision captured in typed ECS state".to_owned()),
            })),
        })?;
    runtime.run_until_stalled()?;
    Ok(runtime.effects().try_recv()?)
}

fn main() -> Result<()> {
    let (mut runtime, agent) = ecs_demo::runtime(false)?;
    install_tool(&mut runtime, agent, "email-tool", SEND_EMAIL, 0)?;
    install_tool(&mut runtime, agent, "delete-tool", DELETE_FILE, 1)?;
    runtime.spawn_policy(
        ecs_demo::id("human-approval-gate")?,
        ecs_demo::tenant()?,
        Policy {
            order: 0,
            revision: 1,
            rule: PolicyRule::RequireApproval {
                point: PolicyPoint::ToolCall,
                prompt: "A human reviewer must decide this tool call".to_owned(),
            },
        },
        agent,
    )?;
    let decision_policy = runtime.spawn_policy(
        ecs_demo::id("apply-human-decision")?,
        ecs_demo::tenant()?,
        Policy {
            order: 1,
            revision: 1,
            rule: PolicyRule::Custom(PolicyPoint::ToolCall),
        },
        agent,
    )?;
    runtime.register_tool_call_policy_responder(
        decision_policy,
        PolicyResponderId::new("apply-human-decision")?,
        apply_reviewer_decision,
    )?;

    let pending = runtime.handle().prompt(
        agent,
        "Email Alice about the budget review, then delete the stale report",
    )?;
    let first_model = ecs_demo::next_effect(&mut runtime)?;
    complete_call(
        &runtime,
        &first_model,
        "email-call",
        SEND_EMAIL,
        serde_json::json!({
            "to": "alice@example.com",
            "subject": "Budget review",
            "body": "Reminder: the review is at 3pm"
        }),
    )?;
    let Some(after_email_review) = service_review(&mut runtime, decision_policy)? else {
        return finish_stopped(&mut runtime, &pending);
    };
    let second_model = if after_email_review.tool_input().is_some() {
        println!(
            "executing approved call with arguments {}",
            after_email_review
                .tool_input()
                .context("checked tool input")?
                .arguments
        );
        ecs_demo::complete_tool(&runtime, &after_email_review, "email sent")?;
        ecs_demo::next_effect(&mut runtime)?
    } else {
        after_email_review
    };
    complete_call(
        &runtime,
        &second_model,
        "delete-call",
        DELETE_FILE,
        serde_json::json!({"path": "/tmp/old_report.csv"}),
    )?;
    let Some(after_delete_review) = service_review(&mut runtime, decision_policy)? else {
        return finish_stopped(&mut runtime, &pending);
    };
    let final_model = if after_delete_review.tool_input().is_some() {
        println!(
            "executing approved call with arguments {}",
            after_delete_review
                .tool_input()
                .context("checked tool input")?
                .arguments
        );
        ecs_demo::complete_tool(&runtime, &after_delete_review, "file deleted")?;
        ecs_demo::next_effect(&mut runtime)?
    } else {
        after_delete_review
    };
    ecs_demo::complete_text(&runtime, &final_model, "operations review completed")?;
    runtime.run_until_stalled()?;
    let run = runtime
        .resolve_run(&pending)
        .context("the prompt should resolve to a run")?;
    match runtime.observe_run(run)? {
        Some(RunState::Completed(output)) => println!("final response: {}", output.text),
        state => println!("run ended without completion: {state:?}"),
    }
    Ok(())
}

fn finish_stopped(
    runtime: &mut rig::runtime::Runtime,
    pending: &rig::runtime::PendingRunHandle,
) -> Result<()> {
    let run = runtime
        .resolve_run(pending)
        .context("the stopped prompt should resolve to a run")?;
    println!("run stopped: {:?}", runtime.observe_run(run)?);
    Ok(())
}
