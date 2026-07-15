//! Stable, read-only diagnostics for live ECS runs.
//!
//! These helpers deliberately expose stable domain identities and normalized
//! phase descriptions, never raw channels, executors, or registered systems.

use bevy_ecs::prelude::Entity;
use serde::Serialize;
use thiserror::Error;

use super::*;

/// Failure to inspect a run or operation in this runtime.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RunDebugError {
    /// The handle belongs to another runtime world.
    #[error("run handle belongs to another runtime")]
    ForeignRuntime,
    /// The run entity is no longer live or lacks required domain state.
    #[error("run is stale")]
    StaleRun,
    /// The operation entity is no longer live.
    #[error("operation is stale")]
    StaleOperation,
}

/// One operation retained by a run.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PendingOperationDebug {
    /// Stable operation identity.
    pub id: StableId,
    /// Normalized operation kind.
    pub kind: String,
    /// Correlation generation currently accepted by ingress.
    pub generation: u64,
    /// Current operation phase.
    pub phase: String,
}

/// One accepted policy revision without runtime-local entity IDs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AcceptedPolicyDebug {
    /// Stable policy identity.
    pub id: StableId,
    /// Exact accepted revision.
    pub revision: u64,
    /// Deterministic ordering key.
    pub order: u32,
    /// Lifecycle point accepted by the operation.
    pub point: PolicyPoint,
}

/// Cursor and phase for a durable policy evaluation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PolicyEvaluationDebug {
    /// Evaluation family.
    pub kind: String,
    /// Next policy index.
    pub cursor: usize,
    /// Number of snapshotted policies.
    pub policy_count: usize,
    /// Current evaluation phase.
    pub phase: String,
}

/// Structured reason a run currently cannot make user-visible progress.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum StalledReason {
    /// The run is eligible for another schedule transition.
    Ready,
    /// Explicit run control prevents internal progression.
    Paused(String),
    /// An external effect has not settled.
    WaitingForEffect(PendingOperationDebug),
    /// An atomic tool batch still has unsettled members.
    WaitingForToolBatch { expected: u32, settled: u32 },
    /// The parent cannot continue until children commit terminal results.
    WaitingForChildren(Vec<StableId>),
    /// The run already has a terminal outcome.
    Terminal(String),
    /// State is internally inconsistent and needs operator attention.
    Inconsistent(String),
}

/// Read-only summary of a live run and its retained work graph.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RunExplanation {
    /// Stable run identity.
    pub run_id: StableId,
    /// Stable owning agent identity.
    pub agent_id: StableId,
    /// Tenant authorization scope.
    pub tenant: TenantId,
    /// Authoritative run state.
    pub state: String,
    /// Orthogonal pause/control state.
    pub control: String,
    /// Next committed model-turn index.
    pub next_turn: u32,
    /// Model calls already committed.
    pub model_calls: u32,
    /// Configured model-call limit, when bounded.
    pub max_model_calls: Option<u32>,
    /// Retained operations in deterministic stable-ID order.
    pub operations: Vec<PendingOperationDebug>,
    /// Current policy cursors retained by those operations.
    pub policy_evaluations: Vec<PolicyEvaluationDebug>,
    /// Immediate diagnostic for lack of progress.
    pub stalled_reason: StalledReason,
}

fn accepted_debug(policies: &[AcceptedPolicy]) -> Vec<AcceptedPolicyDebug> {
    policies
        .iter()
        .map(|policy| AcceptedPolicyDebug {
            id: policy.id.clone(),
            revision: policy.revision,
            order: policy.order,
            point: policy.point,
        })
        .collect()
}

fn debug_entity_id(world: &World, entity: Entity, kind: &str) -> StableId {
    world
        .get::<StableId>(entity)
        .cloned()
        .unwrap_or_else(|| StableId::generated(format!("{kind}-{}", entity.to_bits())))
}

fn escape_dot_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}

fn operation_debug(world: &World, operation: Entity) -> Option<PendingOperationDebug> {
    Some(PendingOperationDebug {
        id: debug_entity_id(world, operation, "operation"),
        kind: format!("{:?}", world.get::<OperationKind>(operation)?),
        generation: world.get::<OperationGeneration>(operation)?.0,
        phase: format!("{:?}", world.get::<OperationState>(operation)?.phase),
    })
}

fn evaluation_debug(world: &World, evaluation: Entity) -> Option<PolicyEvaluationDebug> {
    macro_rules! inspect {
        ($ty:ty, $name:literal) => {
            if let Some(value) = world.get::<$ty>(evaluation) {
                return Some(PolicyEvaluationDebug {
                    kind: $name.to_owned(),
                    cursor: value.cursor,
                    policy_count: value.policies.len(),
                    phase: format!("{:?}", value.phase),
                });
            }
        };
    }
    inspect!(RequestPolicyEvaluation, "request");
    inspect!(ToolCallPolicyEvaluation, "tool_call");
    inspect!(InvalidToolCallPolicyEvaluation, "invalid_tool_call");
    inspect!(ToolResultPolicyEvaluation, "tool_result");
    inspect!(CompletionResponsePolicyEvaluation, "completion_response");
    inspect!(TextDeltaPolicyEvaluation, "text_delta");
    inspect!(ToolCallDeltaPolicyEvaluation, "tool_call_delta");
    None
}

impl Runtime {
    /// Returns accepted policy revisions attached to an operation.
    pub fn accepted_policies(
        &self,
        operation: Entity,
    ) -> Result<Vec<AcceptedPolicyDebug>, RunDebugError> {
        if self.world.get_entity(operation).is_err() {
            return Err(RunDebugError::StaleOperation);
        }
        let mut accepted = Vec::new();
        if let Some(value) = self.world.get::<AcceptedPolicies>(operation) {
            accepted.extend(accepted_debug(&value.0));
        }
        if let Some(value) = self.world.get::<AcceptedToolCallPolicies>(operation) {
            accepted.extend(accepted_debug(&value.0));
        }
        if let Some(value) = self.world.get::<AcceptedInvalidToolCallPolicies>(operation) {
            accepted.extend(accepted_debug(&value.0));
        }
        if let Some(value) = self.world.get::<AcceptedToolResultPolicies>(operation) {
            accepted.extend(accepted_debug(&value.0));
        }
        if let Some(value) = self
            .world
            .get::<AcceptedCompletionResponsePolicies>(operation)
        {
            accepted.extend(accepted_debug(&value.0));
        }
        if let Some(value) = self.world.get::<AcceptedTextDeltaPolicies>(operation) {
            accepted.extend(accepted_debug(&value.0));
        }
        if let Some(value) = self.world.get::<AcceptedToolCallDeltaPolicies>(operation) {
            accepted.extend(accepted_debug(&value.0));
        }
        accepted.sort_by(|left, right| {
            left.point
                .cmp(&right.point)
                .then_with(|| left.order.cmp(&right.order))
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(accepted)
    }

    /// Returns retained operations for a run in stable-ID order.
    pub fn pending_effects(
        &self,
        run: RunHandle,
    ) -> Result<Vec<PendingOperationDebug>, RunDebugError> {
        self.validate_debug_run(run)?;
        let mut operations = self
            .world
            .get::<RunOperations>(run.entity)
            .into_iter()
            .flat_map(|operations| operations.0.iter().copied())
            .filter_map(|operation| operation_debug(&self.world, operation))
            .collect::<Vec<_>>();
        operations.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(operations)
    }

    /// Explains the current run, policy cursors, pending effects, and stall cause.
    pub fn explain_run(&self, run: RunHandle) -> Result<RunExplanation, RunDebugError> {
        self.validate_debug_run(run)?;
        let run_id = self
            .world
            .get::<StableId>(run.entity)
            .cloned()
            .ok_or(RunDebugError::StaleRun)?;
        let tenant = self
            .world
            .get::<TenantId>(run.entity)
            .cloned()
            .ok_or(RunDebugError::StaleRun)?;
        let agent = self
            .world
            .get::<RunOf>(run.entity)
            .ok_or(RunDebugError::StaleRun)?
            .get();
        let agent_id = self
            .world
            .get::<StableId>(agent)
            .cloned()
            .ok_or(RunDebugError::StaleRun)?;
        let state = self
            .world
            .get::<RunState>(run.entity)
            .ok_or(RunDebugError::StaleRun)?;
        let control = self
            .world
            .get::<RunControl>(run.entity)
            .copied()
            .unwrap_or_default();
        let record = self
            .world
            .get::<RunRecord>(run.entity)
            .ok_or(RunDebugError::StaleRun)?;
        let operations = self.pending_effects(run)?;
        let mut policy_evaluations = self
            .world
            .get::<RunOperations>(run.entity)
            .into_iter()
            .flat_map(|operations| operations.0.iter().copied())
            .flat_map(|operation| {
                self.world
                    .get::<OperationPolicyEvaluations>(operation)
                    .into_iter()
                    .flat_map(|evaluations| evaluations.0.iter().copied())
            })
            .filter_map(|evaluation| evaluation_debug(&self.world, evaluation))
            .collect::<Vec<_>>();
        policy_evaluations.sort_by(|left, right| left.kind.cmp(&right.kind));
        let stalled_reason = self.stalled_reason(run)?;
        Ok(RunExplanation {
            run_id,
            agent_id,
            tenant,
            state: format!("{state:?}"),
            control: format!("{control:?}"),
            next_turn: record.next_turn,
            model_calls: record.next_turn,
            max_model_calls: record.max_model_calls,
            operations,
            policy_evaluations,
            stalled_reason,
        })
    }

    /// Explains why the run is not currently producing user-visible progress.
    pub fn stalled_reason(&self, run: RunHandle) -> Result<StalledReason, RunDebugError> {
        self.validate_debug_run(run)?;
        let state = self
            .world
            .get::<RunState>(run.entity)
            .ok_or(RunDebugError::StaleRun)?;
        let control = self
            .world
            .get::<RunControl>(run.entity)
            .copied()
            .unwrap_or_default();
        if !matches!(control, RunControl::Running) {
            return Ok(StalledReason::Paused(format!("{control:?}")));
        }
        if let Some(children) = self.world.get::<ChildRuns>(run.entity)
            && self.world.get::<WaitingForChildren>(run.entity).is_some()
        {
            let mut ids = children
                .0
                .iter()
                .filter_map(|child| self.world.get::<StableId>(*child).cloned())
                .collect::<Vec<_>>();
            ids.sort();
            return Ok(StalledReason::WaitingForChildren(ids));
        }
        match state {
            RunState::Queued => Ok(StalledReason::Ready),
            RunState::WaitingModel { operation } | RunState::WaitingStore { operation } => {
                operation_debug(&self.world, *operation)
                    .map(StalledReason::WaitingForEffect)
                    .ok_or(RunDebugError::StaleOperation)
            }
            RunState::WaitingTools { batch } => {
                let batch_state = self
                    .world
                    .get::<ToolBatchState>(*batch)
                    .ok_or(RunDebugError::StaleOperation)?;
                let settled = self
                    .world
                    .get::<BatchOperations>(*batch)
                    .into_iter()
                    .flat_map(|operations| operations.0.iter().copied())
                    .filter(|operation| {
                        self.world
                            .get::<OperationState>(*operation)
                            .is_some_and(|state| {
                                matches!(
                                    state.phase,
                                    OperationPhase::Settled(_) | OperationPhase::Cancelled
                                )
                            })
                    })
                    .count() as u32;
                Ok(StalledReason::WaitingForToolBatch {
                    expected: batch_state.expected,
                    settled,
                })
            }
            RunState::Completed(_) => Ok(StalledReason::Terminal("completed".to_owned())),
            RunState::Failed(_) => Ok(StalledReason::Terminal("failed".to_owned())),
            RunState::Cancelled => Ok(StalledReason::Terminal("cancelled".to_owned())),
        }
    }

    /// Renders the retained run/operation/child topology as Graphviz DOT.
    pub fn dump_run_graph(&self, run: RunHandle) -> Result<String, RunDebugError> {
        let explanation = self.explain_run(run)?;
        let mut dot = format!(
            "digraph rig_run {{\n  \"{}\" [shape=box,label=\"run {}\\n{}\"];\n",
            escape_dot_label(explanation.run_id.as_str()),
            escape_dot_label(explanation.run_id.as_str()),
            escape_dot_label(&explanation.state)
        );
        for operation in &explanation.operations {
            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\\n{}\"];\n  \"{}\" -> \"{}\";\n",
                escape_dot_label(operation.id.as_str()),
                escape_dot_label(&operation.kind),
                escape_dot_label(&operation.phase),
                escape_dot_label(explanation.run_id.as_str()),
                escape_dot_label(operation.id.as_str())
            ));
        }
        let mut children = self
            .world
            .get::<ChildRuns>(run.entity)
            .into_iter()
            .flat_map(|children| children.0.iter().copied())
            .filter_map(|child| {
                Some((
                    self.world.get::<StableId>(child)?.clone(),
                    format!("{:?}", self.world.get::<RunState>(child)?),
                ))
            })
            .collect::<Vec<_>>();
        children.sort_by(|left, right| left.0.cmp(&right.0));
        for (child_id, child_state) in children {
            dot.push_str(&format!(
                "  \"{}\" [shape=box,label=\"child {}\\n{}\"];\n  \"{}\" -> \"{}\" [label=\"child\"];\n",
                escape_dot_label(child_id.as_str()),
                escape_dot_label(child_id.as_str()),
                escape_dot_label(&child_state),
                escape_dot_label(explanation.run_id.as_str()),
                escape_dot_label(child_id.as_str())
            ));
        }
        dot.push_str("}\n");
        Ok(dot)
    }

    fn validate_debug_run(&self, run: RunHandle) -> Result<(), RunDebugError> {
        if run.runtime_id != self.handle.runtime_id {
            return Err(RunDebugError::ForeignRuntime);
        }
        if self.world.get::<RunState>(run.entity).is_none() {
            return Err(RunDebugError::StaleRun);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> StableId {
        StableId::new(value).unwrap()
    }

    fn tenant() -> TenantId {
        TenantId::new("debug").unwrap()
    }

    #[test]
    fn diagnostics_explain_waiting_effect_and_accepted_policy() {
        let mut runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant(),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "debug".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant(), Agent::default(), model)
            .unwrap();
        runtime
            .spawn_policy(
                id("request-policy"),
                tenant(),
                Policy {
                    order: 4,
                    revision: 7,
                    rule: PolicyRule::PatchRequest(RequestPatch::new().instructions("debugged")),
                },
                agent,
            )
            .unwrap();

        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let effect = runtime.effects().try_recv().unwrap().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        runtime.world_mut().spawn((
            StableId::new("child\\\"\nrun").unwrap(),
            tenant(),
            RunState::Queued,
            ParentRun(run.entity),
        ));
        let explanation = runtime.explain_run(run).unwrap();

        assert!(!explanation.run_id.as_str().is_empty());
        assert_eq!(explanation.agent_id.as_str(), "agent");
        assert_eq!(explanation.operations.len(), 1);
        assert!(matches!(
            explanation.stalled_reason,
            StalledReason::WaitingForEffect(_)
        ));
        assert_eq!(
            runtime.accepted_policies(effect.operation).unwrap(),
            vec![AcceptedPolicyDebug {
                id: id("request-policy"),
                revision: 7,
                order: 4,
                point: PolicyPoint::Request,
            }]
        );
        let dot = runtime.dump_run_graph(run).unwrap();
        assert!(dot.contains(explanation.run_id.as_str()));
        assert!(dot.contains(explanation.operations[0].id.as_str()));
        assert!(dot.contains("child\\\\\\\"\\nrun"));
        assert!(dot.contains("[label=\"child\"]"));
    }
}
