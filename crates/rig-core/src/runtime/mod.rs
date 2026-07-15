//! Bevy ECS-native runtime primitives.
//!
//! [`RigSchedule`] is world-resident and is the single progression engine used
//! by both [`Runtime`] and embedded worlds. External work leaves the world as an
//! owned [`EffectRequest`] and returns through [`EffectCompletion`]; neither
//! type can contain an ECS borrow.
//!
//! Agents, runs, turns, operations, capabilities, grants, policy evaluations,
//! and their topology are ordinary entities, components, and relationships.
//! Steering policies are snapshotted by `(order, StableId)` and evaluated by a
//! durable cursor; observe-only lifecycle events remain separate so observer
//! registration order cannot change decisions. Request patches are rebuilt for
//! each model operation and never mutate the agent baseline.
//!
//! Blocking and streaming facades use the same schedule. Streaming deltas enter
//! through bounded effect ingress, are validated in sequence, then cross
//! entity-targeted observation and policy boundaries before publication.
//! Asynchronous approval, model, tool, store, and discovery work always runs on
//! owned inputs outside the world and returns with operation generations that
//! reject stale or late completions.
//! Serialized provider responses are retained as immutable
//! [`ProviderResponseDiagnostics`] components for response policy and telemetry
//! queries without coupling core progression to provider types.
//!
//! [`RunControl`] and [`AgentControl`] provide independent pause and admission
//! semantics while multiple agents and sibling runs share one world. Active
//! runs can be captured as stable-ID [`ActiveRunSnapshot`] values; runtime-only
//! observers, systems, clients, channels, and secrets must be rebound after
//! restoration. Extension state belongs in typed components related to the run
//! or operation, not in an erased scratchpad.

pub mod adapters;
mod config;
mod control;
mod debug;
mod effects;
mod identity;
mod policy;
mod snapshot;
mod state;
mod topology;

pub use config::RuntimeConfig;
pub use control::*;
pub use debug::{
    AcceptedPolicyDebug, PendingOperationDebug, PolicyEvaluationDebug, RunDebugError,
    RunExplanation, StalledReason,
};
pub use effects::*;
pub use identity::{IdentityError, StableId, TenantId};
pub use policy::*;
pub use snapshot::{
    ActiveRunCheckpointEntry, ActiveRunCheckpointJournal, ActiveRunSnapshot,
    ActiveRunSnapshotCodec, ActiveRunSnapshotError, ActiveRunSnapshotLimits,
    ActiveRunSnapshotProtection, ActiveRunSnapshotSummary, RestoredRuns, SnapshotExtensionSection,
    decode_active_run_snapshot, encode_active_run_snapshot, migrate_active_run_snapshot,
    register_active_run_snapshot_codec, restore_active_run, restore_active_run_with_limits,
    snapshot_active_run, summarize_active_run_snapshot,
};
pub use state::*;
pub use topology::{
    Agent, AgentPolicies, AgentRuns, AgentStoreGrants, AgentToolGrants, BatchOf, BatchOperations,
    ChildOrdinal, ChildRuns, DiscoveredFrom, DiscoveredTools, DiscoveryKey, DiscoveryOperationOf,
    DiscoveryOperations, DiscoverySource, DiscoveryState, GrantForAgent, GrantForTool, ModelAgents,
    ModelCapability, ModelToolChoice, OperationOf, OperationOfBatch, OutputRequirement, ParentRun,
    PolicyFor, PolicyForRun, RetiredCapability, RetrievalRequirement, RunOf, RunOperations,
    RunPolicies, RunSubscriptions, RunToolBatches, RunTurns, StoreCapability, StoreGrant,
    StoreGrantForAgent, StoreGrantForStore, StoreGrants, SubscriptionOf, ToolCapability, ToolGrant,
    ToolGrants, ToolRetrievalRequirement, TurnOf, UsesModel, WaitingForChildren,
};

use effects::{DiscoveryApplied, EffectIngress, OperationKind, PolicyApprovalApplied};
use policy::{
    CompletionResponsePolicyDone, CompletionResponsePolicyInitialized, EffectiveModelOutput,
    EffectiveToolOutput, InvalidToolCallPolicyInitialized, PendingModelRequest, PendingToolCall,
    PreparedToolOperation, RegisteredPolicyResponder, RequestPolicyInitialized,
    ToolCallPolicyInitialized, ToolResultPolicyDone, ToolResultPolicyInitialized,
    accepts_new_policy_evaluations, policy_entities,
};
use state::{EffectDeadline, default_structured_output_retries};
use topology::ChildResultCommitted;

use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, Sender, SyncSender, TryRecvError, TrySendError, channel, sync_channel},
    },
};

use bevy_ecs::{
    message::{MessageReader, MessageWriter, Messages},
    prelude::*,
    relationship::Relationship,
    schedule::{IntoScheduleConfigs, LogLevel, Schedule, ScheduleBuildSettings, ScheduleLabel},
    system::{SystemId, SystemParam},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    completion::Message as CompletionMessage, message::UserContent,
    streaming::ToolCallDeltaContent, tool::ToolOutput,
};

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

/// World-resident runtime schedule.
#[derive(ScheduleLabel, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RigSchedule;

/// Public semantic stages for extension ordering.
///
/// The variants intentionally describe lifecycle boundaries rather than
/// private implementation functions. Extensions can order work at a stable
/// boundary even when the core implementation of that boundary changes.
#[derive(SystemSet, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RigSet {
    /// Pull hosted topology and prompt commands into the world.
    IngestCommands,
    /// Pull hosted pause, resume, cancellation, and bulk-control commands.
    IngestControlCommands,
    /// Pull external effect completions and buffered observations into ECS messages.
    IngestEffects,
    /// Reconcile derived topology and stable identities.
    Reconcile,
    /// Reconcile changed agent admission and bulk-control state.
    ReconcileAgentControl,
    /// Apply requested run-local suspension or resumption.
    ApplyRunControl,
    /// Materialize dynamically requested agents, runs, and child relationships.
    SpawnDynamicAgentsAndRuns,
    /// Prepare run-local memory, retrieval, and prerequisite operations.
    PrepareRun,
    /// Prepare the next immutable model-operation baseline.
    PrepareModel,
    /// Snapshot applicable request policies and create durable evaluations.
    BeginRequestPolicy,
    /// Invoke the next deterministically selected request policy.
    InvokeRequestPolicy,
    /// Reduce request-policy decisions into the effective request.
    ReduceRequestPolicy,
    /// Finalize request decisions and publish pre-dispatch observations.
    FinalizeRequest,
    /// Dispatch immutable model, store, discovery, and approval effects.
    DispatchModel,
    /// Validate and retain model, stream, store, discovery, and approval ingress.
    ApplyModelCompletion,
    /// Begin and advance completion-response and text-delta policy evaluation.
    BeginResponsePolicy,
    /// Begin and advance invalid-tool resolution.
    ResolveInvalidTools,
    /// Commit an accepted model turn or its terminal response.
    CommitModelTurn,
    /// Create logical tool batches and snapshot per-call policies.
    PrepareToolBatch,
    /// Invoke and reduce ordered tool-call policies.
    BeginToolCallPolicy,
    /// Dispatch accepted tool operations.
    DispatchTools,
    /// Validate and retain tool completions.
    ApplyToolCompletions,
    /// Invoke and reduce ordered tool-result policies.
    BeginToolResultPolicy,
    /// Atomically commit settled logical tool batches.
    CommitToolBatch,
    /// Commit store persistence and other durable state.
    Persist,
    /// Publish approved deltas, lifecycle observations, and terminal streams.
    Publish,
    /// Propagate run and operation cancellation.
    Cancel,
    /// Reconcile retirement while retaining referenced revisions.
    Retire,
    /// Despawn only entities whose retention obligations are satisfied.
    Cleanup,
    /// Age message buffers and clear per-pass change tracking.
    MaintainMessages,
}

#[derive(Resource)]
struct CommandIngress(Mutex<Receiver<RuntimeCommand>>);

#[derive(Resource)]
struct CompletionIngress(Mutex<Receiver<EffectIngress>>);

#[derive(Resource, Clone)]
struct EffectOutbox(SyncSender<EffectRequest>);

#[derive(Resource, Clone)]
struct CancellationOutbox(SyncSender<EffectCancellation>);

#[derive(Resource, Default, Clone, Copy, Debug, Eq, PartialEq)]
struct Progress {
    epoch: u64,
}

#[derive(Resource, Default)]
struct StableIdIndex(HashMap<StableId, Entity>);

#[derive(Resource, Default, Clone, Debug, Eq, PartialEq)]
struct InvariantViolations(Vec<RuntimeInvariantError>);

#[derive(Resource, Clone, Copy)]
struct RuntimeIdentity(u64);

#[derive(Resource)]
struct ObservationIngress(Mutex<Receiver<StableId>>);

#[derive(Resource, Clone)]
struct RuntimeWaker(Arc<dyn Fn() + Send + Sync>);

impl RuntimeWaker {
    fn notify(&self) {
        (self.0)();
    }
}

#[derive(Resource, Default, Clone, Copy)]
struct RuntimeClock {
    tick: u64,
}

#[derive(Resource, Clone, Copy)]
struct EffectTimeoutTicks(u64);

#[derive(Resource, Clone, Copy)]
struct EffectDispatchLimits {
    per_pass: usize,
    per_run: usize,
    per_agent: usize,
    per_tenant: usize,
}

#[derive(Resource, Default)]
struct DispatchBudget {
    remaining: usize,
    allowances: HashMap<(String, OperationKind), usize>,
    in_flight_by_run: HashMap<Entity, usize>,
    in_flight_by_agent: HashMap<Entity, usize>,
    in_flight_by_tenant: HashMap<String, usize>,
    cursor: usize,
}

/// Rebuildable lifecycle metrics derived from authoritative ECS state.
#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeMetrics {
    /// Non-terminal runs.
    pub active_runs: usize,
    /// Successfully completed retained runs.
    pub completed_runs: usize,
    /// Failed retained runs.
    pub failed_runs: usize,
    /// Cancelled retained runs.
    pub cancelled_runs: usize,
    /// Operations currently outside the world at the effect boundary.
    pub in_flight_operations: usize,
    /// Prepared operations waiting for bounded queue capacity.
    pub prepared_operations: usize,
    /// Settled operations retained for audit/cleanup.
    pub settled_operations: usize,
    /// Capability versions retired but still retained.
    pub retired_tools: usize,
    /// Stream subscribers dropped for backpressure.
    pub dropped_subscribers: usize,
    /// Terminal runs retained until their result is observed.
    pub unobserved_terminal_runs: usize,
}

#[derive(Message, Clone, Debug)]
struct EffectIngressMessage(EffectIngress);

#[derive(Component)]
struct StreamSink(SyncSender<StreamItem>);

/// Authoritative subscription lifecycle.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubscriptionState {
    /// Eligible to receive deltas and the terminal outcome.
    Active,
    /// Subscriber could not keep pace with its bounded queue.
    DroppedSlowConsumer,
    /// Terminal state was published.
    Finished,
}

#[allow(clippy::large_enum_variant)]
enum RuntimeCommand {
    SpawnAgent {
        id: StableId,
        tenant: TenantId,
        agent: Agent,
        model_id: StableId,
        result: SyncSender<Result<AgentHandle, SpawnError>>,
    },
    Prompt {
        agent: Entity,
        run_id: StableId,
        prompt: PromptPayload,
        history: Vec<serde_json::Value>,
        output_schema: Option<serde_json::Value>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
        subscriber: Option<SyncSender<StreamItem>>,
        parent: Option<(Entity, u64)>,
        run_policies: Vec<RunPolicySpec>,
    },
    Cancel {
        run: Entity,
    },
    PauseRun {
        run: Entity,
        mode: PauseMode,
    },
    ResumeRun {
        run: Entity,
    },
    SetAgentControl {
        agent: Entity,
        control: AgentControl,
    },
    RefreshDiscovery {
        source: Entity,
    },
}

struct PromptPayload {
    message: serde_json::Value,
    text: String,
    plain_text: bool,
}

/// Error returned when bounded runtime ingress cannot accept work.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum SubmitError {
    /// A handle belongs to another runtime world.
    #[error("handle belongs to a different runtime")]
    ForeignRuntime,
    /// The configured bounded queue is full.
    #[error("runtime command queue is full")]
    Backpressure,
    /// The runtime has shut down.
    #[error("runtime command queue is disconnected")]
    Disconnected,
    /// Stable run identity was invalid.
    #[error(transparent)]
    InvalidIdentity(#[from] IdentityError),
    /// A canonical completion message could not be represented at ingress.
    #[error("invalid prompt message: {0}")]
    InvalidPrompt(String),
}

/// A stable handle scoped to one runtime world.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentHandle {
    runtime_id: u64,
    entity: Entity,
}

/// Discovery source handle scoped to one authoritative runtime world.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiscoveryHandle {
    runtime_id: u64,
    entity: Entity,
}

impl DiscoveryHandle {
    /// Underlying Bevy entity for advanced world access.
    pub fn entity(self) -> Entity {
        self.entity
    }
}

impl AgentHandle {
    /// Underlying Bevy entity for advanced world access.
    pub fn entity(self) -> Entity {
        self.entity
    }
}

/// A run handle scoped to one runtime world.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingRunHandle {
    runtime_id: u64,
    stable_id: StableId,
}

/// Result receiver for an agent created through the hosted command boundary.
pub struct PendingAgentHandle {
    receiver: Receiver<Result<AgentHandle, SpawnError>>,
}

impl PendingAgentHandle {
    /// Returns the spawn outcome once command ingestion has processed it.
    pub fn try_resolve(&self) -> Result<Option<Result<AgentHandle, SpawnError>>, SubmitError> {
        match self.receiver.try_recv() {
            Ok(result) => Ok(Some(result)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(SubmitError::Disconnected),
        }
    }
}

impl PendingRunHandle {
    /// Persistent run identity, usable before command ingestion.
    pub fn stable_id(&self) -> &StableId {
        &self.stable_id
    }
}

/// Thread-safe command facade for hosted runtimes.
#[derive(Clone)]
pub struct RuntimeHandle {
    runtime_id: u64,
    commands: SyncSender<RuntimeCommand>,
    observations: Sender<StableId>,
    next_run_id: Arc<AtomicU64>,
    subscriber_capacity: usize,
    waker: RuntimeWaker,
}

impl fmt::Debug for RuntimeHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeHandle")
            .field("runtime_id", &self.runtime_id)
            .finish_non_exhaustive()
    }
}

impl RuntimeHandle {
    fn submit(&self, command: RuntimeCommand) -> Result<(), SubmitError> {
        self.commands
            .try_send(command)
            .map_err(map_command_send_error)?;
        self.waker.notify();
        Ok(())
    }

    /// Creates an agent through command ingress without exposing mutable world access.
    ///
    /// The model is addressed by persistent identity and resolved with tenant
    /// validation when [`RigSchedule`] ingests the command.
    pub fn spawn_agent(
        &self,
        id: StableId,
        tenant: TenantId,
        agent: Agent,
        model_id: StableId,
    ) -> Result<PendingAgentHandle, SubmitError> {
        let (result, receiver) = sync_channel(1);
        self.submit(RuntimeCommand::SpawnAgent {
            id,
            tenant,
            agent,
            model_id,
            result,
        })?;
        Ok(PendingAgentHandle { receiver })
    }

    /// Creates a run with a caller-selected stable identity.
    pub fn spawn_run(
        &self,
        run_id: StableId,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
    ) -> Result<PendingRunHandle, SubmitError> {
        if agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        self.submit(RuntimeCommand::Prompt {
            agent: agent.entity,
            run_id: run_id.clone(),
            prompt: prompt_payload(prompt.into())?,
            history: Vec::new(),
            output_schema: None,
            max_model_calls: None,
            conversation: None,
            subscriber: None,
            parent: None,
            run_policies: Vec::new(),
        })?;
        Ok(PendingRunHandle {
            runtime_id: self.runtime_id,
            stable_id: run_id,
        })
    }

    /// Submits a prompt without exposing concurrent world access.
    pub fn prompt(
        &self,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
    ) -> Result<PendingRunHandle, SubmitError> {
        if agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        let sequence = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_id = StableId::new(format!("run-{}-{sequence}", self.runtime_id))?;
        self.submit(RuntimeCommand::Prompt {
            agent: agent.entity,
            run_id: run_id.clone(),
            prompt: prompt_payload(prompt.into())?,
            history: Vec::new(),
            output_schema: None,
            max_model_calls: None,
            conversation: None,
            subscriber: None,
            parent: None,
            run_policies: Vec::new(),
        })?;
        Ok(PendingRunHandle {
            runtime_id: self.runtime_id,
            stable_id: run_id,
        })
    }

    /// Submits a prompt with owned history and a per-run output requirement.
    ///
    /// Every value is copied into the run entity at ingress; subsequent caller
    /// mutation cannot change an accepted operation.
    pub fn prompt_configured(
        &self,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
        history: impl IntoIterator<Item = CompletionMessage>,
        output_schema: Option<serde_json::Value>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
    ) -> Result<PendingRunHandle, SubmitError> {
        self.prompt_configured_with_run_policies(
            agent,
            prompt,
            history,
            output_schema,
            max_model_calls,
            conversation,
            Vec::new(),
        )
    }

    /// Submits a configured prompt and atomically installs policies on its run.
    #[allow(clippy::too_many_arguments)]
    pub fn prompt_configured_with_run_policies(
        &self,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
        history: impl IntoIterator<Item = CompletionMessage>,
        output_schema: Option<serde_json::Value>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
        run_policies: Vec<RunPolicySpec>,
    ) -> Result<PendingRunHandle, SubmitError> {
        if agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        let history = history
            .into_iter()
            .map(|message| {
                serde_json::to_value(message)
                    .map_err(|error| SubmitError::InvalidPrompt(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sequence = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_id = StableId::new(format!("run-{}-{sequence}", self.runtime_id))?;
        self.submit(RuntimeCommand::Prompt {
            agent: agent.entity,
            run_id: run_id.clone(),
            prompt: prompt_payload(prompt.into())?,
            history,
            output_schema,
            max_model_calls,
            conversation,
            subscriber: None,
            parent: None,
            run_policies,
        })?;
        Ok(PendingRunHandle {
            runtime_id: self.runtime_id,
            stable_id: run_id,
        })
    }

    /// Submits a prompt in a stable conversation, enabling ECS store operations.
    pub fn prompt_in_conversation(
        &self,
        agent: AgentHandle,
        conversation: StableId,
        prompt: impl Into<CompletionMessage>,
    ) -> Result<PendingRunHandle, SubmitError> {
        if agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        let sequence = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_id = StableId::new(format!("run-{}-{sequence}", self.runtime_id))?;
        self.submit(RuntimeCommand::Prompt {
            agent: agent.entity,
            run_id: run_id.clone(),
            prompt: prompt_payload(prompt.into())?,
            history: Vec::new(),
            output_schema: None,
            max_model_calls: None,
            conversation: Some(conversation),
            subscriber: None,
            parent: None,
            run_policies: Vec::new(),
        })?;
        Ok(PendingRunHandle {
            runtime_id: self.runtime_id,
            stable_id: run_id,
        })
    }

    /// Delegates a child run while the parent and other runs continue independently.
    pub fn spawn_child_run(
        &self,
        parent: RunHandle,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
    ) -> Result<PendingRunHandle, SubmitError> {
        if parent.runtime_id != self.runtime_id || agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        let sequence = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_id = StableId::new(format!("run-{}-{sequence}", self.runtime_id))?;
        self.submit(RuntimeCommand::Prompt {
            agent: agent.entity,
            run_id: run_id.clone(),
            prompt: prompt_payload(prompt.into())?,
            history: Vec::new(),
            output_schema: None,
            max_model_calls: None,
            conversation: None,
            subscriber: None,
            parent: Some((parent.entity, sequence)),
            run_policies: Vec::new(),
        })?;
        Ok(PendingRunHandle {
            runtime_id: self.runtime_id,
            stable_id: run_id,
        })
    }

    /// Submits a prompt and creates a bounded incremental subscriber.
    pub fn prompt_stream(
        &self,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
    ) -> Result<(PendingRunHandle, RunStream), SubmitError> {
        if agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        let sequence = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_id = StableId::new(format!("run-{}-{sequence}", self.runtime_id))?;
        let (sender, receiver) = sync_channel(self.subscriber_capacity);
        self.submit(RuntimeCommand::Prompt {
            agent: agent.entity,
            run_id: run_id.clone(),
            prompt: prompt_payload(prompt.into())?,
            history: Vec::new(),
            output_schema: None,
            max_model_calls: None,
            conversation: None,
            subscriber: Some(sender),
            parent: None,
            run_policies: Vec::new(),
        })?;
        Ok((
            PendingRunHandle {
                runtime_id: self.runtime_id,
                stable_id: run_id.clone(),
            },
            RunStream {
                receiver,
                observations: self.observations.clone(),
                run_id: run_id.clone(),
                waker: self.waker.clone(),
                acknowledged: AtomicBool::new(false),
            },
        ))
    }

    /// Submits a streaming prompt with owned history and per-run schema.
    pub fn prompt_stream_configured(
        &self,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
        history: impl IntoIterator<Item = CompletionMessage>,
        output_schema: Option<serde_json::Value>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
    ) -> Result<(PendingRunHandle, RunStream), SubmitError> {
        self.prompt_stream_configured_with_run_policies(
            agent,
            prompt,
            history,
            output_schema,
            max_model_calls,
            conversation,
            Vec::new(),
        )
    }

    /// Submits a configured stream and atomically installs policies on its run.
    #[allow(clippy::too_many_arguments)]
    pub fn prompt_stream_configured_with_run_policies(
        &self,
        agent: AgentHandle,
        prompt: impl Into<CompletionMessage>,
        history: impl IntoIterator<Item = CompletionMessage>,
        output_schema: Option<serde_json::Value>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
        run_policies: Vec<RunPolicySpec>,
    ) -> Result<(PendingRunHandle, RunStream), SubmitError> {
        if agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        let history = history
            .into_iter()
            .map(|message| {
                serde_json::to_value(message)
                    .map_err(|error| SubmitError::InvalidPrompt(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sequence = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let run_id = StableId::new(format!("run-{}-{sequence}", self.runtime_id))?;
        let (sender, receiver) = sync_channel(self.subscriber_capacity);
        self.submit(RuntimeCommand::Prompt {
            agent: agent.entity,
            run_id: run_id.clone(),
            prompt: prompt_payload(prompt.into())?,
            history,
            output_schema,
            max_model_calls,
            conversation,
            subscriber: Some(sender),
            parent: None,
            run_policies,
        })?;
        Ok((
            PendingRunHandle {
                runtime_id: self.runtime_id,
                stable_id: run_id.clone(),
            },
            RunStream {
                receiver,
                observations: self.observations.clone(),
                run_id,
                waker: self.waker.clone(),
                acknowledged: AtomicBool::new(false),
            },
        ))
    }

    /// Requests cancellation of a known run entity.
    pub fn cancel(&self, run: RunHandle) -> Result<(), SubmitError> {
        if run.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        self.submit(RuntimeCommand::Cancel { run: run.entity })
    }

    /// Requests the default drain pause for a known run.
    pub fn pause(&self, run: RunHandle) -> Result<(), SubmitError> {
        self.pause_with_mode(run, PauseMode::Drain)
    }

    /// Requests an explicit suspension behavior for a known run.
    pub fn pause_with_mode(&self, run: RunHandle, mode: PauseMode) -> Result<(), SubmitError> {
        if run.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        self.submit(RuntimeCommand::PauseRun {
            run: run.entity,
            mode,
        })
    }

    /// Resumes a suspended run without changing its execution phase.
    pub fn resume(&self, run: RunHandle) -> Result<(), SubmitError> {
        if run.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        self.submit(RuntimeCommand::ResumeRun { run: run.entity })
    }

    /// Rejects new runs while allowing existing runs to continue.
    pub fn reject_new_runs(&self, agent: AgentHandle) -> Result<(), SubmitError> {
        self.set_agent_control(agent, AgentControl::RejectNewRuns)
    }

    /// Requests suspension of every active run belonging to an agent.
    pub fn pause_agent(&self, agent: AgentHandle, mode: PauseMode) -> Result<(), SubmitError> {
        self.set_agent_control(agent, AgentControl::PauseExistingRuns(mode))
    }

    /// Reopens admission for an agent; paused runs must be resumed explicitly.
    pub fn resume_agent(&self, agent: AgentHandle) -> Result<(), SubmitError> {
        self.set_agent_control(agent, AgentControl::Running)
    }

    fn set_agent_control(
        &self,
        agent: AgentHandle,
        control: AgentControl,
    ) -> Result<(), SubmitError> {
        if agent.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        self.submit(RuntimeCommand::SetAgentControl {
            agent: agent.entity,
            control,
        })
    }

    /// Requests a new generation of dynamic capability discovery.
    pub fn refresh_discovery(&self, source: DiscoveryHandle) -> Result<(), SubmitError> {
        if source.runtime_id != self.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        self.submit(RuntimeCommand::RefreshDiscovery {
            source: source.entity,
        })
    }
}

fn prompt_payload(message: CompletionMessage) -> Result<PromptPayload, SubmitError> {
    let text = message.rag_text().unwrap_or_default();
    let plain_text = matches!(
        &message,
        CompletionMessage::User { content }
            if content.len() == 1
                && matches!(content.first_ref(), UserContent::Text(_))
    );
    let message = serde_json::to_value(message)
        .map_err(|error| SubmitError::InvalidPrompt(error.to_string()))?;
    Ok(PromptPayload {
        message,
        text,
        plain_text,
    })
}

fn map_command_send_error(error: TrySendError<RuntimeCommand>) -> SubmitError {
    match error {
        TrySendError::Full(_) => SubmitError::Backpressure,
        TrySendError::Disconnected(_) => SubmitError::Disconnected,
    }
}

/// Entity-backed run handle after command ingestion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunHandle {
    runtime_id: u64,
    entity: Entity,
}

impl RunHandle {
    /// Underlying Bevy entity.
    pub fn entity(self) -> Entity {
        self.entity
    }
}

/// Incremental observation from the same authoritative run state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamItem {
    /// Ordered model text delta.
    Delta {
        /// Model-operation-local sequence.
        sequence: u64,
        /// Incremental text.
        text: String,
    },
    /// Ordered tool-call name or argument fragment.
    ToolCallDelta {
        /// Model-operation-local sequence.
        sequence: u64,
        /// Provider-facing tool-call ID.
        id: String,
        /// Rig correlation ID.
        internal_call_id: String,
        /// Tool name or argument fragment.
        content: ToolCallDeltaContent,
    },
    /// Terminal run state published after commit.
    Finished(StreamTerminal),
}

/// Terminal streaming observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamTerminal {
    /// Same output visible to a blocking observer.
    Completed(RunOutput),
    /// Same canonical failure visible on the run entity.
    Failed(CanonicalError),
    /// Run was cancelled.
    Cancelled,
}

/// Bounded receiver returned by [`RuntimeHandle::prompt_stream`].
pub struct RunStream {
    receiver: Receiver<StreamItem>,
    observations: Sender<StableId>,
    run_id: StableId,
    waker: RuntimeWaker,
    acknowledged: AtomicBool,
}

impl RunStream {
    fn acknowledge(&self) {
        if !self.acknowledged.swap(true, Ordering::AcqRel)
            && self.observations.send(self.run_id.clone()).is_ok()
        {
            self.waker.notify();
        }
    }

    /// Receives the next available item without blocking.
    pub fn try_recv(&self) -> Result<Option<StreamItem>, StreamReceiveError> {
        match self.receiver.try_recv() {
            Ok(item) => {
                if matches!(item, StreamItem::Finished(_)) {
                    self.acknowledge();
                }
                Ok(Some(item))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.acknowledge();
                Err(StreamReceiveError::Disconnected)
            }
        }
    }
}

impl Drop for RunStream {
    fn drop(&mut self) {
        self.acknowledge();
    }
}

/// Stream receiver failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum StreamReceiveError {
    /// Runtime dropped the subscription channel.
    #[error("stream subscription disconnected")]
    Disconnected,
}

/// External side of the owned effect boundary.
pub struct EffectIo {
    requests: Mutex<Receiver<EffectRequest>>,
    cancellations: Mutex<Receiver<EffectCancellation>>,
    ingress: SyncSender<EffectIngress>,
    waker: RuntimeWaker,
}

impl EffectIo {
    /// Attempts to receive submitted owned work without blocking.
    pub fn try_recv(&self) -> Result<Option<EffectRequest>, EffectIoError> {
        let receiver = self.requests.lock().map_err(|_| EffectIoError::Poisoned)?;
        match receiver.try_recv() {
            Ok(request) => Ok(Some(request)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(EffectIoError::Disconnected),
        }
    }

    /// Attempts to receive a correlated cancellation request without blocking.
    pub fn try_recv_cancellation(&self) -> Result<Option<EffectCancellation>, EffectIoError> {
        let receiver = self
            .cancellations
            .lock()
            .map_err(|_| EffectIoError::Poisoned)?;
        match receiver.try_recv() {
            Ok(cancellation) => Ok(Some(cancellation)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(EffectIoError::Disconnected),
        }
    }

    /// Returns a cloneable completion sender for an executor task.
    pub fn completion_sender(&self) -> EffectCompletionSender {
        EffectCompletionSender {
            ingress: self.ingress.clone(),
            waker: self.waker.clone(),
        }
    }

    /// Returns a cloneable delta sender using the same ordered ingress queue.
    pub fn delta_sender(&self) -> EffectDeltaSender {
        EffectDeltaSender {
            ingress: self.ingress.clone(),
            waker: self.waker.clone(),
        }
    }
}

/// Cloneable completion ingress owned by an executor.
#[derive(Clone)]
pub struct EffectCompletionSender {
    ingress: SyncSender<EffectIngress>,
    waker: RuntimeWaker,
}

impl EffectCompletionSender {
    /// Sends a completion with explicit bounded-queue backpressure.
    pub fn try_send(&self, completion: EffectCompletion) -> Result<(), EffectIoError> {
        self.ingress
            .try_send(EffectIngress::Completion(completion))
            .map_err(|error| match error {
                TrySendError::Full(_) => EffectIoError::Backpressure,
                TrySendError::Disconnected(_) => EffectIoError::Disconnected,
            })?;
        self.waker.notify();
        Ok(())
    }

    /// Sends provider-specific diagnostics before the correlated completion.
    pub fn try_send_provider_diagnostics(
        &self,
        diagnostics: ProviderDiagnosticsIngress,
    ) -> Result<(), EffectIoError> {
        self.ingress
            .try_send(EffectIngress::ProviderDiagnostics(diagnostics))
            .map_err(|error| match error {
                TrySendError::Full(_) => EffectIoError::Backpressure,
                TrySendError::Disconnected(_) => EffectIoError::Disconnected,
            })?;
        self.waker.notify();
        Ok(())
    }

    /// Reports a caught executor panic as a canonical completion.
    pub fn try_send_panic(
        &self,
        operation: Entity,
        generation: u64,
        message: impl Into<String>,
    ) -> Result<(), EffectIoError> {
        self.try_send(EffectCompletion {
            operation,
            generation,
            result: Err(CanonicalError::ExecutorPanicked(message.into())),
        })
    }
}

/// Cloneable ordered delta ingress owned by a streaming executor.
#[derive(Clone)]
pub struct EffectDeltaSender {
    ingress: SyncSender<EffectIngress>,
    waker: RuntimeWaker,
}

impl EffectDeltaSender {
    /// Sends a delta through the same bounded queue as final completions.
    pub fn try_send(&self, delta: EffectDelta) -> Result<(), EffectIoError> {
        self.ingress
            .try_send(EffectIngress::Delta(delta))
            .map_err(|error| match error {
                TrySendError::Full(_) => EffectIoError::Backpressure,
                TrySendError::Disconnected(_) => EffectIoError::Disconnected,
            })?;
        self.waker.notify();
        Ok(())
    }

    /// Sends provider-specific diagnostics through the ordered model ingress.
    pub fn try_send_provider_diagnostics(
        &self,
        diagnostics: ProviderDiagnosticsIngress,
    ) -> Result<(), EffectIoError> {
        self.ingress
            .try_send(EffectIngress::ProviderDiagnostics(diagnostics))
            .map_err(|error| match error {
                TrySendError::Full(_) => EffectIoError::Backpressure,
                TrySendError::Disconnected(_) => EffectIoError::Disconnected,
            })?;
        self.waker.notify();
        Ok(())
    }
}

/// Effect boundary error.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum EffectIoError {
    /// Queue is currently full.
    #[error("effect queue is full")]
    Backpressure,
    /// Runtime or executor has shut down.
    #[error("effect queue is disconnected")]
    Disconnected,
    /// Another thread panicked while holding the queue lock.
    #[error("effect queue lock is poisoned")]
    Poisoned,
}

/// Runtime installation error.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum InstallError {
    /// Rig was already installed into this world.
    #[error("Rig runtime is already installed in this world")]
    AlreadyInstalled,
    /// Queue capacities and progress limits must be non-zero.
    #[error("runtime capacities and progress limit must be non-zero")]
    InvalidCapacity,
}

/// Runtime invariant violation found by scheduled validation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RuntimeInvariantError {
    /// Stable IDs must be unique inside one authoritative world.
    #[error("duplicate stable id `{0}`")]
    DuplicateStableId(String),
    /// A relationship points at a stale entity or an entity of the wrong kind.
    #[error("{relationship} on {entity:?} points at invalid target {target:?}")]
    InvalidRelationship {
        /// Entity containing the relationship.
        entity: Entity,
        /// Relationship name.
        relationship: &'static str,
        /// Referenced entity.
        target: Entity,
    },
    /// A relationship crosses tenant scopes.
    #[error("{relationship} on {entity:?} crosses tenant scope")]
    RelationshipTenantMismatch {
        /// Entity containing the relationship.
        entity: Entity,
        /// Relationship name.
        relationship: &'static str,
    },
}

/// Resources returned when schedules are installed into an existing world.
pub struct InstalledRuntime {
    /// Hosted command facade.
    pub handle: RuntimeHandle,
    /// External executor interface.
    pub effects: EffectIo,
}

impl InstalledRuntime {
    fn validate_world(&self, world: &World) -> Result<(), SpawnError> {
        if world
            .get_resource::<RuntimeIdentity>()
            .is_none_or(|identity| identity.0 != self.handle.runtime_id)
        {
            return Err(SpawnError::ForeignRuntime);
        }
        Ok(())
    }

    /// Spawns a model capability into the existing host world.
    pub fn spawn_model(
        &self,
        world: &mut World,
        id: StableId,
        tenant: TenantId,
        model: ModelCapability,
    ) -> Result<Entity, SpawnError> {
        self.validate_world(world)?;
        ensure_unique_id(world, &id)?;
        Ok(world.spawn((id, tenant, model)).id())
    }

    /// Spawns an agent and returns a handle scoped to this installed runtime.
    pub fn spawn_agent(
        &self,
        world: &mut World,
        id: StableId,
        tenant: TenantId,
        agent: Agent,
        model: Entity,
    ) -> Result<AgentHandle, SpawnError> {
        self.validate_world(world)?;
        ensure_unique_id(world, &id)?;
        let Some(model_tenant) = world.get::<TenantId>(model) else {
            return Err(SpawnError::StaleEntity(model));
        };
        if model_tenant != &tenant {
            return Err(SpawnError::TenantMismatch);
        }
        let entity = world
            .spawn((id, tenant, agent, AgentControl::default(), UsesModel(model)))
            .id();
        Ok(AgentHandle {
            runtime_id: self.handle.runtime_id,
            entity,
        })
    }

    /// Resolves a submitted run after the host has driven [`RigSchedule`].
    pub fn resolve_run(
        &self,
        world: &World,
        pending: &PendingRunHandle,
    ) -> Result<Option<RunHandle>, SubmitError> {
        if pending.runtime_id != self.handle.runtime_id
            || world
                .get_resource::<RuntimeIdentity>()
                .is_none_or(|identity| identity.0 != self.handle.runtime_id)
        {
            return Err(SubmitError::ForeignRuntime);
        }
        Ok(world
            .resource::<StableIdIndex>()
            .0
            .get(&pending.stable_id)
            .copied()
            .map(|entity| RunHandle {
                runtime_id: self.handle.runtime_id,
                entity,
            }))
    }

    /// Clones a run state and acknowledges terminal observation in a host world.
    pub fn observe_run(
        &self,
        world: &mut World,
        run: RunHandle,
    ) -> Result<Option<RunState>, SubmitError> {
        if run.runtime_id != self.handle.runtime_id
            || world
                .get_resource::<RuntimeIdentity>()
                .is_none_or(|identity| identity.0 != self.handle.runtime_id)
        {
            return Err(SubmitError::ForeignRuntime);
        }
        let Some(state) = world.get::<RunState>(run.entity).cloned() else {
            return Ok(None);
        };
        if matches!(
            state,
            RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
        ) && let Some(mut record) = world.get_mut::<RunRecord>(run.entity)
        {
            record.observed = true;
        }
        Ok(Some(state))
    }
}

/// Persistence record containing only stable identities and canonical values.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DomainSnapshot {
    /// Dynamic discovery sources reconstructed before their capabilities.
    pub discovery_sources: Vec<PersistedDiscoverySource>,
    /// Configured model capabilities.
    pub models: Vec<PersistedModel>,
    /// Agents and their stable model relationships.
    pub agents: Vec<PersistedAgent>,
    /// Policy instances and their stable agent relationships.
    pub policies: Vec<PersistedPolicy>,
    /// Executable tool capability metadata.
    pub tools: Vec<PersistedTool>,
    /// Stable agent-to-tool access relationships.
    pub tool_grants: Vec<PersistedToolGrant>,
    /// Addressable store capabilities.
    pub stores: Vec<PersistedStore>,
    /// Stable agent-to-store access relationships.
    pub store_grants: Vec<PersistedStoreGrant>,
}

/// Persisted discovery source configuration without runtime operation handles.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedDiscoverySource {
    /// Persistent identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// External discovery adapter kind.
    pub kind: String,
    /// Last accepted refresh generation.
    pub generation: u64,
}

/// Persisted model record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedModel {
    /// Persistent identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Canonical model configuration.
    pub capability: ModelCapability,
}

/// Persisted agent record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedAgent {
    /// Persistent identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Canonical agent configuration.
    pub agent: Agent,
    /// Persisted admission and bulk-control state.
    pub control: AgentControl,
    /// Optional invalid-tool retry configuration for future runs.
    pub invalid_tool_call_budget: Option<InvalidToolCallBudget>,
    /// Optional structured-output retry budget for future runs.
    #[serde(default)]
    pub structured_output_retry_budget: Option<StructuredOutputRetryBudget>,
    /// Stable model identity remapped during loading.
    pub model_id: StableId,
    /// Optional structured-output configuration.
    pub output_requirement: Option<OutputRequirement>,
    /// Optional vector retrieval configuration.
    pub retrieval_requirement: Option<RetrievalRequirement>,
    /// Optional semantic tool-selection configuration for future runs.
    #[serde(default)]
    pub tool_retrieval_requirement: Option<ToolRetrievalRequirement>,
}

/// Persisted policy record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedPolicy {
    /// Persistent identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Canonical policy data.
    pub policy: Policy,
    /// Admission state retained so retired revisions remain addressable after restore.
    #[serde(default)]
    pub status: PolicyStatus,
    /// Stable agent identity remapped during loading.
    pub agent_id: StableId,
}

/// Persisted tool capability record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedTool {
    /// Persistent identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Provider-facing capability metadata.
    pub capability: ToolCapability,
    /// Stable discovery source identity for dynamically discovered versions.
    pub source_id: Option<StableId>,
    /// Source-local capability key, present exactly when `source_id` is present.
    pub discovery_key: Option<String>,
}

/// Persisted agent-to-tool grant record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedToolGrant {
    /// Persistent grant identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Grant metadata.
    pub grant: ToolGrant,
    /// Stable agent identity.
    pub agent_id: StableId,
    /// Stable tool identity.
    pub tool_id: StableId,
}

/// Persisted store capability record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedStore {
    /// Persistent identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Store capability metadata.
    pub capability: StoreCapability,
}

/// Persisted agent-to-store grant record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedStoreGrant {
    /// Persistent grant identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Grant metadata.
    pub grant: StoreGrant,
    /// Stable agent identity.
    pub agent_id: StableId,
    /// Stable store identity.
    pub store_id: StableId,
}

/// Stable IDs mapped to newly reconstructed runtime entities.
#[derive(Clone, Debug, Default)]
pub struct RestoredEntities(pub HashMap<StableId, Entity>);

/// Explicit persistence/remapping failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PersistenceError {
    /// Snapshot contains a duplicate or conflicts with current world state.
    #[error("conflicting stable id `{0}`")]
    ConflictingStableId(String),
    /// Relationship target is absent from both snapshot and current world.
    #[error("missing stable reference `{0}`")]
    MissingReference(String),
    /// Related persisted records cross tenant scopes.
    #[error("tenant mismatch for stable reference `{0}`")]
    TenantMismatch(String),
    /// Live ECS relationship is stale while taking a snapshot.
    #[error("stale runtime relationship on entity {0:?}")]
    StaleRelationship(Entity),
}

/// Creates deterministic persistence records without serializing runtime-only state.
pub fn snapshot_domain(world: &mut World) -> Result<DomainSnapshot, PersistenceError> {
    let mut discovery_query = world.query::<(Entity, &StableId, &TenantId, &DiscoverySource)>();
    let discovery_rows = discovery_query
        .iter(world)
        .map(|(entity, id, tenant, source)| {
            (
                entity,
                PersistedDiscoverySource {
                    id: id.clone(),
                    tenant: tenant.clone(),
                    kind: source.kind.clone(),
                    generation: source.generation,
                },
            )
        })
        .collect::<Vec<_>>();
    let discovery_ids = discovery_rows
        .iter()
        .map(|(entity, record)| (*entity, record.id.clone()))
        .collect::<HashMap<_, _>>();

    let mut model_query = world.query::<(Entity, &StableId, &TenantId, &ModelCapability)>();
    let model_rows = model_query
        .iter(world)
        .map(|(entity, id, tenant, capability)| {
            (
                entity,
                PersistedModel {
                    id: id.clone(),
                    tenant: tenant.clone(),
                    capability: capability.clone(),
                },
            )
        })
        .collect::<Vec<_>>();
    let model_ids = model_rows
        .iter()
        .map(|(entity, record)| (*entity, record.id.clone()))
        .collect::<HashMap<_, _>>();

    let mut agent_query = world.query::<(
        Entity,
        &StableId,
        &TenantId,
        &Agent,
        Option<&AgentControl>,
        Option<&InvalidToolCallBudget>,
        Option<&StructuredOutputRetryBudget>,
        &UsesModel,
        Option<&OutputRequirement>,
        Option<&RetrievalRequirement>,
        Option<&ToolRetrievalRequirement>,
    )>();
    let agent_rows = agent_query
        .iter(world)
        .map(
            |(
                entity,
                id,
                tenant,
                agent,
                control,
                invalid_tool_call_budget,
                structured_output_retry_budget,
                model,
                output_requirement,
                retrieval_requirement,
                tool_retrieval_requirement,
            )| {
                let model_id = model_ids
                    .get(&model.get())
                    .cloned()
                    .ok_or(PersistenceError::StaleRelationship(model.get()))?;
                Ok((
                    entity,
                    PersistedAgent {
                        id: id.clone(),
                        tenant: tenant.clone(),
                        agent: agent.clone(),
                        control: control.copied().unwrap_or_default(),
                        invalid_tool_call_budget: invalid_tool_call_budget.copied(),
                        structured_output_retry_budget: structured_output_retry_budget.copied(),
                        model_id,
                        output_requirement: output_requirement.cloned(),
                        retrieval_requirement: retrieval_requirement.cloned(),
                        tool_retrieval_requirement: tool_retrieval_requirement.cloned(),
                    },
                ))
            },
        )
        .collect::<Result<Vec<_>, PersistenceError>>()?;
    let agent_ids = agent_rows
        .iter()
        .map(|(entity, record)| (*entity, record.id.clone()))
        .collect::<HashMap<_, _>>();

    let mut policy_query = world.query::<(
        &StableId,
        &TenantId,
        &Policy,
        Option<&PolicyStatus>,
        &PolicyFor,
    )>();
    let mut policies = policy_query
        .iter(world)
        .map(|(id, tenant, policy, status, relation)| {
            let agent_id = agent_ids
                .get(&relation.get())
                .cloned()
                .ok_or(PersistenceError::StaleRelationship(relation.get()))?;
            Ok(PersistedPolicy {
                id: id.clone(),
                tenant: tenant.clone(),
                policy: policy.clone(),
                status: status.copied().unwrap_or_default(),
                agent_id,
            })
        })
        .collect::<Result<Vec<_>, PersistenceError>>()?;

    let mut tool_query = world.query::<(
        Entity,
        &StableId,
        &TenantId,
        &ToolCapability,
        Option<&DiscoveredFrom>,
        Option<&DiscoveryKey>,
    )>();
    let tool_rows = tool_query
        .iter(world)
        .map(|(entity, id, tenant, capability, source, key)| {
            let provenance = match (source, key) {
                (None, None) => (None, None),
                (Some(source), Some(key)) => (
                    Some(
                        discovery_ids
                            .get(&source.get())
                            .cloned()
                            .ok_or(PersistenceError::StaleRelationship(source.get()))?,
                    ),
                    Some(key.0.clone()),
                ),
                (Some(source), None) => {
                    return Err(PersistenceError::StaleRelationship(source.get()));
                }
                (None, Some(_)) => return Err(PersistenceError::StaleRelationship(entity)),
            };
            Ok((
                entity,
                PersistedTool {
                    id: id.clone(),
                    tenant: tenant.clone(),
                    capability: capability.clone(),
                    source_id: provenance.0,
                    discovery_key: provenance.1,
                },
            ))
        })
        .collect::<Result<Vec<_>, PersistenceError>>()?;
    let tool_ids = tool_rows
        .iter()
        .map(|(entity, record)| (*entity, record.id.clone()))
        .collect::<HashMap<_, _>>();
    let mut tool_grant_query = world.query::<(
        &StableId,
        &TenantId,
        &ToolGrant,
        &GrantForAgent,
        &GrantForTool,
    )>();
    let mut tool_grants = tool_grant_query
        .iter(world)
        .map(|(id, tenant, grant, agent, tool)| {
            let agent_id = agent_ids
                .get(&agent.get())
                .cloned()
                .ok_or(PersistenceError::StaleRelationship(agent.get()))?;
            let tool_id = tool_ids
                .get(&tool.get())
                .cloned()
                .ok_or(PersistenceError::StaleRelationship(tool.get()))?;
            Ok(PersistedToolGrant {
                id: id.clone(),
                tenant: tenant.clone(),
                grant: grant.clone(),
                agent_id,
                tool_id,
            })
        })
        .collect::<Result<Vec<_>, PersistenceError>>()?;

    let mut store_query = world.query::<(Entity, &StableId, &TenantId, &StoreCapability)>();
    let store_rows = store_query
        .iter(world)
        .map(|(entity, id, tenant, capability)| {
            (
                entity,
                PersistedStore {
                    id: id.clone(),
                    tenant: tenant.clone(),
                    capability: capability.clone(),
                },
            )
        })
        .collect::<Vec<_>>();
    let store_ids = store_rows
        .iter()
        .map(|(entity, record)| (*entity, record.id.clone()))
        .collect::<HashMap<_, _>>();
    let mut store_grant_query = world.query::<(
        &StableId,
        &TenantId,
        &StoreGrant,
        &StoreGrantForAgent,
        &StoreGrantForStore,
    )>();
    let mut store_grants = store_grant_query
        .iter(world)
        .map(|(id, tenant, grant, agent, store)| {
            let agent_id = agent_ids
                .get(&agent.get())
                .cloned()
                .ok_or(PersistenceError::StaleRelationship(agent.get()))?;
            let store_id = store_ids
                .get(&store.get())
                .cloned()
                .ok_or(PersistenceError::StaleRelationship(store.get()))?;
            Ok(PersistedStoreGrant {
                id: id.clone(),
                tenant: tenant.clone(),
                grant: grant.clone(),
                agent_id,
                store_id,
            })
        })
        .collect::<Result<Vec<_>, PersistenceError>>()?;

    let mut discovery_sources = discovery_rows
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut models = model_rows
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut agents = agent_rows
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut tools = tool_rows
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    let mut stores = store_rows
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();
    discovery_sources.sort_by(|left, right| left.id.cmp(&right.id));
    models.sort_by(|left, right| left.id.cmp(&right.id));
    agents.sort_by(|left, right| left.id.cmp(&right.id));
    policies.sort_by(|left, right| left.id.cmp(&right.id));
    tools.sort_by(|left, right| left.id.cmp(&right.id));
    tool_grants.sort_by(|left, right| left.id.cmp(&right.id));
    stores.sort_by(|left, right| left.id.cmp(&right.id));
    store_grants.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(DomainSnapshot {
        discovery_sources,
        models,
        agents,
        policies,
        tools,
        tool_grants,
        stores,
        store_grants,
    })
}

/// Restores entities first and then remaps stable relationships in a second pass.
pub fn restore_domain(
    world: &mut World,
    snapshot: DomainSnapshot,
) -> Result<RestoredEntities, PersistenceError> {
    let DomainSnapshot {
        discovery_sources,
        models,
        agents,
        policies,
        tools,
        tool_grants,
        stores,
        store_grants,
    } = snapshot;
    let mut existing_query = world.query::<(Entity, &StableId)>();
    let mut mapped = existing_query
        .iter(world)
        .map(|(entity, id)| (id.clone(), entity))
        .collect::<HashMap<_, _>>();
    let mut snapshot_ids = HashSet::new();
    for id in discovery_sources
        .iter()
        .map(|record| &record.id)
        .chain(models.iter().map(|record| &record.id))
        .chain(agents.iter().map(|record| &record.id))
        .chain(policies.iter().map(|record| &record.id))
        .chain(tools.iter().map(|record| &record.id))
        .chain(tool_grants.iter().map(|record| &record.id))
        .chain(stores.iter().map(|record| &record.id))
        .chain(store_grants.iter().map(|record| &record.id))
    {
        if mapped.contains_key(id) || !snapshot_ids.insert(id.clone()) {
            return Err(PersistenceError::ConflictingStableId(
                id.as_str().to_owned(),
            ));
        }
    }

    let mut model_tenants = HashMap::new();
    let mut existing_models = world.query::<(&StableId, &TenantId, &ModelCapability)>();
    for (id, tenant, _) in existing_models.iter(world) {
        model_tenants.insert(id.clone(), tenant.clone());
    }
    for record in &models {
        model_tenants.insert(record.id.clone(), record.tenant.clone());
    }
    let mut agent_tenants = HashMap::new();
    let mut existing_agents = world.query::<(&StableId, &TenantId, &Agent)>();
    for (id, tenant, _) in existing_agents.iter(world) {
        agent_tenants.insert(id.clone(), tenant.clone());
    }
    for record in &agents {
        let model_tenant = model_tenants.get(&record.model_id).ok_or_else(|| {
            PersistenceError::MissingReference(record.model_id.as_str().to_owned())
        })?;
        if model_tenant != &record.tenant {
            return Err(PersistenceError::TenantMismatch(
                record.model_id.as_str().to_owned(),
            ));
        }
        agent_tenants.insert(record.id.clone(), record.tenant.clone());
    }
    for record in &policies {
        let agent_tenant = agent_tenants.get(&record.agent_id).ok_or_else(|| {
            PersistenceError::MissingReference(record.agent_id.as_str().to_owned())
        })?;
        if agent_tenant != &record.tenant {
            return Err(PersistenceError::TenantMismatch(
                record.agent_id.as_str().to_owned(),
            ));
        }
    }

    let mut discovery_tenants = HashMap::new();
    let mut existing_sources = world.query::<(&StableId, &TenantId, &DiscoverySource)>();
    for (id, tenant, _) in existing_sources.iter(world) {
        discovery_tenants.insert(id.clone(), tenant.clone());
    }
    for record in &discovery_sources {
        discovery_tenants.insert(record.id.clone(), record.tenant.clone());
    }

    let mut tool_tenants = HashMap::new();
    let mut existing_tools = world.query::<(&StableId, &TenantId, &ToolCapability)>();
    for (id, tenant, _) in existing_tools.iter(world) {
        tool_tenants.insert(id.clone(), tenant.clone());
    }
    for record in &tools {
        match (&record.source_id, &record.discovery_key) {
            (None, None) => {}
            (Some(source_id), Some(_)) => {
                let source_tenant = discovery_tenants.get(source_id).ok_or_else(|| {
                    PersistenceError::MissingReference(source_id.as_str().to_owned())
                })?;
                if source_tenant != &record.tenant {
                    return Err(PersistenceError::TenantMismatch(
                        source_id.as_str().to_owned(),
                    ));
                }
            }
            (Some(source_id), None) => {
                return Err(PersistenceError::MissingReference(
                    source_id.as_str().to_owned(),
                ));
            }
            (None, Some(key)) => {
                return Err(PersistenceError::MissingReference(key.clone()));
            }
        }
        tool_tenants.insert(record.id.clone(), record.tenant.clone());
    }
    for record in &tool_grants {
        validate_persisted_grant(
            &record.tenant,
            &record.agent_id,
            &record.tool_id,
            &agent_tenants,
            &tool_tenants,
        )?;
    }

    let mut store_tenants = HashMap::new();
    let mut existing_stores = world.query::<(&StableId, &TenantId, &StoreCapability)>();
    for (id, tenant, _) in existing_stores.iter(world) {
        store_tenants.insert(id.clone(), tenant.clone());
    }
    for record in &stores {
        store_tenants.insert(record.id.clone(), record.tenant.clone());
    }
    for record in &store_grants {
        validate_persisted_grant(
            &record.tenant,
            &record.agent_id,
            &record.store_id,
            &agent_tenants,
            &store_tenants,
        )?;
    }

    for record in discovery_sources {
        let entity = world
            .spawn((
                record.id.clone(),
                record.tenant,
                DiscoverySource {
                    kind: record.kind,
                    generation: record.generation,
                    state: DiscoveryState::Idle,
                },
            ))
            .id();
        mapped.insert(record.id, entity);
    }
    for record in models {
        let entity = world
            .spawn((record.id.clone(), record.tenant, record.capability))
            .id();
        mapped.insert(record.id, entity);
    }
    for record in tools {
        let mut entity = world.spawn((record.id.clone(), record.tenant, record.capability));
        match (record.source_id, record.discovery_key) {
            (Some(source_id), Some(key)) => {
                let source = mapped_reference(&mapped, &source_id)?;
                entity.insert((DiscoveredFrom(source), DiscoveryKey(key)));
            }
            (None, None) => {}
            (Some(source_id), None) => {
                return Err(PersistenceError::MissingReference(
                    source_id.as_str().to_owned(),
                ));
            }
            (None, Some(key)) => return Err(PersistenceError::MissingReference(key)),
        }
        let entity = entity.id();
        mapped.insert(record.id, entity);
    }
    for record in stores {
        let entity = world
            .spawn((record.id.clone(), record.tenant, record.capability))
            .id();
        mapped.insert(record.id, entity);
    }
    for record in agents {
        let Some(model) = mapped.get(&record.model_id).copied() else {
            return Err(PersistenceError::MissingReference(
                record.model_id.as_str().to_owned(),
            ));
        };
        let mut entity = world.spawn((
            record.id.clone(),
            record.tenant,
            record.agent,
            record.control,
            UsesModel(model),
        ));
        if let Some(output_requirement) = record.output_requirement {
            entity.insert(output_requirement);
        }
        if let Some(invalid_tool_call_budget) = record.invalid_tool_call_budget {
            entity.insert(invalid_tool_call_budget);
        }
        if let Some(structured_output_retry_budget) = record.structured_output_retry_budget {
            entity.insert(structured_output_retry_budget);
        }
        if let Some(retrieval_requirement) = record.retrieval_requirement {
            entity.insert(retrieval_requirement);
        }
        if let Some(tool_retrieval_requirement) = record.tool_retrieval_requirement {
            entity.insert(tool_retrieval_requirement);
        }
        let entity = entity.id();
        mapped.insert(record.id, entity);
    }
    for record in policies {
        let Some(agent) = mapped.get(&record.agent_id).copied() else {
            return Err(PersistenceError::MissingReference(
                record.agent_id.as_str().to_owned(),
            ));
        };
        let entity = world
            .spawn((
                record.id.clone(),
                record.tenant,
                record.policy,
                record.status,
                PolicyFor(agent),
            ))
            .id();
        mapped.insert(record.id, entity);
    }
    for record in tool_grants {
        let agent = mapped_reference(&mapped, &record.agent_id)?;
        let tool = mapped_reference(&mapped, &record.tool_id)?;
        let entity = world
            .spawn((
                record.id.clone(),
                record.tenant,
                record.grant,
                GrantForAgent(agent),
                GrantForTool(tool),
            ))
            .id();
        mapped.insert(record.id, entity);
    }
    for record in store_grants {
        let agent = mapped_reference(&mapped, &record.agent_id)?;
        let store = mapped_reference(&mapped, &record.store_id)?;
        let entity = world
            .spawn((
                record.id.clone(),
                record.tenant,
                record.grant,
                StoreGrantForAgent(agent),
                StoreGrantForStore(store),
            ))
            .id();
        mapped.insert(record.id, entity);
    }
    Ok(RestoredEntities(mapped))
}

fn validate_persisted_grant(
    tenant: &TenantId,
    agent_id: &StableId,
    capability_id: &StableId,
    agent_tenants: &HashMap<StableId, TenantId>,
    capability_tenants: &HashMap<StableId, TenantId>,
) -> Result<(), PersistenceError> {
    let agent_tenant = agent_tenants
        .get(agent_id)
        .ok_or_else(|| PersistenceError::MissingReference(agent_id.as_str().to_owned()))?;
    let capability_tenant = capability_tenants
        .get(capability_id)
        .ok_or_else(|| PersistenceError::MissingReference(capability_id.as_str().to_owned()))?;
    if agent_tenant != tenant || capability_tenant != tenant {
        return Err(PersistenceError::TenantMismatch(
            capability_id.as_str().to_owned(),
        ));
    }
    Ok(())
}

fn mapped_reference(
    mapped: &HashMap<StableId, Entity>,
    id: &StableId,
) -> Result<Entity, PersistenceError> {
    mapped
        .get(id)
        .copied()
        .ok_or_else(|| PersistenceError::MissingReference(id.as_str().to_owned()))
}

fn count_prepared_request(
    event: On<CompletionRequestPrepared>,
    runs: Query<&RunOf>,
    mut agents: Query<&mut LifecycleTelemetry>,
) {
    let Ok(run_of) = runs.get(event.run) else {
        return;
    };
    if let Ok(mut telemetry) = agents.get_mut(run_of.get()) {
        telemetry.requests_prepared = telemetry.requests_prepared.saturating_add(1);
    }
}

fn count_committed_model_turn(
    event: On<ModelTurnFinished>,
    runs: Query<&RunOf>,
    mut agents: Query<&mut LifecycleTelemetry>,
) {
    let Ok(run_of) = runs.get(event.run) else {
        return;
    };
    if let Ok(mut telemetry) = agents.get_mut(run_of.get()) {
        telemetry.model_turns_committed = telemetry.model_turns_committed.saturating_add(1);
    }
}

fn count_committed_tool_batch(
    event: On<ToolBatchCommitted>,
    runs: Query<&RunOf>,
    mut agents: Query<&mut LifecycleTelemetry>,
) {
    let Ok(run_of) = runs.get(event.run) else {
        return;
    };
    if let Ok(mut telemetry) = agents.get_mut(run_of.get()) {
        telemetry.tool_batches_committed = telemetry.tool_batches_committed.saturating_add(1);
    }
}

fn count_completed_run(
    event: On<RunCompleted>,
    runs: Query<&RunOf>,
    mut agents: Query<&mut LifecycleTelemetry>,
) {
    let Ok(run_of) = runs.get(event.run) else {
        return;
    };
    if let Ok(mut telemetry) = agents.get_mut(run_of.get()) {
        telemetry.runs_completed = telemetry.runs_completed.saturating_add(1);
    }
}

fn count_failed_run(
    event: On<RunFailed>,
    runs: Query<&RunOf>,
    mut agents: Query<&mut LifecycleTelemetry>,
) {
    let Ok(run_of) = runs.get(event.run) else {
        return;
    };
    if let Ok(mut telemetry) = agents.get_mut(run_of.get()) {
        telemetry.runs_failed = telemetry.runs_failed.saturating_add(1);
    }
}

fn count_cancelled_run(
    event: On<RunCancelled>,
    runs: Query<&RunOf>,
    mut agents: Query<&mut LifecycleTelemetry>,
) {
    let Ok(run_of) = runs.get(event.run) else {
        return;
    };
    if let Ok(mut telemetry) = agents.get_mut(run_of.get()) {
        telemetry.runs_cancelled = telemetry.runs_cancelled.saturating_add(1);
    }
}

/// Installs Rig's resources and world-resident schedule into an existing world.
pub fn install_runtime(
    world: &mut World,
    config: RuntimeConfig,
) -> Result<InstalledRuntime, InstallError> {
    install_runtime_with_waker(world, config, || {})
}

/// Installs Rig and invokes `waker` after each accepted command, delta, or completion.
pub fn install_runtime_with_waker(
    world: &mut World,
    config: RuntimeConfig,
    waker: impl Fn() + Send + Sync + 'static,
) -> Result<InstalledRuntime, InstallError> {
    if world.contains_resource::<RuntimeIdentity>() {
        return Err(InstallError::AlreadyInstalled);
    }
    if config.command_capacity == 0
        || config.effect_capacity == 0
        || config.completion_capacity == 0
        || config.subscriber_capacity == 0
        || config.progress_limit == 0
        || config.effect_timeout_ticks == 0
        || config.max_effect_dispatches_per_pass == 0
        || config.per_run_effect_limit == 0
        || config.per_agent_effect_limit == 0
        || config.per_tenant_effect_limit == 0
    {
        return Err(InstallError::InvalidCapacity);
    }

    let runtime_id = NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed);
    let (command_tx, command_rx) = sync_channel(config.command_capacity);
    let (observation_tx, observation_rx) = channel();
    let (effect_tx, effect_rx) = sync_channel(config.effect_capacity);
    let (cancellation_tx, cancellation_rx) = sync_channel(config.effect_capacity);
    let (completion_tx, completion_rx) = sync_channel(config.completion_capacity);
    let waker = RuntimeWaker(Arc::new(waker));

    world.insert_resource(RuntimeIdentity(runtime_id));
    world.insert_resource(CommandIngress(Mutex::new(command_rx)));
    world.insert_resource(ObservationIngress(Mutex::new(observation_rx)));
    world.insert_resource(CompletionIngress(Mutex::new(completion_rx)));
    world.insert_resource(EffectOutbox(effect_tx));
    world.insert_resource(CancellationOutbox(cancellation_tx));
    world.insert_resource(Progress::default());
    world.insert_resource(StableIdIndex::default());
    world.insert_resource(InvariantViolations::default());
    world.insert_resource(RuntimeMetrics::default());
    world.insert_resource(RuntimeClock::default());
    world.insert_resource(EffectTimeoutTicks(config.effect_timeout_ticks));
    world.insert_resource(EffectDispatchLimits {
        per_pass: config.max_effect_dispatches_per_pass,
        per_run: config.per_run_effect_limit,
        per_agent: config.per_agent_effect_limit,
        per_tenant: config.per_tenant_effect_limit,
    });
    world.insert_resource(DispatchBudget::default());
    world.insert_resource(snapshot::SnapshotExtensionCodecs::default());
    world.insert_resource(waker.clone());
    world.insert_resource(Messages::<EffectIngressMessage>::default());
    world.add_observer(bind_builtin_policy_responder);
    world.add_observer(dematerialize_removed_policy);
    world.add_observer(unbind_policy_responders);
    world.add_observer(count_prepared_request);
    world.add_observer(count_committed_model_turn);
    world.add_observer(count_committed_tool_batch);
    world.add_observer(count_completed_run);
    world.add_observer(count_failed_run);
    world.add_observer(count_cancelled_run);

    let mut schedule = Schedule::new(RigSchedule);
    schedule.set_build_settings(ScheduleBuildSettings {
        ambiguity_detection: LogLevel::Error,
        ..ScheduleBuildSettings::default()
    });
    schedule.configure_sets(
        (
            RigSet::IngestCommands,
            RigSet::IngestControlCommands,
            RigSet::IngestEffects,
            RigSet::Reconcile,
            RigSet::ReconcileAgentControl,
            RigSet::ApplyRunControl,
            RigSet::SpawnDynamicAgentsAndRuns,
            RigSet::PrepareRun,
            RigSet::PrepareModel,
            RigSet::BeginRequestPolicy,
            RigSet::InvokeRequestPolicy,
            RigSet::ReduceRequestPolicy,
            RigSet::FinalizeRequest,
        )
            .chain(),
    );
    schedule.configure_sets(
        (
            RigSet::DispatchModel,
            RigSet::ApplyModelCompletion,
            RigSet::BeginResponsePolicy,
            RigSet::CommitModelTurn,
            RigSet::ResolveInvalidTools,
            RigSet::PrepareToolBatch,
            RigSet::BeginToolCallPolicy,
            RigSet::DispatchTools,
            RigSet::ApplyToolCompletions,
            RigSet::BeginToolResultPolicy,
            RigSet::CommitToolBatch,
        )
            .chain()
            .after(RigSet::FinalizeRequest),
    );
    schedule.configure_sets(
        (
            RigSet::Persist,
            RigSet::Publish,
            RigSet::Cancel,
            RigSet::Retire,
            RigSet::Cleanup,
            RigSet::MaintainMessages,
        )
            .chain()
            .after(RigSet::CommitToolBatch),
    );
    schedule.add_systems(
        (advance_runtime_clock, ingest_commands)
            .chain()
            .in_set(RigSet::IngestCommands),
    );
    schedule.add_systems(
        (ingest_observations, ingest_completions)
            .chain()
            .in_set(RigSet::IngestEffects),
    );
    schedule.add_systems(
        (reconcile_discovery_operations, reconcile_stable_ids)
            .chain()
            .in_set(RigSet::Reconcile),
    );
    schedule.add_systems(reconcile_agent_control.in_set(RigSet::ReconcileAgentControl));
    schedule.add_systems(reconcile_run_control.in_set(RigSet::ApplyRunControl));
    schedule.add_systems(prepare_store_operations.in_set(RigSet::PrepareRun));
    schedule.add_systems(prepare_model_operations.in_set(RigSet::PrepareModel));
    schedule.add_systems(initialize_request_policy_evaluations.in_set(RigSet::BeginRequestPolicy));
    schedule.add_systems(evaluate_request_policies.in_set(RigSet::InvokeRequestPolicy));
    schedule.add_systems(publish_prepared_model_observations.in_set(RigSet::FinalizeRequest));
    schedule.add_systems(
        (
            dispatch_model_operations,
            dispatch_discovery_operations,
            dispatch_store_operations,
            dispatch_policy_approval_operations,
        )
            .chain()
            .in_set(RigSet::DispatchModel),
    );
    schedule.add_systems(
        (
            apply_effect_ingress,
            apply_policy_approval_results,
            expire_effects,
        )
            .chain()
            .in_set(RigSet::ApplyModelCompletion),
    );
    schedule.add_systems(
        (
            initialize_completion_response_policy_evaluations,
            evaluate_completion_response_policies,
            evaluate_text_delta_policies,
            evaluate_tool_call_delta_policies,
            publish_applied_model_policy_observations,
        )
            .chain()
            .in_set(RigSet::BeginResponsePolicy),
    );
    schedule.add_systems(
        (
            publish_invalid_call_observations,
            initialize_invalid_tool_call_policy_evaluations,
            evaluate_invalid_tool_call_policies,
        )
            .chain()
            .in_set(RigSet::ResolveInvalidTools),
    );
    schedule.add_systems(
        (commit_model_operations, commit_child_results)
            .chain()
            .in_set(RigSet::CommitModelTurn),
    );
    schedule.add_systems(initialize_tool_call_policy_evaluations.in_set(RigSet::PrepareToolBatch));
    schedule.add_systems(evaluate_tool_call_policies.in_set(RigSet::BeginToolCallPolicy));
    schedule.add_systems(
        (publish_prepared_tool_observations, dispatch_tool_operations)
            .chain()
            .in_set(RigSet::DispatchTools),
    );
    schedule.add_systems(
        (
            initialize_tool_result_policy_evaluations,
            evaluate_tool_result_policies,
            publish_applied_tool_policy_observations,
        )
            .chain()
            .in_set(RigSet::BeginToolResultPolicy),
    );
    schedule.add_systems(commit_tool_batches.in_set(RigSet::CommitToolBatch));
    schedule.add_systems(commit_store_operations.in_set(RigSet::Persist));
    schedule.add_systems(
        (
            publish_run_terminal_observations,
            publish_terminal_streams,
            update_runtime_metrics,
        )
            .chain()
            .in_set(RigSet::Publish),
    );
    schedule.add_systems(
        (propagate_parent_cancellation, propagate_cancellation)
            .chain()
            .in_set(RigSet::Cancel),
    );
    schedule.add_systems(cleanup_retired_tools.in_set(RigSet::Retire));
    schedule.add_systems(cleanup_observed_runs.in_set(RigSet::Cleanup));
    schedule.add_systems(update_effect_messages.in_set(RigSet::MaintainMessages));
    world.add_schedule(schedule);

    Ok(InstalledRuntime {
        handle: RuntimeHandle {
            runtime_id,
            commands: command_tx,
            observations: observation_tx,
            next_run_id: Arc::new(AtomicU64::new(1)),
            subscriber_capacity: config.subscriber_capacity,
            waker: waker.clone(),
        },
        effects: EffectIo {
            requests: Mutex::new(effect_rx),
            cancellations: Mutex::new(cancellation_rx),
            ingress: completion_tx,
            waker,
        },
    })
}

/// Convenience owner for a standalone authoritative world.
pub struct Runtime {
    world: World,
    handle: RuntimeHandle,
    effects: EffectIo,
    progress_limit: usize,
}

impl Runtime {
    /// Creates a standalone runtime using the same installer available to hosts.
    pub fn new(config: RuntimeConfig) -> Result<Self, InstallError> {
        let progress_limit = config.progress_limit;
        let mut world = World::new();
        let installed = install_runtime(&mut world, config)?;
        Ok(Self {
            world,
            handle: installed.handle,
            effects: installed.effects,
            progress_limit,
        })
    }

    /// Returns the hosted command facade.
    pub fn handle(&self) -> RuntimeHandle {
        self.handle.clone()
    }

    /// Returns the external side of the effect boundary.
    pub fn effects(&self) -> &EffectIo {
        &self.effects
    }

    /// Provides synchronous advanced access at a safe point.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Provides synchronous construction/extension access at a safe point.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Installs a thin ECS extension into this runtime's authoritative world.
    pub fn install_extension(
        &mut self,
        extension: &impl RigExtension,
    ) -> Result<(), ExtensionInstallError> {
        extension.install(&mut self.world)
    }

    /// Runs one schedule pass.
    pub fn update(&mut self) {
        self.world.run_schedule(RigSchedule);
    }

    /// Drives immediate work until no system reports progress.
    pub fn run_until_stalled(&mut self) -> Result<usize, DriveError> {
        for passes in 1..=self.progress_limit {
            let before = self.world.resource::<Progress>().epoch;
            self.update();
            let after = self.world.resource::<Progress>().epoch;
            if after == before {
                return Ok(passes);
            }
        }
        Err(DriveError::ProgressLimit(self.progress_limit))
    }

    /// Spawns a model entity into this authoritative world.
    pub fn spawn_model(
        &mut self,
        id: StableId,
        tenant: TenantId,
        model: ModelCapability,
    ) -> Result<Entity, SpawnError> {
        ensure_unique_id(&mut self.world, &id)?;
        Ok(self.world.spawn((id, tenant, model)).id())
    }

    /// Spawns an agent related to an existing model entity.
    pub fn spawn_agent(
        &mut self,
        id: StableId,
        tenant: TenantId,
        agent: Agent,
        model: Entity,
    ) -> Result<AgentHandle, SpawnError> {
        ensure_unique_id(&mut self.world, &id)?;
        let Some(model_tenant) = self.world.get::<TenantId>(model) else {
            return Err(SpawnError::StaleEntity(model));
        };
        if model_tenant != &tenant {
            return Err(SpawnError::TenantMismatch);
        }
        let entity = self
            .world
            .spawn((id, tenant, agent, AgentControl::default(), UsesModel(model)))
            .id();
        Ok(AgentHandle {
            runtime_id: self.handle.runtime_id,
            entity,
        })
    }

    /// Spawns a policy instance related to an agent.
    pub fn spawn_policy(
        &mut self,
        id: StableId,
        tenant: TenantId,
        policy: Policy,
        agent: AgentHandle,
    ) -> Result<Entity, SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        ensure_unique_id(&mut self.world, &id)?;
        let Some(agent_tenant) = self.world.get::<TenantId>(agent.entity) else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        if agent_tenant != &tenant {
            return Err(SpawnError::TenantMismatch);
        }
        Ok(self
            .world
            .spawn((
                id,
                tenant,
                policy,
                PolicyStatus::Enabled,
                PolicyFor(agent.entity),
            ))
            .id())
    }

    /// Spawns an extension-owned typed policy related to an agent.
    ///
    /// `components` may contain any extension-defined policy facts. The
    /// declared capabilities determine which lifecycle snapshots include the
    /// entity; bind one explicit responder for every participating point.
    /// Extension-owned policy state is runtime-only unless the extension also
    /// supplies a stable persistence codec.
    pub fn spawn_typed_policy<B>(
        &mut self,
        id: StableId,
        tenant: TenantId,
        meta: PolicyMeta,
        capabilities: PolicyCapabilities,
        components: B,
        agent: AgentHandle,
    ) -> Result<Entity, SpawnError>
    where
        B: Bundle,
    {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        ensure_unique_id(&mut self.world, &id)?;
        let Some(agent_tenant) = self.world.get::<TenantId>(agent.entity) else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        if agent_tenant != &tenant {
            return Err(SpawnError::TenantMismatch);
        }
        Ok(self
            .world
            .spawn((
                id,
                tenant,
                meta,
                capabilities,
                components,
                PolicyStatus::Enabled,
                PolicyFor(agent.entity),
            ))
            .id())
    }

    /// Spawns a policy revision scoped to one live run.
    pub fn spawn_run_policy(
        &mut self,
        id: StableId,
        tenant: TenantId,
        policy: Policy,
        run: RunHandle,
    ) -> Result<Entity, SpawnError> {
        if run.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        ensure_unique_id(&mut self.world, &id)?;
        let Some(run_tenant) = self.world.get::<TenantId>(run.entity) else {
            return Err(SpawnError::StaleEntity(run.entity));
        };
        if self.world.get::<RunState>(run.entity).is_none() {
            return Err(SpawnError::StaleEntity(run.entity));
        }
        if run_tenant != &tenant {
            return Err(SpawnError::TenantMismatch);
        }
        Ok(self
            .world
            .spawn((
                id,
                tenant,
                policy,
                PolicyStatus::Enabled,
                PolicyForRun(run.entity),
            ))
            .id())
    }

    fn register_typed_policy_responder<E, D, M>(
        &mut self,
        policy: Entity,
        point: PolicyPoint,
        id: PolicyResponderId,
        system: impl IntoSystem<In<E>, Option<D>, M> + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        E: Send + Sync + 'static,
        D: 'static,
        M: 'static,
    {
        let Some(tenant) = self.world.get::<TenantId>(policy).cloned() else {
            return Err(PolicyResponderRegistrationError::StalePolicy(policy));
        };
        if self.world.get::<PolicyMeta>(policy).is_none() {
            return Err(PolicyResponderRegistrationError::StalePolicy(policy));
        }
        let Some(capabilities) = self.world.get::<PolicyCapabilities>(policy) else {
            return Err(PolicyResponderRegistrationError::StalePolicy(policy));
        };
        if !capabilities.contains(point) {
            return Err(PolicyResponderRegistrationError::CapabilityMismatch(point));
        }
        if let Some(bindings) = self.world.get::<PolicyResponders>(policy) {
            for binding in bindings.iter() {
                if self
                    .world
                    .get::<ResponderPoint>(binding)
                    .is_some_and(|candidate| candidate.0 == point)
                {
                    return Err(PolicyResponderRegistrationError::ConflictingResponder(
                        point,
                    ));
                }
                if self
                    .world
                    .get::<PolicyResponderId>(binding)
                    .is_some_and(|candidate| candidate == &id)
                {
                    return Err(PolicyResponderRegistrationError::DuplicateResponderId(
                        id.as_str().to_owned(),
                    ));
                }
            }
        }
        let responder = self.world.register_system(system).entity();
        Ok(self
            .world
            .spawn((
                PolicyResponderFor(policy),
                ResponderPoint(point),
                id,
                tenant,
                RegisteredPolicyResponder(responder),
            ))
            .id())
    }

    /// Binds the exact registered system used to steer one request policy.
    pub fn register_request_policy_responder<M>(
        &mut self,
        policy: Entity,
        id: PolicyResponderId,
        system: impl IntoSystem<In<RequestPolicyInvocation>, Option<RequestPolicyDecision>, M> + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        M: 'static,
    {
        self.register_typed_policy_responder(policy, PolicyPoint::Request, id, system)
    }

    /// Binds the exact registered system used to steer one tool-call policy.
    pub fn register_tool_call_policy_responder<M>(
        &mut self,
        policy: Entity,
        id: PolicyResponderId,
        system: impl IntoSystem<In<ToolCallPolicyInvocation>, Option<ToolCallPolicyDecision>, M>
        + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        M: 'static,
    {
        self.register_typed_policy_responder(policy, PolicyPoint::ToolCall, id, system)
    }

    /// Binds the exact registered system used to steer one invalid-tool policy.
    pub fn register_invalid_tool_call_policy_responder<M>(
        &mut self,
        policy: Entity,
        id: PolicyResponderId,
        system: impl IntoSystem<
            In<InvalidToolCallPolicyInvocation>,
            Option<InvalidToolCallPolicyDecision>,
            M,
        > + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        M: 'static,
    {
        self.register_typed_policy_responder(policy, PolicyPoint::InvalidToolCall, id, system)
    }

    /// Binds the exact registered system used to steer one tool-result policy.
    pub fn register_tool_result_policy_responder<M>(
        &mut self,
        policy: Entity,
        id: PolicyResponderId,
        system: impl IntoSystem<In<ToolResultPolicyInvocation>, Option<ToolResultPolicyDecision>, M>
        + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        M: 'static,
    {
        self.register_typed_policy_responder(policy, PolicyPoint::ToolResult, id, system)
    }

    /// Binds the exact registered system used to steer one completion-response policy.
    pub fn register_completion_response_policy_responder<M>(
        &mut self,
        policy: Entity,
        id: PolicyResponderId,
        system: impl IntoSystem<
            In<CompletionResponsePolicyInvocation>,
            Option<CompletionResponsePolicyDecision>,
            M,
        > + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        M: 'static,
    {
        self.register_typed_policy_responder(policy, PolicyPoint::CompletionResponse, id, system)
    }

    /// Binds the exact registered system used to steer one streaming text-delta policy.
    pub fn register_text_delta_policy_responder<M>(
        &mut self,
        policy: Entity,
        id: PolicyResponderId,
        system: impl IntoSystem<In<TextDeltaPolicyInvocation>, Option<TextDeltaPolicyDecision>, M>
        + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        M: 'static,
    {
        self.register_typed_policy_responder(policy, PolicyPoint::TextDelta, id, system)
    }

    /// Binds the exact registered system used to steer one streaming tool-call-delta policy.
    pub fn register_tool_call_delta_policy_responder<M>(
        &mut self,
        policy: Entity,
        id: PolicyResponderId,
        system: impl IntoSystem<
            In<ToolCallDeltaPolicyInvocation>,
            Option<ToolCallDeltaPolicyDecision>,
            M,
        > + 'static,
    ) -> Result<Entity, PolicyResponderRegistrationError>
    where
        M: 'static,
    {
        self.register_typed_policy_responder(policy, PolicyPoint::ToolCallDelta, id, system)
    }

    /// Retires a policy from future snapshots while preserving accepted work.
    pub fn retire_policy(&mut self, policy: Entity) -> Result<(), SpawnError> {
        let Some(mut entity) = self.world.get_entity_mut(policy).ok() else {
            return Err(SpawnError::StaleEntity(policy));
        };
        if !entity.contains::<PolicyMeta>() {
            return Err(SpawnError::StaleEntity(policy));
        }
        entity.insert(PolicyStatus::Retired);
        Ok(())
    }

    /// Sets an agent's structured-output requirement for future operations.
    pub fn set_output_requirement(
        &mut self,
        agent: AgentHandle,
        requirement: OutputRequirement,
    ) -> Result<(), SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        let Some(mut entity) = self.world.get_entity_mut(agent.entity).ok() else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        if !entity.contains::<Agent>() {
            return Err(SpawnError::StaleEntity(agent.entity));
        }
        entity.insert(requirement);
        if !entity.contains::<StructuredOutputRetryBudget>() {
            entity.insert(StructuredOutputRetryBudget::default());
        }
        Ok(())
    }

    /// Sets the structured-output retry budget accepted by future runs.
    pub fn set_structured_output_retry_budget(
        &mut self,
        agent: AgentHandle,
        max_retries: u32,
    ) -> Result<(), SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        let Some(mut entity) = self.world.get_entity_mut(agent.entity).ok() else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        if !entity.contains::<Agent>() {
            return Err(SpawnError::StaleEntity(agent.entity));
        }
        entity.insert(StructuredOutputRetryBudget { max_retries });
        Ok(())
    }

    /// Sets the invalid-tool retry budget accepted by future runs.
    pub fn set_invalid_tool_call_budget(
        &mut self,
        agent: AgentHandle,
        max_retries: u32,
    ) -> Result<(), SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        let Some(mut entity) = self.world.get_entity_mut(agent.entity).ok() else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        if !entity.contains::<Agent>() {
            return Err(SpawnError::StaleEntity(agent.entity));
        }
        entity.insert(InvalidToolCallBudget { max_retries });
        Ok(())
    }

    /// Sets vector retrieval for future runs of an agent.
    pub fn set_retrieval_requirement(
        &mut self,
        agent: AgentHandle,
        requirement: RetrievalRequirement,
    ) -> Result<(), SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        let Some(mut entity) = self.world.get_entity_mut(agent.entity).ok() else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        if !entity.contains::<Agent>() {
            return Err(SpawnError::StaleEntity(agent.entity));
        }
        entity.insert(requirement);
        Ok(())
    }

    /// Sets semantic tool retrieval for future runs of an agent.
    pub fn set_tool_retrieval_requirement(
        &mut self,
        agent: AgentHandle,
        requirement: ToolRetrievalRequirement,
    ) -> Result<(), SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        let Some(mut entity) = self.world.get_entity_mut(agent.entity).ok() else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        if !entity.contains::<Agent>() {
            return Err(SpawnError::StaleEntity(agent.entity));
        }
        entity.insert(requirement);
        Ok(())
    }

    /// Spawns an executable tool capability entity.
    pub fn spawn_tool(
        &mut self,
        id: StableId,
        tenant: TenantId,
        tool: ToolCapability,
    ) -> Result<Entity, SpawnError> {
        ensure_unique_id(&mut self.world, &id)?;
        Ok(self.world.spawn((id, tenant, tool)).id())
    }

    /// Spawns a dynamic discovery source entity.
    pub fn spawn_discovery_source(
        &mut self,
        id: StableId,
        tenant: TenantId,
        kind: impl Into<String>,
    ) -> Result<DiscoveryHandle, SpawnError> {
        ensure_unique_id(&mut self.world, &id)?;
        let entity = self
            .world
            .spawn((
                id,
                tenant,
                DiscoverySource {
                    kind: kind.into(),
                    generation: 0,
                    state: DiscoveryState::Idle,
                },
            ))
            .id();
        Ok(DiscoveryHandle {
            runtime_id: self.handle.runtime_id,
            entity,
        })
    }

    /// Spawns independent many-to-many access metadata for an agent and tool.
    pub fn grant_tool(
        &mut self,
        id: StableId,
        tenant: TenantId,
        grant: ToolGrant,
        agent: AgentHandle,
        tool: Entity,
    ) -> Result<Entity, SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        ensure_unique_id(&mut self.world, &id)?;
        let Some(agent_tenant) = self.world.get::<TenantId>(agent.entity) else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        let Some(tool_tenant) = self.world.get::<TenantId>(tool) else {
            return Err(SpawnError::StaleEntity(tool));
        };
        if agent_tenant != &tenant || tool_tenant != &tenant {
            return Err(SpawnError::TenantMismatch);
        }
        Ok(self
            .world
            .spawn((
                id,
                tenant,
                grant,
                GrantForAgent(agent.entity),
                GrantForTool(tool),
            ))
            .id())
    }

    /// Spawns an addressable store capability entity.
    pub fn spawn_store(
        &mut self,
        id: StableId,
        tenant: TenantId,
        store: StoreCapability,
    ) -> Result<Entity, SpawnError> {
        ensure_unique_id(&mut self.world, &id)?;
        Ok(self.world.spawn((id, tenant, store)).id())
    }

    /// Spawns independent many-to-many access metadata for an agent and store.
    pub fn grant_store(
        &mut self,
        id: StableId,
        tenant: TenantId,
        grant: StoreGrant,
        agent: AgentHandle,
        store: Entity,
    ) -> Result<Entity, SpawnError> {
        if agent.runtime_id != self.handle.runtime_id {
            return Err(SpawnError::ForeignRuntime);
        }
        ensure_unique_id(&mut self.world, &id)?;
        let Some(agent_tenant) = self.world.get::<TenantId>(agent.entity) else {
            return Err(SpawnError::StaleEntity(agent.entity));
        };
        let Some(store_tenant) = self.world.get::<TenantId>(store) else {
            return Err(SpawnError::StaleEntity(store));
        };
        if agent_tenant != &tenant || store_tenant != &tenant {
            return Err(SpawnError::TenantMismatch);
        }
        Ok(self
            .world
            .spawn((
                id,
                tenant,
                grant,
                StoreGrantForAgent(agent.entity),
                StoreGrantForStore(store),
            ))
            .id())
    }

    /// Resolves a pending facade handle after ingestion.
    pub fn resolve_run(&mut self, pending: &PendingRunHandle) -> Option<RunHandle> {
        if pending.runtime_id != self.handle.runtime_id {
            return None;
        }
        let entity = self
            .world
            .resource::<StableIdIndex>()
            .0
            .get(&pending.stable_id)
            .copied()?;
        Some(RunHandle {
            runtime_id: self.handle.runtime_id,
            entity,
        })
    }

    /// Validates an advanced world entity and returns a runtime-scoped run handle.
    pub fn run_handle(&self, entity: Entity) -> Result<RunHandle, ActiveRunSnapshotError> {
        if self.world.get::<RunOf>(entity).is_none()
            || self.world.get::<RunState>(entity).is_none()
            || self.world.get::<RunRecord>(entity).is_none()
        {
            return Err(ActiveRunSnapshotError::NotRun);
        }
        Ok(RunHandle {
            runtime_id: self.handle.runtime_id,
            entity,
        })
    }

    /// Clones the current run state and releases a terminal result for cleanup.
    pub fn observe_run(&mut self, run: RunHandle) -> Result<Option<RunState>, SubmitError> {
        if run.runtime_id != self.handle.runtime_id {
            return Err(SubmitError::ForeignRuntime);
        }
        let Some(state) = self.world.get::<RunState>(run.entity).cloned() else {
            return Ok(None);
        };
        if matches!(
            state,
            RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
        ) && let Some(mut record) = self.world.get_mut::<RunRecord>(run.entity)
        {
            record.observed = true;
        }
        Ok(Some(state))
    }

    pub(crate) fn run_transcript(&self, run: RunHandle) -> Option<Vec<TranscriptEntry>> {
        self.world
            .get::<RunRecord>(run.entity)
            .map(|record| record.transcript.clone())
    }

    pub(crate) fn run_committed_turns(&mut self, run: RunHandle) -> Vec<CommittedTurn> {
        let mut turns = self.world.query::<(&TurnOf, &CommittedTurn)>();
        let mut turns = turns
            .iter(&self.world)
            .filter(|(relation, _)| relation.get() == run.entity)
            .map(|(_, turn)| turn.clone())
            .collect::<Vec<_>>();
        turns.sort_by_key(|turn| turn.index);
        turns
    }

    /// Returns invariant violations found by the reconciliation stage.
    pub fn validate_invariants(&self) -> Result<(), Vec<RuntimeInvariantError>> {
        let violations = &self.world.resource::<InvariantViolations>().0;
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations.clone())
        }
    }

    /// Returns the latest schedule-derived lifecycle metrics.
    pub fn metrics(&self) -> RuntimeMetrics {
        *self.world.resource::<RuntimeMetrics>()
    }

    /// Captures persistent domain configuration using stable identities.
    pub fn snapshot(&mut self) -> Result<DomainSnapshot, PersistenceError> {
        snapshot_domain(&mut self.world)
    }

    /// Reconstructs persistent domain entities and remaps their relationships.
    pub fn restore(
        &mut self,
        snapshot: DomainSnapshot,
    ) -> Result<RestoredEntities, PersistenceError> {
        restore_domain(&mut self.world, snapshot)
    }

    /// Captures a run and its descendants without serializing runtime entities.
    pub fn snapshot_active_run(
        &mut self,
        run: RunHandle,
    ) -> Result<ActiveRunSnapshot, ActiveRunSnapshotError> {
        if run.runtime_id != self.handle.runtime_id {
            return Err(ActiveRunSnapshotError::NotRun);
        }
        snapshot_active_run(&mut self.world, run.entity)
    }

    /// Returns a structured size, topology, state, and operation inventory.
    pub fn snapshot_summary(
        &mut self,
        run: RunHandle,
    ) -> Result<ActiveRunSnapshotSummary, ActiveRunSnapshotError> {
        let snapshot = self.snapshot_active_run(run)?;
        summarize_active_run_snapshot(&snapshot)
    }

    /// Restores a validated active-run graph against the current domain.
    pub fn restore_active_run(
        &mut self,
        snapshot: ActiveRunSnapshot,
    ) -> Result<RestoredRuns, ActiveRunSnapshotError> {
        restore_active_run(&mut self.world, snapshot)
    }

    /// Restores a validated active-run graph with caller-selected resource limits.
    pub fn restore_active_run_with_limits(
        &mut self,
        snapshot: ActiveRunSnapshot,
        limits: ActiveRunSnapshotLimits,
    ) -> Result<RestoredRuns, ActiveRunSnapshotError> {
        restore_active_run_with_limits(&mut self.world, snapshot, limits)
    }
}

/// Entity construction error.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum SpawnError {
    /// Stable ID already exists in this world.
    #[error("stable id `{0}` already exists")]
    DuplicateStableId(String),
    /// A stable relationship target was not present in this world.
    #[error("missing stable reference `{0}`")]
    MissingStableReference(String),
    /// Related entity is stale or missing required components.
    #[error("stale entity {0:?}")]
    StaleEntity(Entity),
    /// Cross-tenant relationship was rejected.
    #[error("tenant scope mismatch")]
    TenantMismatch,
    /// Handle belongs to another authoritative world.
    #[error("handle belongs to a different runtime")]
    ForeignRuntime,
}

/// Validation failure while binding an exact steering responder to a policy.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PolicyResponderRegistrationError {
    /// Policy entity is stale or lacks required policy metadata.
    #[error("stale policy entity {0:?}")]
    StalePolicy(Entity),
    /// The policy does not declare the requested lifecycle capability.
    #[error("policy does not declare the `{0:?}` capability")]
    CapabilityMismatch(PolicyPoint),
    /// The policy already has a responder at this lifecycle point.
    #[error("policy already has a responder for `{0:?}`")]
    ConflictingResponder(PolicyPoint),
    /// The stable binding identity is already used by this policy.
    #[error("policy responder id `{0}` already exists")]
    DuplicateResponderId(String),
}

/// Local driver error.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum DriveError {
    /// Systems continued making immediate progress beyond the guard.
    #[error("runtime exceeded the progress limit of {0} schedule passes")]
    ProgressLimit(usize),
}

fn ensure_unique_id(world: &mut World, id: &StableId) -> Result<(), SpawnError> {
    let mut query = world.query::<&StableId>();
    if query.iter(world).any(|existing| existing == id) {
        return Err(SpawnError::DuplicateStableId(id.as_str().to_owned()));
    }
    Ok(())
}

fn mark_progress(progress: &mut Progress) {
    progress.epoch = progress.epoch.wrapping_add(1);
}

fn control_allows_internal_progress(control: Option<&RunControl>) -> bool {
    matches!(
        control,
        None | Some(RunControl::Running | RunControl::PauseRequested(PauseMode::Drain))
    )
}

fn world_allows_internal_progress(world: &World, run: Entity) -> bool {
    world.get::<WaitingForChildren>(run).is_none()
        && control_allows_internal_progress(world.get::<RunControl>(run))
}

fn advance_runtime_clock(world: &mut World) {
    world.resource_mut::<RuntimeClock>().tick =
        world.resource::<RuntimeClock>().tick.saturating_add(1);
    let limits = *world.resource::<EffectDispatchLimits>();
    let mut query = world.query::<(
        &OperationState,
        Option<&OperationKind>,
        Option<&OperationOf>,
        Option<&DiscoveryEffectInput>,
    )>();
    let mut active_candidates =
        HashMap::<(String, OperationKind), Vec<(Option<Entity>, Option<Entity>)>>::new();
    let mut in_flight_by_run = HashMap::<Entity, usize>::new();
    let mut in_flight_by_agent = HashMap::<Entity, usize>::new();
    let mut in_flight_by_tenant = HashMap::<String, usize>::new();
    for (state, kind, owner, discovery) in query.iter(world) {
        let Some(kind) = kind.copied() else {
            continue;
        };
        let run = owner.map(Relationship::get);
        let tenant = run
            .and_then(|run| world.get::<TenantId>(run))
            .map(|tenant| tenant.as_str())
            .or_else(|| discovery.map(|input| input.tenant.as_str()));
        let Some(tenant) = tenant else {
            continue;
        };
        let dispatchable = match (kind, run) {
            (OperationKind::Discovery, None) => discovery.is_some(),
            (OperationKind::Model, Some(run)) => {
                world.get::<WaitingForChildren>(run).is_none()
                    && matches!(
                        world.get::<RunState>(run),
                        Some(RunState::WaitingModel { .. })
                    )
            }
            (OperationKind::Tool, Some(run)) => {
                world.get::<WaitingForChildren>(run).is_none()
                    && matches!(
                        world.get::<RunState>(run),
                        Some(RunState::WaitingTools { .. })
                    )
            }
            (OperationKind::Store, Some(run)) => {
                world.get::<WaitingForChildren>(run).is_none()
                    && matches!(
                        world.get::<RunState>(run),
                        Some(RunState::WaitingStore { .. })
                    )
            }
            (OperationKind::PolicyApproval, Some(run)) => {
                world.get::<WaitingForChildren>(run).is_none()
            }
            _ => false,
        };
        if matches!(state.phase, OperationPhase::Prepared)
            && dispatchable
            && run
                .is_none_or(|run| matches!(world.get::<RunControl>(run), Some(RunControl::Running)))
        {
            active_candidates
                .entry((tenant.to_owned(), kind))
                .or_default()
                .push((
                    run,
                    run.and_then(|run| world.get::<RunOf>(run).map(Relationship::get)),
                ));
        }
        if matches!(state.phase, OperationPhase::InFlight) {
            *in_flight_by_tenant.entry(tenant.to_owned()).or_default() += 1;
            if let Some(run) = run {
                *in_flight_by_run.entry(run).or_default() += 1;
                if let Some(agent) = world.get::<RunOf>(run).map(Relationship::get) {
                    *in_flight_by_agent.entry(agent).or_default() += 1;
                }
            }
        }
    }
    let mut active_keys = active_candidates
        .into_iter()
        .filter(|((tenant, _), candidates)| {
            in_flight_by_tenant.get(tenant).copied().unwrap_or(0) < limits.per_tenant
                && candidates.iter().any(|(run, agent)| {
                    run.is_none_or(|run| {
                        in_flight_by_run.get(&run).copied().unwrap_or(0) < limits.per_run
                    }) && agent.is_none_or(|agent| {
                        in_flight_by_agent.get(&agent).copied().unwrap_or(0) < limits.per_agent
                    })
                })
        })
        .map(|(key, _)| key)
        .collect::<Vec<_>>();
    active_keys.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| operation_kind_order(left.1).cmp(&operation_kind_order(right.1)))
    });
    let cursor = world.resource::<DispatchBudget>().cursor;
    let allocated_keys = active_keys
        .iter()
        .cycle()
        .skip(cursor)
        .take(limits.per_pass)
        .cloned()
        .collect::<Vec<_>>();
    let mut budget = world.resource_mut::<DispatchBudget>();
    budget.remaining = limits.per_pass;
    budget.allowances.clear();
    budget.in_flight_by_run = in_flight_by_run;
    budget.in_flight_by_agent = in_flight_by_agent;
    budget.in_flight_by_tenant = in_flight_by_tenant;
    if active_keys.is_empty() {
        budget.cursor = 0;
        return;
    }
    for key in allocated_keys {
        *budget.allowances.entry(key).or_default() += 1;
    }
    budget.cursor = (budget.cursor + limits.per_pass) % active_keys.len();
}

const fn operation_kind_order(kind: OperationKind) -> u8 {
    match kind {
        OperationKind::Model => 0,
        OperationKind::Tool => 1,
        OperationKind::Discovery => 2,
        OperationKind::Store => 3,
        OperationKind::PolicyApproval => 4,
    }
}

type IngestAgents<'world, 'state> = Query<
    'world,
    'state,
    (
        &'static TenantId,
        &'static Agent,
        Option<&'static InvalidToolCallBudget>,
        Option<&'static StructuredOutputRetryBudget>,
        Option<&'static mut AgentControl>,
    ),
>;

// Bevy system parameters are independently borrow-checked runtime inputs; grouping
// them would obscure their access pattern without reducing system complexity.
#[allow(clippy::too_many_arguments)]
fn ingest_commands(
    mut commands: Commands,
    ingress: Res<CommandIngress>,
    runtime: Res<RuntimeIdentity>,
    clock: Res<RuntimeClock>,
    identities: Query<&StableId>,
    models: Query<(Entity, &StableId, &TenantId), With<ModelCapability>>,
    mut agents: IngestAgents<'_, '_>,
    mut runs: Query<(
        &StableId,
        &mut RunState,
        &mut RunRecord,
        Option<&mut RunControl>,
    )>,
    mut sources: Query<(&StableId, &TenantId, &mut DiscoverySource)>,
    mut discovery_operations: Query<&mut OperationState, With<DiscoveryEffectInput>>,
    mut progress: ResMut<Progress>,
) {
    let Ok(receiver) = ingress.0.lock() else {
        return;
    };
    let mut accepted_ids = identities.iter().cloned().collect::<HashSet<_>>();
    loop {
        match receiver.try_recv() {
            Ok(RuntimeCommand::SpawnAgent {
                id,
                tenant,
                agent,
                model_id,
                result,
            }) => {
                let spawn_result = if !accepted_ids.insert(id.clone()) {
                    Err(SpawnError::DuplicateStableId(id.as_str().to_owned()))
                } else if let Some((model, _, model_tenant)) = models
                    .iter()
                    .find(|(_, existing, _)| *existing == &model_id)
                {
                    if model_tenant != &tenant {
                        accepted_ids.remove(&id);
                        Err(SpawnError::TenantMismatch)
                    } else {
                        let entity = commands
                            .spawn((id, tenant, agent, AgentControl::default(), UsesModel(model)))
                            .id();
                        mark_progress(&mut progress);
                        Ok(AgentHandle {
                            runtime_id: runtime.0,
                            entity,
                        })
                    }
                } else {
                    accepted_ids.remove(&id);
                    Err(SpawnError::MissingStableReference(
                        model_id.as_str().to_owned(),
                    ))
                };
                let _ = result.try_send(spawn_result);
            }
            Ok(RuntimeCommand::Prompt {
                agent,
                run_id,
                prompt,
                history,
                output_schema,
                max_model_calls,
                conversation,
                subscriber,
                parent,
                run_policies,
            }) => {
                let transcript_entry = if prompt.plain_text {
                    TranscriptEntry::User(prompt.text.clone())
                } else {
                    TranscriptEntry::UserMessage(prompt.message.clone())
                };
                let mut transcript = history
                    .into_iter()
                    .map(TranscriptEntry::Message)
                    .collect::<Vec<_>>();
                transcript.push(transcript_entry);
                let streaming = subscriber.is_some();
                let mut record = RunRecord {
                    transcript,
                    prompt: prompt.message,
                    prompt_text: prompt.text,
                    output_schema,
                    max_model_calls,
                    observed: false,
                    usage: Usage::default(),
                    next_turn: 0,
                    streaming,
                    invalid_tool_call_retries: 0,
                    max_invalid_tool_call_retries: 0,
                    structured_output_retries: 0,
                    max_structured_output_retries: default_structured_output_retries(),
                    conversation,
                    memory_loaded: false,
                    memory_store: None,
                    retrieval_loaded: false,
                    retrieval_store: None,
                    retrieved_documents: Vec::new(),
                    tool_retrieval_loaded: false,
                    tool_retrieval_store: None,
                    retrieved_tool_names: Vec::new(),
                    pending_output: None,
                    pending_tool_results: Vec::new(),
                };
                let (run, run_tenant) = match agents.get_mut(agent) {
                    Ok((tenant, _, invalid_budget, output_retry_budget, control))
                        if control
                            .as_deref()
                            .is_none_or(|control| matches!(control, AgentControl::Running)) =>
                    {
                        record.max_invalid_tool_call_retries =
                            invalid_budget.map_or(0, |budget| budget.max_retries);
                        record.max_structured_output_retries = output_retry_budget
                            .map_or_else(default_structured_output_retries, |budget| {
                                budget.max_retries
                            });
                        let run = commands
                            .spawn((
                                run_id,
                                tenant.clone(),
                                RunOf(agent),
                                RunState::Queued,
                                RunControl::Running,
                                RunPriority::default(),
                                ReadyAt(i128::from(clock.tick)),
                                record,
                            ))
                            .id();
                        (run, Some(tenant.clone()))
                    }
                    Ok((tenant, _, _, _, _)) => (
                        commands
                            .spawn((
                                run_id,
                                tenant.clone(),
                                RunOf(agent),
                                RunState::Failed(CanonicalError::AgentAdmissionDenied),
                                RunControl::Running,
                                RunPriority::default(),
                                ReadyAt(i128::from(clock.tick)),
                                record,
                            ))
                            .id(),
                        None,
                    ),
                    Err(_) => (
                        commands
                            .spawn((
                                run_id,
                                RunState::Failed(CanonicalError::StaleEntity("agent".to_owned())),
                                RunControl::Running,
                                RunPriority::default(),
                                ReadyAt(i128::from(clock.tick)),
                                record,
                            ))
                            .id(),
                        None,
                    ),
                };
                if let Some(tenant) = run_tenant {
                    let mut local_ids = HashSet::new();
                    let conflict = run_policies.iter().find(|spec| {
                        accepted_ids.contains(&spec.id) || !local_ids.insert(spec.id.clone())
                    });
                    if let Some(conflict) = conflict {
                        commands.entity(run).insert(RunState::Failed(
                            CanonicalError::InvalidRunPolicy(conflict.id.as_str().to_owned()),
                        ));
                    } else {
                        for spec in run_policies {
                            accepted_ids.insert(spec.id.clone());
                            commands.spawn((
                                spec.id,
                                tenant.clone(),
                                spec.policy,
                                PolicyStatus::Enabled,
                                PolicyForRun(run),
                            ));
                        }
                    }
                }
                if let Some(subscriber) = subscriber {
                    commands.spawn((
                        SubscriptionOf(run),
                        StreamSink(subscriber),
                        SubscriptionState::Active,
                    ));
                }
                if let Some((parent, ordinal)) = parent {
                    if runs.get_mut(parent).is_ok() {
                        commands.entity(parent).insert(WaitingForChildren);
                        commands
                            .entity(run)
                            .insert((ParentRun(parent), ChildOrdinal(ordinal)));
                    } else {
                        commands
                            .entity(run)
                            .insert(RunState::Failed(CanonicalError::StaleEntity(
                                "parent run".to_owned(),
                            )));
                    }
                }
                mark_progress(&mut progress);
            }
            Ok(RuntimeCommand::Cancel { run }) => {
                if let Ok((_, mut state, _, _)) = runs.get_mut(run)
                    && !matches!(*state, RunState::Completed(_) | RunState::Failed(_))
                {
                    *state = RunState::Cancelled;
                    mark_progress(&mut progress);
                }
            }
            Ok(RuntimeCommand::PauseRun { run, mode }) => {
                if let Ok((_, state, _, control)) = runs.get_mut(run)
                    && !matches!(
                        *state,
                        RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
                    )
                    && let Some(mut control) = control
                {
                    *control = RunControl::PauseRequested(mode);
                    mark_progress(&mut progress);
                }
            }
            Ok(RuntimeCommand::ResumeRun { run }) => {
                if let Ok((_, state, _, control)) = runs.get_mut(run)
                    && !matches!(
                        *state,
                        RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
                    )
                    && let Some(mut control) = control
                    && !matches!(*control, RunControl::Running)
                {
                    *control = RunControl::Running;
                    mark_progress(&mut progress);
                }
            }
            Ok(RuntimeCommand::SetAgentControl { agent, control }) => {
                if let Ok((_, _, _, _, existing)) = agents.get_mut(agent)
                    && let Some(mut existing) = existing
                    && *existing != control
                {
                    *existing = control;
                    mark_progress(&mut progress);
                }
            }
            Ok(RuntimeCommand::RefreshDiscovery { source }) => {
                let Ok((source_id, tenant, mut discovery)) = sources.get_mut(source) else {
                    continue;
                };
                if let DiscoveryState::Refreshing { operation } = discovery.state
                    && let Ok(mut old_operation) = discovery_operations.get_mut(operation)
                    && !matches!(
                        old_operation.phase,
                        OperationPhase::Settled(_)
                            | OperationPhase::Cancelled
                            | OperationPhase::Superseded
                    )
                {
                    old_operation.phase = OperationPhase::Superseded;
                }
                discovery.generation = discovery.generation.saturating_add(1);
                let generation = discovery.generation;
                let operation = commands
                    .spawn((
                        DiscoveryOperationOf(source),
                        OperationKind::Discovery,
                        OperationGeneration(generation),
                        OperationState {
                            phase: OperationPhase::Prepared,
                        },
                        DiscoveryEffectInput {
                            source,
                            source_id: source_id.clone(),
                            kind: discovery.kind.clone(),
                            generation,
                            tenant: tenant.clone(),
                        },
                    ))
                    .id();
                discovery.state = DiscoveryState::Refreshing { operation };
                mark_progress(&mut progress);
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
        }
    }
    let _ = runtime.0;
}

fn ingest_observations(
    ingress: Res<ObservationIngress>,
    mut runs: Query<(&StableId, &mut RunRecord)>,
    mut progress: ResMut<Progress>,
) {
    let Ok(receiver) = ingress.0.lock() else {
        return;
    };
    while let Ok(run_id) = receiver.try_recv() {
        if let Some((_, mut record)) = runs.iter_mut().find(|(stable_id, _)| **stable_id == run_id)
            && !record.observed
        {
            record.observed = true;
            mark_progress(&mut progress);
        }
    }
}

fn ingest_completions(
    ingress: Res<CompletionIngress>,
    mut writer: MessageWriter<EffectIngressMessage>,
    mut progress: ResMut<Progress>,
) {
    let Ok(receiver) = ingress.0.lock() else {
        return;
    };
    while let Ok(message) = receiver.try_recv() {
        writer.write(EffectIngressMessage(message));
        mark_progress(&mut progress);
    }
}

fn reconcile_agent_control(
    agents: Query<(&AgentControl, Option<&AgentRuns>), Changed<AgentControl>>,
    mut runs: Query<(&RunState, &mut RunControl)>,
    mut progress: ResMut<Progress>,
) {
    for (agent_control, agent_runs) in &agents {
        let Some(agent_runs) = agent_runs else {
            continue;
        };
        for run in agent_runs.iter() {
            let Ok((state, mut run_control)) = runs.get_mut(run) else {
                continue;
            };
            if matches!(
                state,
                RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
            ) {
                continue;
            }
            let next = match agent_control {
                AgentControl::Running => RunControl::Running,
                AgentControl::RejectNewRuns => continue,
                AgentControl::PauseExistingRuns(mode) => RunControl::PauseRequested(*mode),
            };
            if *run_control != next {
                *run_control = next;
                mark_progress(&mut progress);
            }
        }
    }
}

fn reconcile_run_control(
    mut commands: Commands,
    mut runs: Query<(
        Entity,
        &mut RunControl,
        Option<&RunOperations>,
        &mut RunState,
    )>,
    mut operations: Query<(
        &mut OperationState,
        &OperationGeneration,
        Option<&EffectDeadline>,
    )>,
    cancellations: Res<CancellationOutbox>,
    mut progress: ResMut<Progress>,
) {
    for (_run, mut control, run_operations, mut run_state) in &mut runs {
        let RunControl::PauseRequested(mode) = *control else {
            continue;
        };
        let operation_entities = run_operations
            .map(|operations| operations.iter().collect::<Vec<_>>())
            .unwrap_or_default();
        match mode {
            PauseMode::FreezeAfterIngress => {
                *control = RunControl::Paused(mode);
                mark_progress(&mut progress);
            }
            PauseMode::Drain => {
                let has_in_flight = operation_entities.iter().any(|entity| {
                    operations
                        .get(*entity)
                        .is_ok_and(|(state, _, _)| matches!(state.phase, OperationPhase::InFlight))
                });
                if !has_in_flight {
                    *control = RunControl::Paused(mode);
                    mark_progress(&mut progress);
                }
            }
            PauseMode::CancelAndSuspend => {
                let generation_exhausted = operation_entities.iter().any(|entity| {
                    operations.get(*entity).is_ok_and(|(state, generation, _)| {
                        matches!(state.phase, OperationPhase::InFlight) && generation.0 == u64::MAX
                    })
                });
                for operation_entity in operation_entities {
                    let Ok((mut operation, generation, deadline)) =
                        operations.get_mut(operation_entity)
                    else {
                        continue;
                    };
                    if !matches!(operation.phase, OperationPhase::InFlight) {
                        continue;
                    }
                    let _ = cancellations.0.try_send(EffectCancellation {
                        operation: operation_entity,
                        generation: generation.0,
                    });
                    if generation_exhausted {
                        operation.phase = OperationPhase::Cancelled;
                    } else if let Some(next_generation) = generation.0.checked_add(1) {
                        commands
                            .entity(operation_entity)
                            .insert(OperationGeneration(next_generation));
                        operation.phase = OperationPhase::Prepared;
                    }
                    if deadline.is_some() {
                        commands.entity(operation_entity).remove::<EffectDeadline>();
                    }
                }
                if generation_exhausted {
                    *run_state = RunState::Cancelled;
                }
                *control = RunControl::Paused(mode);
                mark_progress(&mut progress);
            }
        }
    }
}

fn update_effect_messages(mut messages: ResMut<Messages<EffectIngressMessage>>) {
    messages.update();
}

#[allow(clippy::type_complexity)]
fn reconcile_discovery_operations(
    mut commands: Commands,
    mut sources: Query<(Entity, &StableId, &TenantId, &mut DiscoverySource)>,
    operations: Query<
        (
            Entity,
            &DiscoveryOperationOf,
            &OperationGeneration,
            &OperationState,
        ),
        (With<DiscoveryEffectInput>, Without<DiscoveryApplied>),
    >,
    mut discovered_tools: Query<(
        Entity,
        &DiscoveryKey,
        &DiscoveredFrom,
        &mut ToolCapability,
        Option<&RetiredCapability>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (operation_entity, operation_of, generation, operation) in &operations {
        let OperationPhase::Settled(outcome) = &operation.phase else {
            continue;
        };
        let Ok((source_entity, source_id, tenant, mut source)) =
            sources.get_mut(operation_of.get())
        else {
            commands.entity(operation_entity).insert(DiscoveryApplied);
            continue;
        };
        if !matches!(
            source.state,
            DiscoveryState::Refreshing { operation } if operation == operation_entity
        ) || source.generation != generation.0
        {
            commands.entity(operation_entity).insert(DiscoveryApplied);
            continue;
        }
        let OperationOutcome::Success(EffectOutput::Discovery(output)) = outcome else {
            source.state = match outcome {
                OperationOutcome::Failure(error) => DiscoveryState::Failed(error.clone()),
                OperationOutcome::Success(_) => {
                    DiscoveryState::Failed(CanonicalError::EffectKindMismatch)
                }
            };
            commands.entity(operation_entity).insert(DiscoveryApplied);
            mark_progress(&mut progress);
            continue;
        };

        let mut definitions = output.tools.clone();
        definitions.sort_by(|left, right| {
            left.order
                .cmp(&right.order)
                .then_with(|| left.name.cmp(&right.name))
                .then_with(|| left.key.cmp(&right.key))
        });
        let mut keys = HashSet::new();
        if let Some(duplicate) = definitions
            .iter()
            .find(|definition| !keys.insert(definition.key.clone()))
        {
            source.state = DiscoveryState::Failed(CanonicalError::InvalidDiscovery(format!(
                "duplicate source key `{}`",
                duplicate.key
            )));
            commands.entity(operation_entity).insert(DiscoveryApplied);
            mark_progress(&mut progress);
            continue;
        }

        let mut desired = definitions
            .into_iter()
            .map(|definition| (definition.key.clone(), definition))
            .collect::<HashMap<_, _>>();
        for (entity, key, discovered_from, mut capability, retired) in &mut discovered_tools {
            if discovered_from.get() != source_entity || retired.is_some() {
                continue;
            }
            let unchanged = desired.get(&key.0).is_some_and(|definition| {
                capability.name == definition.name
                    && capability.description == definition.description
                    && capability.order == definition.order
                    && capability.revision == definition.revision
            });
            if unchanged {
                desired.remove(&key.0);
                continue;
            }
            capability.retired = true;
            commands.entity(entity).insert(RetiredCapability {
                generation: source.generation,
            });
        }
        let mut remaining = desired.into_values().collect::<Vec<_>>();
        remaining.sort_by(|left, right| {
            left.order
                .cmp(&right.order)
                .then_with(|| left.name.cmp(&right.name))
                .then_with(|| left.key.cmp(&right.key))
        });
        for definition in remaining {
            let stable_id = StableId::generated(format!(
                "discovery/{}/{}/{}",
                source_id.as_str(),
                source.generation,
                definition.key
            ));
            commands.spawn((
                stable_id,
                tenant.clone(),
                DiscoveryKey(definition.key),
                DiscoveredFrom(source_entity),
                ToolCapability {
                    name: definition.name,
                    description: definition.description,
                    parameters: definition.parameters,
                    order: definition.order,
                    revision: definition.revision,
                    retired: false,
                },
            ));
        }
        source.state = DiscoveryState::Idle;
        commands.entity(operation_entity).insert(DiscoveryApplied);
        mark_progress(&mut progress);
    }
}

fn reconcile_stable_ids(world: &mut World) {
    let mut entities = world.query::<(Entity, &StableId)>();
    let rows = entities
        .iter(world)
        .map(|(entity, stable_id)| (entity, stable_id.clone()))
        .collect::<Vec<_>>();
    let mut rebuilt = HashMap::new();
    let mut found = Vec::new();
    let mut duplicates = HashSet::new();
    for (entity, stable_id) in rows {
        if rebuilt.insert(stable_id.clone(), entity).is_some() {
            duplicates.insert(stable_id.as_str().to_owned());
        }
    }
    let mut duplicates: Vec<_> = duplicates.into_iter().collect();
    duplicates.sort();
    found.extend(
        duplicates
            .into_iter()
            .map(RuntimeInvariantError::DuplicateStableId),
    );

    macro_rules! validate_relation {
        ($relation:ty, $target:ty, $label:literal) => {{
            let mut query = world.query::<(Entity, &TenantId, &$relation)>();
            let relations = query
                .iter(world)
                .map(|(entity, tenant, relation)| (entity, tenant.clone(), relation.get()))
                .collect::<Vec<_>>();
            for (entity, tenant, target) in relations {
                match (world.get::<TenantId>(target), world.get::<$target>(target)) {
                    (Some(target_tenant), Some(_)) if target_tenant == &tenant => {}
                    (Some(_), Some(_)) => {
                        found.push(RuntimeInvariantError::RelationshipTenantMismatch {
                            entity,
                            relationship: $label,
                        })
                    }
                    _ => found.push(RuntimeInvariantError::InvalidRelationship {
                        entity,
                        relationship: $label,
                        target,
                    }),
                }
            }
        }};
    }

    validate_relation!(UsesModel, ModelCapability, "UsesModel");
    validate_relation!(PolicyFor, Agent, "PolicyFor");
    validate_relation!(PolicyForRun, RunState, "PolicyForRun");
    validate_relation!(GrantForAgent, Agent, "GrantForAgent");
    validate_relation!(GrantForTool, ToolCapability, "GrantForTool");
    validate_relation!(StoreGrantForAgent, Agent, "StoreGrantForAgent");
    validate_relation!(StoreGrantForStore, StoreCapability, "StoreGrantForStore");
    validate_relation!(DiscoveredFrom, DiscoverySource, "DiscoveredFrom");
    validate_relation!(RunOf, Agent, "RunOf");
    validate_relation!(ParentRun, RunState, "ParentRun");
    validate_relation!(TurnOf, RunState, "TurnOf");
    validate_relation!(OperationOf, RunState, "OperationOf");
    validate_relation!(BatchOf, RunState, "BatchOf");
    validate_relation!(SubscriptionOf, RunState, "SubscriptionOf");
    validate_relation!(
        EvaluationOfOperation,
        OperationState,
        "EvaluationOfOperation"
    );

    world.resource_mut::<StableIdIndex>().0 = rebuilt;
    world.resource_mut::<InvariantViolations>().0 = found;
}

#[allow(clippy::type_complexity)]
fn prepare_store_operations(
    mut commands: Commands,
    mut runs: Query<
        (
            Entity,
            &TenantId,
            &RunOf,
            &RunControl,
            &mut RunRecord,
            &mut RunState,
        ),
        Without<WaitingForChildren>,
    >,
    agents: Query<(
        Option<&AgentStoreGrants>,
        Option<&RetrievalRequirement>,
        Option<&ToolRetrievalRequirement>,
    )>,
    grants: Query<(&StableId, &TenantId, &StoreGrant, &StoreGrantForStore)>,
    stores: Query<(&StableId, &TenantId, &StoreCapability)>,
    mut progress: ResMut<Progress>,
) {
    for (run_entity, run_tenant, run_of, control, mut record, mut run_state) in &mut runs {
        if !matches!(*control, RunControl::Running) || !matches!(*run_state, RunState::Queued) {
            continue;
        }
        let Ok((agent_grants, retrieval_requirement, tool_retrieval_requirement)) =
            agents.get(run_of.get())
        else {
            continue;
        };
        let select_store = |kind: &str| {
            let mut candidates = agent_grants
                .into_iter()
                .flat_map(|agent_store_grants| agent_store_grants.iter())
                .filter_map(|grant_entity| {
                    let (grant_id, grant_tenant, grant, store_relation) =
                        grants.get(grant_entity).ok()?;
                    if !grant.enabled || grant_tenant != run_tenant {
                        return None;
                    }
                    let (store_id, store_tenant, store) = stores.get(store_relation.get()).ok()?;
                    if store_tenant != run_tenant || store.retired || store.kind != kind {
                        return None;
                    }
                    Some((
                        grant.order,
                        grant_id.clone(),
                        StoreDecision {
                            store_entity: store_relation.get(),
                            store_id: store_id.clone(),
                            revision: store.revision,
                            kind: store.kind.clone(),
                            tenant: run_tenant.clone(),
                        },
                    ))
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|(left_order, left_id, _), (right_order, right_id, _)| {
                left_order
                    .cmp(right_order)
                    .then_with(|| left_id.cmp(right_id))
            });
            candidates
                .into_iter()
                .next()
                .map(|(_, _, decision)| decision)
        };

        if !record.memory_loaded {
            if let Some(conversation) = record.conversation.clone()
                && let Some(decision) = select_store("conversation-memory")
            {
                let operation = commands
                    .spawn((
                        OperationOf(run_entity),
                        OperationKind::Store,
                        OperationGeneration(0),
                        OperationState {
                            phase: OperationPhase::Prepared,
                        },
                        StoreEffectInput {
                            decision: decision.clone(),
                            operation: StoreOperation::LoadConversation { conversation },
                        },
                    ))
                    .id();
                record.memory_store = Some(decision);
                *run_state = RunState::WaitingStore { operation };
                mark_progress(&mut progress);
                continue;
            }
            record.memory_loaded = true;
            mark_progress(&mut progress);
            continue;
        }

        if !record.retrieval_loaded {
            if let Some(requirement) = retrieval_requirement
                && let Some(decision) = select_store("vector-search")
            {
                let operation = commands
                    .spawn((
                        OperationOf(run_entity),
                        OperationKind::Store,
                        OperationGeneration(0),
                        OperationState {
                            phase: OperationPhase::Prepared,
                        },
                        StoreEffectInput {
                            decision: decision.clone(),
                            operation: StoreOperation::Retrieve {
                                query: record.prompt_text.clone(),
                                limit: requirement.limit,
                            },
                        },
                    ))
                    .id();
                record.retrieval_store = Some(decision);
                *run_state = RunState::WaitingStore { operation };
                mark_progress(&mut progress);
                continue;
            }
            record.retrieval_loaded = true;
            mark_progress(&mut progress);
        }

        if !record.tool_retrieval_loaded {
            if let Some(requirement) = tool_retrieval_requirement
                && let Some(decision) = select_store("tool-vector-search")
            {
                let operation = commands
                    .spawn((
                        OperationOf(run_entity),
                        OperationKind::Store,
                        OperationGeneration(0),
                        OperationState {
                            phase: OperationPhase::Prepared,
                        },
                        StoreEffectInput {
                            decision: decision.clone(),
                            operation: StoreOperation::Retrieve {
                                query: record.prompt_text.clone(),
                                limit: requirement.limit,
                            },
                        },
                    ))
                    .id();
                record.tool_retrieval_store = Some(decision);
                *run_state = RunState::WaitingStore { operation };
                mark_progress(&mut progress);
                continue;
            }
            record.tool_retrieval_loaded = true;
            mark_progress(&mut progress);
        }
    }
}

#[allow(clippy::type_complexity)]
fn prepare_model_operations(
    mut commands: Commands,
    mut runs: Query<
        (
            Entity,
            &TenantId,
            &RunOf,
            &RunControl,
            &mut RunRecord,
            &mut RunState,
        ),
        Without<WaitingForChildren>,
    >,
    agents: Query<(
        &TenantId,
        &Agent,
        &UsesModel,
        Option<&AgentToolGrants>,
        Option<&OutputRequirement>,
        Option<&ToolRetrievalRequirement>,
    )>,
    models: Query<(&StableId, &TenantId, &ModelCapability)>,
    grants: Query<(&StableId, &TenantId, &ToolGrant, &GrantForTool)>,
    tools: Query<(&StableId, &TenantId, &ToolCapability)>,
    mut progress: ResMut<Progress>,
) {
    for (run_entity, run_tenant, run_of, control, mut record, mut run_state) in &mut runs {
        if !matches!(*control, RunControl::Running)
            || !matches!(*run_state, RunState::Queued)
            || !record.memory_loaded
            || !record.retrieval_loaded
            || !record.tool_retrieval_loaded
        {
            continue;
        }
        let Ok((
            agent_tenant,
            agent,
            model_relation,
            agent_grants,
            output_requirement,
            tool_retrieval_requirement,
        )) = agents.get(run_of.get())
        else {
            *run_state = RunState::Failed(CanonicalError::StaleEntity("agent".to_owned()));
            mark_progress(&mut progress);
            continue;
        };
        let Ok((model_id, model_tenant, model)) = models.get(model_relation.get()) else {
            *run_state = RunState::Failed(CanonicalError::StaleEntity("model".to_owned()));
            mark_progress(&mut progress);
            continue;
        };
        if agent_tenant != run_tenant || model_tenant != run_tenant {
            *run_state = RunState::Failed(CanonicalError::TenantMismatch);
            mark_progress(&mut progress);
            continue;
        }
        if model.retired {
            *run_state = RunState::Failed(CanonicalError::RetiredCapability);
            mark_progress(&mut progress);
            continue;
        }
        let model_call_limit = record.max_model_calls.unwrap_or(agent.max_model_calls);
        if record.next_turn >= model_call_limit {
            *run_state = RunState::Failed(CanonicalError::ModelCallBudget {
                limit: model_call_limit,
            });
            mark_progress(&mut progress);
            continue;
        }
        let decision = ModelDecision {
            model_entity: model_relation.get(),
            model_id: model_id.clone(),
            revision: model.revision,
            provider: model.provider.clone(),
            model: model.model.clone(),
            tenant: run_tenant.clone(),
        };
        let mut candidates = Vec::new();
        if let Some(agent_grants) = agent_grants {
            for grant_entity in agent_grants.iter() {
                let Ok((grant_id, grant_tenant, grant, tool_relation)) = grants.get(grant_entity)
                else {
                    continue;
                };
                if !grant.enabled || grant_tenant != run_tenant {
                    continue;
                }
                let Ok((tool_id, tool_tenant, tool)) = tools.get(tool_relation.get()) else {
                    continue;
                };
                if tool_tenant != run_tenant || tool.retired {
                    continue;
                }
                let retrieved_candidate = tool_retrieval_requirement.is_some_and(|requirement| {
                    requirement.candidates.iter().any(|name| name == &tool.name)
                });
                if retrieved_candidate
                    && !record
                        .retrieved_tool_names
                        .iter()
                        .any(|name| name == &tool.name)
                {
                    continue;
                }
                candidates.push((
                    !retrieved_candidate,
                    grant.order,
                    tool.order,
                    tool.name.clone(),
                    grant_id.clone(),
                    ToolDecision {
                        tool_entity: tool_relation.get(),
                        tool_id: tool_id.clone(),
                        revision: tool.revision,
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                        order: 0,
                    },
                ));
            }
        }
        candidates.sort_by(
            |(left_static, left_grant, left_tool, left_name, left_id, _),
             (right_static, right_grant, right_tool, right_name, right_id, _)| {
                left_static
                    .cmp(right_static)
                    .then_with(|| left_grant.cmp(right_grant))
                    .then_with(|| left_tool.cmp(right_tool))
                    .then_with(|| left_name.cmp(right_name))
                    .then_with(|| left_id.cmp(right_id))
            },
        );
        let mut names = HashSet::new();
        let tools = candidates
            .into_iter()
            .filter_map(|(_, _, _, name, _, mut decision)| {
                if !names.insert(name) {
                    return None;
                }
                decision.order = u32::try_from(names.len() - 1).unwrap_or(u32::MAX);
                Some(decision)
            })
            .collect::<Vec<_>>();
        let mut documents = agent.documents.clone();
        documents.extend(record.retrieved_documents.iter().cloned());
        let tool_results = std::mem::take(&mut record.pending_tool_results);
        let operation = commands
            .spawn((
                OperationOf(run_entity),
                OperationGeneration(0),
                OperationState {
                    phase: OperationPhase::Prepared,
                },
                OperationKind::Model,
                ModelStreamState {
                    streaming: record.streaming,
                    ..ModelStreamState::default()
                },
                decision,
                PendingModelRequest(ModelEffectInput {
                    decision: ModelDecision {
                        model_entity: model_relation.get(),
                        model_id: model_id.clone(),
                        revision: model.revision,
                        provider: model.provider.clone(),
                        model: model.model.clone(),
                        tenant: run_tenant.clone(),
                    },
                    instructions: agent.instructions.clone(),
                    prompt: record.prompt.clone(),
                    history: record.transcript.clone(),
                    tools,
                    tool_results,
                    output_schema: record
                        .output_schema
                        .clone()
                        .or_else(|| output_requirement.map(|value| value.schema.clone())),
                    terminal_tool: agent.terminal_tool.clone(),
                    documents,
                    temperature_bits: agent.temperature_bits,
                    max_tokens: agent.max_tokens,
                    tool_choice: agent.tool_choice.clone(),
                    additional_params: agent.additional_params.clone(),
                }),
            ))
            .id();
        if tool_retrieval_requirement.is_some() {
            record.tool_retrieval_loaded = false;
            record.tool_retrieval_store = None;
            record.retrieved_tool_names.clear();
        }
        *run_state = RunState::WaitingModel { operation };
        mark_progress(&mut progress);
    }
}

#[allow(clippy::type_complexity)]
fn initialize_request_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &PendingModelRequest, &OperationState),
        Without<RequestPolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunPolicies>, Option<&RunControl>), Without<WaitingForChildren>>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(
        Entity,
        &StableId,
        &PolicyMeta,
        &PolicyCapabilities,
        Option<&PolicyStatus>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, pending, state) in &operations {
        if !matches!(state.phase, OperationPhase::Prepared) {
            continue;
        }
        let Ok((run_of, run_policies, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = policy_entities(agents.get(run_of.get()).ok().flatten(), run_policies)
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, _, capabilities, status)| {
                accepts_new_policy_evaluations(*status)
                    && capabilities.contains(PolicyPoint::Request)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _, _), (_, right_id, right, _, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
                order: policy.order,
                point: PolicyPoint::Request,
            })
            .collect::<Vec<_>>();
        commands.entity(operation).insert(RequestPolicyInitialized);
        if snapshot.is_empty() {
            commands
                .entity(operation)
                .insert((pending.0.clone(), AcceptedPolicies::default()))
                .remove::<PendingModelRequest>();
        } else {
            commands
                .entity(operation)
                .insert(AcceptedPolicies(snapshot.clone()));
            commands.spawn((
                EvaluationOfOperation(operation),
                RequestPolicyEvaluation {
                    run: operation_of.get(),
                    policies: snapshot,
                    cursor: 0,
                    effective: pending.0.clone(),
                    accumulated: RequestPatch::default(),
                    phase: RequestPolicyEvaluationPhase::Evaluating,
                },
            ));
        }
        mark_progress(&mut progress);
    }
}

fn bind_builtin_policy_responder(
    event: On<Add, Policy>,
    policies: Query<(&StableId, &TenantId, &Policy)>,
    mut commands: Commands,
) {
    let Ok((policy_id, tenant, policy)) = policies.get(event.entity) else {
        return;
    };
    let point = policy.rule.point();
    commands.entity(event.entity).insert((
        PolicyMeta {
            order: policy.order,
            revision: policy.revision,
        },
        PolicyCapabilities::new([point]),
    ));
    match &policy.rule {
        PolicyRule::Allow => {
            commands.entity(event.entity).insert(AllowRequestPolicy);
        }
        PolicyRule::DenyPromptContains(needle) => {
            commands
                .entity(event.entity)
                .insert(DenyPromptContainsPolicy(needle.clone()));
        }
        PolicyRule::PatchRequest(patch) => {
            commands
                .entity(event.entity)
                .insert(RequestPatchPolicy(patch.clone()));
        }
        PolicyRule::RewriteToolArguments { tool, arguments } => {
            commands
                .entity(event.entity)
                .insert(RewriteToolArgumentsPolicy {
                    tool: tool.clone(),
                    arguments: arguments.clone(),
                });
        }
        PolicyRule::SkipToolCall { tool, reason } => {
            commands.entity(event.entity).insert(SkipToolCallPolicy {
                tool: tool.clone(),
                reason: reason.clone(),
            });
        }
        PolicyRule::RepairInvalidTool { from, to } => {
            commands
                .entity(event.entity)
                .insert(RepairInvalidToolPolicy {
                    from: from.clone(),
                    to: to.clone(),
                });
        }
        PolicyRule::RetryInvalidTool { tool, feedback } => {
            commands
                .entity(event.entity)
                .insert(RetryInvalidToolPolicy {
                    tool: tool.clone(),
                    feedback: feedback.clone(),
                });
        }
        PolicyRule::SkipInvalidTool { tool, reason } => {
            commands.entity(event.entity).insert(SkipInvalidToolPolicy {
                tool: tool.clone(),
                reason: reason.clone(),
            });
        }
        PolicyRule::RewriteToolResult { tool, presentation } => {
            commands
                .entity(event.entity)
                .insert(RewriteToolResultPolicy {
                    tool: tool.clone(),
                    presentation: presentation.clone(),
                });
        }
        PolicyRule::StopToolResult { tool, reason } => {
            commands.entity(event.entity).insert(StopToolResultPolicy {
                tool: tool.clone(),
                reason: reason.clone(),
            });
        }
        PolicyRule::RewriteCompletionText { text } => {
            commands
                .entity(event.entity)
                .insert(RewriteCompletionTextPolicy(text.clone()));
        }
        PolicyRule::StopCompletionContains { needle, reason } => {
            commands
                .entity(event.entity)
                .insert(StopCompletionContainsPolicy {
                    needle: needle.clone(),
                    reason: reason.clone(),
                });
        }
        PolicyRule::StopTextDeltaContains { needle, reason } => {
            commands
                .entity(event.entity)
                .insert(StopTextDeltaContainsPolicy {
                    needle: needle.clone(),
                    reason: reason.clone(),
                });
        }
        PolicyRule::StopToolCallDeltaContains { needle, reason } => {
            commands
                .entity(event.entity)
                .insert(StopToolCallDeltaContainsPolicy {
                    needle: needle.clone(),
                    reason: reason.clone(),
                });
        }
        PolicyRule::RequireApproval { point, prompt } => {
            commands.entity(event.entity).insert(RequireApprovalPolicy {
                point: *point,
                prompt: prompt.clone(),
            });
        }
        PolicyRule::Custom(point) => {
            commands.entity(event.entity).insert(CustomPolicy(*point));
        }
    }
    let responder = match &policy.rule {
        PolicyRule::Custom(_) => None,
        _ if point == PolicyPoint::Request => Some(
            commands
                .register_system(apply_builtin_request_policy)
                .entity(),
        ),
        _ if point == PolicyPoint::ToolCall => Some(
            commands
                .register_system(apply_builtin_tool_call_policy)
                .entity(),
        ),
        _ if point == PolicyPoint::InvalidToolCall => Some(
            commands
                .register_system(apply_builtin_invalid_tool_call_policy)
                .entity(),
        ),
        _ if point == PolicyPoint::ToolResult => Some(
            commands
                .register_system(apply_builtin_tool_result_policy)
                .entity(),
        ),
        _ if point == PolicyPoint::CompletionResponse => Some(
            commands
                .register_system(apply_builtin_completion_response_policy)
                .entity(),
        ),
        _ if point == PolicyPoint::TextDelta => Some(
            commands
                .register_system(apply_builtin_text_delta_policy)
                .entity(),
        ),
        _ if point == PolicyPoint::ToolCallDelta => Some(
            commands
                .register_system(apply_builtin_tool_call_delta_policy)
                .entity(),
        ),
        _ => None,
    };
    if let Some(responder) = responder {
        commands.spawn((
            PolicyResponderFor(event.entity),
            ResponderPoint(point),
            PolicyResponderId::generated(format!(
                "builtin:{}:{}",
                policy_id.as_str(),
                point.as_str()
            )),
            tenant.clone(),
            RegisteredPolicyResponder(responder),
        ));
    }
}

fn dematerialize_removed_policy(event: On<Remove, Policy>, mut commands: Commands) {
    commands.entity(event.entity).remove::<PolicyMeta>();
}

fn unbind_policy_responders(
    event: On<Remove, PolicyMeta>,
    bindings: Query<&PolicyResponders>,
    responders: Query<&RegisteredPolicyResponder>,
    mut commands: Commands,
) {
    cleanup_policy_responders(event.entity, &bindings, &responders, &mut commands);
}

fn cleanup_policy_responders(
    policy: Entity,
    bindings: &Query<&PolicyResponders>,
    responders: &Query<&RegisteredPolicyResponder>,
    commands: &mut Commands,
) {
    let Ok(bindings) = bindings.get(policy) else {
        return;
    };
    for binding in bindings.iter() {
        if let Ok(responder) = responders.get(binding) {
            commands.unregister_system(SystemId::<(), ()>::from_entity(responder.0));
            commands.entity(responder.0).despawn();
        }
        commands.entity(binding).despawn();
    }
}

#[allow(clippy::type_complexity)]
fn apply_builtin_request_policy(
    In(invocation): In<RequestPolicyInvocation>,
    policies: Query<(
        Option<&AllowRequestPolicy>,
        Option<&DenyPromptContainsPolicy>,
        Option<&RequestPatchPolicy>,
        Option<&RequireApprovalPolicy>,
    )>,
) -> Option<RequestPolicyDecision> {
    let Ok((allow, deny, patch, approval)) = policies.get(invocation.policy) else {
        return None;
    };
    Some(match (allow, deny, patch, approval) {
        (Some(_), None, None, None) => RequestPolicyDecision::Continue,
        (None, None, Some(patch), None) => RequestPolicyDecision::Patch(patch.0.clone()),
        (None, Some(DenyPromptContainsPolicy(needle)), None, None) => {
            let denied =
                serde_json::from_value::<CompletionMessage>(invocation.request.prompt.clone())
                    .ok()
                    .and_then(|message| message.rag_text())
                    .is_some_and(|text| text.contains(needle));
            if denied {
                RequestPolicyDecision::Stop(format!("prompt contains `{needle}`"))
            } else {
                RequestPolicyDecision::Continue
            }
        }
        (None, None, None, Some(approval)) if approval.point == PolicyPoint::Request => {
            RequestPolicyDecision::AwaitApproval(approval.prompt.clone())
        }
        _ => return None,
    })
}

fn apply_builtin_tool_call_policy(
    In(invocation): In<ToolCallPolicyInvocation>,
    policies: Query<(
        Option<&RewriteToolArgumentsPolicy>,
        Option<&SkipToolCallPolicy>,
        Option<&RequireApprovalPolicy>,
    )>,
) -> Option<ToolCallPolicyDecision> {
    let Ok((rewrite, skip, approval)) = policies.get(invocation.policy) else {
        return None;
    };
    Some(match (rewrite, skip, approval) {
        (Some(rewrite), None, None)
            if rewrite
                .tool
                .as_ref()
                .is_none_or(|name| name == &invocation.call.decision.name) =>
        {
            ToolCallPolicyDecision::Rewrite(rewrite.arguments.clone())
        }
        (None, Some(skip), None)
            if skip
                .tool
                .as_ref()
                .is_none_or(|name| name == &invocation.call.decision.name) =>
        {
            ToolCallPolicyDecision::Skip(skip.reason.clone())
        }
        (Some(_), None, None) | (None, Some(_), None) => ToolCallPolicyDecision::Run,
        (None, None, Some(approval)) if approval.point == PolicyPoint::ToolCall => {
            ToolCallPolicyDecision::AwaitApproval(approval.prompt.clone())
        }
        _ => return None,
    })
}

#[allow(clippy::type_complexity)]
fn apply_builtin_invalid_tool_call_policy(
    In(invocation): In<InvalidToolCallPolicyInvocation>,
    policies: Query<(
        Option<&RepairInvalidToolPolicy>,
        Option<&RetryInvalidToolPolicy>,
        Option<&SkipInvalidToolPolicy>,
        Option<&RequireApprovalPolicy>,
    )>,
) -> Option<InvalidToolCallPolicyDecision> {
    let Ok((repair, retry, skip, approval)) = policies.get(invocation.policy) else {
        return None;
    };
    let emitted = &invocation.invalid.call.name;
    Some(match (repair, retry, skip, approval) {
        (Some(repair), None, None, None)
            if repair.from.as_ref().is_none_or(|name| name == emitted) =>
        {
            InvalidToolCallPolicyDecision::Repair(repair.to.clone())
        }
        (None, Some(retry), None, None)
            if retry.tool.as_ref().is_none_or(|name| name == emitted) =>
        {
            InvalidToolCallPolicyDecision::Retry(retry.feedback.clone())
        }
        (None, None, Some(skip), None) if skip.tool.as_ref().is_none_or(|name| name == emitted) => {
            InvalidToolCallPolicyDecision::Skip(skip.reason.clone())
        }
        (Some(_), None, None, None) | (None, Some(_), None, None) | (None, None, Some(_), None) => {
            InvalidToolCallPolicyDecision::Continue
        }
        (None, None, None, Some(approval)) if approval.point == PolicyPoint::InvalidToolCall => {
            InvalidToolCallPolicyDecision::AwaitApproval(approval.prompt.clone())
        }
        _ => return None,
    })
}

fn apply_builtin_tool_result_policy(
    In(invocation): In<ToolResultPolicyInvocation>,
    policies: Query<(
        Option<&RewriteToolResultPolicy>,
        Option<&StopToolResultPolicy>,
        Option<&RequireApprovalPolicy>,
    )>,
) -> Option<ToolResultPolicyDecision> {
    let Ok((rewrite, stop, approval)) = policies.get(invocation.policy) else {
        return None;
    };
    let tool_name = &invocation.input.decision.name;
    Some(match (rewrite, stop, approval) {
        (Some(rewrite), None, None)
            if rewrite.tool.as_ref().is_none_or(|name| name == tool_name) =>
        {
            ToolResultPolicyDecision::Rewrite(rewrite.presentation.clone())
        }
        (None, Some(stop), None) if stop.tool.as_ref().is_none_or(|name| name == tool_name) => {
            ToolResultPolicyDecision::Stop(stop.reason.clone())
        }
        (Some(_), None, None) | (None, Some(_), None) => ToolResultPolicyDecision::Keep,
        (None, None, Some(approval)) if approval.point == PolicyPoint::ToolResult => {
            ToolResultPolicyDecision::AwaitApproval(approval.prompt.clone())
        }
        _ => return None,
    })
}

fn apply_builtin_completion_response_policy(
    In(invocation): In<CompletionResponsePolicyInvocation>,
    policies: Query<(
        Option<&RewriteCompletionTextPolicy>,
        Option<&StopCompletionContainsPolicy>,
        Option<&RequireApprovalPolicy>,
    )>,
) -> Option<CompletionResponsePolicyDecision> {
    let Ok((rewrite, stop, approval)) = policies.get(invocation.policy) else {
        return None;
    };
    Some(match (rewrite, stop, approval) {
        (Some(rewrite), None, None) => {
            CompletionResponsePolicyDecision::RewriteText(rewrite.0.clone())
        }
        (None, Some(stop), None) if invocation.response.text.contains(&stop.needle) => {
            CompletionResponsePolicyDecision::Stop(stop.reason.clone())
        }
        (None, Some(_), None) => CompletionResponsePolicyDecision::Continue,
        (None, None, Some(approval)) if approval.point == PolicyPoint::CompletionResponse => {
            CompletionResponsePolicyDecision::AwaitApproval(approval.prompt.clone())
        }
        _ => return None,
    })
}

fn apply_builtin_text_delta_policy(
    In(invocation): In<TextDeltaPolicyInvocation>,
    policies: Query<(
        Option<&StopTextDeltaContainsPolicy>,
        Option<&RequireApprovalPolicy>,
    )>,
) -> Option<TextDeltaPolicyDecision> {
    let Ok((stop, approval)) = policies.get(invocation.policy) else {
        return None;
    };
    Some(match (stop, approval) {
        (Some(stop), None) if invocation.delta.contains(&stop.needle) => {
            TextDeltaPolicyDecision::Stop(stop.reason.clone())
        }
        (Some(_), None) => TextDeltaPolicyDecision::Continue,
        (None, Some(approval)) if approval.point == PolicyPoint::TextDelta => {
            TextDeltaPolicyDecision::AwaitApproval(approval.prompt.clone())
        }
        _ => return None,
    })
}

fn apply_builtin_tool_call_delta_policy(
    In(invocation): In<ToolCallDeltaPolicyInvocation>,
    policies: Query<(
        Option<&StopToolCallDeltaContainsPolicy>,
        Option<&RequireApprovalPolicy>,
    )>,
) -> Option<ToolCallDeltaPolicyDecision> {
    let Ok((stop, approval)) = policies.get(invocation.policy) else {
        return None;
    };
    let fragment = match &invocation.content {
        ToolCallDeltaContent::Name(name) | ToolCallDeltaContent::Delta(name) => name,
    };
    Some(match (stop, approval) {
        (Some(stop), None) if fragment.contains(&stop.needle) => {
            ToolCallDeltaPolicyDecision::Stop(stop.reason.clone())
        }
        (Some(_), None) => ToolCallDeltaPolicyDecision::Continue,
        (None, Some(approval)) if approval.point == PolicyPoint::ToolCallDelta => {
            ToolCallDeltaPolicyDecision::AwaitApproval(approval.prompt.clone())
        }
        _ => return None,
    })
}

fn apply_request_patch(input: &mut ModelEffectInput, patch: RequestPatch) {
    if let Some(instructions) = patch.instructions {
        input.instructions = instructions;
    }
    if let Some(temperature_bits) = patch.temperature_bits {
        input.temperature_bits = Some(temperature_bits);
    }
    if let Some(max_tokens) = patch.max_tokens {
        input.max_tokens = Some(max_tokens);
    }
    if let Some(tool_choice) = patch.tool_choice {
        input.tool_choice = Some(tool_choice);
    }
    if let Some(active_tools) = patch.active_tools {
        let active_tools = active_tools.into_iter().collect::<HashSet<_>>();
        input.tools.retain(|tool| active_tools.contains(&tool.name));
    }
    if let Some(patch_params) = patch.additional_params {
        input.additional_params = match (input.additional_params.take(), patch_params) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(patch)) => {
                base.extend(patch);
                Some(serde_json::Value::Object(base))
            }
            (_, replacement) => Some(replacement),
        };
    }
    input.documents.extend(patch.extra_context);
    if let Some(mut history) = patch.history {
        history.push(TranscriptEntry::Message(input.prompt.clone()));
        input.history = history;
    }
}

fn policy_termination_error(
    world: &World,
    run: Entity,
    operation: Entity,
    policy_id: StableId,
    revision: u64,
    point: PolicyPoint,
    reason: String,
) -> CanonicalError {
    let entity_id = |entity: Entity, kind: &str| {
        world
            .get::<StableId>(entity)
            .cloned()
            .unwrap_or_else(|| StableId::generated(format!("{kind}-{}", entity.to_bits())))
    };
    CanonicalError::PolicyTerminated {
        termination: Box::new(PolicyTermination {
            policy_id,
            revision,
            point,
            reason,
            run_id: entity_id(run, "run"),
            operation_id: entity_id(operation, "operation"),
            history: world
                .get::<RunRecord>(run)
                .map(|record| record.transcript.clone())
                .unwrap_or_default(),
        }),
    }
}

fn reduce_request_patch(
    input: &mut ModelEffectInput,
    accumulated: &mut RequestPatch,
    patch: RequestPatch,
    policy: &StableId,
) {
    macro_rules! replace_last_writer {
        ($field:ident) => {
            if let Some(value) = patch.$field.clone() {
                if accumulated.$field.is_some() {
                    tracing::warn!(
                        policy = policy.as_str(),
                        field = stringify!($field),
                        "multiple request policies set a last-writer-wins field"
                    );
                }
                accumulated.$field = Some(value);
            }
        };
    }

    replace_last_writer!(instructions);
    replace_last_writer!(temperature_bits);
    replace_last_writer!(max_tokens);
    replace_last_writer!(tool_choice);
    replace_last_writer!(history);

    if let Some(active_tools) = patch.active_tools.clone() {
        accumulated.active_tools = Some(match accumulated.active_tools.take() {
            Some(existing) => {
                let active = active_tools.into_iter().collect::<HashSet<_>>();
                existing
                    .into_iter()
                    .filter(|tool| active.contains(tool))
                    .collect()
            }
            None => active_tools,
        });
    }
    if let Some(patch_params) = patch.additional_params.clone() {
        accumulated.additional_params = match (accumulated.additional_params.take(), patch_params) {
            (Some(serde_json::Value::Object(mut base)), serde_json::Value::Object(patch)) => {
                base.extend(patch);
                Some(serde_json::Value::Object(base))
            }
            (_, replacement) => Some(replacement),
        };
    }
    accumulated
        .extra_context
        .extend(patch.extra_context.iter().cloned());
    apply_request_patch(input, patch);
}

fn begin_policy_approval(
    world: &mut World,
    evaluation: Entity,
    run: Entity,
    policy: &AcceptedPolicy,
    point: PolicyPoint,
    prompt: String,
) -> Entity {
    world
        .spawn((
            OperationOf(run),
            ApprovalForEvaluation(evaluation),
            OperationKind::PolicyApproval,
            OperationGeneration(0),
            OperationState {
                phase: OperationPhase::Prepared,
            },
            PolicyApprovalEffectInput {
                policy_id: policy.id.clone(),
                revision: policy.revision,
                point,
                prompt,
            },
        ))
        .id()
}

fn run_policy_responder<E, D>(
    world: &mut World,
    invocation: E,
    policy: Entity,
    policy_id: &StableId,
    point: PolicyPoint,
) -> Option<D>
where
    E: Send + Sync + 'static,
    D: 'static,
{
    let Some(bindings) = world.get::<PolicyResponders>(policy) else {
        tracing::warn!(
            policy = policy_id.as_str(),
            "steering policy has no responder binding"
        );
        return None;
    };
    if !world
        .get::<PolicyCapabilities>(policy)
        .is_some_and(|capabilities| capabilities.contains(point))
    {
        tracing::warn!(
            policy = policy_id.as_str(),
            point = point.as_str(),
            "steering invocation does not match the policy capability"
        );
        return None;
    }
    let matching = bindings
        .iter()
        .filter_map(|binding| {
            let binding_point = world.get::<ResponderPoint>(binding)?;
            let responder = world.get::<RegisteredPolicyResponder>(binding)?;
            (binding_point.0 == point).then_some((binding, responder.0))
        })
        .collect::<Vec<_>>();
    let [(binding, responder)] = matching.as_slice() else {
        tracing::warn!(
            policy = policy_id.as_str(),
            point = point.as_str(),
            responders = matching.len(),
            "steering policy must have exactly one explicit responder binding"
        );
        return None;
    };
    if world.get_entity(*binding).is_err() || world.get_entity(*responder).is_err() {
        tracing::warn!(
            policy = policy_id.as_str(),
            point = point.as_str(),
            "steering policy responder binding is stale"
        );
        return None;
    }
    match world.run_system_with(
        SystemId::<In<E>, Option<D>>::from_entity(*responder),
        invocation,
    ) {
        Ok(decision) => decision,
        Err(error) => {
            tracing::warn!(
                policy = policy_id.as_str(),
                point = point.as_str(),
                %error,
                "steering policy responder failed"
            );
            None
        }
    }
}

fn evaluate_request_policies(world: &mut World) {
    let mut query = world.query::<(Entity, &EvaluationOfOperation, &RequestPolicyEvaluation)>();
    let mut ready = query
        .iter(world)
        .filter(|(_, _, evaluation)| {
            matches!(evaluation.phase, RequestPolicyEvaluationPhase::Evaluating)
        })
        .map(|(entity, operation, evaluation)| {
            (entity, operation.get(), evaluation.run, evaluation.cursor)
        })
        .collect::<Vec<_>>();
    ready.retain(|(_, _, run, _)| world_allows_internal_progress(world, *run));
    ready.sort_by(|left, right| {
        let left_id = world.get::<StableId>(left.2);
        let right_id = world.get::<StableId>(right.2);
        left_id
            .cmp(&right_id)
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.0.to_bits().cmp(&right.0.to_bits()))
    });

    for (evaluation_entity, operation, run, cursor) in ready {
        let Some(evaluation) = world.get::<RequestPolicyEvaluation>(evaluation_entity) else {
            continue;
        };
        let Some(policy) = evaluation.policies.get(cursor).cloned() else {
            let effective = evaluation.effective.clone();
            world.entity_mut(operation).insert(effective);
            world.entity_mut(operation).remove::<PendingModelRequest>();
            if let Some(mut evaluation) =
                world.get_mut::<RequestPolicyEvaluation>(evaluation_entity)
            {
                evaluation.phase = RequestPolicyEvaluationPhase::Accepted;
            }
            mark_progress(&mut world.resource_mut::<Progress>());
            continue;
        };
        let invocation = RequestPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            request: evaluation.effective.clone(),
        };
        world.trigger(RequestPolicyInvoked {
            policy: policy.entity,
            evaluation: evaluation_entity,
            policy_id: policy.id.clone(),
            revision: policy.revision,
            cursor,
        });
        let decision =
            run_policy_responder(world, invocation, policy.entity, &policy.id, policy.point);
        world.trigger(RequestPolicyDecided {
            policy: policy.entity,
            evaluation: evaluation_entity,
            policy_id: policy.id.clone(),
            revision: policy.revision,
            decision: decision.clone(),
        });
        match decision {
            Some(RequestPolicyDecision::Continue) => {
                if let Some(mut evaluation) =
                    world.get_mut::<RequestPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.cursor += 1;
                }
            }
            Some(RequestPolicyDecision::Patch(patch)) => {
                if let Some(mut evaluation) =
                    world.get_mut::<RequestPolicyEvaluation>(evaluation_entity)
                {
                    let mut effective = evaluation.effective.clone();
                    reduce_request_patch(
                        &mut effective,
                        &mut evaluation.accumulated,
                        patch,
                        &policy.id,
                    );
                    evaluation.effective = effective;
                    evaluation.cursor += 1;
                }
            }
            Some(RequestPolicyDecision::AwaitApproval(prompt)) => {
                let operation = begin_policy_approval(
                    world,
                    evaluation_entity,
                    run,
                    &policy,
                    PolicyPoint::Request,
                    prompt,
                );
                if let Some(mut evaluation) =
                    world.get_mut::<RequestPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = RequestPolicyEvaluationPhase::WaitingApproval {
                        operation,
                        policy: policy.id.clone(),
                    };
                }
            }
            decision @ (Some(RequestPolicyDecision::Stop(_)) | None) => {
                let reason = match decision {
                    Some(RequestPolicyDecision::Stop(reason)) => reason,
                    _ => "policy did not provide a steering decision".to_owned(),
                };
                if let Some(mut evaluation) =
                    world.get_mut::<RequestPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = RequestPolicyEvaluationPhase::Rejected {
                        policy: policy.id.clone(),
                        reason: reason.clone(),
                    };
                }
                let error = policy_termination_error(
                    world,
                    run,
                    operation,
                    policy.id.clone(),
                    policy.revision,
                    PolicyPoint::Request,
                    reason,
                );
                if let Some(mut state) = world.get_mut::<OperationState>(operation) {
                    state.phase = OperationPhase::Settled(OperationOutcome::Failure(error.clone()));
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(error);
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

#[allow(clippy::type_complexity)]
fn initialize_tool_call_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &PendingToolCall, &OperationState),
        Without<ToolCallPolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunPolicies>, Option<&RunControl>), Without<WaitingForChildren>>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(
        Entity,
        &StableId,
        &PolicyMeta,
        &PolicyCapabilities,
        Option<&PolicyStatus>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, pending, state) in &operations {
        if !matches!(state.phase, OperationPhase::Prepared) {
            continue;
        }
        let Ok((run_of, run_policies, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = policy_entities(agents.get(run_of.get()).ok().flatten(), run_policies)
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, _, capabilities, status)| {
                accepts_new_policy_evaluations(*status)
                    && capabilities.contains(PolicyPoint::ToolCall)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _, _), (_, right_id, right, _, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
                order: policy.order,
                point: PolicyPoint::ToolCall,
            })
            .collect::<Vec<_>>();
        commands.entity(operation).insert(ToolCallPolicyInitialized);
        if snapshot.is_empty() {
            commands
                .entity(operation)
                .insert((pending.0.clone(), AcceptedToolCallPolicies::default()))
                .remove::<PendingToolCall>();
        } else {
            commands
                .entity(operation)
                .insert(AcceptedToolCallPolicies(snapshot.clone()));
            commands.spawn((
                EvaluationOfOperation(operation),
                ToolCallPolicyEvaluation {
                    run: operation_of.get(),
                    policies: snapshot,
                    cursor: 0,
                    effective: pending.0.clone(),
                    phase: ToolCallPolicyEvaluationPhase::Evaluating,
                },
            ));
        }
        mark_progress(&mut progress);
    }
}

fn evaluate_tool_call_policies(world: &mut World) {
    let mut query = world.query::<(Entity, &EvaluationOfOperation, &ToolCallPolicyEvaluation)>();
    let mut ready = query
        .iter(world)
        .filter(|(_, _, evaluation)| {
            matches!(evaluation.phase, ToolCallPolicyEvaluationPhase::Evaluating)
        })
        .map(|(entity, operation, evaluation)| {
            (
                entity,
                operation.get(),
                evaluation.run,
                evaluation.cursor,
                evaluation.effective.index,
            )
        })
        .collect::<Vec<_>>();
    ready.retain(|(_, _, run, _, _)| world_allows_internal_progress(world, *run));
    ready.sort_by(|left, right| {
        let left_id = world.get::<StableId>(left.2);
        let right_id = world.get::<StableId>(right.2);
        left_id
            .cmp(&right_id)
            .then_with(|| left.4.cmp(&right.4))
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.0.to_bits().cmp(&right.0.to_bits()))
    });

    for (evaluation_entity, operation, run, cursor, _) in ready {
        let Some(evaluation) = world.get::<ToolCallPolicyEvaluation>(evaluation_entity) else {
            continue;
        };
        let Some(policy) = evaluation.policies.get(cursor).cloned() else {
            let effective = evaluation.effective.clone();
            world.entity_mut(operation).insert(effective);
            world.entity_mut(operation).remove::<PendingToolCall>();
            if let Some(mut evaluation) =
                world.get_mut::<ToolCallPolicyEvaluation>(evaluation_entity)
            {
                evaluation.phase = ToolCallPolicyEvaluationPhase::Accepted;
            }
            mark_progress(&mut world.resource_mut::<Progress>());
            continue;
        };
        let invocation = ToolCallPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            call: evaluation.effective.clone(),
        };
        let decision =
            run_policy_responder(world, invocation, policy.entity, &policy.id, policy.point);
        match decision {
            Some(ToolCallPolicyDecision::Run) => {
                if let Some(mut evaluation) =
                    world.get_mut::<ToolCallPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.cursor += 1;
                }
            }
            Some(ToolCallPolicyDecision::Rewrite(arguments)) => {
                if let Some(mut evaluation) =
                    world.get_mut::<ToolCallPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.effective.arguments = arguments;
                    evaluation.cursor += 1;
                }
            }
            Some(ToolCallPolicyDecision::AwaitApproval(prompt)) => {
                let approval = begin_policy_approval(
                    world,
                    evaluation_entity,
                    run,
                    &policy,
                    PolicyPoint::ToolCall,
                    prompt,
                );
                if let Some(mut evaluation) =
                    world.get_mut::<ToolCallPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = ToolCallPolicyEvaluationPhase::WaitingApproval {
                        operation: approval,
                        policy: policy.id.clone(),
                    };
                }
            }
            Some(ToolCallPolicyDecision::Skip(reason)) => {
                let Some(effective) = world
                    .get::<ToolCallPolicyEvaluation>(evaluation_entity)
                    .map(|evaluation| evaluation.effective.clone())
                else {
                    continue;
                };
                world.entity_mut(operation).insert(effective.clone());
                world.entity_mut(operation).remove::<PendingToolCall>();
                if let Some(mut state) = world.get_mut::<OperationState>(operation) {
                    state.phase = OperationPhase::Settled(OperationOutcome::Success(
                        EffectOutput::Tool(ToolEffectOutput {
                            call_id: effective.call_id,
                            provider_result_id: effective.provider_result_id,
                            provider_call_id: effective.provider_call_id,
                            name: effective.decision.name,
                            raw: serde_json::json!({"skipped": true, "reason": reason}).into(),
                            presentation: reason.clone().into(),
                            failure: None,
                        }),
                    ));
                }
                if let Some(mut evaluation) =
                    world.get_mut::<ToolCallPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = ToolCallPolicyEvaluationPhase::Skipped(reason);
                }
            }
            decision @ (Some(ToolCallPolicyDecision::Stop(_)) | None) => {
                let reason = match decision {
                    Some(ToolCallPolicyDecision::Stop(reason)) => reason,
                    _ => "policy did not provide a steering decision".to_owned(),
                };
                if let Some(mut evaluation) =
                    world.get_mut::<ToolCallPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = ToolCallPolicyEvaluationPhase::Rejected {
                        policy: policy.id.clone(),
                        reason: reason.clone(),
                    };
                }
                let error = policy_termination_error(
                    world,
                    run,
                    operation,
                    policy.id.clone(),
                    policy.revision,
                    PolicyPoint::ToolCall,
                    reason,
                );
                if let Some(mut state) = world.get_mut::<OperationState>(operation) {
                    state.phase = OperationPhase::Settled(OperationOutcome::Failure(error.clone()));
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(error);
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

#[allow(clippy::type_complexity)]
fn initialize_invalid_tool_call_policy_evaluations(
    mut commands: Commands,
    mut operations: Query<
        (
            Entity,
            &OperationOf,
            &PendingInvalidToolCall,
            &mut OperationState,
        ),
        Without<InvalidToolCallPolicyInitialized>,
    >,
    mut runs: Query<
        (
            &RunOf,
            Option<&RunPolicies>,
            Option<&RunControl>,
            &mut RunState,
        ),
        Without<WaitingForChildren>,
    >,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(
        Entity,
        &StableId,
        &PolicyMeta,
        &PolicyCapabilities,
        Option<&PolicyStatus>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, invalid, mut state) in &mut operations {
        if !matches!(state.phase, OperationPhase::Prepared) {
            continue;
        }
        let Ok((run_of, run_policies, control, mut run_state)) = runs.get_mut(operation_of.get())
        else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = policy_entities(agents.get(run_of.get()).ok().flatten(), run_policies)
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, _, capabilities, status)| {
                accepts_new_policy_evaluations(*status)
                    && capabilities.contains(PolicyPoint::InvalidToolCall)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _, _), (_, right_id, right, _, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
                order: policy.order,
                point: PolicyPoint::InvalidToolCall,
            })
            .collect::<Vec<_>>();
        commands.entity(operation).insert((
            InvalidToolCallPolicyInitialized,
            AcceptedInvalidToolCallPolicies(snapshot.clone()),
        ));
        if snapshot.is_empty() {
            let error = CanonicalError::UnknownTool(invalid.call.name.clone());
            state.phase = OperationPhase::Settled(OperationOutcome::Failure(error.clone()));
            *run_state = RunState::Failed(error);
        } else {
            commands.spawn((
                EvaluationOfOperation(operation),
                InvalidToolCallPolicyEvaluation {
                    run: operation_of.get(),
                    policies: snapshot,
                    cursor: 0,
                    invalid: invalid.clone(),
                    phase: InvalidToolCallPolicyEvaluationPhase::Evaluating,
                },
            ));
        }
        mark_progress(&mut progress);
    }
}

fn synthetic_invalid_tool_input(
    operation: Entity,
    invalid: &PendingInvalidToolCall,
) -> ToolEffectInput {
    ToolEffectInput {
        decision: ToolDecision {
            tool_entity: operation,
            tool_id: StableId::generated(format!("invalid-tool-call/{}", invalid.call.id)),
            revision: 0,
            name: invalid.call.name.clone(),
            description: "synthetic invalid tool-call result".to_owned(),
            parameters: serde_json::Value::Null,
            order: invalid.index,
        },
        call_id: invalid.call.id.clone(),
        provider_result_id: invalid.call.provider_result_id.clone(),
        provider_call_id: invalid.call.provider_call_id.clone(),
        arguments: invalid.call.arguments.clone(),
        index: invalid.index,
    }
}

fn tool_choice_allows(choice: Option<&ModelToolChoice>, name: &str) -> bool {
    match choice {
        Some(ModelToolChoice::None) => false,
        Some(ModelToolChoice::Specific(names)) => names.iter().any(|candidate| candidate == name),
        None | Some(ModelToolChoice::Auto | ModelToolChoice::Required) => true,
    }
}

fn settle_invalid_tool_as_feedback(
    world: &mut World,
    operation: Entity,
    invalid: &PendingInvalidToolCall,
    kind: &str,
    feedback: String,
) {
    let input = synthetic_invalid_tool_input(operation, invalid);
    world.entity_mut(operation).insert(input.clone());
    world
        .entity_mut(operation)
        .remove::<PendingInvalidToolCall>();
    if let Some(mut state) = world.get_mut::<OperationState>(operation) {
        state.phase = OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Tool(
            ToolEffectOutput {
                call_id: input.call_id,
                provider_result_id: input.provider_result_id,
                provider_call_id: input.provider_call_id,
                name: input.decision.name,
                raw: serde_json::json!({"invalid_tool_call": kind, "feedback": feedback}).into(),
                presentation: feedback.into(),
                failure: None,
            },
        )));
    }
}

fn fail_invalid_tool_call(
    world: &mut World,
    evaluation_entity: Entity,
    operation: Entity,
    run: Entity,
    error: CanonicalError,
) {
    if let Some(mut evaluation) =
        world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
    {
        evaluation.phase = InvalidToolCallPolicyEvaluationPhase::Failed;
    }
    if let Some(mut state) = world.get_mut::<OperationState>(operation) {
        state.phase = OperationPhase::Settled(OperationOutcome::Failure(error.clone()));
    }
    if let Some(mut state) = world.get_mut::<RunState>(run) {
        *state = RunState::Failed(error);
    }
}

fn evaluate_invalid_tool_call_policies(world: &mut World) {
    let mut query = world.query::<(
        Entity,
        &EvaluationOfOperation,
        &InvalidToolCallPolicyEvaluation,
    )>();
    let mut ready = query
        .iter(world)
        .filter(|(_, _, evaluation)| {
            matches!(
                evaluation.phase,
                InvalidToolCallPolicyEvaluationPhase::Evaluating
            )
        })
        .map(|(entity, operation, evaluation)| {
            (
                entity,
                operation.get(),
                evaluation.run,
                evaluation.cursor,
                evaluation.invalid.index,
            )
        })
        .collect::<Vec<_>>();
    ready.retain(|(_, _, run, _, _)| world_allows_internal_progress(world, *run));
    ready.sort_by(|left, right| {
        let left_id = world.get::<StableId>(left.2);
        let right_id = world.get::<StableId>(right.2);
        left_id
            .cmp(&right_id)
            .then_with(|| left.4.cmp(&right.4))
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.0.to_bits().cmp(&right.0.to_bits()))
    });

    for (evaluation_entity, operation, run, cursor, _) in ready {
        let Some(evaluation) = world
            .get::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
            .cloned()
        else {
            continue;
        };
        let Some(policy) = evaluation.policies.get(cursor).cloned() else {
            fail_invalid_tool_call(
                world,
                evaluation_entity,
                operation,
                run,
                CanonicalError::UnknownTool(evaluation.invalid.call.name),
            );
            mark_progress(&mut world.resource_mut::<Progress>());
            continue;
        };
        let invocation = InvalidToolCallPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            invalid: evaluation.invalid.clone(),
        };
        let decision =
            run_policy_responder(world, invocation, policy.entity, &policy.id, policy.point);
        match decision {
            Some(InvalidToolCallPolicyDecision::Continue) => {
                if let Some(mut evaluation) =
                    world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.cursor += 1;
                }
            }
            Some(InvalidToolCallPolicyDecision::AwaitApproval(prompt)) => {
                let approval = begin_policy_approval(
                    world,
                    evaluation_entity,
                    run,
                    &policy,
                    PolicyPoint::InvalidToolCall,
                    prompt,
                );
                if let Some(mut evaluation) =
                    world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = InvalidToolCallPolicyEvaluationPhase::WaitingApproval {
                        operation: approval,
                        policy: policy.id.clone(),
                    };
                }
            }
            Some(InvalidToolCallPolicyDecision::Repair(tool_name)) => {
                let Some(decision) = evaluation
                    .invalid
                    .available_tools
                    .iter()
                    .find(|tool| {
                        tool.name == tool_name
                            && tool_choice_allows(
                                evaluation.invalid.tool_choice.as_ref(),
                                &tool.name,
                            )
                    })
                    .cloned()
                else {
                    fail_invalid_tool_call(
                        world,
                        evaluation_entity,
                        operation,
                        run,
                        CanonicalError::UnknownTool(tool_name),
                    );
                    mark_progress(&mut world.resource_mut::<Progress>());
                    continue;
                };
                world
                    .entity_mut(operation)
                    .insert(PendingToolCall(ToolEffectInput {
                        decision,
                        call_id: evaluation.invalid.call.id.clone(),
                        provider_result_id: evaluation.invalid.call.provider_result_id.clone(),
                        provider_call_id: evaluation.invalid.call.provider_call_id.clone(),
                        arguments: evaluation.invalid.call.arguments.clone(),
                        index: evaluation.invalid.index,
                    }));
                world
                    .entity_mut(operation)
                    .remove::<PendingInvalidToolCall>();
                if let Some(mut state) =
                    world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = InvalidToolCallPolicyEvaluationPhase::Resolved;
                }
            }
            Some(InvalidToolCallPolicyDecision::Retry(feedback)) => {
                let retry_allowed = world.get::<RunRecord>(run).is_some_and(|record| {
                    record.invalid_tool_call_retries < record.max_invalid_tool_call_retries
                });
                if !retry_allowed {
                    fail_invalid_tool_call(
                        world,
                        evaluation_entity,
                        operation,
                        run,
                        CanonicalError::UnknownTool(evaluation.invalid.call.name),
                    );
                    mark_progress(&mut world.resource_mut::<Progress>());
                    continue;
                }
                if let Some(mut record) = world.get_mut::<RunRecord>(run) {
                    record.invalid_tool_call_retries =
                        record.invalid_tool_call_retries.saturating_add(1);
                }
                settle_invalid_tool_as_feedback(
                    world,
                    operation,
                    &evaluation.invalid,
                    "retry",
                    feedback,
                );
                if let Some(mut state) =
                    world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = InvalidToolCallPolicyEvaluationPhase::Resolved;
                }
            }
            Some(InvalidToolCallPolicyDecision::Skip(reason)) => {
                settle_invalid_tool_as_feedback(
                    world,
                    operation,
                    &evaluation.invalid,
                    "skip",
                    reason,
                );
                if let Some(mut state) =
                    world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = InvalidToolCallPolicyEvaluationPhase::Resolved;
                }
            }
            Some(InvalidToolCallPolicyDecision::Fail) => {
                fail_invalid_tool_call(
                    world,
                    evaluation_entity,
                    operation,
                    run,
                    CanonicalError::UnknownTool(evaluation.invalid.call.name),
                );
            }
            decision @ (Some(InvalidToolCallPolicyDecision::Stop(_)) | None) => {
                let reason = match decision {
                    Some(InvalidToolCallPolicyDecision::Stop(reason)) => reason,
                    _ => "policy did not provide a steering decision".to_owned(),
                };
                if let Some(mut state) =
                    world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = InvalidToolCallPolicyEvaluationPhase::Rejected {
                        policy: policy.id.clone(),
                        reason: reason.clone(),
                    };
                }
                let mut error = policy_termination_error(
                    world,
                    run,
                    operation,
                    policy.id.clone(),
                    policy.revision,
                    PolicyPoint::InvalidToolCall,
                    reason,
                );
                if let CanonicalError::PolicyTerminated { termination } = &mut error {
                    termination.history = evaluation.invalid.diagnostic_history.clone();
                }
                if let Some(mut state) = world.get_mut::<OperationState>(operation) {
                    state.phase = OperationPhase::Settled(OperationOutcome::Failure(error.clone()));
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(error);
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn apply_dispatch_phase(world: &mut World, entity: Entity, generation: u64, phase: OperationPhase) {
    let dispatched = matches!(phase, OperationPhase::InFlight);
    let generation_matches = world
        .get::<OperationGeneration>(entity)
        .is_some_and(|current| current.0 == generation);
    let updated = if generation_matches
        && let Some(mut state) = world.get_mut::<OperationState>(entity)
        && matches!(state.phase, OperationPhase::Prepared)
    {
        state.phase = phase;
        true
    } else {
        false
    };
    if updated && dispatched {
        let expires_at = world
            .resource::<RuntimeClock>()
            .tick
            .saturating_add(world.resource::<EffectTimeoutTicks>().0);
        world
            .entity_mut(entity)
            .insert(EffectDeadline { expires_at });
    }
}

fn reserve_dispatch_slot(
    world: &mut World,
    run: Option<Entity>,
    tenant: &TenantId,
    kind: OperationKind,
) -> bool {
    let limits = *world.resource::<EffectDispatchLimits>();
    let candidate_agent = run.and_then(|run| world.get::<RunOf>(run).map(Relationship::get));
    let tenant_key = tenant.as_str().to_owned();
    let budget = world.resource::<DispatchBudget>();
    if budget.remaining == 0
        || budget
            .allowances
            .get(&(tenant_key.clone(), kind))
            .copied()
            .unwrap_or(0)
            == 0
        || run.is_some_and(|run| {
            budget.in_flight_by_run.get(&run).copied().unwrap_or(0) >= limits.per_run
        })
        || candidate_agent.is_some_and(|agent| {
            budget.in_flight_by_agent.get(&agent).copied().unwrap_or(0) >= limits.per_agent
        })
        || budget
            .in_flight_by_tenant
            .get(&tenant_key)
            .copied()
            .unwrap_or(0)
            >= limits.per_tenant
    {
        return false;
    }
    let mut budget = world.resource_mut::<DispatchBudget>();
    budget.remaining -= 1;
    *budget
        .allowances
        .entry((tenant_key.clone(), kind))
        .or_default() -= 1;
    *budget.in_flight_by_tenant.entry(tenant_key).or_default() += 1;
    if let Some(run) = run {
        *budget.in_flight_by_run.entry(run).or_default() += 1;
    }
    if let Some(agent) = candidate_agent {
        *budget.in_flight_by_agent.entry(agent).or_default() += 1;
    }
    true
}

fn refund_dispatch_slot(
    world: &mut World,
    run: Option<Entity>,
    tenant: &TenantId,
    kind: OperationKind,
) {
    let limit = world.resource::<EffectDispatchLimits>().per_pass;
    let candidate_agent = run.and_then(|run| world.get::<RunOf>(run).map(Relationship::get));
    let tenant_key = tenant.as_str().to_owned();
    let mut budget = world.resource_mut::<DispatchBudget>();
    budget.remaining = budget.remaining.saturating_add(1).min(limit);
    *budget
        .allowances
        .entry((tenant_key.clone(), kind))
        .or_default() += 1;
    decrement_counter(&mut budget.in_flight_by_tenant, &tenant_key);
    if let Some(run) = run {
        decrement_counter(&mut budget.in_flight_by_run, &run);
    }
    if let Some(agent) = candidate_agent {
        decrement_counter(&mut budget.in_flight_by_agent, &agent);
    }
}

fn decrement_counter<K>(counters: &mut HashMap<K, usize>, key: &K)
where
    K: Eq + std::hash::Hash,
{
    if let Some(value) = counters.get_mut(key) {
        *value = value.saturating_sub(1);
        if *value == 0 {
            counters.remove(key);
        }
    }
}

fn run_dispatch_key(world: &World, run: Entity) -> Option<(Reverse<i32>, ReadyAt, StableId)> {
    Some((
        Reverse(world.get::<RunPriority>(run)?.0),
        *world.get::<ReadyAt>(run)?,
        world.get::<StableId>(run)?.clone(),
    ))
}

fn dispatch_model_operations(world: &mut World) {
    let outbox = world.resource::<EffectOutbox>().0.clone();
    let mut query = world.query_filtered::<(
        Entity,
        &OperationOf,
        &ModelEffectInput,
        &OperationGeneration,
        &OperationState,
    ), (With<ModelEffectInput>, Without<ToolEffectInput>)>();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, run, input, generation, state)| {
            if !matches!(state.phase, OperationPhase::Prepared)
                || world.get::<WaitingForChildren>(run.get()).is_some()
                || !matches!(
                    world.get::<RunControl>(run.get()),
                    Some(RunControl::Running)
                )
                || !matches!(
                    world.get::<RunState>(run.get()),
                    Some(RunState::WaitingModel { .. })
                )
            {
                return None;
            }
            let order = run_dispatch_key(world, run.get())?;
            Some((order, run.get(), entity, generation.0, input.clone()))
        })
        .collect::<Vec<_>>();
    prepared.sort_by(
        |(left_order, _, left_entity, _, _), (right_order, _, right_entity, _, _)| {
            left_order
                .cmp(right_order)
                .then_with(|| left_entity.to_bits().cmp(&right_entity.to_bits()))
        },
    );
    let mut made_progress = false;
    for (_, run, entity, generation, input) in prepared {
        let Some(tenant) = world.get::<TenantId>(run).cloned() else {
            continue;
        };
        if !reserve_dispatch_slot(world, Some(run), &tenant, OperationKind::Model) {
            continue;
        }
        let request = EffectRequest {
            operation: entity,
            generation,
            input: EffectInput::Model(input),
        };
        let phase = match outbox.try_send(request) {
            Ok(()) => {
                made_progress = true;
                Some(OperationPhase::InFlight)
            }
            Err(TrySendError::Full(_)) => {
                refund_dispatch_slot(world, Some(run), &tenant, OperationKind::Model);
                None
            }
            Err(TrySendError::Disconnected(_)) => {
                made_progress = true;
                Some(OperationPhase::Settled(OperationOutcome::Failure(
                    CanonicalError::ExecutorDisconnected,
                )))
            }
        };
        if let Some(phase) = phase {
            let dispatched = matches!(phase, OperationPhase::InFlight);
            apply_dispatch_phase(world, entity, generation, phase);
            if dispatched {
                world.trigger(ModelDispatched {
                    operation: entity,
                    run,
                    generation,
                });
            }
        }
    }
    if made_progress {
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

type PreparedModelOperations<'world, 'state> = Query<
    'world,
    'state,
    (
        Entity,
        &'static OperationOf,
        &'static OperationState,
        &'static ModelEffectInput,
        Option<&'static AcceptedPolicies>,
    ),
    Added<ModelEffectInput>,
>;

type PreparedToolOperations<'world, 'state> = Query<
    'world,
    'state,
    (
        Entity,
        &'static OperationOf,
        &'static OperationState,
        &'static ToolEffectInput,
        Option<&'static AcceptedToolCallPolicies>,
    ),
    Added<ToolEffectInput>,
>;

fn publish_prepared_model_observations(
    mut commands: Commands,
    model_operations: PreparedModelOperations<'_, '_>,
) {
    for (operation, operation_of, state, request, policies) in &model_operations {
        if matches!(state.phase, OperationPhase::Prepared) {
            commands.trigger(CompletionRequestPrepared {
                operation,
                run: operation_of.get(),
                request: request.clone(),
                policies: policies.map_or_else(Vec::new, |value| value.0.clone()),
            });
        }
    }
}

fn publish_prepared_tool_observations(
    mut commands: Commands,
    tool_operations: PreparedToolOperations<'_, '_>,
) {
    for (operation, operation_of, state, call, policies) in &tool_operations {
        if matches!(state.phase, OperationPhase::Prepared) {
            commands.trigger(ToolCallPrepared {
                operation,
                run: operation_of.get(),
                call: call.clone(),
                policies: policies.map_or_else(Vec::new, |value| value.0.clone()),
            });
        }
    }
}

fn publish_invalid_call_observations(
    mut commands: Commands,
    invalid_calls: Query<
        (Entity, &OperationOf, &PendingInvalidToolCall),
        Added<PendingInvalidToolCall>,
    >,
) {
    for (operation, operation_of, invalid) in &invalid_calls {
        commands.trigger(InvalidToolCallDetected {
            operation,
            run: operation_of.get(),
            invalid: invalid.clone(),
        });
    }
}

fn publish_applied_model_policy_observations(
    mut commands: Commands,
    model_operations: Query<
        (
            Entity,
            &OperationOf,
            &OperationState,
            Option<&EffectiveModelOutput>,
        ),
        Added<CompletionResponsePolicyDone>,
    >,
) {
    for (operation, operation_of, state, effective) in &model_operations {
        let OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Model(raw))) =
            &state.phase
        else {
            continue;
        };
        let effective = effective.map_or_else(|| raw.clone(), |value| value.0.clone());
        commands.trigger(CompletionResponseApplied {
            operation,
            run: operation_of.get(),
            raw: raw.clone(),
            effective,
        });
    }
}

fn publish_applied_tool_policy_observations(
    mut commands: Commands,
    tool_operations: Query<
        (
            Entity,
            &OperationOf,
            &OperationState,
            Option<&EffectiveToolOutput>,
        ),
        Added<ToolResultPolicyDone>,
    >,
) {
    for (operation, operation_of, state, effective) in &tool_operations {
        let OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Tool(raw))) =
            &state.phase
        else {
            continue;
        };
        let effective = effective.map_or_else(|| raw.clone(), |value| value.0.clone());
        commands.trigger(ToolResultPresentationFinalized {
            operation,
            run: operation_of.get(),
            status: tool_output_status(raw),
            presentation: effective.presentation.telemetry_summary(),
        });
    }
}

fn tool_output_status(output: &ToolEffectOutput) -> ToolExecutionStatus {
    output.failure.as_ref().map_or(
        ToolExecutionStatus {
            succeeded: true,
            failure_kind: None,
            retryable: None,
            refusal: false,
        },
        |failure| ToolExecutionStatus {
            succeeded: false,
            failure_kind: Some(failure.kind),
            retryable: failure.retryable,
            refusal: failure.refusal,
        },
    )
}

fn published_tool_result(output: &ToolEffectOutput) -> PublishedToolResult {
    PublishedToolResult {
        call_id: output.call_id.clone(),
        provider_result_id: output.provider_result_id.clone(),
        provider_call_id: output.provider_call_id.clone(),
        name: output.name.clone(),
        presentation: output.presentation.telemetry_summary(),
        status: tool_output_status(output),
    }
}

fn tool_execution_status(outcome: &OperationOutcome) -> ToolExecutionStatus {
    match outcome {
        OperationOutcome::Success(EffectOutput::Tool(output)) => tool_output_status(output),
        OperationOutcome::Failure(error) => {
            let (failure_kind, retryable) = match error {
                CanonicalError::Timeout => (Some(crate::tool::ToolErrorKind::Timeout), Some(true)),
                CanonicalError::Tool { retryable, .. } => {
                    (Some(crate::tool::ToolErrorKind::Other), Some(*retryable))
                }
                CanonicalError::ExecutorDisconnected | CanonicalError::ExecutorPanicked(_) => {
                    (Some(crate::tool::ToolErrorKind::Other), Some(false))
                }
                CanonicalError::EffectKindMismatch => {
                    (Some(crate::tool::ToolErrorKind::Other), Some(false))
                }
                _ => (None, None),
            };
            ToolExecutionStatus {
                succeeded: false,
                failure_kind,
                retryable,
                refusal: false,
            }
        }
        OperationOutcome::Success(_) => ToolExecutionStatus {
            succeeded: false,
            failure_kind: Some(crate::tool::ToolErrorKind::Other),
            retryable: Some(false),
            refusal: false,
        },
    }
}

fn dispatch_tool_operations(world: &mut World) {
    let outbox = world.resource::<EffectOutbox>().0.clone();
    let mut batch_query = world.query::<&BatchOf>();
    let mut query = world.query_filtered::<(
        Entity,
        &OperationOfBatch,
        &ToolEffectInput,
        &OperationGeneration,
        &OperationState,
    ), (With<ToolEffectInput>, Without<ModelEffectInput>)>();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, batch, input, generation, state)| {
            let run = batch_query.get(world, batch.get()).ok()?.get();
            if !matches!(state.phase, OperationPhase::Prepared)
                || world.get::<WaitingForChildren>(run).is_some()
                || !matches!(world.get::<RunControl>(run), Some(RunControl::Running))
                || !matches!(
                    world.get::<RunState>(run),
                    Some(RunState::WaitingTools { .. })
                )
            {
                return None;
            }
            Some((
                run_dispatch_key(world, run)?,
                batch.get().to_bits(),
                input.index,
                run,
                entity,
                generation.0,
                input.clone(),
            ))
        })
        .collect::<Vec<_>>();
    prepared.sort_by_key(|(order, batch, index, _, entity, _, _)| {
        (order.clone(), *batch, *index, entity.to_bits())
    });
    let mut made_progress = false;
    for (_, _, index, run, entity, generation, input) in prepared {
        let Some(tenant) = world.get::<TenantId>(run).cloned() else {
            continue;
        };
        if !reserve_dispatch_slot(world, Some(run), &tenant, OperationKind::Tool) {
            continue;
        }
        let request = EffectRequest {
            operation: entity,
            generation,
            input: EffectInput::Tool(input),
        };
        let phase = match outbox.try_send(request) {
            Ok(()) => {
                made_progress = true;
                Some(OperationPhase::InFlight)
            }
            Err(TrySendError::Full(_)) => {
                refund_dispatch_slot(world, Some(run), &tenant, OperationKind::Tool);
                None
            }
            Err(TrySendError::Disconnected(_)) => {
                made_progress = true;
                Some(OperationPhase::Settled(OperationOutcome::Failure(
                    CanonicalError::ExecutorDisconnected,
                )))
            }
        };
        if let Some(phase) = phase {
            let dispatched = matches!(phase, OperationPhase::InFlight);
            apply_dispatch_phase(world, entity, generation, phase);
            if dispatched {
                world.trigger(ToolExecutionStarted {
                    operation: entity,
                    run,
                    generation,
                    index,
                });
            }
        }
    }
    if made_progress {
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn dispatch_discovery_operations(world: &mut World) {
    let outbox = world.resource::<EffectOutbox>().0.clone();
    let mut query = world.query_filtered::<(
        Entity,
        &DiscoveryEffectInput,
        &OperationGeneration,
        &OperationState,
    ), With<DiscoveryEffectInput>>();
    let mut prepared = query
        .iter(world)
        .filter(|(_, _, _, state)| matches!(state.phase, OperationPhase::Prepared))
        .map(|(entity, input, generation, _)| {
            (input.source_id.clone(), entity, generation.0, input.clone())
        })
        .collect::<Vec<_>>();
    prepared.sort_by(
        |(left_id, left_entity, _, _), (right_id, right_entity, _, _)| {
            left_id
                .cmp(right_id)
                .then_with(|| left_entity.to_bits().cmp(&right_entity.to_bits()))
        },
    );
    let mut made_progress = false;
    for (_, entity, generation, input) in prepared {
        if !reserve_dispatch_slot(world, None, &input.tenant, OperationKind::Discovery) {
            continue;
        }
        let tenant = input.tenant.clone();
        let request = EffectRequest {
            operation: entity,
            generation,
            input: EffectInput::Discovery(input),
        };
        let phase = match outbox.try_send(request) {
            Ok(()) => {
                made_progress = true;
                Some(OperationPhase::InFlight)
            }
            Err(TrySendError::Full(_)) => {
                refund_dispatch_slot(world, None, &tenant, OperationKind::Discovery);
                None
            }
            Err(TrySendError::Disconnected(_)) => {
                made_progress = true;
                Some(OperationPhase::Settled(OperationOutcome::Failure(
                    CanonicalError::ExecutorDisconnected,
                )))
            }
        };
        if let Some(phase) = phase {
            apply_dispatch_phase(world, entity, generation, phase);
        }
    }
    if made_progress {
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn dispatch_store_operations(world: &mut World) {
    let outbox = world.resource::<EffectOutbox>().0.clone();
    let mut query = world.query_filtered::<(
        Entity,
        &OperationOf,
        &StoreEffectInput,
        &OperationGeneration,
        &OperationState,
    ), With<StoreEffectInput>>();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, run, input, generation, state)| {
            if !matches!(state.phase, OperationPhase::Prepared)
                || world.get::<WaitingForChildren>(run.get()).is_some()
                || !matches!(
                    world.get::<RunControl>(run.get()),
                    Some(RunControl::Running)
                )
                || !matches!(
                    world.get::<RunState>(run.get()),
                    Some(RunState::WaitingStore { .. })
                )
            {
                return None;
            }
            let order = run_dispatch_key(world, run.get())?;
            Some((order, run.get(), entity, generation.0, input.clone()))
        })
        .collect::<Vec<_>>();
    prepared.sort_by(
        |(left_order, _, left_entity, _, _), (right_order, _, right_entity, _, _)| {
            left_order
                .cmp(right_order)
                .then_with(|| left_entity.to_bits().cmp(&right_entity.to_bits()))
        },
    );
    let mut made_progress = false;
    for (_, run, entity, generation, input) in prepared {
        let Some(tenant) = world.get::<TenantId>(run).cloned() else {
            continue;
        };
        if !reserve_dispatch_slot(world, Some(run), &tenant, OperationKind::Store) {
            continue;
        }
        let request = EffectRequest {
            operation: entity,
            generation,
            input: EffectInput::Store(input),
        };
        let phase = match outbox.try_send(request) {
            Ok(()) => {
                made_progress = true;
                Some(OperationPhase::InFlight)
            }
            Err(TrySendError::Full(_)) => {
                refund_dispatch_slot(world, Some(run), &tenant, OperationKind::Store);
                None
            }
            Err(TrySendError::Disconnected(_)) => {
                made_progress = true;
                Some(OperationPhase::Settled(OperationOutcome::Failure(
                    CanonicalError::ExecutorDisconnected,
                )))
            }
        };
        if let Some(phase) = phase {
            apply_dispatch_phase(world, entity, generation, phase);
        }
    }
    if made_progress {
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn dispatch_policy_approval_operations(world: &mut World) {
    let outbox = world.resource::<EffectOutbox>().0.clone();
    let mut query = world.query::<(
        Entity,
        &OperationOf,
        &PolicyApprovalEffectInput,
        &OperationGeneration,
        &OperationState,
    )>();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, run, input, generation, state)| {
            if !matches!(state.phase, OperationPhase::Prepared)
                || world.get::<WaitingForChildren>(run.get()).is_some()
                || !matches!(
                    world.get::<RunControl>(run.get()),
                    Some(RunControl::Running)
                )
            {
                return None;
            }
            let order = run_dispatch_key(world, run.get())?;
            Some((order, run.get(), entity, generation.0, input.clone()))
        })
        .collect::<Vec<_>>();
    prepared.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.2.to_bits().cmp(&right.2.to_bits()))
    });
    let mut made_progress = false;
    for (_, run, entity, generation, input) in prepared {
        let Some(tenant) = world.get::<TenantId>(run).cloned() else {
            continue;
        };
        if !reserve_dispatch_slot(world, Some(run), &tenant, OperationKind::PolicyApproval) {
            continue;
        }
        let phase = match outbox.try_send(EffectRequest {
            operation: entity,
            generation,
            input: EffectInput::PolicyApproval(input),
        }) {
            Ok(()) => {
                made_progress = true;
                Some(OperationPhase::InFlight)
            }
            Err(TrySendError::Full(_)) => {
                refund_dispatch_slot(world, Some(run), &tenant, OperationKind::PolicyApproval);
                None
            }
            Err(TrySendError::Disconnected(_)) => {
                made_progress = true;
                Some(OperationPhase::Settled(OperationOutcome::Failure(
                    CanonicalError::ExecutorDisconnected,
                )))
            }
        };
        if let Some(phase) = phase {
            apply_dispatch_phase(world, entity, generation, phase);
        }
    }
    if made_progress {
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn propagate_cancellation(
    runs: Query<&RunState>,
    cancellations: Res<CancellationOutbox>,
    mut operations: Query<(
        Entity,
        &OperationOf,
        &OperationGeneration,
        &mut OperationState,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (entity, operation_of, generation, mut operation) in &mut operations {
        let Ok(run) = runs.get(operation_of.get()) else {
            continue;
        };
        if matches!(run, RunState::Cancelled)
            && !matches!(
                operation.phase,
                OperationPhase::Settled(_)
                    | OperationPhase::ExtensionSettled
                    | OperationPhase::Cancelled
            )
        {
            if matches!(operation.phase, OperationPhase::InFlight)
                && matches!(
                    cancellations.0.try_send(EffectCancellation {
                        operation: entity,
                        generation: generation.0,
                    }),
                    Err(TrySendError::Full(_))
                )
            {
                continue;
            }
            operation.phase = OperationPhase::Cancelled;
            mark_progress(&mut progress);
        }
    }
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn apply_effect_ingress(
    mut commands: Commands,
    mut messages: MessageReader<EffectIngressMessage>,
    mut operations: Query<(
        &OperationKind,
        Option<&OperationOf>,
        &OperationGeneration,
        &mut OperationState,
        Option<&mut ModelStreamState>,
        Option<&mut EffectDeadline>,
    )>,
    clock: Res<RuntimeClock>,
    timeout: Res<EffectTimeoutTicks>,
    runs: Query<(&RunOf, Option<&RunPolicies>, &RunRecord)>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(
        Entity,
        &StableId,
        &PolicyMeta,
        &PolicyCapabilities,
        Option<&PolicyStatus>,
    )>,
    run_subscriptions: Query<&RunSubscriptions>,
    mut subscriptions: Query<(&StreamSink, &mut SubscriptionState)>,
    mut progress: ResMut<Progress>,
) {
    for EffectIngressMessage(message) in messages.read() {
        match message {
            EffectIngress::Completion(completion) => {
                let Ok((kind, operation_of, generation, mut state, stream_state, _)) =
                    operations.get_mut(completion.operation)
                else {
                    continue;
                };
                if generation.0 != completion.generation
                    || !matches!(state.phase, OperationPhase::InFlight)
                {
                    continue;
                }
                let stream_finished = match (
                    kind,
                    operation_of,
                    stream_state.as_deref(),
                    &completion.result,
                ) {
                    (
                        OperationKind::Model,
                        Some(operation_of),
                        Some(stream_state),
                        Ok(EffectOutput::Model(output)),
                    ) if stream_state.streaming => Some((operation_of.get(), output.clone())),
                    _ => None,
                };
                state.phase = OperationPhase::Settled(match &completion.result {
                    Ok(output)
                        if matches!(
                            (kind, output),
                            (OperationKind::Model, EffectOutput::Model(_))
                                | (OperationKind::Tool, EffectOutput::Tool(_))
                                | (OperationKind::Discovery, EffectOutput::Discovery(_))
                                | (OperationKind::Store, EffectOutput::Store(_))
                                | (
                                    OperationKind::PolicyApproval,
                                    EffectOutput::PolicyApproval(_)
                                )
                        ) =>
                    {
                        OperationOutcome::Success(output.clone())
                    }
                    Ok(_) => OperationOutcome::Failure(CanonicalError::EffectKindMismatch),
                    Err(error) => OperationOutcome::Failure(error.clone()),
                });
                if let (Some(run), OperationPhase::Settled(outcome)) =
                    (operation_of.map(Relationship::get), &state.phase)
                {
                    match kind {
                        OperationKind::Model => commands.trigger(ModelSettled {
                            operation: completion.operation,
                            run,
                            generation: completion.generation,
                            outcome: outcome.clone(),
                        }),
                        OperationKind::Tool => commands.trigger(ToolExecutionSettled {
                            operation: completion.operation,
                            run,
                            generation: completion.generation,
                            status: tool_execution_status(outcome),
                        }),
                        OperationKind::Discovery
                        | OperationKind::Store
                        | OperationKind::PolicyApproval => {}
                    }
                }
                if let Some((run, output)) = stream_finished {
                    commands.trigger(StreamResponseFinished {
                        operation: completion.operation,
                        run,
                        generation: completion.generation,
                        output,
                    });
                }
                mark_progress(&mut progress);
            }
            EffectIngress::ProviderDiagnostics(diagnostics) => {
                let Ok((kind, _, generation, state, _, _)) = operations.get(diagnostics.operation)
                else {
                    continue;
                };
                if !matches!(kind, OperationKind::Model)
                    || generation.0 != diagnostics.generation
                    || !matches!(state.phase, OperationPhase::InFlight)
                {
                    continue;
                }
                commands
                    .entity(diagnostics.operation)
                    .insert(ProviderResponseDiagnostics(diagnostics.diagnostics.clone()));
                if let Some(typed) = &diagnostics.typed {
                    typed.insert(&mut commands, diagnostics.operation);
                }
                mark_progress(&mut progress);
            }
            EffectIngress::Delta(delta) => {
                let Ok((kind, operation_of, generation, state, stream_state, deadline)) =
                    operations.get_mut(delta.operation)
                else {
                    continue;
                };
                if !matches!(kind, OperationKind::Model)
                    || generation.0 != delta.generation
                    || !matches!(state.phase, OperationPhase::InFlight)
                {
                    continue;
                }
                let Some(mut stream_state) = stream_state else {
                    continue;
                };
                let Some(operation_of) = operation_of else {
                    continue;
                };
                if stream_state.next_sequence != delta.sequence {
                    continue;
                }
                stream_state.next_sequence = stream_state.next_sequence.saturating_add(1);
                if let Some(mut deadline) = deadline {
                    deadline.expires_at = clock.tick.saturating_add(timeout.0);
                }
                let run = operation_of.get();
                let Ok((run_of, run_policies, record)) = runs.get(run) else {
                    continue;
                };
                let turn = record.next_turn;
                match &delta.kind {
                    EffectDeltaKind::Text(text) => {
                        stream_state.aggregated_text.push_str(text);
                        let aggregated = stream_state.aggregated_text.clone();
                        commands.trigger(TextDeltaObserved {
                            operation: delta.operation,
                            run,
                            turn,
                            sequence: delta.sequence,
                            provider_correlation: delta.provider_correlation.clone(),
                            delta: text.clone(),
                            aggregated: aggregated.clone(),
                        });
                        let mut stream_policies =
                            policy_entities(agents.get(run_of.get()).ok().flatten(), run_policies)
                                .filter_map(|entity| policies.get(entity).ok())
                                .filter(|(_, _, _, capabilities, status)| {
                                    accepts_new_policy_evaluations(*status)
                                        && capabilities.contains(PolicyPoint::TextDelta)
                                })
                                .collect::<Vec<_>>();
                        stream_policies.sort_by(
                            |(_, left_id, left, _, _), (_, right_id, right, _, _)| {
                                left.order
                                    .cmp(&right.order)
                                    .then_with(|| left_id.cmp(right_id))
                            },
                        );
                        let snapshot = stream_policies
                            .drain(..)
                            .map(|(entity, id, policy, _, _)| AcceptedPolicy {
                                id: id.clone(),
                                entity,
                                revision: policy.revision,
                                order: policy.order,
                                point: PolicyPoint::TextDelta,
                            })
                            .collect::<Vec<_>>();
                        if !snapshot.is_empty() {
                            commands.spawn((
                                EvaluationOfOperation(delta.operation),
                                AcceptedTextDeltaPolicies(snapshot.clone()),
                                TextDeltaPolicyEvaluation {
                                    run,
                                    operation: delta.operation,
                                    policies: snapshot,
                                    cursor: 0,
                                    turn,
                                    sequence: delta.sequence,
                                    delta: text.clone(),
                                    aggregated,
                                    phase: TextDeltaPolicyEvaluationPhase::Evaluating,
                                },
                            ));
                            mark_progress(&mut progress);
                            continue;
                        }
                        publish_stream_item(
                            &mut commands,
                            run,
                            StreamItem::Delta {
                                sequence: delta.sequence,
                                text: text.clone(),
                            },
                            &run_subscriptions,
                            &mut subscriptions,
                        );
                    }
                    EffectDeltaKind::ToolCall {
                        id,
                        internal_call_id,
                        content,
                    } => {
                        commands.trigger(ToolCallDeltaObserved {
                            operation: delta.operation,
                            run,
                            turn,
                            sequence: delta.sequence,
                            provider_correlation: delta.provider_correlation.clone(),
                            id: id.clone(),
                            internal_call_id: internal_call_id.clone(),
                            content: content.clone(),
                        });
                        let mut stream_policies =
                            policy_entities(agents.get(run_of.get()).ok().flatten(), run_policies)
                                .filter_map(|entity| policies.get(entity).ok())
                                .filter(|(_, _, _, capabilities, status)| {
                                    accepts_new_policy_evaluations(*status)
                                        && capabilities.contains(PolicyPoint::ToolCallDelta)
                                })
                                .collect::<Vec<_>>();
                        stream_policies.sort_by(
                            |(_, left_id, left, _, _), (_, right_id, right, _, _)| {
                                left.order
                                    .cmp(&right.order)
                                    .then_with(|| left_id.cmp(right_id))
                            },
                        );
                        let snapshot = stream_policies
                            .drain(..)
                            .map(|(entity, id, policy, _, _)| AcceptedPolicy {
                                id: id.clone(),
                                entity,
                                revision: policy.revision,
                                order: policy.order,
                                point: PolicyPoint::ToolCallDelta,
                            })
                            .collect::<Vec<_>>();
                        if !snapshot.is_empty() {
                            commands.spawn((
                                EvaluationOfOperation(delta.operation),
                                AcceptedToolCallDeltaPolicies(snapshot.clone()),
                                ToolCallDeltaPolicyEvaluation {
                                    run,
                                    operation: delta.operation,
                                    policies: snapshot,
                                    cursor: 0,
                                    turn,
                                    sequence: delta.sequence,
                                    provider_correlation: delta.provider_correlation.clone(),
                                    id: id.clone(),
                                    internal_call_id: internal_call_id.clone(),
                                    content: content.clone(),
                                    phase: ToolCallDeltaPolicyEvaluationPhase::Evaluating,
                                },
                            ));
                            mark_progress(&mut progress);
                            continue;
                        }
                        publish_stream_item(
                            &mut commands,
                            run,
                            StreamItem::ToolCallDelta {
                                sequence: delta.sequence,
                                id: id.clone(),
                                internal_call_id: internal_call_id.clone(),
                                content: content.clone(),
                            },
                            &run_subscriptions,
                            &mut subscriptions,
                        );
                    }
                }
                mark_progress(&mut progress);
            }
        }
    }
}

fn publish_stream_item(
    commands: &mut Commands,
    run: Entity,
    item: StreamItem,
    run_subscriptions: &Query<&RunSubscriptions>,
    subscriptions: &mut Query<(&StreamSink, &mut SubscriptionState)>,
) {
    let Ok(run_subscriptions) = run_subscriptions.get(run) else {
        return;
    };
    for subscription in run_subscriptions.iter() {
        let Ok((sink, mut subscription_state)) = subscriptions.get_mut(subscription) else {
            continue;
        };
        if !matches!(*subscription_state, SubscriptionState::Active) {
            continue;
        }
        if sink.0.try_send(item.clone()).is_err() {
            *subscription_state = SubscriptionState::DroppedSlowConsumer;
            commands.entity(subscription).remove::<StreamSink>();
        }
    }
}

fn expire_effects(
    clock: Res<RuntimeClock>,
    mut operations: Query<(&mut OperationState, &EffectDeadline)>,
    mut progress: ResMut<Progress>,
) {
    for (mut operation, deadline) in &mut operations {
        if matches!(operation.phase, OperationPhase::InFlight) && clock.tick >= deadline.expires_at
        {
            operation.phase =
                OperationPhase::Settled(OperationOutcome::Failure(CanonicalError::Timeout));
            mark_progress(&mut progress);
        }
    }
}

fn publish_run_terminal_observations(
    mut commands: Commands,
    runs: Query<(Entity, &RunState), Changed<RunState>>,
) {
    for (run, state) in &runs {
        match state {
            RunState::Completed(output) => commands.trigger(RunCompleted {
                run,
                output: output.clone(),
            }),
            RunState::Failed(error) => commands.trigger(RunFailed {
                run,
                error: error.clone(),
            }),
            RunState::Cancelled => commands.trigger(RunCancelled { run }),
            RunState::Queued
            | RunState::WaitingModel { .. }
            | RunState::WaitingTools { .. }
            | RunState::WaitingStore { .. } => {}
        }
    }
}

fn publish_terminal_streams(
    mut commands: Commands,
    runs: Query<(&RunState, Option<&RunSubscriptions>), Changed<RunState>>,
    mut subscriptions: Query<(&StreamSink, &mut SubscriptionState)>,
    mut progress: ResMut<Progress>,
) {
    for (run_state, run_subscriptions) in &runs {
        let terminal = match run_state {
            RunState::Completed(output) => StreamTerminal::Completed(output.clone()),
            RunState::Failed(error) => StreamTerminal::Failed(error.clone()),
            RunState::Cancelled => StreamTerminal::Cancelled,
            RunState::Queued
            | RunState::WaitingModel { .. }
            | RunState::WaitingTools { .. }
            | RunState::WaitingStore { .. } => continue,
        };
        let Some(run_subscriptions) = run_subscriptions else {
            continue;
        };
        for subscription in run_subscriptions.iter() {
            let Ok((sink, mut subscription_state)) = subscriptions.get_mut(subscription) else {
                continue;
            };
            if !matches!(*subscription_state, SubscriptionState::Active) {
                continue;
            }
            *subscription_state = if sink
                .0
                .try_send(StreamItem::Finished(terminal.clone()))
                .is_ok()
            {
                SubscriptionState::Finished
            } else {
                commands.entity(subscription).remove::<StreamSink>();
                SubscriptionState::DroppedSlowConsumer
            };
            mark_progress(&mut progress);
        }
    }
}

fn commit_child_results(
    mut commands: Commands,
    mut parents: Query<(Entity, &ChildRuns, &mut RunRecord), With<WaitingForChildren>>,
    children: Query<(
        &StableId,
        &ChildOrdinal,
        &RunState,
        Option<&ChildResultCommitted>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (parent, child_runs, mut parent_record) in &mut parents {
        let child_count = child_runs.iter().count();
        let mut ordered = child_runs
            .iter()
            .filter_map(|child| {
                let (id, ordinal, state, committed) = children.get(child).ok()?;
                Some((ordinal.0, child, id.clone(), state, committed.is_some()))
            })
            .collect::<Vec<_>>();
        let mut all_terminal = ordered.len() == child_count && child_count > 0;
        ordered.sort_by_key(|(ordinal, child, _, _, _)| (*ordinal, child.to_bits()));
        for (ordinal, child, child_id, state, committed) in ordered {
            if committed {
                continue;
            }
            let result = match state {
                RunState::Completed(output) => Ok(output.text.clone()),
                RunState::Failed(error) => Err(error.to_string()),
                RunState::Cancelled => Err("cancelled".to_owned()),
                RunState::Queued
                | RunState::WaitingModel { .. }
                | RunState::WaitingTools { .. }
                | RunState::WaitingStore { .. } => {
                    all_terminal = false;
                    break;
                }
            };
            parent_record.transcript.push(TranscriptEntry::ChildResult {
                ordinal,
                run_id: child_id.clone(),
                result: result.clone(),
            });
            commands.entity(child).insert(ChildResultCommitted);
            commands.trigger(ChildRunFinished {
                parent,
                child,
                child_id,
                ordinal,
                result,
            });
            mark_progress(&mut progress);
        }
        if all_terminal {
            commands.entity(parent).remove::<WaitingForChildren>();
            mark_progress(&mut progress);
        }
    }
}

fn propagate_parent_cancellation(world: &mut World) {
    let mut query = world.query::<(&RunState, &ChildRuns)>();
    let children = query
        .iter(world)
        .filter(|(state, _)| matches!(state, RunState::Cancelled))
        .flat_map(|(_, children)| children.iter())
        .collect::<Vec<_>>();
    let mut changed = false;
    for child in children {
        if let Some(mut state) = world.get_mut::<RunState>(child)
            && !matches!(
                *state,
                RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
            )
        {
            *state = RunState::Cancelled;
            changed = true;
        }
    }
    if changed {
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn cleanup_observed_runs(
    mut commands: Commands,
    runs: Query<(Entity, &RunState, &RunRecord)>,
    operations: Query<(Entity, &OperationOf, &OperationState)>,
    turns: Query<(Entity, &TurnOf)>,
    batches: Query<(Entity, &BatchOf)>,
    subscriptions: Query<(Entity, &SubscriptionOf, &SubscriptionState)>,
    mut progress: ResMut<Progress>,
) {
    for (run_entity, state, record) in &runs {
        if !record.observed
            || !matches!(
                state,
                RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
            )
        {
            continue;
        }
        let run_operations = operations
            .iter()
            .filter(|(_, relation, _)| relation.get() == run_entity)
            .collect::<Vec<_>>();
        if run_operations.iter().any(|(_, _, operation)| {
            matches!(
                operation.phase,
                OperationPhase::Prepared | OperationPhase::InFlight
            )
        }) {
            continue;
        }
        let run_subscriptions = subscriptions
            .iter()
            .filter(|(_, relation, _)| relation.get() == run_entity)
            .collect::<Vec<_>>();
        if run_subscriptions
            .iter()
            .any(|(_, _, state)| matches!(state, SubscriptionState::Active))
        {
            continue;
        }
        for (entity, _, _) in run_operations {
            commands.entity(entity).despawn();
        }
        for (entity, relation) in &turns {
            if relation.get() == run_entity {
                commands.entity(entity).despawn();
            }
        }
        for (entity, relation) in &batches {
            if relation.get() == run_entity {
                commands.entity(entity).despawn();
            }
        }
        for (entity, _, _) in run_subscriptions {
            commands.entity(entity).despawn();
        }
        commands.entity(run_entity).despawn();
        mark_progress(&mut progress);
    }
}

fn cleanup_retired_tools(
    mut commands: Commands,
    retired_tools: Query<Entity, (With<ToolCapability>, With<RetiredCapability>)>,
    model_inputs: Query<&ModelEffectInput>,
    tool_inputs: Query<&ToolEffectInput>,
    grants: Query<(Entity, &GrantForTool)>,
    mut progress: ResMut<Progress>,
) {
    for tool in &retired_tools {
        let retained_by_model = model_inputs.iter().any(|input| {
            input
                .tools
                .iter()
                .any(|decision| decision.tool_entity == tool)
        });
        let retained_by_call = tool_inputs
            .iter()
            .any(|input| input.decision.tool_entity == tool);
        if retained_by_model || retained_by_call {
            continue;
        }
        for (grant, relation) in &grants {
            if relation.get() == tool {
                commands.entity(grant).despawn();
            }
        }
        commands.entity(tool).despawn();
        mark_progress(&mut progress);
    }
}

fn update_runtime_metrics(
    runs: Query<(&RunState, &RunRecord)>,
    operations: Query<&OperationState>,
    tools: Query<(&ToolCapability, Option<&RetiredCapability>)>,
    subscriptions: Query<&SubscriptionState>,
    mut metrics: ResMut<RuntimeMetrics>,
) {
    let mut next = RuntimeMetrics::default();
    for (run, record) in &runs {
        match run {
            RunState::Queued
            | RunState::WaitingModel { .. }
            | RunState::WaitingTools { .. }
            | RunState::WaitingStore { .. } => next.active_runs += 1,
            RunState::Completed(_) => next.completed_runs += 1,
            RunState::Failed(_) => next.failed_runs += 1,
            RunState::Cancelled => next.cancelled_runs += 1,
        }
        if !record.observed
            && matches!(
                run,
                RunState::Completed(_) | RunState::Failed(_) | RunState::Cancelled
            )
        {
            next.unobserved_terminal_runs += 1;
        }
    }
    for operation in &operations {
        match operation.phase {
            OperationPhase::Prepared => next.prepared_operations += 1,
            OperationPhase::InFlight => next.in_flight_operations += 1,
            OperationPhase::Settled(_) | OperationPhase::ExtensionSettled => {
                next.settled_operations += 1;
            }
            OperationPhase::Cancelled | OperationPhase::Superseded => {}
        }
    }
    next.retired_tools = tools
        .iter()
        .filter(|(tool, retired)| tool.retired || retired.is_some())
        .count();
    next.dropped_subscribers = subscriptions
        .iter()
        .filter(|state| matches!(state, SubscriptionState::DroppedSlowConsumer))
        .count();
    *metrics = next;
}

#[allow(clippy::type_complexity)]
fn initialize_completion_response_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &ModelEffectInput, &OperationState),
        Without<CompletionResponsePolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunPolicies>, Option<&RunControl>), Without<WaitingForChildren>>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(
        Entity,
        &StableId,
        &PolicyMeta,
        &PolicyCapabilities,
        Option<&PolicyStatus>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, request, state) in &operations {
        let OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Model(output))) =
            &state.phase
        else {
            continue;
        };
        let Ok((run_of, run_policies, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = policy_entities(agents.get(run_of.get()).ok().flatten(), run_policies)
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, _, capabilities, status)| {
                accepts_new_policy_evaluations(*status)
                    && capabilities.contains(PolicyPoint::CompletionResponse)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _, _), (_, right_id, right, _, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
                order: policy.order,
                point: PolicyPoint::CompletionResponse,
            })
            .collect::<Vec<_>>();
        commands
            .entity(operation)
            .insert(CompletionResponsePolicyInitialized);
        if snapshot.is_empty() {
            commands.entity(operation).insert((
                AcceptedCompletionResponsePolicies::default(),
                EffectiveModelOutput(output.clone()),
                CompletionResponsePolicyDone,
            ));
        } else {
            commands
                .entity(operation)
                .insert(AcceptedCompletionResponsePolicies(snapshot.clone()));
            commands.spawn((
                EvaluationOfOperation(operation),
                CompletionResponsePolicyEvaluation {
                    run: operation_of.get(),
                    policies: snapshot,
                    cursor: 0,
                    request: request.clone(),
                    effective: output.clone(),
                    phase: CompletionResponsePolicyEvaluationPhase::Evaluating,
                },
            ));
        }
        mark_progress(&mut progress);
    }
}

fn evaluate_completion_response_policies(world: &mut World) {
    let mut query = world.query::<(
        Entity,
        &EvaluationOfOperation,
        &CompletionResponsePolicyEvaluation,
    )>();
    let mut ready = query
        .iter(world)
        .filter(|(_, _, evaluation)| {
            matches!(
                evaluation.phase,
                CompletionResponsePolicyEvaluationPhase::Evaluating
            )
        })
        .map(|(entity, operation, evaluation)| {
            (entity, operation.get(), evaluation.run, evaluation.cursor)
        })
        .collect::<Vec<_>>();
    ready.retain(|(_, _, run, _)| world_allows_internal_progress(world, *run));
    ready.sort_by(|left, right| {
        let left_id = world.get::<StableId>(left.2);
        let right_id = world.get::<StableId>(right.2);
        left_id
            .cmp(&right_id)
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.0.to_bits().cmp(&right.0.to_bits()))
    });

    for (evaluation_entity, operation, run, cursor) in ready {
        let Some(evaluation) = world
            .get::<CompletionResponsePolicyEvaluation>(evaluation_entity)
            .cloned()
        else {
            continue;
        };
        let Some(policy) = evaluation.policies.get(cursor).cloned() else {
            world.entity_mut(operation).insert((
                EffectiveModelOutput(evaluation.effective),
                CompletionResponsePolicyDone,
            ));
            if let Some(mut evaluation) =
                world.get_mut::<CompletionResponsePolicyEvaluation>(evaluation_entity)
            {
                evaluation.phase = CompletionResponsePolicyEvaluationPhase::Accepted;
            }
            mark_progress(&mut world.resource_mut::<Progress>());
            continue;
        };
        let invocation = CompletionResponsePolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            request: evaluation.request,
            response: evaluation.effective,
        };
        let decision =
            run_policy_responder(world, invocation, policy.entity, &policy.id, policy.point);
        match decision {
            Some(CompletionResponsePolicyDecision::Continue) => {
                if let Some(mut evaluation) =
                    world.get_mut::<CompletionResponsePolicyEvaluation>(evaluation_entity)
                {
                    evaluation.cursor += 1;
                }
            }
            Some(CompletionResponsePolicyDecision::RewriteText(text)) => {
                if let Some(mut evaluation) =
                    world.get_mut::<CompletionResponsePolicyEvaluation>(evaluation_entity)
                {
                    evaluation.effective.text = text;
                    evaluation.cursor += 1;
                }
            }
            Some(CompletionResponsePolicyDecision::AwaitApproval(prompt)) => {
                let approval = begin_policy_approval(
                    world,
                    evaluation_entity,
                    run,
                    &policy,
                    PolicyPoint::CompletionResponse,
                    prompt,
                );
                if let Some(mut evaluation) =
                    world.get_mut::<CompletionResponsePolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = CompletionResponsePolicyEvaluationPhase::WaitingApproval {
                        operation: approval,
                        policy: policy.id.clone(),
                    };
                }
            }
            decision @ (Some(CompletionResponsePolicyDecision::Stop(_)) | None) => {
                let reason = match decision {
                    Some(CompletionResponsePolicyDecision::Stop(reason)) => reason,
                    _ => "policy did not provide a steering decision".to_owned(),
                };
                if let Some(mut evaluation) =
                    world.get_mut::<CompletionResponsePolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = CompletionResponsePolicyEvaluationPhase::Rejected {
                        policy: policy.id.clone(),
                        reason: reason.clone(),
                    };
                }
                let error = policy_termination_error(
                    world,
                    run,
                    operation,
                    policy.id.clone(),
                    policy.revision,
                    PolicyPoint::CompletionResponse,
                    reason,
                );
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(error);
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn publish_evaluated_text_delta(world: &mut World, run: Entity, sequence: u64, text: &str) {
    let subscriptions = world
        .get::<RunSubscriptions>(run)
        .map(|subscriptions| subscriptions.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    for subscription in subscriptions {
        if !matches!(
            world.get::<SubscriptionState>(subscription),
            Some(SubscriptionState::Active)
        ) {
            continue;
        }
        let sender = world
            .get::<StreamSink>(subscription)
            .map(|sink| sink.0.clone());
        let Some(sender) = sender else {
            continue;
        };
        if sender
            .try_send(StreamItem::Delta {
                sequence,
                text: text.to_owned(),
            })
            .is_err()
        {
            if let Some(mut state) = world.get_mut::<SubscriptionState>(subscription) {
                *state = SubscriptionState::DroppedSlowConsumer;
            }
            world.entity_mut(subscription).remove::<StreamSink>();
        }
    }
}

fn evaluate_text_delta_policies(world: &mut World) {
    let mut query = world.query::<(Entity, &TextDeltaPolicyEvaluation)>();
    let mut blocked_operations = query
        .iter(world)
        .filter_map(|(_, evaluation)| {
            matches!(
                evaluation.phase,
                TextDeltaPolicyEvaluationPhase::WaitingApproval { .. }
            )
            .then_some(evaluation.operation)
        })
        .collect::<HashSet<_>>();
    let mut ready = query
        .iter(world)
        .filter(|(_, evaluation)| {
            matches!(evaluation.phase, TextDeltaPolicyEvaluationPhase::Evaluating)
        })
        .map(|(entity, evaluation)| {
            (
                entity,
                evaluation.run,
                evaluation.operation,
                evaluation.sequence,
                evaluation.cursor,
            )
        })
        .collect::<Vec<_>>();
    ready.retain(|(_, run, _, _, _)| world_allows_internal_progress(world, *run));
    ready.sort_by(|left, right| {
        let left_id = world.get::<StableId>(left.1);
        let right_id = world.get::<StableId>(right.1);
        left_id
            .cmp(&right_id)
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.4.cmp(&right.4))
            .then_with(|| left.0.to_bits().cmp(&right.0.to_bits()))
    });
    for (evaluation_entity, run, operation, _, cursor) in ready {
        if blocked_operations.contains(&operation) {
            continue;
        }
        let Some(evaluation) = world
            .get::<TextDeltaPolicyEvaluation>(evaluation_entity)
            .cloned()
        else {
            continue;
        };
        let Some(policy) = evaluation.policies.get(cursor).cloned() else {
            publish_evaluated_text_delta(world, run, evaluation.sequence, &evaluation.delta);
            if let Some(mut state) = world.get_mut::<TextDeltaPolicyEvaluation>(evaluation_entity) {
                state.phase = TextDeltaPolicyEvaluationPhase::Published;
            }
            mark_progress(&mut world.resource_mut::<Progress>());
            continue;
        };
        let invocation = TextDeltaPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation: evaluation.operation,
            turn: evaluation.turn,
            sequence: evaluation.sequence,
            delta: evaluation.delta.clone(),
            aggregated: evaluation.aggregated.clone(),
        };
        let decision =
            run_policy_responder(world, invocation, policy.entity, &policy.id, policy.point);
        match decision {
            Some(TextDeltaPolicyDecision::Continue) => {
                if cursor + 1 == evaluation.policies.len() {
                    publish_evaluated_text_delta(
                        world,
                        run,
                        evaluation.sequence,
                        &evaluation.delta,
                    );
                    if let Some(mut state) =
                        world.get_mut::<TextDeltaPolicyEvaluation>(evaluation_entity)
                    {
                        state.cursor += 1;
                        state.phase = TextDeltaPolicyEvaluationPhase::Published;
                    }
                } else if let Some(mut state) =
                    world.get_mut::<TextDeltaPolicyEvaluation>(evaluation_entity)
                {
                    state.cursor += 1;
                }
            }
            Some(TextDeltaPolicyDecision::AwaitApproval(prompt)) => {
                let approval = begin_policy_approval(
                    world,
                    evaluation_entity,
                    run,
                    &policy,
                    PolicyPoint::TextDelta,
                    prompt,
                );
                if let Some(mut state) =
                    world.get_mut::<TextDeltaPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = TextDeltaPolicyEvaluationPhase::WaitingApproval {
                        operation: approval,
                        policy: policy.id.clone(),
                    };
                }
                blocked_operations.insert(operation);
            }
            decision @ (Some(TextDeltaPolicyDecision::Stop(_)) | None) => {
                let reason = match decision {
                    Some(TextDeltaPolicyDecision::Stop(reason)) => reason,
                    _ => "policy did not provide a steering decision".to_owned(),
                };
                if let Some(mut state) =
                    world.get_mut::<TextDeltaPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = TextDeltaPolicyEvaluationPhase::Rejected {
                        policy: policy.id.clone(),
                        reason: reason.clone(),
                    };
                }
                let error = policy_termination_error(
                    world,
                    run,
                    evaluation.operation,
                    policy.id.clone(),
                    policy.revision,
                    PolicyPoint::TextDelta,
                    reason,
                );
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(error);
                }
                blocked_operations.insert(operation);
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn publish_evaluated_tool_call_delta(
    world: &mut World,
    run: Entity,
    sequence: u64,
    id: &str,
    internal_call_id: &str,
    content: &ToolCallDeltaContent,
) {
    let subscriptions = world
        .get::<RunSubscriptions>(run)
        .map(|subscriptions| subscriptions.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    for subscription in subscriptions {
        if !matches!(
            world.get::<SubscriptionState>(subscription),
            Some(SubscriptionState::Active)
        ) {
            continue;
        }
        let sender = world
            .get::<StreamSink>(subscription)
            .map(|sink| sink.0.clone());
        let Some(sender) = sender else {
            continue;
        };
        if sender
            .try_send(StreamItem::ToolCallDelta {
                sequence,
                id: id.to_owned(),
                internal_call_id: internal_call_id.to_owned(),
                content: content.clone(),
            })
            .is_err()
        {
            if let Some(mut state) = world.get_mut::<SubscriptionState>(subscription) {
                *state = SubscriptionState::DroppedSlowConsumer;
            }
            world.entity_mut(subscription).remove::<StreamSink>();
        }
    }
}

fn evaluate_tool_call_delta_policies(world: &mut World) {
    let mut query = world.query::<(Entity, &ToolCallDeltaPolicyEvaluation)>();
    let mut blocked_operations = query
        .iter(world)
        .filter_map(|(_, evaluation)| {
            matches!(
                evaluation.phase,
                ToolCallDeltaPolicyEvaluationPhase::WaitingApproval { .. }
            )
            .then_some(evaluation.operation)
        })
        .collect::<HashSet<_>>();
    let mut ready = query
        .iter(world)
        .filter(|(_, evaluation)| {
            matches!(
                evaluation.phase,
                ToolCallDeltaPolicyEvaluationPhase::Evaluating
            )
        })
        .map(|(entity, evaluation)| {
            (
                entity,
                evaluation.run,
                evaluation.operation,
                evaluation.sequence,
                evaluation.cursor,
            )
        })
        .collect::<Vec<_>>();
    ready.retain(|(_, run, _, _, _)| world_allows_internal_progress(world, *run));
    ready.sort_by(|left, right| {
        let left_id = world.get::<StableId>(left.1);
        let right_id = world.get::<StableId>(right.1);
        left_id
            .cmp(&right_id)
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.4.cmp(&right.4))
            .then_with(|| left.0.to_bits().cmp(&right.0.to_bits()))
    });
    for (evaluation_entity, run, operation, _, cursor) in ready {
        if blocked_operations.contains(&operation) {
            continue;
        }
        let Some(evaluation) = world
            .get::<ToolCallDeltaPolicyEvaluation>(evaluation_entity)
            .cloned()
        else {
            continue;
        };
        let Some(policy) = evaluation.policies.get(cursor).cloned() else {
            publish_evaluated_tool_call_delta(
                world,
                run,
                evaluation.sequence,
                &evaluation.id,
                &evaluation.internal_call_id,
                &evaluation.content,
            );
            if let Some(mut state) =
                world.get_mut::<ToolCallDeltaPolicyEvaluation>(evaluation_entity)
            {
                state.phase = ToolCallDeltaPolicyEvaluationPhase::Published;
            }
            mark_progress(&mut world.resource_mut::<Progress>());
            continue;
        };
        let invocation = ToolCallDeltaPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation: evaluation.operation,
            turn: evaluation.turn,
            sequence: evaluation.sequence,
            provider_correlation: evaluation.provider_correlation.clone(),
            id: evaluation.id.clone(),
            internal_call_id: evaluation.internal_call_id.clone(),
            content: evaluation.content.clone(),
        };
        let decision =
            run_policy_responder(world, invocation, policy.entity, &policy.id, policy.point);
        match decision {
            Some(ToolCallDeltaPolicyDecision::Continue) => {
                if cursor + 1 == evaluation.policies.len() {
                    publish_evaluated_tool_call_delta(
                        world,
                        run,
                        evaluation.sequence,
                        &evaluation.id,
                        &evaluation.internal_call_id,
                        &evaluation.content,
                    );
                    if let Some(mut state) =
                        world.get_mut::<ToolCallDeltaPolicyEvaluation>(evaluation_entity)
                    {
                        state.cursor += 1;
                        state.phase = ToolCallDeltaPolicyEvaluationPhase::Published;
                    }
                } else if let Some(mut state) =
                    world.get_mut::<ToolCallDeltaPolicyEvaluation>(evaluation_entity)
                {
                    state.cursor += 1;
                }
            }
            Some(ToolCallDeltaPolicyDecision::AwaitApproval(prompt)) => {
                let approval = begin_policy_approval(
                    world,
                    evaluation_entity,
                    run,
                    &policy,
                    PolicyPoint::ToolCallDelta,
                    prompt,
                );
                if let Some(mut state) =
                    world.get_mut::<ToolCallDeltaPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = ToolCallDeltaPolicyEvaluationPhase::WaitingApproval {
                        operation: approval,
                        policy: policy.id.clone(),
                    };
                }
                blocked_operations.insert(operation);
            }
            decision @ (Some(ToolCallDeltaPolicyDecision::Stop(_)) | None) => {
                let reason = match decision {
                    Some(ToolCallDeltaPolicyDecision::Stop(reason)) => reason,
                    _ => "policy did not provide a steering decision".to_owned(),
                };
                if let Some(mut state) =
                    world.get_mut::<ToolCallDeltaPolicyEvaluation>(evaluation_entity)
                {
                    state.phase = ToolCallDeltaPolicyEvaluationPhase::Rejected {
                        policy: policy.id.clone(),
                        reason: reason.clone(),
                    };
                }
                let error = policy_termination_error(
                    world,
                    run,
                    evaluation.operation,
                    policy.id.clone(),
                    policy.revision,
                    PolicyPoint::ToolCallDelta,
                    reason,
                );
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(error);
                }
                blocked_operations.insert(operation);
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

#[allow(clippy::type_complexity)]
fn initialize_tool_result_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &ToolEffectInput, &OperationState),
        Without<ToolResultPolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunPolicies>, Option<&RunControl>), Without<WaitingForChildren>>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(
        Entity,
        &StableId,
        &PolicyMeta,
        &PolicyCapabilities,
        Option<&PolicyStatus>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, input, state) in &operations {
        let OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Tool(output))) =
            &state.phase
        else {
            continue;
        };
        let Ok((run_of, run_policies, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = policy_entities(agents.get(run_of.get()).ok().flatten(), run_policies)
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, _, capabilities, status)| {
                accepts_new_policy_evaluations(*status)
                    && capabilities.contains(PolicyPoint::ToolResult)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _, _), (_, right_id, right, _, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
                order: policy.order,
                point: PolicyPoint::ToolResult,
            })
            .collect::<Vec<_>>();
        commands
            .entity(operation)
            .insert(ToolResultPolicyInitialized);
        if snapshot.is_empty() {
            commands.entity(operation).insert((
                AcceptedToolResultPolicies::default(),
                EffectiveToolOutput(output.clone()),
                ToolResultPolicyDone,
            ));
        } else {
            commands
                .entity(operation)
                .insert(AcceptedToolResultPolicies(snapshot.clone()));
            commands.spawn((
                EvaluationOfOperation(operation),
                ToolResultPolicyEvaluation {
                    run: operation_of.get(),
                    policies: snapshot,
                    cursor: 0,
                    input: input.clone(),
                    effective: output.clone(),
                    phase: ToolResultPolicyEvaluationPhase::Evaluating,
                },
            ));
        }
        mark_progress(&mut progress);
    }
}

fn evaluate_tool_result_policies(world: &mut World) {
    let mut query = world.query::<(Entity, &EvaluationOfOperation, &ToolResultPolicyEvaluation)>();
    let mut ready = query
        .iter(world)
        .filter(|(_, _, evaluation)| {
            matches!(
                evaluation.phase,
                ToolResultPolicyEvaluationPhase::Evaluating
            )
        })
        .map(|(entity, operation, evaluation)| {
            (
                entity,
                operation.get(),
                evaluation.run,
                evaluation.cursor,
                evaluation.input.index,
            )
        })
        .collect::<Vec<_>>();
    ready.retain(|(_, _, run, _, _)| world_allows_internal_progress(world, *run));
    ready.sort_by(|left, right| {
        let left_id = world.get::<StableId>(left.2);
        let right_id = world.get::<StableId>(right.2);
        left_id
            .cmp(&right_id)
            .then_with(|| left.4.cmp(&right.4))
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.0.to_bits().cmp(&right.0.to_bits()))
    });

    for (evaluation_entity, operation, run, cursor, _) in ready {
        let Some(evaluation) = world
            .get::<ToolResultPolicyEvaluation>(evaluation_entity)
            .cloned()
        else {
            continue;
        };
        let Some(policy) = evaluation.policies.get(cursor).cloned() else {
            world.entity_mut(operation).insert((
                EffectiveToolOutput(evaluation.effective),
                ToolResultPolicyDone,
            ));
            if let Some(mut evaluation) =
                world.get_mut::<ToolResultPolicyEvaluation>(evaluation_entity)
            {
                evaluation.phase = ToolResultPolicyEvaluationPhase::Accepted;
            }
            mark_progress(&mut world.resource_mut::<Progress>());
            continue;
        };
        let invocation = ToolResultPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            input: evaluation.input,
            result: evaluation.effective,
        };
        let decision =
            run_policy_responder(world, invocation, policy.entity, &policy.id, policy.point);
        match decision {
            Some(ToolResultPolicyDecision::Keep) => {
                if let Some(mut evaluation) =
                    world.get_mut::<ToolResultPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.cursor += 1;
                }
            }
            Some(ToolResultPolicyDecision::Rewrite(presentation)) => {
                if let Some(mut evaluation) =
                    world.get_mut::<ToolResultPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.effective.presentation = presentation;
                    evaluation.cursor += 1;
                }
            }
            Some(ToolResultPolicyDecision::AwaitApproval(prompt)) => {
                let approval = begin_policy_approval(
                    world,
                    evaluation_entity,
                    run,
                    &policy,
                    PolicyPoint::ToolResult,
                    prompt,
                );
                if let Some(mut evaluation) =
                    world.get_mut::<ToolResultPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = ToolResultPolicyEvaluationPhase::WaitingApproval {
                        operation: approval,
                        policy: policy.id.clone(),
                    };
                }
            }
            decision @ (Some(ToolResultPolicyDecision::Stop(_)) | None) => {
                let reason = match decision {
                    Some(ToolResultPolicyDecision::Stop(reason)) => reason,
                    _ => "policy did not provide a steering decision".to_owned(),
                };
                if let Some(mut evaluation) =
                    world.get_mut::<ToolResultPolicyEvaluation>(evaluation_entity)
                {
                    evaluation.phase = ToolResultPolicyEvaluationPhase::Rejected {
                        policy: policy.id.clone(),
                        reason: reason.clone(),
                    };
                }
                let error = policy_termination_error(
                    world,
                    run,
                    operation,
                    policy.id.clone(),
                    policy.revision,
                    PolicyPoint::ToolResult,
                    reason,
                );
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(error);
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn apply_policy_approval_results(world: &mut World) {
    let mut query = world.query_filtered::<(
        Entity,
        &ApprovalForEvaluation,
        &OperationOf,
        &PolicyApprovalEffectInput,
        &OperationState,
    ), Without<PolicyApprovalApplied>>();
    let settled = query
        .iter(world)
        .filter_map(|(operation, evaluation, run, input, state)| {
            let OperationPhase::Settled(outcome) = &state.phase else {
                return None;
            };
            Some((
                operation,
                evaluation.get(),
                run.get(),
                input.clone(),
                outcome.clone(),
            ))
        })
        .collect::<Vec<_>>();

    for (operation, evaluation_entity, run, input, outcome) in settled {
        let approved = matches!(
            &outcome,
            OperationOutcome::Success(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                ..
            }))
        );
        let rejection_reason = match &outcome {
            OperationOutcome::Success(EffectOutput::PolicyApproval(output)) => output
                .reason
                .clone()
                .unwrap_or_else(|| "approval denied".to_owned()),
            OperationOutcome::Failure(error) => error.to_string(),
            OperationOutcome::Success(_) => "approval effect kind mismatch".to_owned(),
        };
        let mut matched = false;

        if let Some(mut evaluation) = world.get_mut::<RequestPolicyEvaluation>(evaluation_entity)
            && matches!(
                evaluation.phase,
                RequestPolicyEvaluationPhase::WaitingApproval {
                    operation: waiting,
                    ..
                } if waiting == operation
            )
        {
            matched = true;
            if approved {
                evaluation.cursor += 1;
                evaluation.phase = RequestPolicyEvaluationPhase::Evaluating;
            } else {
                evaluation.phase = RequestPolicyEvaluationPhase::Rejected {
                    policy: input.policy_id.clone(),
                    reason: rejection_reason.clone(),
                };
            }
        }
        if let Some(mut evaluation) = world.get_mut::<ToolCallPolicyEvaluation>(evaluation_entity)
            && matches!(
                evaluation.phase,
                ToolCallPolicyEvaluationPhase::WaitingApproval {
                    operation: waiting,
                    ..
                } if waiting == operation
            )
        {
            matched = true;
            if approved {
                evaluation.cursor += 1;
                evaluation.phase = ToolCallPolicyEvaluationPhase::Evaluating;
            } else {
                evaluation.phase = ToolCallPolicyEvaluationPhase::Rejected {
                    policy: input.policy_id.clone(),
                    reason: rejection_reason.clone(),
                };
            }
        }
        if let Some(mut evaluation) =
            world.get_mut::<InvalidToolCallPolicyEvaluation>(evaluation_entity)
            && matches!(
                evaluation.phase,
                InvalidToolCallPolicyEvaluationPhase::WaitingApproval {
                    operation: waiting,
                    ..
                } if waiting == operation
            )
        {
            matched = true;
            if approved {
                evaluation.cursor += 1;
                evaluation.phase = InvalidToolCallPolicyEvaluationPhase::Evaluating;
            } else {
                evaluation.phase = InvalidToolCallPolicyEvaluationPhase::Rejected {
                    policy: input.policy_id.clone(),
                    reason: rejection_reason.clone(),
                };
            }
        }
        if let Some(mut evaluation) = world.get_mut::<ToolResultPolicyEvaluation>(evaluation_entity)
            && matches!(
                evaluation.phase,
                ToolResultPolicyEvaluationPhase::WaitingApproval {
                    operation: waiting,
                    ..
                } if waiting == operation
            )
        {
            matched = true;
            if approved {
                evaluation.cursor += 1;
                evaluation.phase = ToolResultPolicyEvaluationPhase::Evaluating;
            } else {
                evaluation.phase = ToolResultPolicyEvaluationPhase::Rejected {
                    policy: input.policy_id.clone(),
                    reason: rejection_reason.clone(),
                };
            }
        }
        if let Some(mut evaluation) =
            world.get_mut::<CompletionResponsePolicyEvaluation>(evaluation_entity)
            && matches!(
                evaluation.phase,
                CompletionResponsePolicyEvaluationPhase::WaitingApproval {
                    operation: waiting,
                    ..
                } if waiting == operation
            )
        {
            matched = true;
            if approved {
                evaluation.cursor += 1;
                evaluation.phase = CompletionResponsePolicyEvaluationPhase::Evaluating;
            } else {
                evaluation.phase = CompletionResponsePolicyEvaluationPhase::Rejected {
                    policy: input.policy_id.clone(),
                    reason: rejection_reason.clone(),
                };
            }
        }
        if let Some(mut evaluation) = world.get_mut::<TextDeltaPolicyEvaluation>(evaluation_entity)
            && matches!(
                evaluation.phase,
                TextDeltaPolicyEvaluationPhase::WaitingApproval {
                    operation: waiting,
                    ..
                } if waiting == operation
            )
        {
            matched = true;
            if approved {
                evaluation.cursor += 1;
                evaluation.phase = TextDeltaPolicyEvaluationPhase::Evaluating;
            } else {
                evaluation.phase = TextDeltaPolicyEvaluationPhase::Rejected {
                    policy: input.policy_id.clone(),
                    reason: rejection_reason.clone(),
                };
            }
        }
        if let Some(mut evaluation) =
            world.get_mut::<ToolCallDeltaPolicyEvaluation>(evaluation_entity)
            && matches!(
                evaluation.phase,
                ToolCallDeltaPolicyEvaluationPhase::WaitingApproval {
                    operation: waiting,
                    ..
                } if waiting == operation
            )
        {
            matched = true;
            if approved {
                evaluation.cursor += 1;
                evaluation.phase = ToolCallDeltaPolicyEvaluationPhase::Evaluating;
            } else {
                evaluation.phase = ToolCallDeltaPolicyEvaluationPhase::Rejected {
                    policy: input.policy_id.clone(),
                    reason: rejection_reason.clone(),
                };
            }
        }

        if matched && !approved {
            let evaluated_operation = world
                .get::<EvaluationOfOperation>(evaluation_entity)
                .map(Relationship::get)
                .unwrap_or(operation);
            let error = policy_termination_error(
                world,
                run,
                evaluated_operation,
                input.policy_id.clone(),
                input.revision,
                input.point,
                rejection_reason,
            );
            if let Some(mut state) = world.get_mut::<RunState>(run) {
                *state = RunState::Failed(error);
            }
        }
        world.entity_mut(operation).insert(PolicyApprovalApplied);
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn commit_store_operations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &StoreEffectInput, &OperationState),
        With<StoreEffectInput>,
    >,
    mut runs: Query<
        (&mut RunState, &mut RunRecord, Option<&RunControl>),
        Without<WaitingForChildren>,
    >,
    mut progress: ResMut<Progress>,
) {
    for (operation_entity, operation_of, input, operation) in &operations {
        let OperationPhase::Settled(outcome) = &operation.phase else {
            continue;
        };
        let Ok((mut run_state, mut record, control)) = runs.get_mut(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        if !matches!(
            *run_state,
            RunState::WaitingStore { operation } if operation == operation_entity
        ) {
            continue;
        }
        match (&input.operation, outcome) {
            (
                StoreOperation::LoadConversation { .. },
                OperationOutcome::Success(EffectOutput::Store(
                    StoreEffectOutput::LoadedConversation(history),
                )),
            ) => {
                let mut transcript = history.clone();
                transcript.append(&mut record.transcript);
                record.transcript = transcript;
                record.memory_loaded = true;
                *run_state = RunState::Queued;
            }
            (
                StoreOperation::Retrieve { limit, .. },
                OperationOutcome::Success(EffectOutput::Store(StoreEffectOutput::Retrieved(
                    documents,
                ))),
            ) => {
                if documents.len() > *limit {
                    *run_state = RunState::Failed(CanonicalError::Store {
                        message: format!(
                            "retrieval returned {} documents for limit {limit}",
                            documents.len()
                        ),
                        retryable: false,
                    });
                } else if input.decision.kind == "tool-vector-search" {
                    record.retrieved_tool_names = documents
                        .iter()
                        .map(|document| document.id.clone())
                        .collect();
                    record.tool_retrieval_loaded = true;
                    *run_state = RunState::Queued;
                } else {
                    record.retrieved_documents = documents.clone();
                    record.retrieval_loaded = true;
                    *run_state = RunState::Queued;
                }
            }
            (
                StoreOperation::PersistConversation { .. },
                OperationOutcome::Success(EffectOutput::Store(StoreEffectOutput::Persisted)),
            ) => {
                let Some(output) = record.pending_output.take() else {
                    *run_state = RunState::Failed(CanonicalError::EffectKindMismatch);
                    mark_progress(&mut progress);
                    continue;
                };
                *run_state = RunState::Completed(output);
            }
            (_, OperationOutcome::Failure(error)) => {
                *run_state = RunState::Failed(error.clone());
            }
            _ => {
                *run_state = RunState::Failed(CanonicalError::EffectKindMismatch);
            }
        }
        commands.trigger(PersistenceSettled {
            operation: operation_entity,
            run: operation_of.get(),
            outcome: outcome.clone(),
        });
        mark_progress(&mut progress);
    }
}

#[allow(clippy::type_complexity)]
fn commit_model_operations(
    mut commands: Commands,
    operations: Query<
        (
            Entity,
            &OperationOf,
            &OperationState,
            &ModelEffectInput,
            Option<&ModelStreamState>,
            Option<&EffectiveModelOutput>,
            Option<&CompletionResponsePolicyDone>,
            Option<&AcceptedPolicies>,
            Option<&AcceptedCompletionResponsePolicies>,
        ),
        With<ModelEffectInput>,
    >,
    mut runs: Query<
        (&RunOf, &mut RunState, &mut RunRecord, Option<&RunControl>),
        Without<WaitingForChildren>,
    >,
    agents: Query<&Agent>,
    mut progress: ResMut<Progress>,
) {
    for (
        operation_entity,
        operation_of,
        operation,
        input,
        stream_state,
        effective_output,
        response_policy_done,
        request_policies,
        response_policies,
    ) in &operations
    {
        let OperationPhase::Settled(outcome) = &operation.phase else {
            continue;
        };
        let Ok((run_of, mut run_state, mut record, control)) = runs.get_mut(operation_of.get())
        else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let RunState::WaitingModel {
            operation: waiting_operation,
        } = *run_state
        else {
            continue;
        };
        if waiting_operation != operation_entity {
            continue;
        }
        match outcome {
            OperationOutcome::Success(EffectOutput::Model(output)) => {
                if response_policy_done.is_none() {
                    continue;
                }
                let output = effective_output.map_or(output, |effective| &effective.0);
                let mut call_ids = HashSet::with_capacity(output.tool_calls.len());
                if let Some(duplicate) = output
                    .tool_calls
                    .iter()
                    .find(|call| !call_ids.insert(call.id.as_str()))
                {
                    *run_state =
                        RunState::Failed(CanonicalError::DuplicateToolCall(duplicate.id.clone()));
                    mark_progress(&mut progress);
                    continue;
                }
                record.usage.input_tokens = record
                    .usage
                    .input_tokens
                    .saturating_add(output.usage.input_tokens);
                record.usage.output_tokens = record
                    .usage
                    .output_tokens
                    .saturating_add(output.usage.output_tokens);
                if let Some(message) = &output.assistant_message {
                    record
                        .transcript
                        .push(TranscriptEntry::AssistantMessage(message.clone()));
                } else {
                    record
                        .transcript
                        .push(TranscriptEntry::Assistant(output.text.clone()));
                    record
                        .transcript
                        .extend(output.tool_calls.iter().map(|call| {
                            TranscriptEntry::AssistantToolCall {
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            }
                        }));
                }
                let committed_turn = record.next_turn;
                commands.spawn((
                    TurnOf(operation_of.get()),
                    CommittedTurn {
                        index: committed_turn,
                        output: output.clone(),
                    },
                ));
                commands.trigger(ModelTurnFinished {
                    run: operation_of.get(),
                    operation: operation_entity,
                    turn: committed_turn,
                    output: output.clone(),
                    request_policies: request_policies
                        .map_or_else(Vec::new, |policies| policies.0.clone()),
                    response_policies: response_policies
                        .map_or_else(Vec::new, |policies| policies.0.clone()),
                });
                record.next_turn = record.next_turn.saturating_add(1);
                let terminal_text = if output.tool_calls.is_empty() {
                    Some((
                        if input
                            .terminal_tool
                            .as_ref()
                            .is_some_and(|name| !name.is_empty())
                        {
                            String::new()
                        } else {
                            output.text.clone()
                        },
                        input
                            .terminal_tool
                            .as_ref()
                            .is_none_or(|name| name.is_empty()),
                    ))
                } else {
                    input.terminal_tool.as_ref().and_then(|terminal_tool| {
                        output
                            .tool_calls
                            .iter()
                            .find(|call| &call.name == terminal_tool)
                            .map(|call| (call.arguments.to_string(), true))
                    })
                };
                if let Some((terminal_text, validate_output)) = terminal_text {
                    if validate_output
                        && let Some(schema) = &input.output_schema
                        && let Err(error) = validate_structured_output(schema, &terminal_text)
                    {
                        let model_call_limit = record.max_model_calls.unwrap_or_else(|| {
                            agents
                                .get(run_of.get())
                                .map_or(u32::MAX, |agent| agent.max_model_calls)
                        });
                        if record.structured_output_retries < record.max_structured_output_retries
                            && record.next_turn < model_call_limit
                        {
                            record.structured_output_retries =
                                record.structured_output_retries.saturating_add(1);
                            record.transcript.push(TranscriptEntry::User(format!(
                                "The previous response did not satisfy the required output schema: {error}. Return a corrected structured response that satisfies every schema constraint."
                            )));
                            *run_state = RunState::Queued;
                        } else {
                            *run_state = RunState::Failed(error);
                        }
                        mark_progress(&mut progress);
                        continue;
                    }
                    let final_output = RunOutput {
                        text: terminal_text,
                        usage: record.usage,
                    };
                    if let (Some(conversation), Some(decision)) =
                        (record.conversation.clone(), record.memory_store.clone())
                    {
                        let operation = commands
                            .spawn((
                                OperationOf(operation_of.get()),
                                OperationKind::Store,
                                OperationGeneration(0),
                                OperationState {
                                    phase: OperationPhase::Prepared,
                                },
                                StoreEffectInput {
                                    decision,
                                    operation: StoreOperation::PersistConversation {
                                        conversation,
                                        entries: record.transcript.clone(),
                                    },
                                },
                            ))
                            .id();
                        record.pending_output = Some(final_output);
                        *run_state = RunState::WaitingStore { operation };
                    } else {
                        *run_state = RunState::Completed(final_output);
                    }
                } else {
                    record.structured_output_retries = 0;
                    let mut prepared = Vec::with_capacity(output.tool_calls.len());
                    for (index, call) in output.tool_calls.iter().enumerate() {
                        let index = u32::try_from(index).unwrap_or(u32::MAX);
                        prepared.push(
                            if let Some(decision) = input.tools.iter().find(|tool| {
                                tool.name == call.name
                                    && tool_choice_allows(input.tool_choice.as_ref(), &tool.name)
                            }) {
                                PreparedToolOperation::Valid(ToolEffectInput {
                                    decision: decision.clone(),
                                    call_id: call.id.clone(),
                                    provider_result_id: call.provider_result_id.clone(),
                                    provider_call_id: call.provider_call_id.clone(),
                                    arguments: call.arguments.clone(),
                                    index,
                                })
                            } else {
                                PreparedToolOperation::Invalid(PendingInvalidToolCall {
                                    call: call.clone(),
                                    available_tools: input.tools.clone(),
                                    index,
                                    source_model_operation: operation_entity,
                                    turn: committed_turn,
                                    tool_choice: input.tool_choice.clone(),
                                    diagnostic_history: record.transcript.clone(),
                                    streaming_origin: stream_state.is_some(),
                                    retry_count: record.invalid_tool_call_retries,
                                    max_retries: record.max_invalid_tool_call_retries,
                                })
                            },
                        );
                    }
                    let expected = u32::try_from(prepared.len()).unwrap_or(u32::MAX);
                    let batch = commands
                        .spawn((
                            BatchOf(operation_of.get()),
                            ToolBatchState {
                                source_model_operation: operation_entity,
                                expected,
                                committed: false,
                            },
                        ))
                        .id();
                    for tool_input in prepared {
                        let mut operation = commands.spawn((
                            OperationOf(operation_of.get()),
                            OperationOfBatch(batch),
                            OperationKind::Tool,
                            OperationGeneration(0),
                            OperationState {
                                phase: OperationPhase::Prepared,
                            },
                        ));
                        match tool_input {
                            PreparedToolOperation::Valid(input) => {
                                operation.insert(PendingToolCall(input));
                            }
                            PreparedToolOperation::Invalid(invalid) => {
                                operation.insert(invalid);
                            }
                        }
                    }
                    *run_state = RunState::WaitingTools { batch };
                }
            }
            OperationOutcome::Success(
                EffectOutput::Tool(_)
                | EffectOutput::Discovery(_)
                | EffectOutput::Store(_)
                | EffectOutput::PolicyApproval(_),
            ) => {
                *run_state = RunState::Failed(CanonicalError::EffectKindMismatch);
            }
            OperationOutcome::Failure(error) => {
                *run_state = RunState::Failed(error.clone());
            }
        }
        mark_progress(&mut progress);
    }
}

fn validate_structured_output(
    schema: &serde_json::Value,
    text: &str,
) -> Result<(), CanonicalError> {
    let value = serde_json::from_str::<serde_json::Value>(text)
        .or_else(|original| {
            text.char_indices()
                .filter(|(_, character)| matches!(character, '{' | '['))
                .find_map(|(index, _)| {
                    serde_json::Deserializer::from_str(&text[index..])
                        .into_iter::<serde_json::Value>()
                        .next()
                        .and_then(Result::ok)
                })
                .ok_or(original)
        })
        .map_err(|error| {
            CanonicalError::InvalidStructuredOutput(format!("response was not JSON: {error}"))
        })?;
    let validator = jsonschema::validator_for(schema).map_err(|error| {
        CanonicalError::InvalidStructuredOutput(format!("invalid output schema: {error}"))
    })?;
    validator.validate(&value).map_err(|error| {
        CanonicalError::InvalidStructuredOutput(format!("schema validation failed: {error}"))
    })
}

fn commit_tool_batches(
    mut commands: Commands,
    mut batches: Query<(Entity, &BatchOf, &BatchOperations, &mut ToolBatchState)>,
    operations: Query<(
        &ToolEffectInput,
        &OperationState,
        Option<&EffectiveToolOutput>,
        Option<&ToolResultPolicyDone>,
    )>,
    agents: Query<&Agent>,
    mut runs: Query<
        (&RunOf, &mut RunState, &mut RunRecord, Option<&RunControl>),
        Without<WaitingForChildren>,
    >,
    mut progress: ResMut<Progress>,
) {
    'batches: for (batch_entity, batch_of, batch_operations, mut batch) in &mut batches {
        if batch.committed
            || usize::try_from(batch.expected).ok() != Some(batch_operations.iter().count())
        {
            continue;
        }
        let mut settled = Vec::with_capacity(batch_operations.iter().count());
        let mut all_settled = true;
        for operation_entity in batch_operations.iter() {
            let Ok((input, state, effective, policy_done)) = operations.get(operation_entity)
            else {
                all_settled = false;
                break;
            };
            let OperationPhase::Settled(outcome) = &state.phase else {
                all_settled = false;
                break;
            };
            if matches!(outcome, OperationOutcome::Success(EffectOutput::Tool(_)))
                && policy_done.is_none()
            {
                all_settled = false;
                break;
            }
            settled.push((input.index, input, outcome, effective));
        }
        if !all_settled {
            continue;
        }
        settled.sort_by_key(|(index, _, _, _)| *index);
        let Ok((run_of, mut run_state, mut record, control)) = runs.get_mut(batch_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        if !matches!(*run_state, RunState::WaitingTools { batch } if batch == batch_entity) {
            continue;
        }
        if let Some((_, _, OperationOutcome::Failure(error), _)) = settled
            .iter()
            .find(|(_, _, outcome, _)| matches!(outcome, OperationOutcome::Failure(_)))
        {
            *run_state = RunState::Failed(error.clone());
            batch.committed = true;
            mark_progress(&mut progress);
            continue;
        }
        let mut results = Vec::with_capacity(settled.len());
        for (_, input, outcome, effective) in settled {
            let OperationOutcome::Success(EffectOutput::Tool(output)) = outcome else {
                *run_state = RunState::Failed(CanonicalError::EffectKindMismatch);
                batch.committed = true;
                mark_progress(&mut progress);
                continue 'batches;
            };
            if output.call_id != input.call_id
                || output.provider_result_id != input.provider_result_id
                || output.provider_call_id != input.provider_call_id
                || output.name != input.decision.name
            {
                *run_state = RunState::Failed(CanonicalError::EffectKindMismatch);
                batch.committed = true;
                mark_progress(&mut progress);
                continue 'batches;
            }
            let presentation_overrides_raw = output.presentation != output.raw
                || effective
                    .is_some_and(|effective| effective.0.presentation != output.presentation);
            results.push((
                effective.map_or_else(|| output.clone(), |effective| effective.0.clone()),
                presentation_overrides_raw,
            ));
        }
        for (result, presentation_overrides_raw) in &results {
            record.transcript.push(TranscriptEntry::ToolResult {
                call_id: result.call_id.clone(),
                provider_result_id: result.provider_result_id.clone(),
                provider_call_id: result.provider_call_id.clone(),
                name: result.name.clone(),
                raw: result.raw.clone(),
                content: result.presentation.clone(),
                presentation_overrides_raw: *presentation_overrides_raw,
            });
        }
        let results = results
            .into_iter()
            .map(|(result, _)| result)
            .collect::<Vec<_>>();
        commands.trigger(ToolBatchCommitted {
            batch: batch_entity,
            run: batch_of.get(),
            results: results.iter().map(published_tool_result).collect(),
        });
        record.pending_tool_results = results;
        let Some(model_call_limit) = agents
            .get(run_of.get())
            .ok()
            .map(|agent| record.max_model_calls.unwrap_or(agent.max_model_calls))
        else {
            *run_state = RunState::Failed(CanonicalError::StaleEntity("agent".to_owned()));
            batch.committed = true;
            mark_progress(&mut progress);
            continue;
        };
        if record.next_turn >= model_call_limit {
            *run_state = RunState::Failed(CanonicalError::ModelCallBudget {
                limit: model_call_limit,
            });
            batch.committed = true;
            mark_progress(&mut progress);
            continue;
        }
        *run_state = RunState::Queued;
        batch.committed = true;
        mark_progress(&mut progress);
    }
}

#[cfg(test)]
mod tests;
