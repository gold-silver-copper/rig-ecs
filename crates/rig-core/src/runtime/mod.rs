//! Bevy ECS-native runtime primitives.
//!
//! [`RigSchedule`] is world-resident and is the single progression engine used
//! by both [`Runtime`] and embedded worlds. External work leaves the world as an
//! owned [`EffectRequest`] and returns through [`EffectCompletion`]; neither
//! type can contain an ECS borrow.

pub mod adapters;
mod snapshot;

pub use snapshot::{
    ActiveRunSnapshot, ActiveRunSnapshotError, RestoredRuns, restore_active_run,
    snapshot_active_run,
};

use std::{
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
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    completion::Message as CompletionMessage, message::UserContent, streaming::ToolCallDeltaContent,
};

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

/// Persistent identity for a domain entity.
#[derive(Component, Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct StableId(String);

impl StableId {
    pub(crate) fn generated(value: String) -> Self {
        debug_assert!(!value.is_empty());
        Self(value)
    }

    /// Creates a stable identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(IdentityError::Empty);
        }
        Ok(Self(value))
    }

    /// Returns the serialized identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Tenant scope participating in runtime authorization decisions.
#[derive(Component, Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TenantId(String);

impl TenantId {
    /// Creates a tenant scope.
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(IdentityError::Empty);
        }
        Ok(Self(value))
    }

    /// Returns the tenant identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Invalid stable or tenant identity.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum IdentityError {
    /// Empty identifiers are not valid persistence keys.
    #[error("identity must not be empty")]
    Empty,
}

/// Agent configuration stored directly on an entity.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Agent {
    /// Human-readable identity used by telemetry and extension queries.
    pub name: Option<String>,
    /// Human-readable purpose used by composition tooling.
    pub description: Option<String>,
    /// Instructions used for newly prepared turns.
    pub instructions: String,
    /// Optional sampling temperature encoded as IEEE-754 bits for exact snapshots.
    pub temperature_bits: Option<u64>,
    /// Optional provider output-token limit.
    pub max_tokens: Option<u64>,
    /// Provider tool-selection behavior.
    pub tool_choice: Option<ModelToolChoice>,
    /// Provider-specific parameters copied into each accepted model operation.
    pub additional_params: Option<serde_json::Value>,
    /// Static context documents supplied to every model operation.
    pub documents: Vec<RetrievedDocument>,
    /// Maximum number of model operations accepted for one run.
    pub max_model_calls: u32,
    /// Tool whose arguments terminate a structured extraction run.
    pub terminal_tool: Option<String>,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            instructions: String::new(),
            temperature_bits: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            documents: Vec::new(),
            max_model_calls: 16,
            terminal_tool: None,
        }
    }
}

/// Provider-independent tool selection accepted as ECS configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ModelToolChoice {
    /// Let the provider decide whether to call a tool.
    Auto,
    /// Do not permit tool calls.
    None,
    /// Require at least one tool call.
    Required,
    /// Restrict selection to the named provider-facing tools.
    Specific(Vec<String>),
}

/// Native structured-output requirement attached to an agent entity.
///
/// Preparation copies this value into each immutable model effect so later
/// mutations affect only future operations.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OutputRequirement {
    /// Provider-facing JSON Schema and the schema used for final commit checks.
    pub schema: serde_json::Value,
}

/// Retrieval configuration attached to an agent entity.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RetrievalRequirement {
    /// Maximum deterministically ordered documents requested from vector search.
    pub limit: usize,
}

/// A configured model capability.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelCapability {
    /// Provider identifier used by effect executors.
    pub provider: String,
    /// Provider model identifier.
    pub model: String,
    /// Immutable capability revision.
    pub revision: u64,
    /// Retired capabilities cannot be selected for new turns.
    pub retired: bool,
}

/// Provider-facing tool metadata stored on a tool entity.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolCapability {
    /// Provider-facing name.
    pub name: String,
    /// Provider-facing description.
    pub description: String,
    /// JSON Schema advertised to the model for tool arguments.
    pub parameters: serde_json::Value,
    /// Deterministic ordering key within an agent's visible tools.
    pub order: u32,
    /// Immutable executable revision.
    pub revision: u64,
    /// Retired tools are excluded from new decisions.
    pub retired: bool,
}

/// Dynamic discovery/MCP source lifecycle.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct DiscoverySource {
    /// Source kind used by external executors, such as `mcp`.
    pub kind: String,
    /// Monotonic refresh generation.
    pub generation: u64,
    /// Authoritative refresh/liveness state.
    pub state: DiscoveryState,
}

/// Mutually exclusive discovery source state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiscoveryState {
    /// Ready to accept a refresh request.
    Idle,
    /// Waiting for this generation-correlated operation.
    Refreshing { operation: Entity },
    /// Source is unavailable and publishes no new capabilities.
    Disconnected,
    /// Most recent refresh failed; a later refresh may retry.
    Failed(CanonicalError),
}

/// Stable source-local identity of a discovered capability.
#[derive(Component, Clone, Debug, Eq, Hash, PartialEq)]
pub struct DiscoveryKey(pub String);

/// Marks a retired capability version while in-flight decisions retain it.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetiredCapability {
    /// Source generation that retired this version.
    pub generation: u64,
}

/// An addressable infrastructure/store capability.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StoreCapability {
    /// Capability kind, such as `conversation-memory` or `vector-search`.
    pub kind: String,
    /// Immutable capability revision.
    pub revision: u64,
    /// Retired stores cannot be selected for new operations.
    pub retired: bool,
}

/// Relationship from an agent to its selected model.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = ModelAgents)]
pub struct UsesModel(pub Entity);

/// Agents selecting a model.
#[derive(Component, Debug)]
#[relationship_target(relationship = UsesModel)]
pub struct ModelAgents(Vec<Entity>);

/// Relationship from a run to its agent.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = AgentRuns)]
pub struct RunOf(pub Entity);

/// Runs created for an agent.
#[derive(Component, Debug)]
#[relationship_target(relationship = RunOf)]
pub struct AgentRuns(Vec<Entity>);

/// Relationship from a delegated child run to its parent run.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = ChildRuns)]
pub struct ParentRun(pub Entity);

/// Child runs delegated by a parent run.
#[derive(Component, Debug)]
#[relationship_target(relationship = ParentRun)]
pub struct ChildRuns(Vec<Entity>);

/// Deterministic creation order for child-result reduction.
#[derive(Component, Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ChildOrdinal(pub u64);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct ChildResultCommitted;

/// Relationship from a committed turn to its run.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = RunTurns)]
pub struct TurnOf(pub Entity);

/// Committed turns belonging to a run.
#[derive(Component, Debug)]
#[relationship_target(relationship = TurnOf)]
pub struct RunTurns(Vec<Entity>);

/// Relationship from an operation to its run.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = RunOperations)]
pub struct OperationOf(pub Entity);

/// Operations belonging to a run.
#[derive(Component, Debug)]
#[relationship_target(relationship = OperationOf)]
pub struct RunOperations(Vec<Entity>);

/// Relationship from a logical tool batch to its run.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = RunToolBatches)]
pub struct BatchOf(pub Entity);

/// Logical tool batches belonging to a run.
#[derive(Component, Debug)]
#[relationship_target(relationship = BatchOf)]
pub struct RunToolBatches(Vec<Entity>);

/// Relationship from a tool operation to its logical batch.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = BatchOperations)]
pub struct OperationOfBatch(pub Entity);

/// Tool operations belonging to one atomic batch.
#[derive(Component, Debug)]
#[relationship_target(relationship = OperationOfBatch)]
pub struct BatchOperations(Vec<Entity>);

/// Relationship from a discovered tool version to its source.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = DiscoveredTools)]
pub struct DiscoveredFrom(pub Entity);

/// Tool versions produced by a discovery source.
#[derive(Component, Debug)]
#[relationship_target(relationship = DiscoveredFrom)]
pub struct DiscoveredTools(Vec<Entity>);

/// Relationship from a discovery refresh operation to its source.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = DiscoveryOperations)]
pub struct DiscoveryOperationOf(pub Entity);

/// Refresh operations belonging to a discovery source.
#[derive(Component, Debug)]
#[relationship_target(relationship = DiscoveryOperationOf)]
pub struct DiscoveryOperations(Vec<Entity>);

/// Relationship from a stream subscription to its run.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = RunSubscriptions)]
pub struct SubscriptionOf(pub Entity);

/// Stream subscriptions observing a run.
#[derive(Component, Debug)]
#[relationship_target(relationship = SubscriptionOf)]
pub struct RunSubscriptions(Vec<Entity>);

/// Relationship from a policy instance to the agent it governs.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = AgentPolicies)]
pub struct PolicyFor(pub Entity);

/// Policy instances governing an agent.
#[derive(Component, Debug)]
#[relationship_target(relationship = PolicyFor)]
pub struct AgentPolicies(Vec<Entity>);

/// Relationship from a grant entity to its agent.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = AgentToolGrants)]
pub struct GrantForAgent(pub Entity);

/// Tool grant entities belonging to an agent.
#[derive(Component, Debug)]
#[relationship_target(relationship = GrantForAgent)]
pub struct AgentToolGrants(Vec<Entity>);

/// Relationship from a grant entity to the exact tool capability it authorizes.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = ToolGrants)]
pub struct GrantForTool(pub Entity);

/// Grants referring to a tool capability.
#[derive(Component, Debug)]
#[relationship_target(relationship = GrantForTool)]
pub struct ToolGrants(Vec<Entity>);

/// Independent many-to-many access metadata between an agent and a tool.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolGrant {
    /// Explicit precedence for collision resolution.
    pub order: u32,
    /// Disabled grants are excluded from new decisions.
    pub enabled: bool,
}

/// Relationship from store access metadata to its agent.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = AgentStoreGrants)]
pub struct StoreGrantForAgent(pub Entity);

/// Store grants belonging to an agent.
#[derive(Component, Debug)]
#[relationship_target(relationship = StoreGrantForAgent)]
pub struct AgentStoreGrants(Vec<Entity>);

/// Relationship from store access metadata to its store capability.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = StoreGrants)]
pub struct StoreGrantForStore(pub Entity);

/// Grants referring to a store capability.
#[derive(Component, Debug)]
#[relationship_target(relationship = StoreGrantForStore)]
pub struct StoreGrants(Vec<Entity>);

/// Independent access metadata between an agent and store.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StoreGrant {
    /// Explicit selection precedence.
    pub order: u32,
    /// Disabled grants are excluded from new operations.
    pub enabled: bool,
}

/// Ordered ECS policy instance.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[component(immutable)]
pub struct Policy {
    /// Explicit composition order. Stable ID breaks ties.
    pub order: u32,
    /// Immutable policy revision recorded by accepted decisions.
    pub revision: u64,
    /// Data interpreted by the policy system.
    pub rule: PolicyRule,
}

/// Admission status for a policy revision.
#[derive(Component, Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum PolicyStatus {
    /// Eligible for future policy snapshots.
    #[default]
    Enabled,
    /// Excluded from future snapshots but retained for accepted evaluations.
    Retired,
}

fn accepts_new_policy_evaluations(status: Option<&PolicyStatus>) -> bool {
    !matches!(status, Some(PolicyStatus::Retired))
}

/// Built-in policy data. Extensions can add components and systems in [`RigSet::Policy`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PolicyRule {
    /// Permit progression without modification.
    Allow,
    /// Deny prompts containing this substring.
    DenyPromptContains(String),
    /// Apply a non-sticky patch to each newly prepared model request.
    PatchRequest(RequestPatch),
    /// Replace arguments for matching tool calls before dispatch.
    RewriteToolArguments {
        /// Optional tool name; `None` applies to every advertised tool.
        tool: Option<String>,
        /// Replacement JSON arguments.
        arguments: serde_json::Value,
    },
    /// Skip matching tool calls and return synthetic feedback to the model.
    SkipToolCall {
        /// Optional tool name; `None` applies to every advertised tool.
        tool: Option<String>,
        /// Model-visible feedback.
        reason: String,
    },
    /// Repair an invalid model-emitted tool name.
    RepairInvalidTool {
        /// Optional emitted name; `None` matches every invalid call.
        from: Option<String>,
        /// Replacement name resolved against the immutable advertised snapshot.
        to: String,
    },
    /// Retry the model after an invalid tool call.
    RetryInvalidTool {
        /// Optional emitted name; `None` matches every invalid call.
        tool: Option<String>,
        /// Corrective feedback returned as a synthetic result.
        feedback: String,
    },
    /// Treat an invalid tool call as skipped.
    SkipInvalidTool {
        /// Optional emitted name; `None` matches every invalid call.
        tool: Option<String>,
        /// Model-visible feedback.
        reason: String,
    },
    /// Rewrite only the model-visible presentation of matching tool results.
    RewriteToolResult {
        /// Optional tool name; `None` applies to every result.
        tool: Option<String>,
        /// Replacement model-visible presentation.
        presentation: String,
    },
    /// Stop a run after a matching tool result settles but before commit.
    StopToolResult {
        /// Optional tool name; `None` applies to every result.
        tool: Option<String>,
        /// Audit-only stop reason.
        reason: String,
    },
    /// Replace normalized completion text before the model turn commits.
    RewriteCompletionText {
        /// Replacement canonical text.
        text: String,
    },
    /// Stop when normalized completion text contains a substring.
    StopCompletionContains {
        /// Substring matched against normalized completion text.
        needle: String,
        /// Audit-only stop reason.
        reason: String,
    },
    /// Stop a streaming run when a text delta contains a substring.
    StopTextDeltaContains {
        /// Substring matched against the new delta.
        needle: String,
        /// Audit-only stop reason.
        reason: String,
    },
    /// Await an external approval operation at one lifecycle point.
    RequireApproval {
        /// Lifecycle point whose progression is suspended.
        point: PolicyPoint,
        /// Owned prompt sent to the approval executor.
        prompt: String,
    },
    /// Delegate one lifecycle point to an entity-targeted observer.
    ///
    /// Runtime-only observers must be rebound after restoring a domain snapshot.
    Custom(PolicyPoint),
}

/// Lifecycle point governed by a custom policy entity.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PolicyPoint {
    /// Before a model request is dispatched.
    Request,
    /// Before one tool call is dispatched.
    ToolCall,
    /// When a model emits a tool call outside its immutable snapshot.
    InvalidToolCall,
    /// After tool execution and before transcript commit.
    ToolResult,
    /// After a model response settles and before turn commit.
    CompletionResponse,
    /// Before an accepted text delta is published to subscribers.
    TextDelta,
}

impl PolicyRule {
    fn applies_to(&self, point: PolicyPoint) -> bool {
        match self {
            Self::Allow | Self::DenyPromptContains(_) | Self::PatchRequest(_) => {
                point == PolicyPoint::Request
            }
            Self::RewriteToolArguments { .. } | Self::SkipToolCall { .. } => {
                point == PolicyPoint::ToolCall
            }
            Self::RepairInvalidTool { .. }
            | Self::RetryInvalidTool { .. }
            | Self::SkipInvalidTool { .. } => point == PolicyPoint::InvalidToolCall,
            Self::RewriteToolResult { .. } | Self::StopToolResult { .. } => {
                point == PolicyPoint::ToolResult
            }
            Self::RewriteCompletionText { .. } | Self::StopCompletionContains { .. } => {
                point == PolicyPoint::CompletionResponse
            }
            Self::StopTextDeltaContains { .. } => point == PolicyPoint::TextDelta,
            Self::RequireApproval {
                point: approval_point,
                ..
            } => *approval_point == point,
            Self::Custom(custom) => *custom == point,
        }
    }
}

/// Non-sticky changes contributed by one request policy.
///
/// Patches are reduced in accepted policy order. Context is appended,
/// provider parameters are shallow-merged, tool allow-lists are intersected,
/// and scalar/history values use last-writer-wins semantics. The result is
/// applied only to the operation being evaluated.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RequestPatch {
    /// Replacement instructions for this request.
    pub instructions: Option<String>,
    /// Sampling temperature encoded as IEEE-754 bits.
    pub temperature_bits: Option<u64>,
    /// Replacement output-token limit.
    pub max_tokens: Option<u64>,
    /// Replacement tool-selection behavior.
    pub tool_choice: Option<ModelToolChoice>,
    /// Allow-list intersected with the snapshotted tools.
    pub active_tools: Option<Vec<String>>,
    /// Provider-specific top-level parameters.
    pub additional_params: Option<serde_json::Value>,
    /// Context documents appended to the request.
    pub extra_context: Vec<RetrievedDocument>,
    /// Replacement canonical history.
    pub history: Option<Vec<TranscriptEntry>>,
}

impl RequestPatch {
    /// Creates an empty patch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces instructions for this operation.
    pub fn instructions(mut self, value: impl Into<String>) -> Self {
        self.instructions = Some(value.into());
        self
    }

    /// Sets the sampling temperature for this operation.
    pub fn temperature(mut self, value: f64) -> Self {
        self.temperature_bits = Some(value.to_bits());
        self
    }

    /// Sets the output-token limit for this operation.
    pub fn max_tokens(mut self, value: u64) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Replaces the tool-selection behavior for this operation.
    pub fn tool_choice(mut self, value: ModelToolChoice) -> Self {
        self.tool_choice = Some(value);
        self
    }

    /// Narrows the tools advertised for this operation.
    pub fn active_tools<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.active_tools = Some(values.into_iter().map(Into::into).collect());
        self
    }

    /// Applies provider-specific request parameters.
    pub fn additional_params(mut self, value: serde_json::Value) -> Self {
        self.additional_params = Some(value);
        self
    }

    /// Appends request-local context documents.
    pub fn extra_context<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = RetrievedDocument>,
    {
        self.extra_context.extend(values);
        self
    }

    /// Replaces canonical history for this operation.
    pub fn history<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = TranscriptEntry>,
    {
        self.history = Some(values.into_iter().collect());
        self
    }
}

/// Typed result returned by a request-policy observer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RequestPolicyDecision {
    /// Accept the current effective request unchanged.
    Continue,
    /// Merge a non-sticky patch into the effective request.
    Patch(RequestPatch),
    /// Fail the run before model dispatch.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Entity-targeted invocation of exactly one policy in deterministic order.
///
/// Steering observers set [`Self::decision`]. Audit observers should instead
/// consume the observation events published after reduction.
#[derive(EntityEvent, Clone, Debug)]
pub struct RequestPolicyInvocation {
    /// Policy entity targeted by this invocation.
    #[event_target]
    pub policy: Entity,
    /// Durable evaluation entity owning the cursor and effective request.
    pub evaluation: Entity,
    /// Run being evaluated.
    pub run: Entity,
    /// Model operation awaiting a finalized request.
    pub operation: Entity,
    /// Current effective request, including all earlier rewrites.
    pub request: ModelEffectInput,
    /// Decision written by the policy's single steering observer.
    pub decision: Option<RequestPolicyDecision>,
}

/// Observe-only notification emitted immediately before a request policy runs.
#[derive(EntityEvent, Clone, Debug)]
pub struct RequestPolicyInvoked {
    /// Policy entity targeted by the core evaluator.
    #[event_target]
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Stable policy identity.
    pub policy_id: StableId,
    /// Exact snapshotted revision.
    pub revision: u64,
    /// Deterministic cursor position.
    pub cursor: usize,
}

/// Observe-only notification emitted after a request decision is reduced.
#[derive(EntityEvent, Clone, Debug)]
pub struct RequestPolicyDecided {
    /// Policy entity whose decision was reduced.
    #[event_target]
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Stable policy identity.
    pub policy_id: StableId,
    /// Exact snapshotted revision.
    pub revision: u64,
    /// Decision recorded by the evaluator, or `None` when no responder existed.
    pub decision: Option<RequestPolicyDecision>,
}

/// Immutable record of the ordered policy instances applied to an operation.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AcceptedPolicies(pub Vec<AcceptedPolicy>);

/// Immutable tool-call policy snapshot accepted for one operation.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AcceptedToolCallPolicies(pub Vec<AcceptedPolicy>);

/// Immutable invalid-call policy snapshot accepted for one operation.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AcceptedInvalidToolCallPolicies(pub Vec<AcceptedPolicy>);

/// One accepted policy revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedPolicy {
    /// Persistent policy identity.
    pub id: StableId,
    /// Runtime-local policy identity retained for correlation.
    pub entity: Entity,
    /// Exact revision applied.
    pub revision: u64,
}

/// Relationship from a durable policy evaluation to its operation.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = OperationPolicyEvaluations)]
pub struct EvaluationOfOperation(pub Entity);

/// Policy evaluations retained by an operation for audit and resumption.
#[derive(Component, Debug)]
#[relationship_target(relationship = EvaluationOfOperation)]
pub struct OperationPolicyEvaluations(Vec<Entity>);

/// Relationship from an asynchronous approval operation to its evaluation.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = EvaluationApprovalOperations)]
pub struct ApprovalForEvaluation(pub Entity);

/// Approval operations retained by a durable policy evaluation.
#[derive(Component, Debug)]
#[relationship_target(relationship = ApprovalForEvaluation)]
pub struct EvaluationApprovalOperations(Vec<Entity>);

/// Authoritative phase of a request policy evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RequestPolicyEvaluationPhase {
    /// The policy at `cursor` is ready to be invoked.
    Evaluating,
    /// Waiting for a correlated approval operation.
    WaitingApproval { operation: Entity, policy: StableId },
    /// Every snapshotted policy accepted the effective request.
    Accepted,
    /// A policy stopped the run.
    Rejected { policy: StableId, reason: String },
}

/// Durable ordered request-policy state.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct RequestPolicyEvaluation {
    /// Run whose operation is being evaluated.
    pub run: Entity,
    /// Deterministically sorted immutable policy snapshot.
    pub policies: Vec<AcceptedPolicy>,
    /// Index of the next policy to invoke.
    pub cursor: usize,
    /// Effective request visible to the next policy.
    pub effective: ModelEffectInput,
    /// Authoritative evaluation phase.
    pub phase: RequestPolicyEvaluationPhase,
}

#[derive(Component, Clone, Debug, Eq, PartialEq)]
struct PendingModelRequest(ModelEffectInput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct RequestPolicyInitialized;

/// Typed result returned by a tool-call policy observer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolCallPolicyDecision {
    /// Execute with the current effective arguments.
    Run,
    /// Replace arguments before later policies inspect the call.
    Rewrite(serde_json::Value),
    /// Do not execute and return model-visible feedback.
    Skip(String),
    /// Stop the complete run before dispatch.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Entity-targeted invocation for one tool call and one policy.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolCallPolicyInvocation {
    /// Policy entity targeted by the core evaluator.
    #[event_target]
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Run containing the logical tool batch.
    pub run: Entity,
    /// Tool operation awaiting finalized arguments.
    pub operation: Entity,
    /// Effective call including all earlier rewrites.
    pub call: ToolEffectInput,
    /// Decision written by the policy's steering observer.
    pub decision: Option<ToolCallPolicyDecision>,
}

/// Authoritative phase of a tool-call policy evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolCallPolicyEvaluationPhase {
    /// The policy at `cursor` is ready to run.
    Evaluating,
    /// Waiting for a correlated approval operation.
    WaitingApproval { operation: Entity, policy: StableId },
    /// The call is accepted for dispatch.
    Accepted,
    /// Execution is skipped with synthetic feedback.
    Skipped(String),
    /// The run was stopped by policy.
    Rejected { policy: StableId, reason: String },
}

/// Durable ordered tool-call policy state.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct ToolCallPolicyEvaluation {
    /// Run containing the operation.
    pub run: Entity,
    /// Deterministically sorted immutable policy snapshot.
    pub policies: Vec<AcceptedPolicy>,
    /// Index of the next policy to invoke.
    pub cursor: usize,
    /// Effective call visible to the next policy.
    pub effective: ToolEffectInput,
    /// Authoritative evaluation phase.
    pub phase: ToolCallPolicyEvaluationPhase,
}

#[derive(Component, Clone, Debug, Eq, PartialEq)]
struct PendingToolCall(ToolEffectInput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct ToolCallPolicyInitialized;

/// Typed resolution for a model-emitted call outside its tool snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvalidToolCallPolicyDecision {
    /// This policy does not resolve the call; continue ordered evaluation.
    Continue,
    /// Preserve fail-fast behavior.
    Fail,
    /// Return corrective feedback and retry the model.
    Retry(String),
    /// Replace the emitted name with a snapshotted tool name.
    Repair(String),
    /// Return synthetic feedback without executing a tool.
    Skip(String),
    /// Stop the complete run.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Immutable invalid-call context retained while policy resolution proceeds.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct PendingInvalidToolCall {
    /// Model-emitted call.
    pub call: ModelToolCall,
    /// Exact tools advertised on the producing model operation.
    pub available_tools: Vec<ToolDecision>,
    /// Explicit position in the logical batch.
    pub index: u32,
    /// Producing model operation retained for audit and resumption.
    pub source_model_operation: Entity,
    /// Committed model-call position that emitted this invalid call.
    pub turn: u32,
    /// Tool-selection behavior accepted by the producing request.
    pub tool_choice: Option<ModelToolChoice>,
    /// Complete canonical history at detection time.
    pub diagnostic_history: Vec<TranscriptEntry>,
    /// Whether any provider delta was accepted for the producing operation.
    pub streaming_origin: bool,
    /// Retry count before resolving this call.
    pub retry_count: u32,
    /// Accepted run-local invalid-call retry limit.
    pub max_retries: u32,
}

/// Entity-targeted invocation for one invalid call and one policy.
#[derive(EntityEvent, Clone, Debug)]
pub struct InvalidToolCallPolicyInvocation {
    /// Policy entity targeted by the core evaluator.
    #[event_target]
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Run containing the invalid call.
    pub run: Entity,
    /// Operation retaining the invalid-call context.
    pub operation: Entity,
    /// Immutable invalid-call context.
    pub invalid: PendingInvalidToolCall,
    /// Decision written by the policy's steering observer.
    pub decision: Option<InvalidToolCallPolicyDecision>,
}

/// Authoritative invalid-call evaluation phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvalidToolCallPolicyEvaluationPhase {
    /// The policy at `cursor` is ready to run.
    Evaluating,
    /// Waiting for a correlated approval operation.
    WaitingApproval { operation: Entity, policy: StableId },
    /// The call was resolved to a real or synthetic result.
    Resolved,
    /// No policy resolved the call or one explicitly failed it.
    Failed,
    /// A policy stopped the run.
    Rejected { policy: StableId, reason: String },
}

/// Durable ordered invalid-call resolution state.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct InvalidToolCallPolicyEvaluation {
    /// Run containing the operation.
    pub run: Entity,
    /// Deterministically sorted immutable policy snapshot.
    pub policies: Vec<AcceptedPolicy>,
    /// Index of the next policy to invoke.
    pub cursor: usize,
    /// Immutable invalid-call context.
    pub invalid: PendingInvalidToolCall,
    /// Authoritative evaluation phase.
    pub phase: InvalidToolCallPolicyEvaluationPhase,
}

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct InvalidToolCallPolicyInitialized;

enum PreparedToolOperation {
    Valid(ToolEffectInput),
    Invalid(PendingInvalidToolCall),
}

/// Typed result returned by a tool-result policy observer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolResultPolicyDecision {
    /// Preserve the current effective presentation.
    Keep,
    /// Replace only the model-visible presentation.
    Rewrite(String),
    /// Stop the run without publishing raw result content.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Entity-targeted invocation for one settled tool result and one policy.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolResultPolicyInvocation {
    /// Policy entity targeted by the core evaluator.
    #[event_target]
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Run containing the result.
    pub run: Entity,
    /// Settled tool operation.
    pub operation: Entity,
    /// Exact input used by execution.
    pub input: ToolEffectInput,
    /// Immutable raw result and current effective presentation.
    pub result: ToolEffectOutput,
    /// Decision written by the policy's steering observer.
    pub decision: Option<ToolResultPolicyDecision>,
}

/// Authoritative tool-result evaluation phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolResultPolicyEvaluationPhase {
    /// The policy at `cursor` is ready to run.
    Evaluating,
    /// Waiting for a correlated approval operation.
    WaitingApproval { operation: Entity, policy: StableId },
    /// The effective presentation is accepted for commit.
    Accepted,
    /// A policy stopped the run.
    Rejected { policy: StableId, reason: String },
}

/// Durable ordered tool-result policy state.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct ToolResultPolicyEvaluation {
    /// Run containing the operation.
    pub run: Entity,
    /// Deterministically sorted immutable policy snapshot.
    pub policies: Vec<AcceptedPolicy>,
    /// Index of the next policy to invoke.
    pub cursor: usize,
    /// Exact executed input.
    pub input: ToolEffectInput,
    /// Raw result plus current effective presentation.
    pub effective: ToolEffectOutput,
    /// Authoritative evaluation phase.
    pub phase: ToolResultPolicyEvaluationPhase,
}

/// Immutable tool-result policy snapshot accepted for one operation.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AcceptedToolResultPolicies(pub Vec<AcceptedPolicy>);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct ToolResultPolicyInitialized;

#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
struct EffectiveToolOutput(ToolEffectOutput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct ToolResultPolicyDone;

/// Typed result returned by a completion-response policy observer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionResponsePolicyDecision {
    /// Accept the current normalized response.
    Continue,
    /// Replace normalized text before later policies and commit.
    RewriteText(String),
    /// Stop the run before committing the model turn.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Entity-targeted invocation for one settled model response and one policy.
#[derive(EntityEvent, Clone, Debug)]
pub struct CompletionResponsePolicyInvocation {
    /// Policy entity targeted by the core evaluator.
    #[event_target]
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Run containing the model operation.
    pub run: Entity,
    /// Settled model operation.
    pub operation: Entity,
    /// Exact request sent to the provider.
    pub request: ModelEffectInput,
    /// Current normalized response, including earlier rewrites.
    pub response: ModelEffectOutput,
    /// Decision written by the policy's steering observer.
    pub decision: Option<CompletionResponsePolicyDecision>,
}

/// Authoritative completion-response evaluation phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionResponsePolicyEvaluationPhase {
    /// The policy at `cursor` is ready to run.
    Evaluating,
    /// Waiting for a correlated approval operation.
    WaitingApproval { operation: Entity, policy: StableId },
    /// The effective response is accepted for commit.
    Accepted,
    /// A policy stopped the run.
    Rejected { policy: StableId, reason: String },
}

/// Durable ordered completion-response policy state.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct CompletionResponsePolicyEvaluation {
    /// Run containing the operation.
    pub run: Entity,
    /// Deterministically sorted immutable policy snapshot.
    pub policies: Vec<AcceptedPolicy>,
    /// Index of the next policy to invoke.
    pub cursor: usize,
    /// Exact dispatched request.
    pub request: ModelEffectInput,
    /// Effective normalized response.
    pub effective: ModelEffectOutput,
    /// Authoritative evaluation phase.
    pub phase: CompletionResponsePolicyEvaluationPhase,
}

/// Immutable completion-response policy snapshot accepted for one operation.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AcceptedCompletionResponsePolicies(pub Vec<AcceptedPolicy>);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct CompletionResponsePolicyInitialized;

#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
struct EffectiveModelOutput(ModelEffectOutput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct CompletionResponsePolicyDone;

/// Observe-only event emitted after a model turn commits on either surface.
#[derive(EntityEvent, Clone, Debug)]
pub struct ModelTurnFinished {
    /// Run receiving the committed turn.
    #[event_target]
    pub run: Entity,
    /// Model operation that produced the turn.
    pub operation: Entity,
    /// Zero-based committed turn index.
    pub turn: u32,
    /// Canonical committed output.
    pub output: ModelEffectOutput,
    /// Request-policy revisions accepted for the operation.
    pub request_policies: Vec<AcceptedPolicy>,
    /// Response-policy revisions accepted for the operation.
    pub response_policies: Vec<AcceptedPolicy>,
}

/// Observe-only event emitted when a terminal child result commits to its parent.
#[derive(EntityEvent, Clone, Debug)]
pub struct ChildRunFinished {
    /// Parent run receiving the result.
    #[event_target]
    pub parent: Entity,
    /// Terminal child run.
    pub child: Entity,
    /// Stable child identity.
    pub child_id: StableId,
    /// Deterministic creation ordinal.
    pub ordinal: u64,
    /// Terminal child outcome.
    pub result: Result<String, String>,
}

/// Observe-only notification for a validated text delta.
#[derive(EntityEvent, Clone, Debug)]
pub struct TextDeltaObserved {
    /// Producing model operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Committed model-call position at ingress.
    pub turn: u32,
    /// Explicit operation-local sequence.
    pub sequence: u64,
    /// Provider correlation, when supplied.
    pub provider_correlation: Option<String>,
    /// Newly validated text.
    pub delta: String,
    /// Canonical aggregate including this delta.
    pub aggregated: String,
}

/// Observe-only notification for a validated tool-call fragment.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolCallDeltaObserved {
    /// Producing model operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Committed model-call position at ingress.
    pub turn: u32,
    /// Explicit operation-local sequence.
    pub sequence: u64,
    /// Provider correlation, when supplied.
    pub provider_correlation: Option<String>,
    /// Provider-facing tool-call ID.
    pub id: String,
    /// Rig correlation ID retained through invalid-call recovery.
    pub internal_call_id: String,
    /// Tool name or argument fragment.
    pub content: ToolCallDeltaContent,
}

/// Observe-only notification after a streaming provider response settles.
#[derive(EntityEvent, Clone, Debug)]
pub struct StreamResponseFinished {
    /// Producing model operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Exact accepted operation generation.
    pub generation: u64,
    /// Canonical final response before response-policy rewriting.
    pub output: ModelEffectOutput,
}

/// Observe-only notification after request policy finalizes immutable input.
#[derive(EntityEvent, Clone, Debug)]
pub struct CompletionRequestPrepared {
    /// Model operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Exact immutable request.
    pub request: ModelEffectInput,
    /// Accepted request-policy revisions.
    pub policies: Vec<AcceptedPolicy>,
}

/// Observe-only notification after a model effect enters the outbox.
#[derive(EntityEvent, Clone, Debug)]
pub struct ModelDispatched {
    /// Model operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Exact operation generation.
    pub generation: u64,
}

/// Observe-only notification after a model effect completion is validated.
#[derive(EntityEvent, Clone, Debug)]
pub struct ModelSettled {
    /// Model operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Exact operation generation.
    pub generation: u64,
    /// Canonical outcome before response policy.
    pub outcome: OperationOutcome,
}

/// Observe-only notification after response policy finalizes presentation.
#[derive(EntityEvent, Clone, Debug)]
pub struct CompletionResponseApplied {
    /// Model operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Immutable provider-normalized result.
    pub raw: ModelEffectOutput,
    /// Effective result accepted for commit.
    pub effective: ModelEffectOutput,
}

/// Observe-only notification when an unknown or disallowed call is retained.
#[derive(EntityEvent, Clone, Debug)]
pub struct InvalidToolCallDetected {
    /// Pending tool operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Immutable invalid-call context.
    pub invalid: PendingInvalidToolCall,
}

/// Observe-only notification after tool-call policy finalizes immutable input.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolCallPrepared {
    /// Tool operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Exact immutable call.
    pub call: ToolEffectInput,
    /// Accepted call-policy revisions.
    pub policies: Vec<AcceptedPolicy>,
}

/// Observe-only notification after a real tool effect enters the outbox.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolExecutionStarted {
    /// Tool operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Exact operation generation.
    pub generation: u64,
    /// Logical batch position.
    pub index: u32,
}

/// Observe-only notification after a real tool effect completion is validated.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolExecutionSettled {
    /// Tool operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Exact operation generation.
    pub generation: u64,
    /// Immutable raw outcome before result policy.
    pub outcome: OperationOutcome,
}

/// Observe-only notification after tool-result presentation policy finalizes.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolResultPresentationFinalized {
    /// Tool operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Immutable raw result.
    pub raw: ToolEffectOutput,
    /// Final model-visible presentation.
    pub presentation: String,
}

/// Observe-only notification after an atomic tool batch commits successfully.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolBatchCommitted {
    /// Tool-batch entity.
    #[event_target]
    pub batch: Entity,
    /// Owning run.
    pub run: Entity,
    /// Results in logical call order.
    pub results: Vec<ToolEffectOutput>,
}

/// Observe-only notification after a store operation settles and is applied.
#[derive(EntityEvent, Clone, Debug)]
pub struct PersistenceSettled {
    /// Store operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Canonical outcome.
    pub outcome: OperationOutcome,
}

/// Observe-only terminal success notification.
#[derive(EntityEvent, Clone, Debug)]
pub struct RunCompleted {
    /// Terminal run.
    #[event_target]
    pub run: Entity,
    /// Canonical output.
    pub output: RunOutput,
}

/// Observe-only terminal failure notification.
#[derive(EntityEvent, Clone, Debug)]
pub struct RunFailed {
    /// Terminal run.
    #[event_target]
    pub run: Entity,
    /// Canonical error.
    pub error: CanonicalError,
}

/// Observe-only terminal cancellation notification.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct RunCancelled {
    /// Terminal run.
    #[event_target]
    pub run: Entity,
}

/// Typed steering result for a streaming text delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextDeltaPolicyDecision {
    /// Publish the accepted delta.
    Continue,
    /// Stop the run before publishing this delta.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Entity-targeted invocation for one accepted, not-yet-published text delta.
#[derive(EntityEvent, Clone, Debug)]
pub struct TextDeltaPolicyInvocation {
    /// Policy entity targeted by the core evaluator.
    #[event_target]
    pub policy: Entity,
    /// Durable delta evaluation identity.
    pub evaluation: Entity,
    /// Run receiving the stream.
    pub run: Entity,
    /// Model operation producing the stream.
    pub operation: Entity,
    /// Committed model-call index at ingress time.
    pub turn: u32,
    /// Monotonic operation-local sequence.
    pub sequence: u64,
    /// Newly accepted text.
    pub delta: String,
    /// Authoritative aggregate including this delta.
    pub aggregated: String,
    /// Decision written by the policy's steering observer.
    pub decision: Option<TextDeltaPolicyDecision>,
}

/// Authoritative text-delta evaluation phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextDeltaPolicyEvaluationPhase {
    /// The policy at `cursor` is ready to run.
    Evaluating,
    /// Waiting for a correlated approval operation.
    WaitingApproval { operation: Entity, policy: StableId },
    /// The delta was published once.
    Published,
    /// A policy stopped the run.
    Rejected { policy: StableId, reason: String },
}

/// Durable ordered stream-delta policy state.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct TextDeltaPolicyEvaluation {
    /// Run receiving the delta.
    pub run: Entity,
    /// Producing model operation.
    pub operation: Entity,
    /// Deterministically sorted immutable policy snapshot.
    pub policies: Vec<AcceptedPolicy>,
    /// Index of the next policy to invoke.
    pub cursor: usize,
    /// Committed model-call index at ingress time.
    pub turn: u32,
    /// Operation-local sequence.
    pub sequence: u64,
    /// New text.
    pub delta: String,
    /// Aggregate including the new text.
    pub aggregated: String,
    /// Authoritative evaluation phase.
    pub phase: TextDeltaPolicyEvaluationPhase,
}

/// Immutable text-delta policy snapshot accepted for one delta.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AcceptedTextDeltaPolicies(pub Vec<AcceptedPolicy>);

/// Authoritative run phase and outcome.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub enum RunState {
    /// Accepted but not prepared.
    Queued,
    /// Waiting for an external model operation.
    WaitingModel { operation: Entity },
    /// Waiting for every operation in one logical tool batch.
    WaitingTools { batch: Entity },
    /// Waiting for conversation-memory load or persistence.
    WaitingStore { operation: Entity },
    /// Completed and retained for observation.
    Completed(RunOutput),
    /// Terminal failure.
    Failed(CanonicalError),
    /// Cancellation prevents further dispatch and commit.
    Cancelled,
}

/// Explicit semantics for suspending a run.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PauseMode {
    /// Stop new dispatch, allow in-flight work to settle and commit to a safe boundary.
    Drain,
    /// Retain validated completions but defer policy evaluation and commit until resume.
    FreezeAfterIngress,
    /// Cancel in-flight effects and retain a redispatchable prepared checkpoint.
    CancelAndSuspend,
}

/// Authoritative run-local control orthogonal to [`RunState`].
#[derive(Component, Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum RunControl {
    /// Normal progression is eligible.
    #[default]
    Running,
    /// A suspension transition is being reconciled.
    PauseRequested(PauseMode),
    /// Progression is suspended while ingress and maintenance remain active.
    Paused(PauseMode),
}

/// Admission and bulk-control policy for an agent definition.
#[derive(Component, Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum AgentControl {
    /// Admit new runs and leave existing runs unchanged.
    #[default]
    Running,
    /// Reject new runs while existing runs continue.
    RejectNewRuns,
    /// Reject new runs and request this pause mode for active runs.
    PauseExistingRuns(PauseMode),
}

/// Per-agent invalid-tool retry budget snapshotted onto each admitted run.
#[derive(Component, Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct InvalidToolCallBudget {
    /// Maximum `Retry` resolutions before later invalid calls fail closed.
    pub max_retries: u32,
}

/// A run's authoritative input and committed transcript.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunRecord {
    /// Canonical serialized user message accepted for this run.
    pub prompt: serde_json::Value,
    /// First textual prompt part used only for retrieval and text policy matching.
    pub prompt_text: String,
    /// Optional per-run structured-output requirement.
    pub output_schema: Option<serde_json::Value>,
    /// Optional per-run model-call budget override.
    pub max_model_calls: Option<u32>,
    /// Deterministically committed entries.
    pub transcript: Vec<TranscriptEntry>,
    /// Whether the terminal result has been observed by its facade.
    pub observed: bool,
    /// Usage accumulated across every model operation in the run.
    pub usage: Usage,
    /// Explicit next committed model-turn position.
    pub next_turn: u32,
    /// Whether this run was admitted with a streaming subscription.
    pub streaming: bool,
    /// Invalid tool calls already resolved with a model retry.
    pub invalid_tool_call_retries: u32,
    /// Immutable retry budget accepted when the run was admitted.
    pub max_invalid_tool_call_retries: u32,
    /// Optional stable conversation identity.
    pub conversation: Option<StableId>,
    /// Prevents repeated memory resolution/load.
    pub memory_loaded: bool,
    /// Immutable store choice accepted for this run.
    pub memory_store: Option<StoreDecision>,
    /// Prevents repeated vector retrieval.
    pub retrieval_loaded: bool,
    /// Immutable vector-search store choice accepted for this run.
    pub retrieval_store: Option<StoreDecision>,
    /// Ordered documents returned by the accepted retrieval operation.
    pub retrieved_documents: Vec<RetrievedDocument>,
    /// Output retained while required persistence settles.
    pub pending_output: Option<RunOutput>,
}

/// Canonical transcript entry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TranscriptEntry {
    /// Original user input.
    User(String),
    /// A multimodal user message retained without flattening provider content.
    UserMessage(serde_json::Value),
    /// A canonical message supplied as pre-existing run history.
    Message(serde_json::Value),
    /// Committed model output.
    Assistant(String),
    /// Complete canonical assistant message when provider-significant parts
    /// such as reasoning signatures must survive a tool round trip.
    AssistantMessage(serde_json::Value),
    /// Exact provider tool call advertised before its result.
    AssistantToolCall {
        /// Provider correlation identifier.
        call_id: String,
        /// Provider-facing tool name.
        name: String,
        /// Structured arguments emitted by the model.
        arguments: serde_json::Value,
    },
    /// Atomically committed model-visible tool result.
    ToolResult {
        /// Provider correlation identifier.
        call_id: String,
        /// Provider-facing result identifier before runtime normalization.
        provider_result_id: String,
        /// Provider-specific call ID, when represented separately.
        provider_call_id: Option<String>,
        /// Provider-facing tool name.
        name: String,
        /// Canonical structured content retained for provider round trips.
        raw: serde_json::Value,
        /// Presentation returned to the next model call.
        content: String,
    },
    /// Deterministically reduced terminal child-run outcome.
    ChildResult {
        /// Creation ordinal independent of completion order.
        ordinal: u64,
        /// Stable child run identity.
        run_id: StableId,
        /// Terminal output or canonical terminal error text.
        result: Result<String, String>,
    },
}

/// Terminal run output.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunOutput {
    /// Final model text.
    pub text: String,
    /// Provider-reported usage.
    pub usage: Usage,
}

/// Canonical token usage.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Usage {
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
}

fn placeholder_entity() -> Entity {
    Entity::PLACEHOLDER
}

/// Immutable decision accepted before model dispatch.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelDecision {
    /// Entity identity is retained for in-memory correlation.
    #[serde(skip, default = "placeholder_entity")]
    pub model_entity: Entity,
    /// Stable identity is retained for audit and persistence.
    pub model_id: StableId,
    /// Exact capability revision used by the effect.
    pub revision: u64,
    /// Owned provider identifier.
    pub provider: String,
    /// Owned provider model identifier.
    pub model: String,
    /// Tenant scope accepted by preparation.
    pub tenant: TenantId,
}

/// Exact tool capability advertised and retained for one model operation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolDecision {
    /// In-memory capability identity.
    #[serde(skip, default = "placeholder_entity")]
    pub tool_entity: Entity,
    /// Stable capability identity.
    pub tool_id: StableId,
    /// Exact executable revision.
    pub revision: u64,
    /// Provider-facing name.
    pub name: String,
    /// Provider-facing description.
    pub description: String,
    /// Exact argument schema advertised for this accepted revision.
    pub parameters: serde_json::Value,
    /// Explicit provider-facing position.
    pub order: u32,
}

/// Exact store capability accepted for a run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StoreDecision {
    /// In-memory store identity.
    #[serde(skip, default = "placeholder_entity")]
    pub store_entity: Entity,
    /// Persistent store identity.
    pub store_id: StableId,
    /// Exact capability revision.
    pub revision: u64,
    /// Narrow capability kind.
    pub kind: String,
    /// Accepted tenant scope.
    pub tenant: TenantId,
}

/// Authoritative lifecycle of an external operation.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OperationState {
    /// Incremented whenever a logical operation is superseded or retried.
    pub generation: u64,
    /// Mutually exclusive phase/outcome.
    pub phase: OperationPhase,
}

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
struct EffectDeadline {
    expires_at: u64,
}

/// Mutually exclusive operation state.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
// Schedules move outcomes between phases once; boxing would add an allocation
// to every completed effect solely to reduce the enum's stack footprint.
#[allow(clippy::large_enum_variant)]
pub enum OperationPhase {
    /// Has authoritative correlation state but has not entered the queue.
    Prepared,
    /// Submitted to the external effect boundary.
    InFlight,
    /// Terminal result, written once for this generation.
    Settled(OperationOutcome),
    /// Cancellation prevents a late result from mutating the run.
    Cancelled,
    /// A newer generation replaced this operation.
    Superseded,
}

/// Canonical terminal operation outcome.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum OperationOutcome {
    /// Successful typed effect output.
    Success(EffectOutput),
    /// Failed external operation.
    Failure(CanonicalError),
}

/// Canonical error safe for policy, telemetry, and persistence.
#[derive(Clone, Debug, Deserialize, Error, Eq, PartialEq, Serialize)]
pub enum CanonicalError {
    /// Referenced entity was absent or stale.
    #[error("stale entity reference: {0}")]
    StaleEntity(String),
    /// Tenant scopes were not authorized to interact.
    #[error("tenant scope mismatch")]
    TenantMismatch,
    /// Capability is retired.
    #[error("capability is retired")]
    RetiredCapability,
    /// Effect queue was disconnected.
    #[error("effect executor disconnected")]
    ExecutorDisconnected,
    /// An external effect exceeded its runtime-owned deadline.
    #[error("external effect timed out")]
    Timeout,
    /// The executor caught a panic while running an effect task.
    #[error("external effect task panicked: {0}")]
    ExecutorPanicked(String),
    /// A typed provider returned content the canonical runtime cannot represent.
    #[error("unsupported model content: {0}")]
    UnsupportedModelContent(String),
    /// A model's terminal text did not satisfy the accepted output requirement.
    #[error("structured output was invalid: {0}")]
    InvalidStructuredOutput(String),
    /// A run exhausted the model-call budget accepted from its agent.
    #[error("model-call budget exhausted after {limit} calls")]
    ModelCallBudget {
        /// Maximum accepted model operations.
        limit: u32,
    },
    /// External provider failure.
    #[error("provider failure: {message}")]
    Provider {
        /// Operator-facing diagnostic; model-visible presentation is separate.
        message: String,
        /// Whether policy may retry the failure.
        retryable: bool,
    },
    /// Typed tool execution failed outside the world.
    #[error("tool failure: {message}")]
    Tool {
        /// Operator-facing diagnostic.
        message: String,
        /// Whether policy may retry the failure.
        retryable: bool,
    },
    /// Typed store execution failed outside the world.
    #[error("store failure: {message}")]
    Store {
        /// Operator-facing diagnostic.
        message: String,
        /// Whether policy may retry the failure.
        retryable: bool,
    },
    /// Typed discovery refresh failed outside the world.
    #[error("discovery failure: {message}")]
    Discovery {
        /// Operator-facing diagnostic.
        message: String,
        /// Whether policy may retry the failure.
        retryable: bool,
    },
    /// Ordered policy denied dispatch.
    #[error("policy `{policy}` denied the operation")]
    PolicyDenied {
        /// Persistent policy identity.
        policy: String,
    },
    /// Agent admission control rejected a newly submitted run.
    #[error("agent is not accepting new runs")]
    AgentAdmissionDenied,
    /// Model requested a tool that was not in its immutable snapshot.
    #[error("unknown or unavailable tool `{0}`")]
    UnknownTool(String),
    /// Tool-call identifiers must be unique within a logical batch.
    #[error("duplicate tool call id `{0}`")]
    DuplicateToolCall(String),
    /// Completion payload did not match its authoritative operation kind.
    #[error("effect completion kind did not match operation kind")]
    EffectKindMismatch,
    /// Discovery payload violated source reconciliation invariants.
    #[error("invalid discovery payload: {0}")]
    InvalidDiscovery(String),
}

/// Owned request submitted at the asynchronous effect boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectRequest {
    /// Correlates the completion to current ECS state.
    pub operation: Entity,
    /// Rejects stale retries and duplicate generations.
    pub generation: u64,
    /// Fully owned immutable input.
    pub input: EffectInput,
}

/// Correlated request for an executor to cancel in-flight work when supported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectCancellation {
    /// Operation entity used only for runtime-local correlation.
    pub operation: Entity,
    /// Exact operation generation to cancel.
    pub generation: u64,
}

impl EffectRequest {
    /// Returns model input when this is a model operation.
    pub fn model_input(&self) -> Option<&ModelEffectInput> {
        match &self.input {
            EffectInput::Model(input) => Some(input),
            EffectInput::Tool(_)
            | EffectInput::Discovery(_)
            | EffectInput::Store(_)
            | EffectInput::PolicyApproval(_) => None,
        }
    }

    /// Returns tool input when this is a tool operation.
    pub fn tool_input(&self) -> Option<&ToolEffectInput> {
        match &self.input {
            EffectInput::Model(_)
            | EffectInput::Discovery(_)
            | EffectInput::Store(_)
            | EffectInput::PolicyApproval(_) => None,
            EffectInput::Tool(input) => Some(input),
        }
    }

    /// Returns discovery input when this is a refresh operation.
    pub fn discovery_input(&self) -> Option<&DiscoveryEffectInput> {
        match &self.input {
            EffectInput::Discovery(input) => Some(input),
            EffectInput::Model(_)
            | EffectInput::Tool(_)
            | EffectInput::Store(_)
            | EffectInput::PolicyApproval(_) => None,
        }
    }

    /// Returns store input when this is a memory/store operation.
    pub fn store_input(&self) -> Option<&StoreEffectInput> {
        match &self.input {
            EffectInput::Store(input) => Some(input),
            EffectInput::Model(_)
            | EffectInput::Tool(_)
            | EffectInput::Discovery(_)
            | EffectInput::PolicyApproval(_) => None,
        }
    }

    /// Returns an approval input when this is a policy operation.
    pub fn policy_approval_input(&self) -> Option<&PolicyApprovalEffectInput> {
        match &self.input {
            EffectInput::PolicyApproval(input) => Some(input),
            EffectInput::Model(_)
            | EffectInput::Tool(_)
            | EffectInput::Discovery(_)
            | EffectInput::Store(_) => None,
        }
    }
}

/// Fully owned typed input crossing the effect boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum EffectInput {
    /// Provider model call.
    Model(ModelEffectInput),
    /// Exact snapshotted tool execution.
    Tool(ToolEffectInput),
    /// Dynamic capability discovery refresh.
    Discovery(DiscoveryEffectInput),
    /// Store/memory operation.
    Store(StoreEffectInput),
    /// External policy/approval decision.
    PolicyApproval(PolicyApprovalEffectInput),
}

/// Fully owned model effect input.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[component(immutable)]
pub struct ModelEffectInput {
    /// Accepted immutable model decision.
    pub decision: ModelDecision,
    /// Instructions accepted for this turn.
    pub instructions: String,
    /// Canonical serialized user message.
    pub prompt: serde_json::Value,
    /// Canonical committed history prepared for this model call.
    pub history: Vec<TranscriptEntry>,
    /// Deterministically ordered exact tool snapshot.
    pub tools: Vec<ToolDecision>,
    /// Ordered results from the preceding logical tool batch.
    pub tool_results: Vec<ToolEffectOutput>,
    /// Structured-output schema accepted for this operation, if configured.
    pub output_schema: Option<serde_json::Value>,
    /// Tool whose arguments are accepted as terminal structured output.
    pub terminal_tool: Option<String>,
    /// Ordered retrieval context accepted before model dispatch.
    pub documents: Vec<RetrievedDocument>,
    /// Sampling temperature accepted for this operation.
    pub temperature_bits: Option<u64>,
    /// Provider output-token limit accepted for this operation.
    pub max_tokens: Option<u64>,
    /// Tool-selection behavior accepted for this operation.
    pub tool_choice: Option<ModelToolChoice>,
    /// Provider-specific parameters accepted for this operation.
    pub additional_params: Option<serde_json::Value>,
}

/// Fully owned tool execution input assembled from an immutable decision.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[component(immutable)]
pub struct ToolEffectInput {
    /// Exact capability accepted before the model call.
    pub decision: ToolDecision,
    /// Provider tool-call correlation identifier.
    pub call_id: String,
    /// Provider-facing tool-result identifier before runtime normalization.
    pub provider_result_id: String,
    /// Provider-specific call ID, when the API represents it separately.
    pub provider_call_id: Option<String>,
    /// Provider-supplied arguments.
    pub arguments: serde_json::Value,
    /// Explicit position in the model call's logical batch.
    pub index: u32,
}

/// Owned discovery refresh input.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryEffectInput {
    /// Source entity used only for runtime correlation.
    pub source: Entity,
    /// Persistent source identity for the external executor.
    pub source_id: StableId,
    /// Source kind.
    pub kind: String,
    /// Monotonic requested generation.
    pub generation: u64,
    /// Tenant scope accepted for this refresh.
    pub tenant: TenantId,
}

/// Fully owned store effect input.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StoreEffectInput {
    /// Exact store decision accepted for this run.
    pub decision: StoreDecision,
    /// Owned operation payload.
    pub operation: StoreOperation,
}

/// Fully owned approval request crossing the effect boundary.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PolicyApprovalEffectInput {
    /// Stable policy identity.
    pub policy_id: StableId,
    /// Exact policy revision snapshotted by the evaluation.
    pub revision: u64,
    /// Lifecycle point awaiting approval.
    pub point: PolicyPoint,
    /// Owned approver-facing prompt.
    pub prompt: String,
}

/// Provider-independent conversation store operation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum StoreOperation {
    /// Load committed history before model preparation.
    LoadConversation {
        /// Stable conversation identity.
        conversation: StableId,
    },
    /// Persist the complete committed transcript before terminal success.
    PersistConversation {
        /// Stable conversation identity.
        conversation: StableId,
        /// Canonical transcript snapshot.
        entries: Vec<TranscriptEntry>,
    },
    /// Retrieve ordered context documents for a prompt.
    Retrieve {
        /// Owned semantic query.
        query: String,
        /// Maximum result count.
        limit: usize,
    },
}

/// Provider-independent retrieved document.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RetrievedDocument {
    /// Stable store-local document identity.
    pub id: String,
    /// Text supplied to the model.
    pub text: String,
    /// Additional deterministic string metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Owned effect completion sent back to the runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectCompletion {
    /// Operation entity copied only for correlation.
    pub operation: Entity,
    /// Operation generation copied at submission.
    pub generation: u64,
    /// Owned completion value.
    pub result: Result<EffectOutput, CanonicalError>,
}

/// Ordered incremental model delta entering through the effect boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectDelta {
    /// Model operation identity.
    pub operation: Entity,
    /// Operation generation copied at submission.
    pub generation: u64,
    /// Monotonic sequence within this model operation.
    pub sequence: u64,
    /// Optional provider correlation for diagnostics and audit.
    pub provider_correlation: Option<String>,
    /// Owned incremental content; deltas are not entities.
    pub kind: EffectDeltaKind,
}

/// Canonical high-volume streaming delta content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectDeltaKind {
    /// Incremental assistant text.
    Text(String),
    /// Incremental tool name or JSON arguments.
    ToolCall {
        /// Provider-facing tool-call ID.
        id: String,
        /// Rig correlation ID preserved through repair and execution.
        internal_call_id: String,
        /// Tool name or argument fragment.
        content: ToolCallDeltaContent,
    },
}

#[derive(Clone, Debug)]
enum EffectIngress {
    Completion(EffectCompletion),
    Delta(EffectDelta),
}

/// Typed provider-independent effect result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum EffectOutput {
    /// Model operation result.
    Model(ModelEffectOutput),
    /// Tool operation result.
    Tool(ToolEffectOutput),
    /// Reconciled discovery payload.
    Discovery(DiscoveryEffectOutput),
    /// Store operation result.
    Store(StoreEffectOutput),
    /// External approval decision.
    PolicyApproval(PolicyApprovalEffectOutput),
}

/// Provider-independent model output.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelEffectOutput {
    /// Complete canonical assistant message, including provider-significant
    /// reasoning and correlation data when available.
    pub assistant_message: Option<serde_json::Value>,
    /// Model-visible text.
    pub text: String,
    /// Usage finalized with the call.
    pub usage: Usage,
    /// Ordered tool calls requested by this model operation.
    pub tool_calls: Vec<ModelToolCall>,
}

/// Provider-independent requested tool call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelToolCall {
    /// Provider correlation identifier.
    pub id: String,
    /// Provider-facing tool-result identifier before runtime normalization.
    pub provider_result_id: String,
    /// Provider-specific call ID, when the API represents it separately.
    pub provider_call_id: Option<String>,
    /// Advertised provider-facing name.
    pub name: String,
    /// Structured provider arguments.
    pub arguments: serde_json::Value,
}

/// Provider-independent tool execution result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolEffectOutput {
    /// Provider correlation identifier copied from the call.
    pub call_id: String,
    /// Provider-facing tool-result identifier before runtime normalization.
    pub provider_result_id: String,
    /// Provider-specific call ID, when the API represents it separately.
    pub provider_call_id: Option<String>,
    /// Provider-facing tool name.
    pub name: String,
    /// Raw canonical model-visible output retained independently of rendering.
    pub raw: serde_json::Value,
    /// Independently rewritable model-visible presentation.
    pub presentation: String,
    /// Operator and retry metadata kept separate from model-visible content.
    pub failure: Option<ToolEffectFailure>,
}

/// Canonical tool failure metadata retained for policy, telemetry, and audit.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolEffectFailure {
    /// Operator-facing diagnostic, which is never copied into the transcript.
    pub message: String,
    /// Whether a retry policy may repeat the operation.
    pub retryable: Option<bool>,
    /// Stable normalized failure classification.
    pub kind: crate::tool::ToolErrorKind,
    /// Whether the tool intentionally refused the request.
    pub refusal: bool,
}

/// Owned discovery result for one source generation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DiscoveryEffectOutput {
    /// Complete capability set observed for this generation.
    pub tools: Vec<DiscoveredTool>,
}

/// Provider-independent discovered tool definition.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DiscoveredTool {
    /// Stable key within the discovery source.
    pub key: String,
    /// Provider-facing name.
    pub name: String,
    /// Provider-facing description.
    pub description: String,
    /// JSON Schema advertised for tool arguments.
    pub parameters: serde_json::Value,
    /// Explicit source-defined order.
    pub order: u32,
    /// External executable revision.
    pub revision: u64,
}

/// Provider-independent store result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum StoreEffectOutput {
    /// Loaded committed conversation history.
    LoadedConversation(Vec<TranscriptEntry>),
    /// Required transcript persistence completed.
    Persisted,
    /// Ordered retrieval results.
    Retrieved(Vec<RetrievedDocument>),
}

/// Provider-independent approval result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PolicyApprovalEffectOutput {
    /// Whether progression is approved.
    pub approved: bool,
    /// Optional audit explanation.
    pub reason: Option<String>,
}

/// Operation kind used to reject type-confused completions.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
enum OperationKind {
    Model,
    Tool,
    Discovery,
    Store,
    PolicyApproval,
}

#[derive(Component)]
struct DiscoveryApplied;

#[derive(Component)]
struct PolicyApprovalApplied;

/// Authoritative state for an atomic logical tool batch.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct ToolBatchState {
    /// Model operation that produced this batch and owns its immutable snapshot.
    pub source_model_operation: Entity,
    /// Number of operations that must settle before commit.
    pub expected: u32,
    /// Prevents a settled batch from committing twice.
    pub committed: bool,
}

/// Ordered delta state owned by a model operation entity.
#[derive(Component, Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelStreamState {
    /// Next accepted sequence; duplicates and gaps are rejected.
    pub next_sequence: u64,
    /// Accepted text accumulated independently of subscriber delivery.
    pub aggregated_text: String,
    /// Whether the owning run uses the streaming facade.
    pub streaming: bool,
}

/// A committed turn entity.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct CommittedTurn {
    /// Explicit order within the run.
    pub index: u32,
    /// Model output committed at this position.
    pub output: ModelEffectOutput,
}

/// World-resident runtime schedule.
#[derive(ScheduleLabel, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RigSchedule;

/// Public semantic stages for extension ordering.
#[derive(SystemSet, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RigSet {
    /// Ingest external commands and effect completions.
    Ingest,
    /// Reconcile world structure and stable identities.
    Reconcile,
    /// Prepare immutable decisions and operation entities.
    Prepare,
    /// Apply ordered policy systems.
    Policy,
    /// Submit owned external effects.
    Dispatch,
    /// Validate and apply effect completions.
    Apply,
    /// Commit deterministic outcomes.
    Commit,
    /// Persist required state.
    Persist,
    /// Publish observations.
    Publish,
    /// Cancel, retire, and clean up.
    Cleanup,
}

/// Bounded runtime queue configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    /// Maximum pending facade commands.
    pub command_capacity: usize,
    /// Maximum pending external effects.
    pub effect_capacity: usize,
    /// Maximum pending effect completions.
    pub completion_capacity: usize,
    /// Per-run stream subscription capacity and slow-consumer threshold.
    pub subscriber_capacity: usize,
    /// Maximum schedule passes in [`Runtime::run_until_stalled`].
    pub progress_limit: usize,
    /// Maximum schedule ticks an external effect may remain in flight.
    pub effect_timeout_ticks: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            command_capacity: 256,
            effect_capacity: 256,
            completion_capacity: 256,
            subscriber_capacity: 64,
            progress_limit: 1024,
            effect_timeout_ticks: 1024,
        }
    }
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
    /// Stable model identity remapped during loading.
    pub model_id: StableId,
    /// Optional structured-output configuration.
    pub output_requirement: Option<OutputRequirement>,
    /// Optional vector retrieval configuration.
    pub retrieval_requirement: Option<RetrievalRequirement>,
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
        &UsesModel,
        Option<&OutputRequirement>,
        Option<&RetrievalRequirement>,
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
                model,
                output_requirement,
                retrieval_requirement,
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
                        model_id,
                        output_requirement: output_requirement.cloned(),
                        retrieval_requirement: retrieval_requirement.cloned(),
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
        if let Some(retrieval_requirement) = record.retrieval_requirement {
            entity.insert(retrieval_requirement);
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
    world.insert_resource(waker.clone());
    world.insert_resource(Messages::<EffectIngressMessage>::default());
    world.add_observer(bind_builtin_policy_observer);

    let mut schedule = Schedule::new(RigSchedule);
    schedule.set_build_settings(ScheduleBuildSettings {
        ambiguity_detection: LogLevel::Error,
        ..ScheduleBuildSettings::default()
    });
    schedule.configure_sets(
        (
            RigSet::Ingest,
            RigSet::Reconcile,
            RigSet::Prepare,
            RigSet::Policy,
            RigSet::Dispatch,
            RigSet::Apply,
            RigSet::Commit,
            RigSet::Persist,
            RigSet::Publish,
            RigSet::Cleanup,
        )
            .chain(),
    );
    schedule.add_systems(
        (
            advance_runtime_clock,
            ingest_commands,
            ingest_observations,
            ingest_completions,
        )
            .chain()
            .in_set(RigSet::Ingest),
    );
    schedule.add_systems(
        (
            reconcile_agent_control,
            reconcile_run_control,
            reconcile_discovery_operations,
            reconcile_stable_ids,
        )
            .chain()
            .in_set(RigSet::Reconcile),
    );
    schedule.add_systems(
        (
            prepare_store_operations,
            prepare_model_operations,
            publish_invalid_call_observations,
            initialize_request_policy_evaluations,
            initialize_invalid_tool_call_policy_evaluations,
            initialize_tool_call_policy_evaluations,
        )
            .chain()
            .in_set(RigSet::Prepare),
    );
    schedule.add_systems(
        (
            evaluate_request_policies,
            evaluate_invalid_tool_call_policies,
            evaluate_tool_call_policies,
            evaluate_completion_response_policies,
            evaluate_tool_result_policies,
            evaluate_text_delta_policies,
        )
            .chain()
            .in_set(RigSet::Policy),
    );
    schedule.add_systems(
        (
            publish_prepared_operation_observations,
            dispatch_model_operations,
            dispatch_tool_operations,
            dispatch_discovery_operations,
            dispatch_store_operations,
            dispatch_policy_approval_operations,
        )
            .chain()
            .in_set(RigSet::Dispatch),
    );
    schedule.add_systems(
        (
            apply_effect_ingress,
            apply_policy_approval_results,
            expire_effects,
        )
            .chain()
            .in_set(RigSet::Apply),
    );
    schedule.add_systems(
        (
            initialize_completion_response_policy_evaluations,
            initialize_tool_result_policy_evaluations,
            publish_applied_policy_observations,
            commit_store_operations,
            commit_model_operations,
            commit_tool_batches,
            commit_child_results,
        )
            .chain()
            .in_set(RigSet::Commit),
    );
    schedule.add_systems(
        (
            propagate_parent_cancellation,
            propagate_cancellation,
            cleanup_observed_runs,
            cleanup_retired_tools,
            update_effect_messages,
        )
            .chain()
            .in_set(RigSet::Cleanup),
    );
    schedule.add_systems(
        (
            publish_run_terminal_observations,
            publish_terminal_streams,
            update_runtime_metrics,
        )
            .chain()
            .in_set(RigSet::Publish),
    );
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

    /// Retires a policy from future snapshots while preserving accepted work.
    pub fn retire_policy(&mut self, policy: Entity) -> Result<(), SpawnError> {
        let Some(mut entity) = self.world.get_entity_mut(policy).ok() else {
            return Err(SpawnError::StaleEntity(policy));
        };
        if !entity.contains::<Policy>() {
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

    /// Restores a validated active-run graph against the current domain.
    pub fn restore_active_run(
        &mut self,
        snapshot: ActiveRunSnapshot,
    ) -> Result<RestoredRuns, ActiveRunSnapshotError> {
        restore_active_run(&mut self.world, snapshot)
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
    control_allows_internal_progress(world.get::<RunControl>(run))
}

fn advance_runtime_clock(mut clock: ResMut<RuntimeClock>) {
    clock.tick = clock.tick.saturating_add(1);
}

// Bevy system parameters are independently borrow-checked runtime inputs; grouping
// them would obscure their access pattern without reducing system complexity.
#[allow(clippy::too_many_arguments)]
fn ingest_commands(
    mut commands: Commands,
    ingress: Res<CommandIngress>,
    runtime: Res<RuntimeIdentity>,
    identities: Query<&StableId>,
    models: Query<(Entity, &StableId, &TenantId), With<ModelCapability>>,
    mut agents: Query<(
        &TenantId,
        &Agent,
        Option<&InvalidToolCallBudget>,
        Option<&mut AgentControl>,
    )>,
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
                    conversation,
                    memory_loaded: false,
                    memory_store: None,
                    retrieval_loaded: false,
                    retrieval_store: None,
                    retrieved_documents: Vec::new(),
                    pending_output: None,
                };
                let run = match agents.get_mut(agent) {
                    Ok((tenant, _, invalid_budget, control))
                        if control
                            .as_deref()
                            .is_none_or(|control| matches!(control, AgentControl::Running)) =>
                    {
                        record.max_invalid_tool_call_retries =
                            invalid_budget.map_or(0, |budget| budget.max_retries);
                        commands
                            .spawn((
                                run_id,
                                tenant.clone(),
                                RunOf(agent),
                                RunState::Queued,
                                RunControl::Running,
                                record,
                            ))
                            .id()
                    }
                    Ok((tenant, _, _, _)) => commands
                        .spawn((
                            run_id,
                            tenant.clone(),
                            RunOf(agent),
                            RunState::Failed(CanonicalError::AgentAdmissionDenied),
                            RunControl::Running,
                            record,
                        ))
                        .id(),
                    Err(_) => commands
                        .spawn((
                            run_id,
                            RunState::Failed(CanonicalError::StaleEntity("agent".to_owned())),
                            RunControl::Running,
                            record,
                        ))
                        .id(),
                };
                if let Some(subscriber) = subscriber {
                    commands.spawn((
                        SubscriptionOf(run),
                        StreamSink(subscriber),
                        SubscriptionState::Active,
                    ));
                }
                if let Some((parent, ordinal)) = parent {
                    if runs.get_mut(parent).is_ok() {
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
                if let Ok((_, _, _, existing)) = agents.get_mut(agent)
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
                        OperationState {
                            generation,
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
    mut runs: Query<(Entity, &mut RunControl, Option<&RunOperations>)>,
    mut operations: Query<(&mut OperationState, Option<&EffectDeadline>)>,
    cancellations: Res<CancellationOutbox>,
    mut progress: ResMut<Progress>,
) {
    for (_run, mut control, run_operations) in &mut runs {
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
                        .is_ok_and(|(state, _)| matches!(state.phase, OperationPhase::InFlight))
                });
                if !has_in_flight {
                    *control = RunControl::Paused(mode);
                    mark_progress(&mut progress);
                }
            }
            PauseMode::CancelAndSuspend => {
                for operation_entity in operation_entities {
                    let Ok((mut operation, deadline)) = operations.get_mut(operation_entity) else {
                        continue;
                    };
                    if !matches!(operation.phase, OperationPhase::InFlight) {
                        continue;
                    }
                    let generation = operation.generation;
                    let _ = cancellations.0.try_send(EffectCancellation {
                        operation: operation_entity,
                        generation,
                    });
                    operation.generation = operation.generation.saturating_add(1);
                    operation.phase = OperationPhase::Prepared;
                    if deadline.is_some() {
                        commands.entity(operation_entity).remove::<EffectDeadline>();
                    }
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
        (Entity, &DiscoveryOperationOf, &OperationState),
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
    for (operation_entity, operation_of, operation) in &operations {
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
        ) || source.generation != operation.generation
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
            let stable_id = StableId(format!(
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

fn prepare_store_operations(
    mut commands: Commands,
    mut runs: Query<(
        Entity,
        &TenantId,
        &RunOf,
        &RunControl,
        &mut RunRecord,
        &mut RunState,
    )>,
    agents: Query<(Option<&AgentStoreGrants>, Option<&RetrievalRequirement>)>,
    grants: Query<(&StableId, &TenantId, &StoreGrant, &StoreGrantForStore)>,
    stores: Query<(&StableId, &TenantId, &StoreCapability)>,
    mut progress: ResMut<Progress>,
) {
    for (run_entity, run_tenant, run_of, control, mut record, mut run_state) in &mut runs {
        if !matches!(*control, RunControl::Running) || !matches!(*run_state, RunState::Queued) {
            continue;
        }
        let Ok((agent_grants, retrieval_requirement)) = agents.get(run_of.get()) else {
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
                        OperationState {
                            generation: 0,
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
                        OperationState {
                            generation: 0,
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
    }
}

#[allow(clippy::type_complexity)]
fn prepare_model_operations(
    mut commands: Commands,
    mut runs: Query<(
        Entity,
        &TenantId,
        &RunOf,
        &RunControl,
        &RunRecord,
        &mut RunState,
    )>,
    agents: Query<(
        &TenantId,
        &Agent,
        &UsesModel,
        Option<&AgentToolGrants>,
        Option<&OutputRequirement>,
    )>,
    models: Query<(&StableId, &TenantId, &ModelCapability)>,
    grants: Query<(&StableId, &TenantId, &ToolGrant, &GrantForTool)>,
    tools: Query<(&StableId, &TenantId, &ToolCapability)>,
    mut progress: ResMut<Progress>,
) {
    for (run_entity, run_tenant, run_of, control, record, mut run_state) in &mut runs {
        if !matches!(*control, RunControl::Running)
            || !matches!(*run_state, RunState::Queued)
            || !record.memory_loaded
            || !record.retrieval_loaded
        {
            continue;
        }
        let Ok((agent_tenant, agent, model_relation, agent_grants, output_requirement)) =
            agents.get(run_of.get())
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
                candidates.push((
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
            |(left_grant, left_tool, left_name, left_id, _),
             (right_grant, right_tool, right_name, right_id, _)| {
                left_grant
                    .cmp(right_grant)
                    .then_with(|| left_tool.cmp(right_tool))
                    .then_with(|| left_name.cmp(right_name))
                    .then_with(|| left_id.cmp(right_id))
            },
        );
        let mut names = HashSet::new();
        let tools = candidates
            .into_iter()
            .filter_map(|(_, _, name, _, mut decision)| {
                if !names.insert(name) {
                    return None;
                }
                decision.order = u32::try_from(names.len() - 1).unwrap_or(u32::MAX);
                Some(decision)
            })
            .collect::<Vec<_>>();
        let mut documents = agent.documents.clone();
        documents.extend(record.retrieved_documents.iter().cloned());
        let operation = commands
            .spawn((
                OperationOf(run_entity),
                OperationState {
                    generation: 0,
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
                    tool_results: Vec::new(),
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
        *run_state = RunState::WaitingModel { operation };
        mark_progress(&mut progress);
    }
}

fn initialize_request_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &PendingModelRequest, &OperationState),
        Without<RequestPolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunControl>)>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(Entity, &StableId, &Policy, Option<&PolicyStatus>)>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, pending, state) in &operations {
        if !matches!(state.phase, OperationPhase::Prepared) {
            continue;
        }
        let Ok((run_of, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = agents
            .get(run_of.get())
            .ok()
            .flatten()
            .into_iter()
            .flat_map(|agent_policies| agent_policies.iter())
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, policy, status)| {
                accepts_new_policy_evaluations(*status)
                    && policy.rule.applies_to(PolicyPoint::Request)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _), (_, right_id, right, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
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
                    phase: RequestPolicyEvaluationPhase::Evaluating,
                },
            ));
        }
        mark_progress(&mut progress);
    }
}

fn bind_builtin_policy_observer(
    event: On<Add, Policy>,
    policies: Query<&Policy>,
    mut commands: Commands,
) {
    let Ok(policy) = policies.get(event.entity) else {
        return;
    };
    if policy.rule.applies_to(PolicyPoint::Request) && !matches!(policy.rule, PolicyRule::Custom(_))
    {
        commands
            .entity(event.entity)
            .observe(apply_builtin_request_policy);
    }
    if policy.rule.applies_to(PolicyPoint::ToolCall)
        && !matches!(policy.rule, PolicyRule::Custom(_))
    {
        commands
            .entity(event.entity)
            .observe(apply_builtin_tool_call_policy);
    }
    if policy.rule.applies_to(PolicyPoint::InvalidToolCall)
        && !matches!(policy.rule, PolicyRule::Custom(_))
    {
        commands
            .entity(event.entity)
            .observe(apply_builtin_invalid_tool_call_policy);
    }
    if policy.rule.applies_to(PolicyPoint::ToolResult)
        && !matches!(policy.rule, PolicyRule::Custom(_))
    {
        commands
            .entity(event.entity)
            .observe(apply_builtin_tool_result_policy);
    }
    if policy.rule.applies_to(PolicyPoint::CompletionResponse)
        && !matches!(policy.rule, PolicyRule::Custom(_))
    {
        commands
            .entity(event.entity)
            .observe(apply_builtin_completion_response_policy);
    }
    if policy.rule.applies_to(PolicyPoint::TextDelta)
        && !matches!(policy.rule, PolicyRule::Custom(_))
    {
        commands
            .entity(event.entity)
            .observe(apply_builtin_text_delta_policy);
    }
}

fn apply_builtin_request_policy(
    mut invocation: On<RequestPolicyInvocation>,
    policies: Query<&Policy>,
) {
    let Ok(policy) = policies.get(invocation.policy) else {
        return;
    };
    invocation.decision = Some(match &policy.rule {
        PolicyRule::Allow => RequestPolicyDecision::Continue,
        PolicyRule::PatchRequest(patch) => RequestPolicyDecision::Patch(patch.clone()),
        PolicyRule::DenyPromptContains(needle) => {
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
        PolicyRule::RequireApproval { prompt, .. } => {
            RequestPolicyDecision::AwaitApproval(prompt.clone())
        }
        PolicyRule::RewriteToolArguments { .. }
        | PolicyRule::SkipToolCall { .. }
        | PolicyRule::RepairInvalidTool { .. }
        | PolicyRule::RetryInvalidTool { .. }
        | PolicyRule::SkipInvalidTool { .. }
        | PolicyRule::RewriteToolResult { .. }
        | PolicyRule::StopToolResult { .. }
        | PolicyRule::RewriteCompletionText { .. }
        | PolicyRule::StopCompletionContains { .. }
        | PolicyRule::StopTextDeltaContains { .. }
        | PolicyRule::Custom(_) => return,
    });
}

fn apply_builtin_tool_call_policy(
    mut invocation: On<ToolCallPolicyInvocation>,
    policies: Query<&Policy>,
) {
    let Ok(policy) = policies.get(invocation.policy) else {
        return;
    };
    invocation.decision = Some(match &policy.rule {
        PolicyRule::RewriteToolArguments { tool, arguments }
            if tool
                .as_ref()
                .is_none_or(|name| name == &invocation.call.decision.name) =>
        {
            ToolCallPolicyDecision::Rewrite(arguments.clone())
        }
        PolicyRule::SkipToolCall { tool, reason }
            if tool
                .as_ref()
                .is_none_or(|name| name == &invocation.call.decision.name) =>
        {
            ToolCallPolicyDecision::Skip(reason.clone())
        }
        PolicyRule::RewriteToolArguments { .. } | PolicyRule::SkipToolCall { .. } => {
            ToolCallPolicyDecision::Run
        }
        PolicyRule::RequireApproval { prompt, .. } => {
            ToolCallPolicyDecision::AwaitApproval(prompt.clone())
        }
        PolicyRule::Allow
        | PolicyRule::DenyPromptContains(_)
        | PolicyRule::PatchRequest(_)
        | PolicyRule::RepairInvalidTool { .. }
        | PolicyRule::RetryInvalidTool { .. }
        | PolicyRule::SkipInvalidTool { .. }
        | PolicyRule::RewriteToolResult { .. }
        | PolicyRule::StopToolResult { .. }
        | PolicyRule::RewriteCompletionText { .. }
        | PolicyRule::StopCompletionContains { .. }
        | PolicyRule::StopTextDeltaContains { .. }
        | PolicyRule::Custom(_) => return,
    });
}

fn apply_builtin_invalid_tool_call_policy(
    mut invocation: On<InvalidToolCallPolicyInvocation>,
    policies: Query<&Policy>,
) {
    let Ok(policy) = policies.get(invocation.policy) else {
        return;
    };
    let emitted = &invocation.invalid.call.name;
    invocation.decision = Some(match &policy.rule {
        PolicyRule::RepairInvalidTool { from, to }
            if from.as_ref().is_none_or(|name| name == emitted) =>
        {
            InvalidToolCallPolicyDecision::Repair(to.clone())
        }
        PolicyRule::RetryInvalidTool { tool, feedback }
            if tool.as_ref().is_none_or(|name| name == emitted) =>
        {
            InvalidToolCallPolicyDecision::Retry(feedback.clone())
        }
        PolicyRule::SkipInvalidTool { tool, reason }
            if tool.as_ref().is_none_or(|name| name == emitted) =>
        {
            InvalidToolCallPolicyDecision::Skip(reason.clone())
        }
        PolicyRule::RepairInvalidTool { .. }
        | PolicyRule::RetryInvalidTool { .. }
        | PolicyRule::SkipInvalidTool { .. } => InvalidToolCallPolicyDecision::Continue,
        PolicyRule::RequireApproval { prompt, .. } => {
            InvalidToolCallPolicyDecision::AwaitApproval(prompt.clone())
        }
        PolicyRule::Allow
        | PolicyRule::DenyPromptContains(_)
        | PolicyRule::PatchRequest(_)
        | PolicyRule::RewriteToolArguments { .. }
        | PolicyRule::SkipToolCall { .. }
        | PolicyRule::RewriteToolResult { .. }
        | PolicyRule::StopToolResult { .. }
        | PolicyRule::RewriteCompletionText { .. }
        | PolicyRule::StopCompletionContains { .. }
        | PolicyRule::StopTextDeltaContains { .. }
        | PolicyRule::Custom(_) => return,
    });
}

fn apply_builtin_tool_result_policy(
    mut invocation: On<ToolResultPolicyInvocation>,
    policies: Query<&Policy>,
) {
    let Ok(policy) = policies.get(invocation.policy) else {
        return;
    };
    let tool_name = &invocation.input.decision.name;
    invocation.decision = Some(match &policy.rule {
        PolicyRule::RewriteToolResult { tool, presentation }
            if tool.as_ref().is_none_or(|name| name == tool_name) =>
        {
            ToolResultPolicyDecision::Rewrite(presentation.clone())
        }
        PolicyRule::StopToolResult { tool, reason }
            if tool.as_ref().is_none_or(|name| name == tool_name) =>
        {
            ToolResultPolicyDecision::Stop(reason.clone())
        }
        PolicyRule::RewriteToolResult { .. } | PolicyRule::StopToolResult { .. } => {
            ToolResultPolicyDecision::Keep
        }
        PolicyRule::RequireApproval { prompt, .. } => {
            ToolResultPolicyDecision::AwaitApproval(prompt.clone())
        }
        PolicyRule::Allow
        | PolicyRule::DenyPromptContains(_)
        | PolicyRule::PatchRequest(_)
        | PolicyRule::RewriteToolArguments { .. }
        | PolicyRule::SkipToolCall { .. }
        | PolicyRule::RepairInvalidTool { .. }
        | PolicyRule::RetryInvalidTool { .. }
        | PolicyRule::SkipInvalidTool { .. }
        | PolicyRule::RewriteCompletionText { .. }
        | PolicyRule::StopCompletionContains { .. }
        | PolicyRule::StopTextDeltaContains { .. }
        | PolicyRule::Custom(_) => return,
    });
}

fn apply_builtin_completion_response_policy(
    mut invocation: On<CompletionResponsePolicyInvocation>,
    policies: Query<&Policy>,
) {
    let Ok(policy) = policies.get(invocation.policy) else {
        return;
    };
    invocation.decision = Some(match &policy.rule {
        PolicyRule::RewriteCompletionText { text } => {
            CompletionResponsePolicyDecision::RewriteText(text.clone())
        }
        PolicyRule::StopCompletionContains { needle, reason }
            if invocation.response.text.contains(needle) =>
        {
            CompletionResponsePolicyDecision::Stop(reason.clone())
        }
        PolicyRule::StopCompletionContains { .. } => CompletionResponsePolicyDecision::Continue,
        PolicyRule::RequireApproval { prompt, .. } => {
            CompletionResponsePolicyDecision::AwaitApproval(prompt.clone())
        }
        PolicyRule::Allow
        | PolicyRule::DenyPromptContains(_)
        | PolicyRule::PatchRequest(_)
        | PolicyRule::RewriteToolArguments { .. }
        | PolicyRule::SkipToolCall { .. }
        | PolicyRule::RepairInvalidTool { .. }
        | PolicyRule::RetryInvalidTool { .. }
        | PolicyRule::SkipInvalidTool { .. }
        | PolicyRule::RewriteToolResult { .. }
        | PolicyRule::StopToolResult { .. }
        | PolicyRule::StopTextDeltaContains { .. }
        | PolicyRule::Custom(_) => return,
    });
}

fn apply_builtin_text_delta_policy(
    mut invocation: On<TextDeltaPolicyInvocation>,
    policies: Query<&Policy>,
) {
    let Ok(policy) = policies.get(invocation.policy) else {
        return;
    };
    invocation.decision = Some(match &policy.rule {
        PolicyRule::StopTextDeltaContains { needle, reason }
            if invocation.delta.contains(needle) =>
        {
            TextDeltaPolicyDecision::Stop(reason.clone())
        }
        PolicyRule::StopTextDeltaContains { .. } => TextDeltaPolicyDecision::Continue,
        PolicyRule::RequireApproval { prompt, .. } => {
            TextDeltaPolicyDecision::AwaitApproval(prompt.clone())
        }
        PolicyRule::Allow
        | PolicyRule::DenyPromptContains(_)
        | PolicyRule::PatchRequest(_)
        | PolicyRule::RewriteToolArguments { .. }
        | PolicyRule::SkipToolCall { .. }
        | PolicyRule::RepairInvalidTool { .. }
        | PolicyRule::RetryInvalidTool { .. }
        | PolicyRule::SkipInvalidTool { .. }
        | PolicyRule::RewriteToolResult { .. }
        | PolicyRule::StopToolResult { .. }
        | PolicyRule::RewriteCompletionText { .. }
        | PolicyRule::StopCompletionContains { .. }
        | PolicyRule::Custom(_) => return,
    });
}

fn merge_request_patch(input: &mut ModelEffectInput, patch: RequestPatch) {
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
    if let Some(history) = patch.history {
        input.history = history;
    }
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
            OperationState {
                generation: 0,
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
        let mut invocation = RequestPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            request: evaluation.effective.clone(),
            decision: None,
        };
        world.trigger(RequestPolicyInvoked {
            policy: policy.entity,
            evaluation: evaluation_entity,
            policy_id: policy.id.clone(),
            revision: policy.revision,
            cursor,
        });
        world.trigger_ref(&mut invocation);
        let decision = invocation.decision.clone();
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
                    merge_request_patch(&mut evaluation.effective, patch);
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
                        reason,
                    };
                }
                if let Some(mut state) = world.get_mut::<OperationState>(operation) {
                    state.phase = OperationPhase::Settled(OperationOutcome::Failure(
                        CanonicalError::PolicyDenied {
                            policy: policy.id.as_str().to_owned(),
                        },
                    ));
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(CanonicalError::PolicyDenied {
                        policy: policy.id.as_str().to_owned(),
                    });
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn initialize_tool_call_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &PendingToolCall, &OperationState),
        Without<ToolCallPolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunControl>)>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(Entity, &StableId, &Policy, Option<&PolicyStatus>)>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, pending, state) in &operations {
        if !matches!(state.phase, OperationPhase::Prepared) {
            continue;
        }
        let Ok((run_of, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = agents
            .get(run_of.get())
            .ok()
            .flatten()
            .into_iter()
            .flat_map(|agent_policies| agent_policies.iter())
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, policy, status)| {
                accepts_new_policy_evaluations(*status)
                    && policy.rule.applies_to(PolicyPoint::ToolCall)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _), (_, right_id, right, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
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
        let mut invocation = ToolCallPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            call: evaluation.effective.clone(),
            decision: None,
        };
        world.trigger_ref(&mut invocation);
        match invocation.decision {
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
                let effective = invocation.call.clone();
                world.entity_mut(operation).insert(effective.clone());
                world.entity_mut(operation).remove::<PendingToolCall>();
                if let Some(mut state) = world.get_mut::<OperationState>(operation) {
                    state.phase = OperationPhase::Settled(OperationOutcome::Success(
                        EffectOutput::Tool(ToolEffectOutput {
                            call_id: effective.call_id,
                            provider_result_id: effective.provider_result_id,
                            provider_call_id: effective.provider_call_id,
                            name: effective.decision.name,
                            raw: serde_json::json!({"skipped": true, "reason": reason}),
                            presentation: reason.clone(),
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
                        reason,
                    };
                }
                if let Some(mut state) = world.get_mut::<OperationState>(operation) {
                    state.phase = OperationPhase::Settled(OperationOutcome::Failure(
                        CanonicalError::PolicyDenied {
                            policy: policy.id.as_str().to_owned(),
                        },
                    ));
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(CanonicalError::PolicyDenied {
                        policy: policy.id.as_str().to_owned(),
                    });
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn initialize_invalid_tool_call_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (
            Entity,
            &OperationOf,
            &PendingInvalidToolCall,
            &OperationState,
        ),
        Without<InvalidToolCallPolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunControl>)>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(Entity, &StableId, &Policy, Option<&PolicyStatus>)>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, invalid, state) in &operations {
        if !matches!(state.phase, OperationPhase::Prepared) {
            continue;
        }
        let Ok((run_of, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = agents
            .get(run_of.get())
            .ok()
            .flatten()
            .into_iter()
            .flat_map(|agent_policies| agent_policies.iter())
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, policy, status)| {
                accepts_new_policy_evaluations(*status)
                    && policy.rule.applies_to(PolicyPoint::InvalidToolCall)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _), (_, right_id, right, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
            })
            .collect::<Vec<_>>();
        commands.entity(operation).insert((
            InvalidToolCallPolicyInitialized,
            AcceptedInvalidToolCallPolicies(snapshot.clone()),
        ));
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
                raw: serde_json::json!({"invalid_tool_call": kind, "feedback": feedback}),
                presentation: feedback,
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
        let mut invocation = InvalidToolCallPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            invalid: evaluation.invalid.clone(),
            decision: None,
        };
        world.trigger_ref(&mut invocation);
        match invocation.decision {
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
                        reason,
                    };
                }
                let error = CanonicalError::PolicyDenied {
                    policy: policy.id.as_str().to_owned(),
                };
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
    let updated = if let Some(mut state) = world.get_mut::<OperationState>(entity)
        && state.generation == generation
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

fn dispatch_model_operations(world: &mut World) {
    let outbox = world.resource::<EffectOutbox>().0.clone();
    let mut query = world.query_filtered::<(
        Entity,
        &OperationOf,
        &ModelEffectInput,
        &OperationState,
    ), (With<ModelEffectInput>, Without<ToolEffectInput>)>();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, run, input, state)| {
            if !matches!(state.phase, OperationPhase::Prepared)
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
            let run_id = world.get::<StableId>(run.get())?;
            Some((
                run_id.clone(),
                run.get(),
                entity,
                state.generation,
                input.clone(),
            ))
        })
        .collect::<Vec<_>>();
    prepared.sort_by(
        |(left_id, _, left_entity, _, _), (right_id, _, right_entity, _, _)| {
            left_id
                .cmp(right_id)
                .then_with(|| left_entity.to_bits().cmp(&right_entity.to_bits()))
        },
    );
    let mut made_progress = false;
    for (_, run, entity, generation, input) in prepared {
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
            Err(TrySendError::Full(_)) => None,
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

fn publish_prepared_operation_observations(
    mut commands: Commands,
    model_operations: PreparedModelOperations<'_, '_>,
    tool_operations: PreparedToolOperations<'_, '_>,
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

fn publish_applied_policy_observations(
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
            raw: raw.clone(),
            presentation: effective.presentation,
        });
    }
}

fn dispatch_tool_operations(world: &mut World) {
    let outbox = world.resource::<EffectOutbox>().0.clone();
    let mut batch_query = world.query::<&BatchOf>();
    let mut query = world.query_filtered::<
        (Entity, &OperationOfBatch, &ToolEffectInput, &OperationState),
        (With<ToolEffectInput>, Without<ModelEffectInput>),
    >();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, batch, input, state)| {
            let run = batch_query.get(world, batch.get()).ok()?.get();
            if !matches!(state.phase, OperationPhase::Prepared)
                || !matches!(world.get::<RunControl>(run), Some(RunControl::Running))
                || !matches!(
                    world.get::<RunState>(run),
                    Some(RunState::WaitingTools { .. })
                )
            {
                return None;
            }
            Some((
                batch.get().to_bits(),
                input.index,
                run,
                entity,
                state.generation,
                input.clone(),
            ))
        })
        .collect::<Vec<_>>();
    prepared.sort_by_key(|(batch, index, _, entity, _, _)| (*batch, *index, entity.to_bits()));
    let mut made_progress = false;
    for (_, index, run, entity, generation, input) in prepared {
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
            Err(TrySendError::Full(_)) => None,
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
    let mut query = world.query_filtered::<
        (Entity, &DiscoveryEffectInput, &OperationState),
        With<DiscoveryEffectInput>,
    >();
    let mut prepared = query
        .iter(world)
        .filter(|(_, _, state)| matches!(state.phase, OperationPhase::Prepared))
        .map(|(entity, input, state)| {
            (
                input.source_id.clone(),
                entity,
                state.generation,
                input.clone(),
            )
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
            Err(TrySendError::Full(_)) => None,
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
    let mut query = world.query_filtered::<
        (Entity, &OperationOf, &StoreEffectInput, &OperationState),
        With<StoreEffectInput>,
    >();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, run, input, state)| {
            if !matches!(state.phase, OperationPhase::Prepared)
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
            let run_id = world.get::<StableId>(run.get())?;
            Some((run_id.clone(), entity, state.generation, input.clone()))
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
            Err(TrySendError::Full(_)) => None,
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
        &OperationState,
    )>();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, run, input, state)| {
            if !matches!(state.phase, OperationPhase::Prepared)
                || !matches!(
                    world.get::<RunControl>(run.get()),
                    Some(RunControl::Running)
                )
            {
                return None;
            }
            let run_id = world.get::<StableId>(run.get())?;
            Some((run_id.clone(), entity, state.generation, input.clone()))
        })
        .collect::<Vec<_>>();
    prepared.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.to_bits().cmp(&right.1.to_bits()))
    });
    let mut made_progress = false;
    for (_, entity, generation, input) in prepared {
        let phase = match outbox.try_send(EffectRequest {
            operation: entity,
            generation,
            input: EffectInput::PolicyApproval(input),
        }) {
            Ok(()) => {
                made_progress = true;
                Some(OperationPhase::InFlight)
            }
            Err(TrySendError::Full(_)) => None,
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
    mut operations: Query<(Entity, &OperationOf, &mut OperationState)>,
    mut progress: ResMut<Progress>,
) {
    for (entity, operation_of, mut operation) in &mut operations {
        let Ok(run) = runs.get(operation_of.get()) else {
            continue;
        };
        if matches!(run, RunState::Cancelled)
            && !matches!(
                operation.phase,
                OperationPhase::Settled(_) | OperationPhase::Cancelled
            )
        {
            if matches!(operation.phase, OperationPhase::InFlight)
                && matches!(
                    cancellations.0.try_send(EffectCancellation {
                        operation: entity,
                        generation: operation.generation,
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
        &mut OperationState,
        Option<&mut ModelStreamState>,
        Option<&mut EffectDeadline>,
    )>,
    clock: Res<RuntimeClock>,
    timeout: Res<EffectTimeoutTicks>,
    runs: Query<(&RunOf, &RunRecord)>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(Entity, &StableId, &Policy, Option<&PolicyStatus>)>,
    run_subscriptions: Query<&RunSubscriptions>,
    mut subscriptions: Query<(&StreamSink, &mut SubscriptionState)>,
    mut progress: ResMut<Progress>,
) {
    for EffectIngressMessage(message) in messages.read() {
        match message {
            EffectIngress::Completion(completion) => {
                let Ok((kind, operation_of, mut state, stream_state, _)) =
                    operations.get_mut(completion.operation)
                else {
                    continue;
                };
                if state.generation != completion.generation
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
                            outcome: outcome.clone(),
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
            EffectIngress::Delta(delta) => {
                let Ok((kind, operation_of, state, stream_state, deadline)) =
                    operations.get_mut(delta.operation)
                else {
                    continue;
                };
                if !matches!(kind, OperationKind::Model)
                    || state.generation != delta.generation
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
                let Ok((run_of, record)) = runs.get(run) else {
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
                        let mut stream_policies = agents
                            .get(run_of.get())
                            .ok()
                            .flatten()
                            .into_iter()
                            .flat_map(|agent_policies| agent_policies.iter())
                            .filter_map(|entity| policies.get(entity).ok())
                            .filter(|(_, _, policy, status)| {
                                accepts_new_policy_evaluations(*status)
                                    && policy.rule.applies_to(PolicyPoint::TextDelta)
                            })
                            .collect::<Vec<_>>();
                        stream_policies.sort_by(
                            |(_, left_id, left, _), (_, right_id, right, _)| {
                                left.order
                                    .cmp(&right.order)
                                    .then_with(|| left_id.cmp(right_id))
                            },
                        );
                        let snapshot = stream_policies
                            .drain(..)
                            .map(|(entity, id, policy, _)| AcceptedPolicy {
                                id: id.clone(),
                                entity,
                                revision: policy.revision,
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
    mut parents: Query<(Entity, &ChildRuns, &mut RunRecord)>,
    children: Query<(
        &StableId,
        &ChildOrdinal,
        &RunState,
        Option<&ChildResultCommitted>,
    )>,
    mut progress: ResMut<Progress>,
) {
    for (parent, child_runs, mut parent_record) in &mut parents {
        let mut ordered = child_runs
            .iter()
            .filter_map(|child| {
                let (id, ordinal, state, committed) = children.get(child).ok()?;
                Some((ordinal.0, child, id.clone(), state, committed.is_some()))
            })
            .collect::<Vec<_>>();
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
                | RunState::WaitingStore { .. } => break,
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
            OperationPhase::Settled(_) => next.settled_operations += 1,
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

fn initialize_completion_response_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &ModelEffectInput, &OperationState),
        Without<CompletionResponsePolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunControl>)>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(Entity, &StableId, &Policy, Option<&PolicyStatus>)>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, request, state) in &operations {
        let OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Model(output))) =
            &state.phase
        else {
            continue;
        };
        let Ok((run_of, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = agents
            .get(run_of.get())
            .ok()
            .flatten()
            .into_iter()
            .flat_map(|agent_policies| agent_policies.iter())
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, policy, status)| {
                accepts_new_policy_evaluations(*status)
                    && policy.rule.applies_to(PolicyPoint::CompletionResponse)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _), (_, right_id, right, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
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
        let mut invocation = CompletionResponsePolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            request: evaluation.request,
            response: evaluation.effective,
            decision: None,
        };
        world.trigger_ref(&mut invocation);
        match invocation.decision {
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
                        reason,
                    };
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(CanonicalError::PolicyDenied {
                        policy: policy.id.as_str().to_owned(),
                    });
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
    let mut operations = HashSet::new();
    ready.retain(|(_, _, operation, _, _)| operations.insert(*operation));

    for (evaluation_entity, run, _, _, cursor) in ready {
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
        let mut invocation = TextDeltaPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation: evaluation.operation,
            turn: evaluation.turn,
            sequence: evaluation.sequence,
            delta: evaluation.delta,
            aggregated: evaluation.aggregated,
            decision: None,
        };
        world.trigger_ref(&mut invocation);
        match invocation.decision {
            Some(TextDeltaPolicyDecision::Continue) => {
                if let Some(mut state) =
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
                        reason,
                    };
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(CanonicalError::PolicyDenied {
                        policy: policy.id.as_str().to_owned(),
                    });
                }
            }
        }
        mark_progress(&mut world.resource_mut::<Progress>());
    }
}

fn initialize_tool_result_policy_evaluations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &ToolEffectInput, &OperationState),
        Without<ToolResultPolicyInitialized>,
    >,
    runs: Query<(&RunOf, Option<&RunControl>)>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(Entity, &StableId, &Policy, Option<&PolicyStatus>)>,
    mut progress: ResMut<Progress>,
) {
    for (operation, operation_of, input, state) in &operations {
        let OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Tool(output))) =
            &state.phase
        else {
            continue;
        };
        let Ok((run_of, control)) = runs.get(operation_of.get()) else {
            continue;
        };
        if !control_allows_internal_progress(control) {
            continue;
        }
        let mut ordered = agents
            .get(run_of.get())
            .ok()
            .flatten()
            .into_iter()
            .flat_map(|agent_policies| agent_policies.iter())
            .filter_map(|entity| policies.get(entity).ok())
            .filter(|(_, _, policy, status)| {
                accepts_new_policy_evaluations(*status)
                    && policy.rule.applies_to(PolicyPoint::ToolResult)
            })
            .collect::<Vec<_>>();
        ordered.sort_by(|(_, left_id, left, _), (_, right_id, right, _)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        let snapshot = ordered
            .into_iter()
            .map(|(entity, id, policy, _)| AcceptedPolicy {
                id: id.clone(),
                entity,
                revision: policy.revision,
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
        let mut invocation = ToolResultPolicyInvocation {
            policy: policy.entity,
            evaluation: evaluation_entity,
            run,
            operation,
            input: evaluation.input,
            result: evaluation.effective,
            decision: None,
        };
        world.trigger_ref(&mut invocation);
        match invocation.decision {
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
                        reason,
                    };
                }
                if let Some(mut state) = world.get_mut::<RunState>(run) {
                    *state = RunState::Failed(CanonicalError::PolicyDenied {
                        policy: policy.id.as_str().to_owned(),
                    });
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
                    reason: rejection_reason,
                };
            }
        }

        if matched
            && !approved
            && let Some(mut state) = world.get_mut::<RunState>(run)
        {
            *state = RunState::Failed(CanonicalError::PolicyDenied {
                policy: input.policy_id.as_str().to_owned(),
            });
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
    mut runs: Query<(&mut RunState, &mut RunRecord, Option<&RunControl>)>,
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
    mut runs: Query<(&mut RunState, &mut RunRecord, Option<&RunControl>)>,
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
        let Ok((mut run_state, mut record, control)) = runs.get_mut(operation_of.get()) else {
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
                        *run_state = RunState::Failed(error);
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
                                OperationState {
                                    generation: 0,
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
                                    streaming_origin: stream_state
                                        .is_some_and(|state| state.next_sequence > 0),
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
                            OperationState {
                                generation: 0,
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
    source_models: Query<&ModelEffectInput>,
    agents: Query<&Agent>,
    mut runs: Query<(&RunOf, &mut RunState, &mut RunRecord, Option<&RunControl>)>,
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
            results.push(effective.map_or_else(|| output.clone(), |effective| effective.0.clone()));
        }
        for result in &results {
            record.transcript.push(TranscriptEntry::ToolResult {
                call_id: result.call_id.clone(),
                provider_result_id: result.provider_result_id.clone(),
                provider_call_id: result.provider_call_id.clone(),
                name: result.name.clone(),
                raw: result.raw.clone(),
                content: result.presentation.clone(),
            });
        }
        let Ok(source_input) = source_models.get(batch.source_model_operation) else {
            *run_state = RunState::Failed(CanonicalError::StaleEntity(
                "source model operation".to_owned(),
            ));
            batch.committed = true;
            mark_progress(&mut progress);
            continue;
        };
        commands.trigger(ToolBatchCommitted {
            batch: batch_entity,
            run: batch_of.get(),
            results: results.clone(),
        });
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
        let mut next_input = source_input.clone();
        next_input.history = record.transcript.clone();
        next_input.tool_results = results;
        let next_operation = commands
            .spawn((
                OperationOf(batch_of.get()),
                OperationKind::Model,
                ModelStreamState {
                    streaming: record.streaming,
                    ..ModelStreamState::default()
                },
                OperationState {
                    generation: 0,
                    phase: OperationPhase::Prepared,
                },
                next_input.decision.clone(),
                PendingModelRequest(next_input),
            ))
            .id();
        *run_state = RunState::WaitingModel {
            operation: next_operation,
        };
        batch.committed = true;
        mark_progress(&mut progress);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Component)]
    struct InspectRequestPolicy {
        expected_instructions: String,
        replacement_instructions: String,
    }

    #[derive(Component)]
    struct InspectToolPolicy {
        expected_arguments: serde_json::Value,
        replacement_arguments: serde_json::Value,
    }

    #[derive(Component)]
    struct InspectToolResultPolicy {
        expected_presentation: String,
        replacement_presentation: String,
    }

    #[derive(Resource, Default)]
    struct FinishedTurns(Vec<ModelTurnFinished>);

    #[derive(Resource, Default)]
    struct ObservedStreamDeltas {
        text: Vec<TextDeltaObserved>,
        tool: Vec<ToolCallDeltaObserved>,
        finished: Vec<StreamResponseFinished>,
    }

    #[derive(Resource, Default)]
    struct LifecycleLog(Vec<&'static str>);

    macro_rules! record_lifecycle_event {
        ($function:ident, $event:ty, $name:literal) => {
            fn $function(_: On<$event>, mut log: ResMut<LifecycleLog>) {
                log.0.push($name);
            }
        };
    }

    record_lifecycle_event!(
        record_request_prepared,
        CompletionRequestPrepared,
        "request-prepared"
    );
    record_lifecycle_event!(record_model_dispatched, ModelDispatched, "model-dispatched");
    record_lifecycle_event!(record_model_settled, ModelSettled, "model-settled");
    record_lifecycle_event!(
        record_response_applied,
        CompletionResponseApplied,
        "response-applied"
    );
    record_lifecycle_event!(
        record_invalid_detected,
        InvalidToolCallDetected,
        "invalid-detected"
    );
    record_lifecycle_event!(record_tool_prepared, ToolCallPrepared, "tool-prepared");
    record_lifecycle_event!(record_tool_started, ToolExecutionStarted, "tool-started");
    record_lifecycle_event!(record_tool_settled, ToolExecutionSettled, "tool-settled");
    record_lifecycle_event!(
        record_result_finalized,
        ToolResultPresentationFinalized,
        "result-finalized"
    );
    record_lifecycle_event!(
        record_batch_committed,
        ToolBatchCommitted,
        "batch-committed"
    );
    record_lifecycle_event!(
        record_persistence_settled,
        PersistenceSettled,
        "persistence-settled"
    );
    record_lifecycle_event!(record_run_completed, RunCompleted, "run-completed");
    record_lifecycle_event!(record_run_failed, RunFailed, "run-failed");
    record_lifecycle_event!(record_run_cancelled, RunCancelled, "run-cancelled");

    fn inspect_request_policy(
        mut event: On<RequestPolicyInvocation>,
        policies: Query<&InspectRequestPolicy>,
    ) {
        let Ok(policy) = policies.get(event.policy) else {
            return;
        };
        event.decision = Some(
            if event.request.instructions == policy.expected_instructions {
                RequestPolicyDecision::Patch(
                    RequestPatch::new().instructions(policy.replacement_instructions.clone()),
                )
            } else {
                RequestPolicyDecision::Stop("earlier rewrite was not visible".to_owned())
            },
        );
    }

    fn inspect_tool_policy(
        mut event: On<ToolCallPolicyInvocation>,
        policies: Query<&InspectToolPolicy>,
    ) {
        let Ok(policy) = policies.get(event.policy) else {
            return;
        };
        event.decision = Some(if event.call.arguments == policy.expected_arguments {
            ToolCallPolicyDecision::Rewrite(policy.replacement_arguments.clone())
        } else {
            ToolCallPolicyDecision::Stop("earlier argument rewrite was not visible".to_owned())
        });
    }

    fn inspect_tool_result_policy(
        mut event: On<ToolResultPolicyInvocation>,
        policies: Query<&InspectToolResultPolicy>,
    ) {
        let Ok(policy) = policies.get(event.policy) else {
            return;
        };
        event.decision = Some(
            if event.result.presentation == policy.expected_presentation {
                ToolResultPolicyDecision::Rewrite(policy.replacement_presentation.clone())
            } else {
                ToolResultPolicyDecision::Stop("earlier result rewrite was not visible".to_owned())
            },
        );
    }

    fn record_finished_turn(event: On<ModelTurnFinished>, mut turns: ResMut<FinishedTurns>) {
        turns.0.push(event.event().clone());
    }

    fn record_text_delta(event: On<TextDeltaObserved>, mut observed: ResMut<ObservedStreamDeltas>) {
        observed.text.push(event.event().clone());
    }

    fn record_tool_delta(
        event: On<ToolCallDeltaObserved>,
        mut observed: ResMut<ObservedStreamDeltas>,
    ) {
        observed.tool.push(event.event().clone());
    }

    fn record_stream_finished(
        event: On<StreamResponseFinished>,
        mut observed: ResMut<ObservedStreamDeltas>,
    ) {
        observed.finished.push(event.event().clone());
    }

    fn id(value: &str) -> StableId {
        StableId::new(value).unwrap()
    }

    fn tenant(value: &str) -> TenantId {
        TenantId::new(value).unwrap()
    }

    fn runtime() -> Runtime {
        Runtime::new(RuntimeConfig::default()).unwrap()
    }

    fn runtime_with_tool() -> (Runtime, AgentHandle) {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let tool = runtime
            .spawn_tool(
                id("tool"),
                tenant("a"),
                ToolCapability {
                    name: "lookup".to_owned(),
                    description: "lookup tool".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_tool(
                id("grant"),
                tenant("a"),
                ToolGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                tool,
            )
            .unwrap();
        (runtime, agent)
    }

    fn advance_to_lookup_tool(
        runtime: &mut Runtime,
        agent: AgentHandle,
    ) -> (PendingRunHandle, EffectRequest) {
        let pending = runtime.handle().prompt(agent, "use lookup").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
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
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({}),
                    }],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let tool = runtime.effects().try_recv().unwrap().unwrap();
        (pending, tool)
    }

    #[test]
    fn model_effect_round_trip_uses_world_resident_schedule() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 7,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(
                id("agent"),
                tenant("a"),
                Agent {
                    instructions: "be concise".to_owned(),
                    ..Agent::default()
                },
                model,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();

        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(request.model_input().unwrap().decision.revision, 7);
        assert_eq!(
            request.model_input().unwrap().prompt,
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            })
        );

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "hi".to_owned(),
                    usage: Usage {
                        input_tokens: 3,
                        output_tokens: 1,
                    },
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Completed(RunOutput {
                text: "hi".to_owned(),
                usage: Usage {
                    input_tokens: 3,
                    output_tokens: 1,
                },
            }))
        );
        assert_eq!(
            runtime
                .world()
                .get::<RunRecord>(run.entity())
                .unwrap()
                .transcript,
            vec![
                TranscriptEntry::User("hello".to_owned()),
                TranscriptEntry::Assistant("hi".to_owned()),
            ]
        );
        assert_eq!(runtime.metrics().unobserved_terminal_runs, 1);
        assert!(matches!(
            runtime.observe_run(run).unwrap(),
            Some(RunState::Completed(_))
        ));
        runtime.run_until_stalled().unwrap();
        assert!(runtime.world().get::<RunState>(run.entity()).is_none());
        assert!(
            runtime
                .world()
                .get::<OperationState>(request.operation)
                .is_none()
        );
        assert!(runtime.resolve_run(&pending).is_none());
    }

    #[test]
    fn lifecycle_observations_cover_model_tool_batch_and_terminal_boundaries() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime.world_mut().insert_resource(LifecycleLog::default());
        runtime.world_mut().add_observer(record_request_prepared);
        runtime.world_mut().add_observer(record_model_dispatched);
        runtime.world_mut().add_observer(record_model_settled);
        runtime.world_mut().add_observer(record_response_applied);
        runtime.world_mut().add_observer(record_invalid_detected);
        runtime.world_mut().add_observer(record_tool_prepared);
        runtime.world_mut().add_observer(record_tool_started);
        runtime.world_mut().add_observer(record_tool_settled);
        runtime.world_mut().add_observer(record_result_finalized);
        runtime.world_mut().add_observer(record_batch_committed);
        runtime.world_mut().add_observer(record_persistence_settled);
        runtime.world_mut().add_observer(record_run_completed);
        runtime.world_mut().add_observer(record_run_failed);
        runtime.world_mut().add_observer(record_run_cancelled);

        let (_, tool) = advance_to_lookup_tool(&mut runtime, agent);
        let input = tool.tool_input().unwrap();
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
                    raw: serde_json::json!({"answer": 42}),
                    presentation: "42".to_owned(),
                    failure: None,
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model.operation,
                generation: model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "done".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(
            runtime.world().resource::<LifecycleLog>().0,
            vec![
                "request-prepared",
                "model-dispatched",
                "model-settled",
                "response-applied",
                "tool-prepared",
                "tool-started",
                "tool-settled",
                "result-finalized",
                "batch-committed",
                "request-prepared",
                "model-dispatched",
                "model-settled",
                "response-applied",
                "run-completed",
            ]
        );
    }

    #[test]
    fn streaming_and_blocking_observe_the_same_terminal_state() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let (pending, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let deltas = runtime.effects().delta_sender();
        deltas
            .try_send(EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence: 0,
                provider_correlation: None,
                kind: EffectDeltaKind::Text("he".to_owned()),
            })
            .unwrap();
        deltas
            .try_send(EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence: 1,
                provider_correlation: None,
                kind: EffectDeltaKind::Text("llo".to_owned()),
            })
            .unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "hello".to_owned(),
                    usage: Usage {
                        input_tokens: 2,
                        output_tokens: 1,
                    },
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let output = RunOutput {
            text: "hello".to_owned(),
            usage: Usage {
                input_tokens: 2,
                output_tokens: 1,
            },
        };
        assert_eq!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Delta {
                sequence: 0,
                text: "he".to_owned(),
            })
        );
        assert_eq!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Delta {
                sequence: 1,
                text: "llo".to_owned(),
            })
        );
        assert_eq!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Finished(StreamTerminal::Completed(
                output.clone()
            )))
        );
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Completed(output))
        );
    }

    #[test]
    fn structured_output_is_snapshotted_forwarded_and_validated() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let old_schema = serde_json::json!({
            "type": "object",
            "properties": {"old": {"type": "string"}},
            "required": ["old"]
        });
        let new_schema = serde_json::json!({
            "type": "object",
            "properties": {"new": {"type": "integer"}},
            "required": ["new"]
        });
        runtime
            .set_output_requirement(
                agent,
                OutputRequirement {
                    schema: old_schema.clone(),
                },
            )
            .unwrap();

        let first_pending = runtime.handle().prompt(agent, "first").unwrap();
        runtime.run_until_stalled().unwrap();
        let first = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(first.model_input().unwrap().output_schema, Some(old_schema));
        runtime
            .set_output_requirement(
                agent,
                OutputRequirement {
                    schema: new_schema.clone(),
                },
            )
            .unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: first.operation,
                generation: first.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: r#"{"old":"accepted"}"#.to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let first_run = runtime.resolve_run(&first_pending).unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(first_run.entity()),
            Some(RunState::Completed(_))
        ));

        let second_pending = runtime.handle().prompt(agent, "second").unwrap();
        runtime.run_until_stalled().unwrap();
        let second = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            second.model_input().unwrap().output_schema,
            Some(new_schema)
        );
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: second.operation,
                generation: second.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: r#"{"old":"rejected"}"#.to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let second_run = runtime.resolve_run(&second_pending).unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(second_run.entity()),
            Some(RunState::Failed(CanonicalError::InvalidStructuredOutput(message)))
                if message.contains("new")
        ));
    }

    #[test]
    fn structured_output_validator_resolves_refs_and_composition() {
        let schema = serde_json::json!({
            "$defs": {
                "payload": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"count": {"type": "integer", "minimum": 1}},
                            "required": ["count"],
                            "additionalProperties": false
                        },
                        {"const": "unavailable"}
                    ]
                }
            },
            "$ref": "#/$defs/payload"
        });

        assert!(validate_structured_output(&schema, r#"{"count":2}"#).is_ok());
        assert!(validate_structured_output(&schema, r#""unavailable""#).is_ok());
        assert!(matches!(
            validate_structured_output(&schema, r#"{"count":0,"extra":true}"#),
            Err(CanonicalError::InvalidStructuredOutput(_))
        ));
    }

    #[test]
    fn duplicate_provider_tool_call_ids_fail_before_tool_dispatch() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let tool = runtime
            .spawn_tool(
                id("tool"),
                tenant("a"),
                ToolCapability {
                    name: "lookup".to_owned(),
                    description: String::new(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_tool(
                id("grant"),
                tenant("a"),
                ToolGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                tool,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "lookup twice").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: String::new(),
                    usage: Usage::default(),
                    tool_calls: vec![
                        ModelToolCall {
                            id: "duplicate".to_owned(),
                            provider_result_id: "duplicate".to_owned(),
                            provider_call_id: None,
                            name: "lookup".to_owned(),
                            arguments: serde_json::json!({"id": 1}),
                        },
                        ModelToolCall {
                            id: "duplicate".to_owned(),
                            provider_result_id: "duplicate".to_owned(),
                            provider_call_id: None,
                            name: "lookup".to_owned(),
                            arguments: serde_json::json!({"id": 2}),
                        },
                    ],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::DuplicateToolCall(
                "duplicate".to_owned()
            )))
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
        assert_eq!(runtime.run_committed_turns(run), Vec::new());
    }

    #[test]
    fn vector_retrieval_enriches_the_immutable_model_effect() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let store = runtime
            .spawn_store(
                id("vectors"),
                tenant("a"),
                StoreCapability {
                    kind: "vector-search".to_owned(),
                    revision: 5,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_store(
                id("vector-grant"),
                tenant("a"),
                StoreGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                store,
            )
            .unwrap();
        runtime
            .set_retrieval_requirement(agent, RetrievalRequirement { limit: 2 })
            .unwrap();

        runtime.handle().prompt(agent, "find context").unwrap();
        runtime.run_until_stalled().unwrap();
        let retrieval = runtime.effects().try_recv().unwrap().unwrap();
        assert!(matches!(
            &retrieval.store_input().unwrap().operation,
            StoreOperation::Retrieve { query, limit }
                if query == "find context" && *limit == 2
        ));
        let document = RetrievedDocument {
            id: "doc-1".to_owned(),
            text: "retrieved text".to_owned(),
            metadata: BTreeMap::from([("source".to_owned(), "test".to_owned())]),
        };
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: retrieval.operation,
                generation: retrieval.generation,
                result: Ok(EffectOutput::Store(StoreEffectOutput::Retrieved(vec![
                    document.clone(),
                ]))),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(model.model_input().unwrap().documents, vec![document]);
        assert_eq!(model.model_input().unwrap().decision.revision, 1);
    }

    #[test]
    fn streaming_rejects_duplicate_gap_and_late_deltas() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let (_, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let deltas = runtime.effects().delta_sender();
        for (sequence, text) in [(0, "accepted"), (0, "duplicate"), (2, "gap")] {
            deltas
                .try_send(EffectDelta {
                    operation: request.operation,
                    generation: request.generation,
                    sequence,
                    provider_correlation: None,
                    kind: EffectDeltaKind::Text(text.to_owned()),
                })
                .unwrap();
        }
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "done".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        deltas
            .try_send(EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence: 1,
                provider_correlation: None,
                kind: EffectDeltaKind::Text("late".to_owned()),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Delta {
                sequence: 0,
                text: "accepted".to_owned(),
            })
        );
        assert!(matches!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Finished(StreamTerminal::Completed(_)))
        ));
        assert_eq!(stream.try_recv().unwrap(), None);
    }

    #[test]
    fn text_and_tool_call_deltas_share_sequence_and_observation_pipeline() {
        let mut runtime = runtime();
        runtime
            .world_mut()
            .insert_resource(ObservedStreamDeltas::default());
        runtime.world_mut().add_observer(record_text_delta);
        runtime.world_mut().add_observer(record_stream_finished);
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let (_, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .world_mut()
            .entity_mut(request.operation)
            .observe(record_tool_delta);
        let deltas = runtime.effects().delta_sender();
        for delta in [
            EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence: 0,
                provider_correlation: Some("message-1".to_owned()),
                kind: EffectDeltaKind::Text("hello".to_owned()),
            },
            EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence: 1,
                provider_correlation: Some("call-1".to_owned()),
                kind: EffectDeltaKind::ToolCall {
                    id: "call-1".to_owned(),
                    internal_call_id: "internal-1".to_owned(),
                    content: ToolCallDeltaContent::Name("lookup".to_owned()),
                },
            },
            EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence: 2,
                provider_correlation: Some("call-1".to_owned()),
                kind: EffectDeltaKind::ToolCall {
                    id: "call-1".to_owned(),
                    internal_call_id: "internal-1".to_owned(),
                    content: ToolCallDeltaContent::Delta("{\"q\":".to_owned()),
                },
            },
        ] {
            deltas.try_send(delta).unwrap();
        }
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "hello".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Delta {
                sequence: 0,
                text: "hello".to_owned(),
            })
        );
        assert!(matches!(
            stream.try_recv().unwrap(),
            Some(StreamItem::ToolCallDelta {
                sequence: 1,
                id,
                internal_call_id,
                content: ToolCallDeltaContent::Name(name),
            }) if id == "call-1" && internal_call_id == "internal-1" && name == "lookup"
        ));
        assert!(matches!(
            stream.try_recv().unwrap(),
            Some(StreamItem::ToolCallDelta {
                sequence: 2,
                content: ToolCallDeltaContent::Delta(fragment),
                ..
            }) if fragment == "{\"q\":"
        ));
        let observed = runtime.world().resource::<ObservedStreamDeltas>();
        assert_eq!(observed.text.len(), 1);
        assert_eq!(observed.tool.len(), 2);
        assert_eq!(observed.finished.len(), 1);
        assert_eq!(
            observed
                .text
                .first()
                .and_then(|event| event.provider_correlation.as_deref()),
            Some("message-1")
        );
        assert_eq!(observed.tool.first().map(|event| event.sequence), Some(1));
        assert_eq!(observed.tool.get(1).map(|event| event.sequence), Some(2));
    }

    #[test]
    fn text_delta_policy_preserves_sequence_and_stops_before_publication() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime
            .spawn_policy(
                id("stream-guard"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 2,
                    rule: PolicyRule::StopTextDeltaContains {
                        needle: "blocked".to_owned(),
                        reason: "unsafe stream".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let (_, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let deltas = runtime.effects().delta_sender();
        for (sequence, text) in [(0, "safe"), (1, "blocked")] {
            deltas
                .try_send(EffectDelta {
                    operation: request.operation,
                    generation: request.generation,
                    sequence,
                    provider_correlation: None,
                    kind: EffectDeltaKind::Text(text.to_owned()),
                })
                .unwrap();
        }

        runtime.run_until_stalled().unwrap();
        assert_eq!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Delta {
                sequence: 0,
                text: "safe".to_owned(),
            })
        );
        assert!(matches!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Finished(StreamTerminal::Failed(
                CanonicalError::PolicyDenied { policy }
            ))) if policy == "stream-guard"
        ));
        assert_eq!(stream.try_recv().unwrap(), None);
    }

    #[test]
    fn slow_stream_consumer_is_disconnected_without_cancelling_the_run() {
        let mut runtime = Runtime::new(RuntimeConfig {
            subscriber_capacity: 1,
            ..RuntimeConfig::default()
        })
        .unwrap();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let (pending, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let deltas = runtime.effects().delta_sender();
        for sequence in 0..2 {
            deltas
                .try_send(EffectDelta {
                    operation: request.operation,
                    generation: request.generation,
                    sequence,
                    provider_correlation: None,
                    kind: EffectDeltaKind::Text(sequence.to_string()),
                })
                .unwrap();
        }
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "done".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::Completed(_))
        ));
        let subscription = runtime
            .world()
            .get::<RunSubscriptions>(run.entity())
            .unwrap()
            .iter()
            .next()
            .unwrap();
        assert_eq!(
            runtime.world().get::<SubscriptionState>(subscription),
            Some(&SubscriptionState::DroppedSlowConsumer)
        );
        assert_eq!(
            stream.try_recv().unwrap(),
            Some(StreamItem::Delta {
                sequence: 0,
                text: "0".to_owned(),
            })
        );
        assert_eq!(stream.try_recv(), Err(StreamReceiveError::Disconnected));
        runtime.run_until_stalled().unwrap();
        assert!(runtime.resolve_run(&pending).is_none());
    }

    #[test]
    fn dropping_stream_acknowledges_terminal_run_when_command_queue_is_full() {
        let mut runtime = Runtime::new(RuntimeConfig {
            command_capacity: 1,
            ..RuntimeConfig::default()
        })
        .unwrap();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let (pending, stream) = runtime.handle().prompt_stream(agent, "first").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "done".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        runtime
            .handle()
            .prompt(agent, "fills command queue")
            .unwrap();
        drop(stream);
        runtime.run_until_stalled().unwrap();

        assert!(runtime.resolve_run(&pending).is_none());
    }

    #[test]
    fn replacement_does_not_change_an_accepted_decision() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "old".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        runtime
            .world_mut()
            .entity_mut(model)
            .insert(ModelCapability {
                provider: "fake".to_owned(),
                model: "new".to_owned(),
                revision: 2,
                retired: false,
            });

        let request = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(request.model_input().unwrap().decision.model, "old");
        assert_eq!(request.model_input().unwrap().decision.revision, 1);
    }

    #[test]
    fn duplicate_and_stale_completions_are_idempotently_ignored() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let sender = runtime.effects().completion_sender();
        let completion = EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "once".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        };
        sender.try_send(completion.clone()).unwrap();
        sender.try_send(completion).unwrap();
        sender
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation + 1,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "stale".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        let record = runtime.world().get::<RunRecord>(run.entity()).unwrap();
        assert_eq!(
            record.transcript,
            vec![
                TranscriptEntry::User("hello".to_owned()),
                TranscriptEntry::Assistant("once".to_owned()),
            ]
        );
    }

    #[test]
    fn cross_tenant_agent_model_relationship_is_rejected() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        assert_eq!(
            runtime.spawn_agent(id("agent"), tenant("b"), Agent::default(), model),
            Err(SpawnError::TenantMismatch)
        );
    }

    #[test]
    fn invariant_validation_detects_manual_cross_tenant_and_wrong_kind_relationships() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model-a"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "m".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime.world_mut().spawn((
            id("cross-tenant-agent"),
            tenant("b"),
            Agent::default(),
            UsesModel(model),
        ));
        let wrong_kind = runtime
            .world_mut()
            .spawn((id("not-a-model"), tenant("a"), Agent::default()))
            .id();
        runtime.world_mut().spawn((
            id("wrong-kind-agent"),
            tenant("a"),
            Agent::default(),
            UsesModel(wrong_kind),
        ));

        runtime.update();
        let violations = runtime.validate_invariants().unwrap_err();
        assert!(violations.iter().any(|violation| matches!(
            violation,
            RuntimeInvariantError::RelationshipTenantMismatch {
                relationship: "UsesModel",
                ..
            }
        )));
        assert!(violations.iter().any(|violation| matches!(
            violation,
            RuntimeInvariantError::InvalidRelationship {
                relationship: "UsesModel",
                target,
                ..
            } if *target == wrong_kind
        )));
    }

    #[test]
    fn ordered_policy_denies_before_dispatch_and_records_revision() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let policy = runtime
            .spawn_policy(
                id("deny-secret"),
                tenant("a"),
                Policy {
                    order: 10,
                    revision: 4,
                    rule: PolicyRule::DenyPromptContains("secret".to_owned()),
                },
                agent,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "reveal the secret").unwrap();

        runtime.run_until_stalled().unwrap();
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::PolicyDenied {
                policy: "deny-secret".to_owned(),
            }))
        );
        let operation = runtime
            .world()
            .get::<RunOperations>(run.entity())
            .unwrap()
            .iter()
            .next()
            .unwrap();
        assert_eq!(
            runtime.world().get::<AcceptedPolicies>(operation),
            Some(&AcceptedPolicies(vec![AcceptedPolicy {
                id: id("deny-secret"),
                entity: policy,
                revision: 4,
            }]))
        );
    }

    #[test]
    fn request_patches_reduce_by_order_and_remain_operation_local() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(
                id("agent"),
                tenant("a"),
                Agent {
                    instructions: "baseline".to_owned(),
                    additional_params: Some(serde_json::json!({"baseline": true, "winner": 0})),
                    ..Agent::default()
                },
                model,
            )
            .unwrap();
        runtime
            .spawn_policy(
                id("z-last"),
                tenant("a"),
                Policy {
                    order: 10,
                    revision: 2,
                    rule: PolicyRule::PatchRequest(
                        RequestPatch::new()
                            .instructions("last")
                            .additional_params(serde_json::json!({"winner": 2, "z": true})),
                    ),
                },
                agent,
            )
            .unwrap();
        runtime
            .spawn_policy(
                id("a-first"),
                tenant("a"),
                Policy {
                    order: 10,
                    revision: 1,
                    rule: PolicyRule::PatchRequest(
                        RequestPatch::new()
                            .instructions("first")
                            .additional_params(serde_json::json!({"winner": 1, "a": true})),
                    ),
                },
                agent,
            )
            .unwrap();

        runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let effect = runtime.effects().try_recv().unwrap().unwrap();
        let input = effect.model_input().unwrap();
        assert_eq!(input.instructions, "last");
        assert_eq!(
            input.additional_params,
            Some(serde_json::json!({
                "a": true,
                "baseline": true,
                "winner": 2,
                "z": true
            }))
        );
        assert_eq!(
            runtime
                .world()
                .get::<Agent>(agent.entity())
                .unwrap()
                .instructions,
            "baseline"
        );
    }

    #[test]
    fn targeted_custom_policy_observes_earlier_effective_request() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime
            .spawn_policy(
                id("first"),
                tenant("a"),
                Policy {
                    order: 1,
                    revision: 1,
                    rule: PolicyRule::PatchRequest(
                        RequestPatch::new().instructions("rewritten once"),
                    ),
                },
                agent,
            )
            .unwrap();
        let custom = runtime
            .spawn_policy(
                id("second"),
                tenant("a"),
                Policy {
                    order: 2,
                    revision: 7,
                    rule: PolicyRule::Custom(PolicyPoint::Request),
                },
                agent,
            )
            .unwrap();
        runtime
            .world_mut()
            .entity_mut(custom)
            .insert(InspectRequestPolicy {
                expected_instructions: "rewritten once".to_owned(),
                replacement_instructions: "rewritten twice".to_owned(),
            })
            .observe(inspect_request_policy);

        runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let effect = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            effect.model_input().unwrap().instructions,
            "rewritten twice"
        );
    }

    #[test]
    fn custom_policy_without_a_responder_fails_closed() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime
            .spawn_policy(
                id("unbound"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::Custom(PolicyPoint::Request),
                },
                agent,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();

        runtime.run_until_stalled().unwrap();
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::PolicyDenied {
                policy: "unbound".to_owned(),
            }))
        );
    }

    #[test]
    fn request_approval_operation_suspends_and_resumes_policy_cursor() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime
            .spawn_policy(
                id("approval"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 9,
                    rule: PolicyRule::RequireApproval {
                        point: PolicyPoint::Request,
                        prompt: "approve model call".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let approval = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            approval.policy_approval_input(),
            Some(&PolicyApprovalEffectInput {
                policy_id: id("approval"),
                revision: 9,
                point: PolicyPoint::Request,
                prompt: "approve model call".to_owned(),
            })
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
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
            .unwrap();

        runtime.run_until_stalled().unwrap();
        assert!(
            runtime
                .effects()
                .try_recv()
                .unwrap()
                .unwrap()
                .model_input()
                .is_some()
        );
    }

    #[test]
    fn retired_policy_is_retained_for_accepted_work_and_excluded_from_future_snapshots() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let policy = runtime
            .spawn_policy(
                id("approval"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 7,
                    rule: PolicyRule::RequireApproval {
                        point: PolicyPoint::Request,
                        prompt: "approve accepted request".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();

        runtime.handle().prompt(agent, "first").unwrap();
        runtime.run_until_stalled().unwrap();
        let approval = runtime.effects().try_recv().unwrap().unwrap();
        assert!(approval.policy_approval_input().is_some());

        runtime.retire_policy(policy).unwrap();
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
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert!(
            runtime
                .effects()
                .try_recv()
                .unwrap()
                .unwrap()
                .model_input()
                .is_some()
        );

        runtime.handle().prompt(agent, "second").unwrap();
        runtime.run_until_stalled().unwrap();
        assert!(
            runtime
                .effects()
                .try_recv()
                .unwrap()
                .unwrap()
                .model_input()
                .is_some()
        );

        let snapshot = runtime.snapshot().unwrap();
        assert_eq!(snapshot.policies[0].status, PolicyStatus::Retired);
        let mut restored = Runtime::new(RuntimeConfig::default()).unwrap();
        let restored_entities = restored.restore(snapshot).unwrap();
        let restored_agent = AgentHandle {
            runtime_id: restored.handle.runtime_id,
            entity: restored_entities.0[&id("agent")],
        };
        let restored_policy = restored_entities.0[&id("approval")];
        assert_eq!(
            restored.world().get::<PolicyStatus>(restored_policy),
            Some(&PolicyStatus::Retired)
        );
        restored
            .handle()
            .prompt(restored_agent, "after restore")
            .unwrap();
        restored.run_until_stalled().unwrap();
        assert!(
            restored
                .effects()
                .try_recv()
                .unwrap()
                .unwrap()
                .model_input()
                .is_some()
        );
    }

    #[test]
    fn denied_tool_approval_prevents_tool_dispatch() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .spawn_policy(
                id("tool-approval"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::RequireApproval {
                        point: PolicyPoint::ToolCall,
                        prompt: "approve tool".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "use lookup").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
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
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({}),
                    }],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let approval = runtime.effects().try_recv().unwrap().unwrap();
        assert!(approval.policy_approval_input().is_some());
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: approval.operation,
                generation: approval.generation,
                result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                    approved: false,
                    reason: Some("operator denied".to_owned()),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::PolicyDenied {
                policy: "tool-approval".to_owned(),
            }))
        );
    }

    #[test]
    fn completion_response_policy_runs_before_commit_and_emits_turn_event() {
        let mut runtime = runtime();
        runtime
            .world_mut()
            .insert_resource(FinishedTurns::default());
        runtime.world_mut().add_observer(record_finished_turn);
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime
            .spawn_policy(
                id("rewrite-response"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 5,
                    rule: PolicyRule::RewriteCompletionText {
                        text: "rewritten".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "original".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "rewritten"
        ));
        let turns = &runtime.world().resource::<FinishedTurns>().0;
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].output.text, "rewritten");
        assert_eq!(turns[0].response_policies[0].revision, 5);
    }

    #[test]
    fn stopped_completion_response_never_commits_a_turn() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime
            .spawn_policy(
                id("stop-response"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::StopCompletionContains {
                        needle: "secret".to_owned(),
                        reason: "sensitive completion".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "a secret".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::PolicyDenied {
                policy: "stop-response".to_owned(),
            }))
        );
        assert!(runtime.run_committed_turns(run).is_empty());
    }

    #[test]
    fn tool_argument_rewrites_chain_in_policy_order() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .spawn_policy(
                id("first-rewrite"),
                tenant("a"),
                Policy {
                    order: 1,
                    revision: 1,
                    rule: PolicyRule::RewriteToolArguments {
                        tool: Some("lookup".to_owned()),
                        arguments: serde_json::json!({"step": 1}),
                    },
                },
                agent,
            )
            .unwrap();
        let second = runtime
            .spawn_policy(
                id("second-rewrite"),
                tenant("a"),
                Policy {
                    order: 2,
                    revision: 1,
                    rule: PolicyRule::Custom(PolicyPoint::ToolCall),
                },
                agent,
            )
            .unwrap();
        runtime
            .world_mut()
            .entity_mut(second)
            .insert(InspectToolPolicy {
                expected_arguments: serde_json::json!({"step": 1}),
                replacement_arguments: serde_json::json!({"step": 2}),
            })
            .observe(inspect_tool_policy);

        runtime.handle().prompt(agent, "use lookup").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
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
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({"step": 0}),
                    }],
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        let tool = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            tool.tool_input().unwrap().arguments,
            serde_json::json!({"step": 2})
        );
    }

    #[test]
    fn skipped_tool_call_never_dispatches_and_reenters_model_flow() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .spawn_policy(
                id("skip"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::SkipToolCall {
                        tool: Some("lookup".to_owned()),
                        reason: "approval denied".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();

        runtime.handle().prompt(agent, "use lookup").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
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
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({}),
                    }],
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        let next_model = runtime.effects().try_recv().unwrap().unwrap();
        let input = next_model.model_input().unwrap();
        assert_eq!(input.tool_results.len(), 1);
        assert_eq!(input.tool_results[0].presentation, "approval denied");
        assert_eq!(
            input.tool_results[0].raw,
            serde_json::json!({"skipped": true, "reason": "approval denied"})
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
    }

    #[test]
    fn tool_result_rewrites_chain_without_mutating_raw_audit_data() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .spawn_policy(
                id("first-result"),
                tenant("a"),
                Policy {
                    order: 1,
                    revision: 1,
                    rule: PolicyRule::RewriteToolResult {
                        tool: Some("lookup".to_owned()),
                        presentation: "redacted once".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let second = runtime
            .spawn_policy(
                id("second-result"),
                tenant("a"),
                Policy {
                    order: 2,
                    revision: 1,
                    rule: PolicyRule::Custom(PolicyPoint::ToolResult),
                },
                agent,
            )
            .unwrap();
        runtime
            .world_mut()
            .entity_mut(second)
            .insert(InspectToolResultPolicy {
                expected_presentation: "redacted once".to_owned(),
                replacement_presentation: "redacted twice".to_owned(),
            })
            .observe(inspect_tool_result_policy);
        let (pending, tool) = advance_to_lookup_tool(&mut runtime, agent);
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: tool.operation,
                generation: tool.generation,
                result: Ok(EffectOutput::Tool(ToolEffectOutput {
                    call_id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    raw: serde_json::json!({"secret": 42}),
                    presentation: "unredacted".to_owned(),
                    failure: None,
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        let next_model = runtime.effects().try_recv().unwrap().unwrap();
        let result = &next_model.model_input().unwrap().tool_results[0];
        assert_eq!(result.raw, serde_json::json!({"secret": 42}));
        assert_eq!(result.presentation, "redacted twice");
        let run = runtime.resolve_run(&pending).unwrap();
        assert!(
            runtime
                .world()
                .get::<RunRecord>(run.entity())
                .unwrap()
                .transcript
                .iter()
                .any(|entry| matches!(
                    entry,
                    TranscriptEntry::ToolResult { raw, content, .. }
                        if raw == &serde_json::json!({"secret": 42}) && content == "redacted twice"
                ))
        );
    }

    #[test]
    fn stopped_tool_result_is_not_committed_or_redispatched() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .spawn_policy(
                id("stop-result"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::StopToolResult {
                        tool: None,
                        reason: "sensitive result".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let (pending, tool) = advance_to_lookup_tool(&mut runtime, agent);
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: tool.operation,
                generation: tool.generation,
                result: Ok(EffectOutput::Tool(ToolEffectOutput {
                    call_id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    raw: serde_json::json!({"secret": 42}),
                    presentation: "secret".to_owned(),
                    failure: None,
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::PolicyDenied {
                policy: "stop-result".to_owned(),
            }))
        );
        assert!(
            !runtime
                .world()
                .get::<RunRecord>(run.entity())
                .unwrap()
                .transcript
                .iter()
                .any(|entry| matches!(entry, TranscriptEntry::ToolResult { .. }))
        );
    }

    #[test]
    fn tool_collision_resolution_is_deterministic_and_snapshotted() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let preferred = runtime
            .spawn_tool(
                id("preferred"),
                tenant("a"),
                ToolCapability {
                    name: "lookup".to_owned(),
                    description: "preferred".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 50,
                    revision: 3,
                    retired: false,
                },
            )
            .unwrap();
        let fallback = runtime
            .spawn_tool(
                id("fallback"),
                tenant("a"),
                ToolCapability {
                    name: "lookup".to_owned(),
                    description: "fallback".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 1,
                    revision: 8,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_tool(
                id("grant-preferred"),
                tenant("a"),
                ToolGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                preferred,
            )
            .unwrap();
        runtime
            .grant_tool(
                id("grant-fallback"),
                tenant("a"),
                ToolGrant {
                    order: 1,
                    enabled: true,
                },
                agent,
                fallback,
            )
            .unwrap();

        runtime.handle().prompt(agent, "first").unwrap();
        runtime.run_until_stalled().unwrap();
        runtime
            .world_mut()
            .entity_mut(preferred)
            .insert(ToolCapability {
                name: "lookup".to_owned(),
                description: "mutated".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                order: 50,
                revision: 4,
                retired: true,
            });
        let first = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            first.model_input().unwrap().tools,
            vec![ToolDecision {
                tool_entity: preferred,
                tool_id: id("preferred"),
                revision: 3,
                name: "lookup".to_owned(),
                description: "preferred".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                order: 0,
            }]
        );

        runtime.handle().prompt(agent, "second").unwrap();
        runtime.run_until_stalled().unwrap();
        let second = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(second.model_input().unwrap().tools.len(), 1);
        assert_eq!(
            second.model_input().unwrap().tools[0].tool_id,
            id("fallback")
        );
        assert_eq!(second.model_input().unwrap().tools[0].revision, 8);
    }

    #[test]
    fn parallel_tool_batch_commits_atomically_in_model_call_order() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let first_tool = runtime
            .spawn_tool(
                id("first-tool"),
                tenant("a"),
                ToolCapability {
                    name: "first".to_owned(),
                    description: "first tool".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 3,
                    retired: false,
                },
            )
            .unwrap();
        let second_tool = runtime
            .spawn_tool(
                id("second-tool"),
                tenant("a"),
                ToolCapability {
                    name: "second".to_owned(),
                    description: "second tool".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 1,
                    revision: 4,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_tool(
                id("first-grant"),
                tenant("a"),
                ToolGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                first_tool,
            )
            .unwrap();
        runtime
            .grant_tool(
                id("second-grant"),
                tenant("a"),
                ToolGrant {
                    order: 1,
                    enabled: true,
                },
                agent,
                second_tool,
            )
            .unwrap();

        let pending = runtime.handle().prompt(agent, "use both").unwrap();
        runtime.run_until_stalled().unwrap();
        let initial_model = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: initial_model.operation,
                generation: initial_model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: String::new(),
                    usage: Usage {
                        input_tokens: 4,
                        output_tokens: 1,
                    },
                    tool_calls: vec![
                        ModelToolCall {
                            id: "call-0".to_owned(),
                            provider_result_id: "call-0".to_owned(),
                            provider_call_id: None,
                            name: "first".to_owned(),
                            arguments: serde_json::json!({"value": 0}),
                        },
                        ModelToolCall {
                            id: "call-1".to_owned(),
                            provider_result_id: "call-1".to_owned(),
                            provider_call_id: None,
                            name: "second".to_owned(),
                            arguments: serde_json::json!({"value": 1}),
                        },
                    ],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let first_request = runtime.effects().try_recv().unwrap().unwrap();
        let second_request = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(first_request.tool_input().unwrap().index, 0);
        assert_eq!(second_request.tool_input().unwrap().index, 1);
        assert_eq!(first_request.tool_input().unwrap().decision.revision, 3);

        let _ = runtime.world_mut().despawn(first_tool);
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: second_request.operation,
                generation: second_request.generation,
                result: Ok(EffectOutput::Tool(ToolEffectOutput {
                    call_id: "call-1".to_owned(),
                    provider_result_id: "call-1".to_owned(),
                    provider_call_id: None,
                    name: "second".to_owned(),
                    raw: serde_json::json!({"result": 1}),
                    presentation: "second result".to_owned(),
                    failure: None,
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::WaitingTools { .. })
        ));
        assert!(
            !runtime
                .world()
                .get::<RunRecord>(run.entity())
                .unwrap()
                .transcript
                .iter()
                .any(|entry| matches!(entry, TranscriptEntry::ToolResult { .. }))
        );

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: first_request.operation,
                generation: first_request.generation,
                result: Ok(EffectOutput::Tool(ToolEffectOutput {
                    call_id: "call-0".to_owned(),
                    provider_result_id: "call-0".to_owned(),
                    provider_call_id: None,
                    name: "first".to_owned(),
                    raw: serde_json::json!({"result": 0}),
                    presentation: "first result".to_owned(),
                    failure: None,
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let followup_model = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            followup_model.model_input().unwrap().tool_results,
            vec![
                ToolEffectOutput {
                    call_id: "call-0".to_owned(),
                    provider_result_id: "call-0".to_owned(),
                    provider_call_id: None,
                    name: "first".to_owned(),
                    raw: serde_json::json!({"result": 0}),
                    presentation: "first result".to_owned(),
                    failure: None,
                },
                ToolEffectOutput {
                    call_id: "call-1".to_owned(),
                    provider_result_id: "call-1".to_owned(),
                    provider_call_id: None,
                    name: "second".to_owned(),
                    raw: serde_json::json!({"result": 1}),
                    presentation: "second result".to_owned(),
                    failure: None,
                },
            ]
        );
        assert_eq!(
            &followup_model.model_input().unwrap().history,
            &runtime
                .world()
                .get::<RunRecord>(run.entity())
                .unwrap()
                .transcript
        );
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: followup_model.operation,
                generation: followup_model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "done".to_owned(),
                    usage: Usage {
                        input_tokens: 6,
                        output_tokens: 2,
                    },
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Completed(RunOutput {
                text: "done".to_owned(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 3,
                },
            }))
        );
        let record = runtime.world().get::<RunRecord>(run.entity()).unwrap();
        assert_eq!(
            record.transcript,
            vec![
                TranscriptEntry::User("use both".to_owned()),
                TranscriptEntry::Assistant(String::new()),
                TranscriptEntry::AssistantToolCall {
                    call_id: "call-0".to_owned(),
                    name: "first".to_owned(),
                    arguments: serde_json::json!({"value": 0}),
                },
                TranscriptEntry::AssistantToolCall {
                    call_id: "call-1".to_owned(),
                    name: "second".to_owned(),
                    arguments: serde_json::json!({"value": 1}),
                },
                TranscriptEntry::ToolResult {
                    call_id: "call-0".to_owned(),
                    provider_result_id: "call-0".to_owned(),
                    provider_call_id: None,
                    name: "first".to_owned(),
                    raw: serde_json::json!({"result": 0}),
                    content: "first result".to_owned(),
                },
                TranscriptEntry::ToolResult {
                    call_id: "call-1".to_owned(),
                    provider_result_id: "call-1".to_owned(),
                    provider_call_id: None,
                    name: "second".to_owned(),
                    raw: serde_json::json!({"result": 1}),
                    content: "second result".to_owned(),
                },
                TranscriptEntry::Assistant("done".to_owned()),
            ]
        );
    }

    #[test]
    fn tool_batch_failure_is_atomic_and_selected_in_model_call_order() {
        let (mut runtime, agent) = runtime_with_tool();
        let pending = runtime.handle().prompt(agent, "call twice").unwrap();
        runtime.run_until_stalled().unwrap();
        let model_request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model_request.operation,
                generation: model_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: String::new(),
                    usage: Usage::default(),
                    tool_calls: vec![
                        ModelToolCall {
                            id: "first".to_owned(),
                            provider_result_id: "first".to_owned(),
                            provider_call_id: None,
                            name: "lookup".to_owned(),
                            arguments: serde_json::json!({"order": 0}),
                        },
                        ModelToolCall {
                            id: "second".to_owned(),
                            provider_result_id: "second".to_owned(),
                            provider_call_id: None,
                            name: "lookup".to_owned(),
                            arguments: serde_json::json!({"order": 1}),
                        },
                    ],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let first = runtime.effects().try_recv().unwrap().unwrap();
        let second = runtime.effects().try_recv().unwrap().unwrap();

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: second.operation,
                generation: second.generation,
                result: Err(CanonicalError::Provider {
                    message: "second failure".to_owned(),
                    retryable: false,
                }),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::WaitingTools { .. })
        ));

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: first.operation,
                generation: first.generation,
                result: Err(CanonicalError::Provider {
                    message: "first failure".to_owned(),
                    retryable: false,
                }),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::Provider {
                message: "first failure".to_owned(),
                retryable: false,
            }))
        );
        assert!(
            !runtime
                .world()
                .get::<RunRecord>(run.entity())
                .unwrap()
                .transcript
                .iter()
                .any(|entry| matches!(entry, TranscriptEntry::ToolResult { .. }))
        );
    }

    #[test]
    fn unknown_tool_call_fails_without_dispatching_a_partial_batch() {
        let (mut runtime, agent) = runtime_with_tool();
        let pending = runtime.handle().prompt(agent, "invalid call").unwrap();
        runtime.run_until_stalled().unwrap();
        let model_request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model_request.operation,
                generation: model_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: String::new(),
                    usage: Usage::default(),
                    tool_calls: vec![
                        ModelToolCall {
                            id: "valid".to_owned(),
                            provider_result_id: "valid".to_owned(),
                            provider_call_id: None,
                            name: "lookup".to_owned(),
                            arguments: serde_json::json!({}),
                        },
                        ModelToolCall {
                            id: "invalid".to_owned(),
                            provider_result_id: "invalid".to_owned(),
                            provider_call_id: None,
                            name: "not-advertised".to_owned(),
                            arguments: serde_json::json!({}),
                        },
                    ],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::UnknownTool(
                "not-advertised".to_owned(),
            )))
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
        let batch = runtime
            .world()
            .get::<RunToolBatches>(run.entity())
            .unwrap()
            .iter()
            .next()
            .unwrap();
        assert_eq!(
            runtime
                .world()
                .get::<ToolBatchState>(batch)
                .unwrap()
                .expected,
            2
        );
    }

    #[test]
    fn invalid_tool_repair_resolves_against_the_advertised_snapshot() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .spawn_policy(
                id("repair"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 3,
                    rule: PolicyRule::RepairInvalidTool {
                        from: Some("lookpu".to_owned()),
                        to: "lookup".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        runtime.handle().prompt(agent, "use lookup").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
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
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "lookpu".to_owned(),
                        arguments: serde_json::json!({"query": "x"}),
                    }],
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        let tool = runtime.effects().try_recv().unwrap().unwrap();
        let input = tool.tool_input().unwrap();
        assert_eq!(input.decision.name, "lookup");
        assert_eq!(input.arguments, serde_json::json!({"query": "x"}));
    }

    #[test]
    fn invalid_tool_repair_rejects_target_disallowed_by_tool_choice() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(
                id("agent"),
                tenant("a"),
                Agent {
                    tool_choice: Some(ModelToolChoice::None),
                    ..Agent::default()
                },
                model,
            )
            .unwrap();
        let tool = runtime
            .spawn_tool(
                id("tool"),
                tenant("a"),
                ToolCapability {
                    name: "lookup".to_owned(),
                    description: "lookup".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_tool(
                id("grant"),
                tenant("a"),
                ToolGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                tool,
            )
            .unwrap();
        runtime
            .spawn_policy(
                id("repair"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::RepairInvalidTool {
                        from: None,
                        to: "lookup".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let pending = runtime.handle().prompt(agent, "malicious call").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
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
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "missing".to_owned(),
                        arguments: serde_json::json!({}),
                    }],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::UnknownTool(
                "lookup".to_owned()
            )))
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
    }

    #[test]
    fn invalid_tool_retry_returns_feedback_without_tool_dispatch() {
        let (mut runtime, agent) = runtime_with_tool();
        runtime.set_invalid_tool_call_budget(agent, 1).unwrap();
        runtime
            .spawn_policy(
                id("retry"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::RetryInvalidTool {
                        tool: None,
                        feedback: "choose an advertised tool".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        runtime.handle().prompt(agent, "use lookup").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
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
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "missing".to_owned(),
                        arguments: serde_json::json!({}),
                    }],
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        let next_model = runtime.effects().try_recv().unwrap().unwrap();
        let input = next_model.model_input().unwrap();
        assert_eq!(input.tool_results.len(), 1);
        assert_eq!(
            input.tool_results[0].presentation,
            "choose an advertised tool"
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
    }

    #[test]
    fn invalid_tool_retry_enforces_invalid_and_total_model_budgets() {
        for (invalid_budget, max_model_calls, expected) in [
            (0, 4, CanonicalError::UnknownTool("missing".to_owned())),
            (1, 1, CanonicalError::ModelCallBudget { limit: 1 }),
        ] {
            let (mut runtime, agent) = runtime_with_tool();
            runtime
                .set_invalid_tool_call_budget(agent, invalid_budget)
                .unwrap();
            runtime
                .spawn_policy(
                    id("retry"),
                    tenant("a"),
                    Policy {
                        order: 0,
                        revision: 1,
                        rule: PolicyRule::RetryInvalidTool {
                            tool: None,
                            feedback: "retry".to_owned(),
                        },
                    },
                    agent,
                )
                .unwrap();
            let pending = runtime
                .handle()
                .prompt_configured(
                    agent,
                    "invalid",
                    Vec::<CompletionMessage>::new(),
                    None,
                    Some(max_model_calls),
                    None,
                )
                .unwrap();
            runtime.run_until_stalled().unwrap();
            let model = runtime.effects().try_recv().unwrap().unwrap();
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
                            id: "call".to_owned(),
                            provider_result_id: "call".to_owned(),
                            provider_call_id: None,
                            name: "missing".to_owned(),
                            arguments: serde_json::json!({}),
                        }],
                    })),
                })
                .unwrap();
            runtime.run_until_stalled().unwrap();
            let run = runtime.resolve_run(&pending).unwrap();
            assert_eq!(
                runtime.world().get::<RunState>(run.entity()),
                Some(&RunState::Failed(expected))
            );
            assert_eq!(runtime.effects().try_recv().unwrap(), None);
        }
    }

    #[test]
    fn cross_tenant_tool_grant_is_rejected() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let tool = runtime
            .spawn_tool(
                id("tool"),
                tenant("b"),
                ToolCapability {
                    name: "lookup".to_owned(),
                    description: String::new(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        assert_eq!(
            runtime.grant_tool(
                id("grant"),
                tenant("a"),
                ToolGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                tool,
            ),
            Err(SpawnError::TenantMismatch)
        );
    }

    #[test]
    fn discovery_rejects_stale_generations_and_retires_old_versions() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let source = runtime
            .spawn_discovery_source(id("mcp-source"), tenant("a"), "mcp")
            .unwrap();

        runtime.handle().refresh_discovery(source).unwrap();
        runtime.run_until_stalled().unwrap();
        let stale_request = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(stale_request.discovery_input().unwrap().generation, 1);
        runtime.handle().refresh_discovery(source).unwrap();
        runtime.run_until_stalled().unwrap();
        let current_request = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(current_request.discovery_input().unwrap().generation, 2);
        assert_eq!(
            runtime
                .world()
                .get::<OperationState>(stale_request.operation)
                .unwrap()
                .phase,
            OperationPhase::Superseded
        );

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: stale_request.operation,
                generation: stale_request.generation,
                result: Ok(EffectOutput::Discovery(DiscoveryEffectOutput {
                    tools: vec![DiscoveredTool {
                        key: "stale".to_owned(),
                        name: "stale".to_owned(),
                        description: "must not apply".to_owned(),
                        parameters: serde_json::json!({"type": "object"}),
                        order: 0,
                        revision: 1,
                    }],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let world = runtime.world_mut();
        let mut discovered = world.query::<&DiscoveryKey>();
        assert_eq!(discovered.iter(world).count(), 0);

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: current_request.operation,
                generation: current_request.generation,
                result: Ok(EffectOutput::Discovery(DiscoveryEffectOutput {
                    tools: vec![
                        DiscoveredTool {
                            key: "alpha".to_owned(),
                            name: "alpha".to_owned(),
                            description: "version two".to_owned(),
                            parameters: serde_json::json!({"type": "object"}),
                            order: 0,
                            revision: 2,
                        },
                        DiscoveredTool {
                            key: "beta".to_owned(),
                            name: "beta".to_owned(),
                            description: "removed later".to_owned(),
                            parameters: serde_json::json!({"type": "object"}),
                            order: 1,
                            revision: 2,
                        },
                    ],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let alpha = {
            let world = runtime.world_mut();
            let mut tools = world.query::<(Entity, &DiscoveryKey, &ToolCapability)>();
            tools
                .iter(world)
                .find(|(_, key, tool)| key.0 == "alpha" && !tool.retired)
                .map(|(entity, _, _)| entity)
                .unwrap()
        };
        runtime
            .grant_tool(
                id("alpha-grant"),
                tenant("a"),
                ToolGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                alpha,
            )
            .unwrap();
        runtime.handle().prompt(agent, "snapshot alpha").unwrap();
        runtime.run_until_stalled().unwrap();
        let model_request = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(model_request.model_input().unwrap().tools[0].revision, 2);

        runtime.handle().refresh_discovery(source).unwrap();
        runtime.run_until_stalled().unwrap();
        let replacement_request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: replacement_request.operation,
                generation: replacement_request.generation,
                result: Ok(EffectOutput::Discovery(DiscoveryEffectOutput {
                    tools: vec![DiscoveredTool {
                        key: "alpha".to_owned(),
                        name: "alpha".to_owned(),
                        description: "version three".to_owned(),
                        parameters: serde_json::json!({"type": "object"}),
                        order: 0,
                        revision: 3,
                    }],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let world = runtime.world_mut();
        let mut tools =
            world.query::<(&DiscoveryKey, &ToolCapability, Option<&RetiredCapability>)>();
        let rows = tools
            .iter(world)
            .map(|(key, tool, retired)| (key.0.clone(), tool.revision, retired.is_some()))
            .collect::<Vec<_>>();
        assert!(rows.contains(&("alpha".to_owned(), 2, true)));
        assert!(!rows.iter().any(|(key, _, _)| key == "beta"));
        assert!(rows.contains(&("alpha".to_owned(), 3, false)));
        assert_eq!(model_request.model_input().unwrap().tools[0].revision, 2);
    }

    #[test]
    fn invalid_discovery_snapshot_does_not_partially_reconcile() {
        let mut runtime = runtime();
        let source = runtime
            .spawn_discovery_source(id("source"), tenant("a"), "fake")
            .unwrap();
        runtime.handle().refresh_discovery(source).unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Discovery(DiscoveryEffectOutput {
                    tools: vec![
                        DiscoveredTool {
                            key: "duplicate".to_owned(),
                            name: "first".to_owned(),
                            description: String::new(),
                            parameters: serde_json::json!({"type": "object"}),
                            order: 0,
                            revision: 1,
                        },
                        DiscoveredTool {
                            key: "duplicate".to_owned(),
                            name: "second".to_owned(),
                            description: String::new(),
                            parameters: serde_json::json!({"type": "object"}),
                            order: 1,
                            revision: 1,
                        },
                    ],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(
            runtime
                .world()
                .get::<DiscoverySource>(source.entity())
                .unwrap()
                .state,
            DiscoveryState::Failed(CanonicalError::InvalidDiscovery(
                "duplicate source key `duplicate`".to_owned(),
            ))
        );
        let world = runtime.world_mut();
        let mut tools = world.query::<&DiscoveryKey>();
        assert_eq!(tools.iter(world).count(), 0);
    }

    #[test]
    fn cancellation_rejects_a_late_completion() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();

        runtime.handle().cancel(run).unwrap();
        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.effects().try_recv_cancellation().unwrap(),
            Some(EffectCancellation {
                operation: request.operation,
                generation: request.generation,
            })
        );
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "too late".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Cancelled)
        );
        assert_eq!(
            runtime.world().get::<OperationState>(request.operation),
            Some(&OperationState {
                generation: 0,
                phase: OperationPhase::Cancelled,
            })
        );
    }

    #[test]
    fn cancellation_prevents_dispatch_of_prepared_work() {
        let mut runtime = Runtime::new(RuntimeConfig {
            effect_capacity: 1,
            ..RuntimeConfig::default()
        })
        .unwrap();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime.handle().prompt(agent, "occupy queue").unwrap();
        let cancelled = runtime.handle().prompt(agent, "do not dispatch").unwrap();
        runtime.run_until_stalled().unwrap();

        let _in_flight = runtime.effects().try_recv().unwrap().unwrap();
        let cancelled = runtime.resolve_run(&cancelled).unwrap();
        runtime.handle().cancel(cancelled).unwrap();
        runtime.run_until_stalled().unwrap();

        assert_eq!(runtime.effects().try_recv().unwrap(), None);
        assert_eq!(
            runtime.world().get::<RunState>(cancelled.entity()),
            Some(&RunState::Cancelled)
        );
    }

    #[test]
    fn freeze_after_ingress_retains_completion_until_resume() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        runtime
            .handle()
            .pause_with_mode(run, PauseMode::FreezeAfterIngress)
            .unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "retained".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.world().get::<RunControl>(run.entity()),
            Some(&RunControl::Paused(PauseMode::FreezeAfterIngress))
        );
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::WaitingModel { .. })
        ));
        assert!(runtime.run_committed_turns(run).is_empty());

        runtime.handle().resume(run).unwrap();
        runtime.run_until_stalled().unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "retained"
        ));
    }

    #[test]
    fn cancel_and_suspend_rejects_late_completion_and_redispatches_new_generation() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        runtime
            .handle()
            .pause_with_mode(run, PauseMode::CancelAndSuspend)
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.effects().try_recv_cancellation().unwrap(),
            Some(EffectCancellation {
                operation: request.operation,
                generation: request.generation,
            })
        );
        assert_eq!(
            runtime.world().get::<RunControl>(run.entity()),
            Some(&RunControl::Paused(PauseMode::CancelAndSuspend))
        );
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "late".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        runtime.handle().resume(run).unwrap();
        runtime.run_until_stalled().unwrap();
        let retried = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(retried.operation, request.operation);
        assert_eq!(retried.generation, request.generation + 1);
    }

    #[test]
    fn drain_paused_run_does_not_block_sibling_progress() {
        let (mut runtime, agent) = runtime_with_tool();
        let first_pending = runtime.handle().prompt(agent, "first").unwrap();
        let second_pending = runtime.handle().prompt(agent, "second").unwrap();
        runtime.run_until_stalled().unwrap();
        let first_request = runtime.effects().try_recv().unwrap().unwrap();
        let second_request = runtime.effects().try_recv().unwrap().unwrap();
        let first = runtime.resolve_run(&first_pending).unwrap();
        let second = runtime.resolve_run(&second_pending).unwrap();
        runtime.handle().pause(first).unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: first_request.operation,
                generation: first_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: String::new(),
                    usage: Usage::default(),
                    tool_calls: vec![ModelToolCall {
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({}),
                    }],
                })),
            })
            .unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: second_request.operation,
                generation: second_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "sibling completed".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();

        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.world().get::<RunControl>(first.entity()),
            Some(&RunControl::Paused(PauseMode::Drain))
        );
        assert!(matches!(
            runtime.world().get::<RunState>(first.entity()),
            Some(RunState::WaitingTools { .. })
        ));
        assert!(matches!(
            runtime.world().get::<RunState>(second.entity()),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "sibling completed"
        ));
        assert_eq!(runtime.effects().try_recv().unwrap(), None);

        runtime.handle().resume(first).unwrap();
        runtime.run_until_stalled().unwrap();
        assert!(
            runtime
                .effects()
                .try_recv()
                .unwrap()
                .unwrap()
                .tool_input()
                .is_some()
        );
    }

    #[test]
    fn hosted_commands_spawn_agent_and_named_run_while_another_run_is_active() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let parent_agent = runtime
            .spawn_agent(id("parent-agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let parent_pending = runtime
            .handle()
            .prompt(parent_agent, "parent stays active")
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let parent_request = runtime.effects().try_recv().unwrap().unwrap();
        let parent = runtime.resolve_run(&parent_pending).unwrap();
        assert!(matches!(
            runtime.observe_run(parent).unwrap(),
            Some(RunState::WaitingModel { .. })
        ));

        let pending_agent = runtime
            .handle()
            .spawn_agent(
                id("dynamic-agent"),
                tenant("a"),
                Agent {
                    instructions: "spawned during execution".to_owned(),
                    ..Agent::default()
                },
                id("model"),
            )
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let dynamic_agent = pending_agent.try_resolve().unwrap().unwrap().unwrap();
        let dynamic_pending = runtime
            .handle()
            .spawn_run(id("dynamic-run"), dynamic_agent, "independent")
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let dynamic_request = runtime.effects().try_recv().unwrap().unwrap();
        let dynamic_run = runtime.resolve_run(&dynamic_pending).unwrap();
        assert_eq!(
            runtime.world().get::<StableId>(dynamic_run.entity()),
            Some(&id("dynamic-run"))
        );

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: dynamic_request.operation,
                generation: dynamic_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "dynamic completed".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert!(matches!(
            runtime.observe_run(dynamic_run).unwrap(),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "dynamic completed"
        ));
        assert!(matches!(
            runtime.observe_run(parent).unwrap(),
            Some(RunState::WaitingModel { .. })
        ));
        assert_eq!(
            runtime
                .world()
                .get::<OperationState>(parent_request.operation),
            Some(&OperationState {
                generation: parent_request.generation,
                phase: OperationPhase::InFlight,
            })
        );
    }

    #[test]
    fn child_results_commit_in_creation_order_not_completion_order() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let parent_pending = runtime.handle().prompt(agent, "parent").unwrap();
        runtime.run_until_stalled().unwrap();
        let _parent_request = runtime.effects().try_recv().unwrap().unwrap();
        let parent = runtime.resolve_run(&parent_pending).unwrap();
        let first_pending = runtime
            .handle()
            .spawn_child_run(parent, agent, "first child")
            .unwrap();
        let second_pending = runtime
            .handle()
            .spawn_child_run(parent, agent, "second child")
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let first_request = runtime.effects().try_recv().unwrap().unwrap();
        let second_request = runtime.effects().try_recv().unwrap().unwrap();
        let first = runtime.resolve_run(&first_pending).unwrap();
        let second = runtime.resolve_run(&second_pending).unwrap();
        assert_eq!(
            runtime.world().get::<ParentRun>(first.entity()),
            Some(&ParentRun(parent.entity()))
        );
        assert_eq!(
            runtime.world().get::<ParentRun>(second.entity()),
            Some(&ParentRun(parent.entity()))
        );

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: second_request.operation,
                generation: second_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "second output".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert!(
            !runtime
                .world()
                .get::<RunRecord>(parent.entity())
                .unwrap()
                .transcript
                .iter()
                .any(|entry| matches!(entry, TranscriptEntry::ChildResult { .. }))
        );

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: first_request.operation,
                generation: first_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "first output".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let results = runtime
            .world()
            .get::<RunRecord>(parent.entity())
            .unwrap()
            .transcript
            .iter()
            .filter_map(|entry| match entry {
                TranscriptEntry::ChildResult { result, .. } => result.as_ref().ok().cloned(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(results, vec!["first output", "second output"]);
    }

    #[test]
    fn cancelling_parent_cancels_active_children() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let parent_pending = runtime.handle().prompt(agent, "parent").unwrap();
        runtime.run_until_stalled().unwrap();
        let _ = runtime.effects().try_recv().unwrap().unwrap();
        let parent = runtime.resolve_run(&parent_pending).unwrap();
        let child_pending = runtime
            .handle()
            .spawn_child_run(parent, agent, "child")
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let _ = runtime.effects().try_recv().unwrap().unwrap();
        let child = runtime.resolve_run(&child_pending).unwrap();

        runtime.handle().cancel(parent).unwrap();
        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(child.entity()),
            Some(&RunState::Cancelled)
        );
    }

    #[test]
    fn agent_control_rejects_admission_and_bulk_resumes_runs() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let active_pending = runtime.handle().prompt(agent, "active").unwrap();
        runtime.run_until_stalled().unwrap();
        let _ = runtime.effects().try_recv().unwrap().unwrap();
        let active = runtime.resolve_run(&active_pending).unwrap();

        runtime
            .handle()
            .pause_agent(agent, PauseMode::FreezeAfterIngress)
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.world().get::<RunControl>(active.entity()),
            Some(&RunControl::Paused(PauseMode::FreezeAfterIngress))
        );
        let rejected_pending = runtime.handle().prompt(agent, "rejected").unwrap();
        runtime.run_until_stalled().unwrap();
        let rejected = runtime.resolve_run(&rejected_pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(rejected.entity()),
            Some(&RunState::Failed(CanonicalError::AgentAdmissionDenied))
        );

        runtime.handle().resume_agent(agent).unwrap();
        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.world().get::<RunControl>(active.entity()),
            Some(&RunControl::Running)
        );
    }

    #[test]
    fn accepted_prompt_for_stale_agent_resolves_as_terminal_failure() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        runtime.world_mut().entity_mut(agent.entity()).despawn();

        let pending = runtime.handle().prompt(agent, "hello").unwrap();
        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();

        assert_eq!(
            runtime.observe_run(run).unwrap(),
            Some(RunState::Failed(CanonicalError::StaleEntity(
                "agent".to_owned()
            )))
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
    }

    #[test]
    fn runtime_owned_deadline_times_out_and_rejects_late_completion() {
        let mut runtime = Runtime::new(RuntimeConfig {
            effect_timeout_ticks: 2,
            ..RuntimeConfig::default()
        })
        .unwrap();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let pending = runtime.handle().prompt(agent, "wait").unwrap();
        runtime.run_until_stalled().unwrap();
        let request = runtime.effects().try_recv().unwrap().unwrap();

        runtime.update();
        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::Timeout))
        );

        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "late".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::Timeout))
        );
    }

    #[test]
    fn persistence_uses_stable_ids_and_remaps_relationships() {
        let mut source = runtime();
        let model = source
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 3,
                    retired: false,
                },
            )
            .unwrap();
        let agent = source
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        source
            .spawn_policy(
                id("policy"),
                tenant("a"),
                Policy {
                    order: 1,
                    revision: 2,
                    rule: PolicyRule::Allow,
                },
                agent,
            )
            .unwrap();
        let tool = source
            .spawn_tool(
                id("tool"),
                tenant("a"),
                ToolCapability {
                    name: "lookup".to_owned(),
                    description: "lookup".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 5,
                    retired: false,
                },
            )
            .unwrap();
        source
            .grant_tool(
                id("tool-grant"),
                tenant("a"),
                ToolGrant {
                    order: 2,
                    enabled: true,
                },
                agent,
                tool,
            )
            .unwrap();
        let store = source
            .spawn_store(
                id("store"),
                tenant("a"),
                StoreCapability {
                    kind: "conversation-memory".to_owned(),
                    revision: 6,
                    retired: false,
                },
            )
            .unwrap();
        source
            .grant_store(
                id("store-grant"),
                tenant("a"),
                StoreGrant {
                    order: 3,
                    enabled: true,
                },
                agent,
                store,
            )
            .unwrap();
        let snapshot = source.snapshot().unwrap();
        let serialized = serde_json::to_string(&snapshot).unwrap();
        assert!(serialized.contains("\"model_id\":\"model\""));
        assert!(serialized.contains("\"agent_id\":\"agent\""));
        assert!(serialized.contains("\"tool_id\":\"tool\""));
        assert!(serialized.contains("\"store_id\":\"store\""));
        assert!(!serialized.contains("generation"));

        let mut destination = runtime();
        let restored = destination.restore(snapshot).unwrap();
        let restored_agent = restored.0[&id("agent")];
        let restored_model = restored.0[&id("model")];
        let restored_policy = restored.0[&id("policy")];
        let restored_tool = restored.0[&id("tool")];
        let restored_tool_grant = restored.0[&id("tool-grant")];
        let restored_store = restored.0[&id("store")];
        let restored_store_grant = restored.0[&id("store-grant")];
        assert_eq!(
            destination
                .world()
                .get::<UsesModel>(restored_agent)
                .unwrap()
                .get(),
            restored_model
        );
        assert_eq!(
            destination
                .world()
                .get::<PolicyFor>(restored_policy)
                .unwrap()
                .get(),
            restored_agent
        );
        assert_eq!(
            destination
                .world()
                .get::<GrantForAgent>(restored_tool_grant)
                .unwrap()
                .get(),
            restored_agent
        );
        assert_eq!(
            destination
                .world()
                .get::<GrantForTool>(restored_tool_grant)
                .unwrap()
                .get(),
            restored_tool
        );
        assert_eq!(
            destination
                .world()
                .get::<StoreGrantForAgent>(restored_store_grant)
                .unwrap()
                .get(),
            restored_agent
        );
        assert_eq!(
            destination
                .world()
                .get::<StoreGrantForStore>(restored_store_grant)
                .unwrap()
                .get(),
            restored_store
        );
    }

    #[test]
    fn persistence_remaps_discovery_source_provenance() {
        let mut original = runtime();
        let source = original
            .spawn_discovery_source(id("source"), tenant("a"), "mcp")
            .unwrap();
        original
            .world_mut()
            .get_mut::<DiscoverySource>(source.entity())
            .unwrap()
            .generation = 9;
        original.world_mut().spawn((
            id("discovered-tool"),
            tenant("a"),
            ToolCapability {
                name: "lookup".to_owned(),
                description: "discovered".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                order: 2,
                revision: 4,
                retired: false,
            },
            DiscoveredFrom(source.entity()),
            DiscoveryKey("remote/lookup".to_owned()),
        ));

        let snapshot = original.snapshot().unwrap();
        let mut restored = runtime();
        let entities = restored.restore(snapshot).unwrap();
        let restored_source = entities.0[&id("source")];
        let restored_tool = entities.0[&id("discovered-tool")];
        assert_eq!(
            restored
                .world()
                .get::<DiscoverySource>(restored_source)
                .unwrap()
                .generation,
            9
        );
        assert_eq!(
            restored
                .world()
                .get::<DiscoveredFrom>(restored_tool)
                .unwrap()
                .get(),
            restored_source
        );
        assert_eq!(
            restored.world().get::<DiscoveryKey>(restored_tool),
            Some(&DiscoveryKey("remote/lookup".to_owned()))
        );
    }

    #[test]
    fn conversation_memory_loads_before_model_and_persists_before_success() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let store = runtime
            .spawn_store(
                id("memory"),
                tenant("a"),
                StoreCapability {
                    kind: "conversation-memory".to_owned(),
                    revision: 7,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_store(
                id("memory-grant"),
                tenant("a"),
                StoreGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                store,
            )
            .unwrap();
        let pending = runtime
            .handle()
            .prompt_in_conversation(agent, id("conversation"), "new question")
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let load = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(load.store_input().unwrap().decision.revision, 7);
        assert_eq!(
            load.store_input().unwrap().operation,
            StoreOperation::LoadConversation {
                conversation: id("conversation")
            }
        );
        runtime
            .world_mut()
            .entity_mut(store)
            .insert(StoreCapability {
                kind: "conversation-memory".to_owned(),
                revision: 8,
                retired: true,
            });
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: load.operation,
                generation: load.generation,
                result: Ok(EffectOutput::Store(StoreEffectOutput::LoadedConversation(
                    vec![
                        TranscriptEntry::User("old question".to_owned()),
                        TranscriptEntry::Assistant("old answer".to_owned()),
                    ],
                ))),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let model_request = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            model_request.model_input().unwrap().history,
            vec![
                TranscriptEntry::User("old question".to_owned()),
                TranscriptEntry::Assistant("old answer".to_owned()),
                TranscriptEntry::User("new question".to_owned()),
            ]
        );
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model_request.operation,
                generation: model_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "new answer".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::WaitingStore { .. })
        ));
        let persist = runtime.effects().try_recv().unwrap().unwrap();
        assert_eq!(persist.store_input().unwrap().decision.revision, 7);
        assert_eq!(persist.store_input().unwrap().decision.store_entity, store);
        assert_eq!(
            persist.store_input().unwrap().operation,
            StoreOperation::PersistConversation {
                conversation: id("conversation"),
                entries: vec![
                    TranscriptEntry::User("old question".to_owned()),
                    TranscriptEntry::Assistant("old answer".to_owned()),
                    TranscriptEntry::User("new question".to_owned()),
                    TranscriptEntry::Assistant("new answer".to_owned()),
                ],
            }
        );
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: persist.operation,
                generation: persist.generation,
                result: Ok(EffectOutput::Store(StoreEffectOutput::Persisted)),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        assert!(matches!(
            runtime.world().get::<RunState>(run.entity()),
            Some(RunState::Completed(_))
        ));
    }

    #[test]
    fn conversation_persistence_failure_prevents_terminal_success() {
        let mut runtime = runtime();
        let model = runtime
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = runtime
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let store = runtime
            .spawn_store(
                id("memory"),
                tenant("a"),
                StoreCapability {
                    kind: "conversation-memory".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_store(
                id("grant"),
                tenant("a"),
                StoreGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                store,
            )
            .unwrap();
        let pending = runtime
            .handle()
            .prompt_in_conversation(agent, id("conversation"), "question")
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let load = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: load.operation,
                generation: load.generation,
                result: Ok(EffectOutput::Store(StoreEffectOutput::LoadedConversation(
                    Vec::new(),
                ))),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let model_request = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model_request.operation,
                generation: model_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "answer".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let persist = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: persist.operation,
                generation: persist.generation,
                result: Err(CanonicalError::Provider {
                    message: "storage unavailable".to_owned(),
                    retryable: true,
                }),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();

        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(CanonicalError::Provider {
                message: "storage unavailable".to_owned(),
                retryable: true,
            }))
        );
    }

    #[test]
    fn invalid_snapshot_is_rejected_before_spawning_any_entities() {
        let mut runtime = runtime();
        let snapshot = DomainSnapshot {
            models: Vec::new(),
            agents: vec![PersistedAgent {
                id: id("agent"),
                tenant: tenant("a"),
                agent: Agent::default(),
                control: AgentControl::default(),
                invalid_tool_call_budget: None,
                model_id: id("missing"),
                output_requirement: None,
                retrieval_requirement: None,
            }],
            ..DomainSnapshot::default()
        };
        assert_eq!(
            runtime.restore(snapshot).unwrap_err(),
            PersistenceError::MissingReference("missing".to_owned())
        );
        let world = runtime.world_mut();
        let mut query = world.query::<&StableId>();
        assert_eq!(query.iter(world).count(), 0);
    }

    #[test]
    fn embedded_installer_uses_the_same_schedule() {
        let mut world = World::new();
        let installed = install_runtime(&mut world, RuntimeConfig::default()).unwrap();
        assert!(world.get_resource::<Schedules>().is_some());
        let tenant = tenant("embedded");
        let model = installed
            .spawn_model(
                &mut world,
                id("model"),
                tenant.clone(),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = installed
            .spawn_agent(&mut world, id("agent"), tenant, Agent::default(), model)
            .unwrap();
        let pending = installed.handle.prompt(agent, "hello").unwrap();
        let request = (0..4)
            .find_map(|_| {
                world.run_schedule(RigSchedule);
                installed.effects.try_recv().unwrap()
            })
            .expect("embedded schedule should dispatch the model effect");
        installed
            .effects
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "embedded result".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        world.run_schedule(RigSchedule);
        let run = installed.resolve_run(&world, &pending).unwrap().unwrap();
        assert!(matches!(
            installed.observe_run(&mut world, run).unwrap(),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "embedded result"
        ));
    }

    #[test]
    fn hosted_ingress_notifies_the_supplied_waker() {
        let wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed_wakes = wakes.clone();
        let mut world = World::new();
        let installed =
            install_runtime_with_waker(&mut world, RuntimeConfig::default(), move || {
                observed_wakes.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
        let model = world
            .spawn((
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            ))
            .id();
        let agent = world
            .spawn((id("agent"), tenant("a"), Agent::default(), UsesModel(model)))
            .id();
        let pending = installed
            .handle
            .prompt(
                AgentHandle {
                    runtime_id: installed.handle.runtime_id,
                    entity: agent,
                },
                "wake",
            )
            .unwrap();
        assert_eq!(wakes.load(Ordering::Relaxed), 1);
        for _ in 0..3 {
            world.run_schedule(RigSchedule);
        }
        let request = installed.effects.try_recv().unwrap().unwrap();
        installed
            .effects
            .completion_sender()
            .try_send_panic(request.operation, request.generation, "worker panic")
            .unwrap();
        assert_eq!(wakes.load(Ordering::Relaxed), 2);
        for _ in 0..2 {
            world.run_schedule(RigSchedule);
        }
        let run = world
            .resource::<StableIdIndex>()
            .0
            .get(pending.stable_id())
            .copied()
            .unwrap();
        assert_eq!(
            world.get::<RunState>(run),
            Some(&RunState::Failed(CanonicalError::ExecutorPanicked(
                "worker panic".to_owned()
            )))
        );
    }

    #[test]
    fn executor_shutdown_settles_new_work_as_disconnected() {
        let mut world = World::new();
        let installed = install_runtime(&mut world, RuntimeConfig::default()).unwrap();
        let handle = installed.handle.clone();
        let model = world
            .spawn((
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            ))
            .id();
        let agent = world
            .spawn((id("agent"), tenant("a"), Agent::default(), UsesModel(model)))
            .id();
        drop(installed.effects);
        let pending = handle
            .prompt(
                AgentHandle {
                    runtime_id: handle.runtime_id,
                    entity: agent,
                },
                "shutdown",
            )
            .unwrap();
        for _ in 0..4 {
            world.run_schedule(RigSchedule);
        }
        let run = world
            .resource::<StableIdIndex>()
            .0
            .get(pending.stable_id())
            .copied()
            .unwrap();
        assert_eq!(
            world.get::<RunState>(run),
            Some(&RunState::Failed(CanonicalError::ExecutorDisconnected))
        );
    }

    #[test]
    fn active_run_snapshot_rejects_live_effect_and_resumes_cancelled_checkpoint() {
        let mut source = runtime();
        let model = source
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = source
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let domain = source.snapshot().unwrap();
        let pending = source.handle().prompt(agent, "checkpoint me").unwrap();
        source.run_until_stalled().unwrap();
        let original = source.effects().try_recv().unwrap().unwrap();
        let run = source.resolve_run(&pending).unwrap();
        assert!(matches!(
            source.snapshot_active_run(run),
            Err(ActiveRunSnapshotError::UnsafeInFlightEffect(_))
        ));

        source
            .handle()
            .pause_with_mode(run, PauseMode::CancelAndSuspend)
            .unwrap();
        source.run_until_stalled().unwrap();
        assert!(matches!(
            source.world().get::<RunControl>(run.entity()),
            Some(RunControl::Paused(PauseMode::CancelAndSuspend))
        ));
        let snapshot = source.snapshot_active_run(run).unwrap();
        let encoded = serde_json::to_string(&snapshot).unwrap();
        assert!(!encoded.contains("model_entity"));
        assert!(!encoded.contains("tool_entity"));
        assert!(!encoded.contains("store_entity"));
        let snapshot: ActiveRunSnapshot = serde_json::from_str(&encoded).unwrap();

        let mut restored = runtime();
        restored.restore(domain).unwrap();
        let restored_runs = restored.restore_active_run(snapshot).unwrap();
        let restored_entity = restored_runs.0.get(pending.stable_id()).copied().unwrap();
        let restored_run = RunHandle {
            runtime_id: restored.handle.runtime_id,
            entity: restored_entity,
        };
        restored.handle().resume(restored_run).unwrap();
        restored.run_until_stalled().unwrap();
        let restored_request = restored.effects().try_recv().unwrap().unwrap();
        assert!(restored_request.generation > original.generation);
        restored
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: restored_request.operation,
                generation: restored_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "resumed".to_owned(),
                    usage: Usage {
                        input_tokens: 2,
                        output_tokens: 1,
                    },
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        restored.run_until_stalled().unwrap();
        assert_eq!(
            restored.observe_run(restored_run).unwrap(),
            Some(RunState::Completed(RunOutput {
                text: "resumed".to_owned(),
                usage: Usage {
                    input_tokens: 2,
                    output_tokens: 1,
                },
            }))
        );
        assert_eq!(
            restored.run_transcript(restored_run),
            Some(vec![
                TranscriptEntry::User("checkpoint me".to_owned()),
                TranscriptEntry::Assistant("resumed".to_owned()),
            ])
        );
    }

    #[test]
    fn active_run_snapshot_restores_request_policy_cursor_and_pending_approval() {
        let mut source = runtime();
        let model = source
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = source
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        source
            .spawn_policy(
                id("approve-request"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 4,
                    rule: PolicyRule::RequireApproval {
                        point: PolicyPoint::Request,
                        prompt: "approve request".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let domain = source.snapshot().unwrap();
        let pending = source.handle().prompt(agent, "policy checkpoint").unwrap();
        source.run_until_stalled().unwrap();
        let approval = source.effects().try_recv().unwrap().unwrap();
        assert!(approval.policy_approval_input().is_some());
        let run = source.resolve_run(&pending).unwrap();
        source
            .handle()
            .pause_with_mode(run, PauseMode::CancelAndSuspend)
            .unwrap();
        source.run_until_stalled().unwrap();
        let snapshot: ActiveRunSnapshot = serde_json::from_str(
            &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
        )
        .unwrap();

        let mut restored = runtime();
        restored.restore(domain).unwrap();
        let runs = restored.restore_active_run(snapshot).unwrap();
        let run = RunHandle {
            runtime_id: restored.handle.runtime_id,
            entity: runs.0.get(pending.stable_id()).copied().unwrap(),
        };
        restored.handle().resume(run).unwrap();
        restored.run_until_stalled().unwrap();
        let approval = restored.effects().try_recv().unwrap().unwrap();
        assert_eq!(approval.policy_approval_input().unwrap().revision, 4);
        restored
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: approval.operation,
                generation: approval.generation,
                result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                    approved: true,
                    reason: Some("restored approval".to_owned()),
                })),
            })
            .unwrap();
        restored.run_until_stalled().unwrap();
        let model = restored.effects().try_recv().unwrap().unwrap();
        assert!(model.model_input().is_some());
        restored
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model.operation,
                generation: model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "approved after restore".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        restored.run_until_stalled().unwrap();
        assert!(matches!(
            restored.observe_run(run).unwrap(),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "approved after restore"
        ));
    }

    #[test]
    fn active_run_snapshot_restores_tool_batch_and_result_policy() {
        let (mut source, agent) = runtime_with_tool();
        source
            .spawn_policy(
                id("approve-result"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 2,
                    rule: PolicyRule::RequireApproval {
                        point: PolicyPoint::ToolResult,
                        prompt: "approve tool result".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let domain = source.snapshot().unwrap();
        let (pending, tool) = advance_to_lookup_tool(&mut source, agent);
        let input = tool.tool_input().unwrap();
        source
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
                    raw: serde_json::json!({"answer": 42}),
                    presentation: "42".to_owned(),
                    failure: None,
                })),
            })
            .unwrap();
        source.run_until_stalled().unwrap();
        let approval = source.effects().try_recv().unwrap().unwrap();
        assert!(approval.policy_approval_input().is_some());
        let run = source.resolve_run(&pending).unwrap();
        source
            .handle()
            .pause_with_mode(run, PauseMode::CancelAndSuspend)
            .unwrap();
        source.run_until_stalled().unwrap();
        let snapshot: ActiveRunSnapshot = serde_json::from_str(
            &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
        )
        .unwrap();

        let mut restored = runtime();
        restored.restore(domain).unwrap();
        let runs = restored.restore_active_run(snapshot).unwrap();
        let run = RunHandle {
            runtime_id: restored.handle.runtime_id,
            entity: runs.0.get(pending.stable_id()).copied().unwrap(),
        };
        restored.handle().resume(run).unwrap();
        restored.run_until_stalled().unwrap();
        let approval = restored.effects().try_recv().unwrap().unwrap();
        restored
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
            .unwrap();
        restored.run_until_stalled().unwrap();
        let model = restored.effects().try_recv().unwrap().unwrap();
        assert_eq!(
            model
                .model_input()
                .unwrap()
                .tool_results
                .first()
                .unwrap()
                .presentation,
            "42"
        );
        restored
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model.operation,
                generation: model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "tool batch resumed".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        restored.run_until_stalled().unwrap();
        assert!(matches!(
            restored.observe_run(run).unwrap(),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "tool batch resumed"
        ));
    }

    #[test]
    fn active_run_snapshot_restores_pending_invalid_call_resolution() {
        let (mut source, agent) = runtime_with_tool();
        source
            .spawn_policy(
                id("approve-invalid"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::RequireApproval {
                        point: PolicyPoint::InvalidToolCall,
                        prompt: "approve invalid repair".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        source
            .spawn_policy(
                id("repair-invalid"),
                tenant("a"),
                Policy {
                    order: 1,
                    revision: 3,
                    rule: PolicyRule::RepairInvalidTool {
                        from: Some("lookpu".to_owned()),
                        to: "lookup".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let domain = source.snapshot().unwrap();
        let pending = source
            .handle()
            .prompt(agent, "repair after restore")
            .unwrap();
        source.run_until_stalled().unwrap();
        let model = source.effects().try_recv().unwrap().unwrap();
        source
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
                        id: "invalid-call".to_owned(),
                        provider_result_id: "invalid-result".to_owned(),
                        provider_call_id: Some("provider-call".to_owned()),
                        name: "lookpu".to_owned(),
                        arguments: serde_json::json!({"query": "persist me"}),
                    }],
                })),
            })
            .unwrap();
        source.run_until_stalled().unwrap();
        let approval = source.effects().try_recv().unwrap().unwrap();
        assert!(approval.policy_approval_input().is_some());
        let run = source.resolve_run(&pending).unwrap();
        source
            .handle()
            .pause_with_mode(run, PauseMode::CancelAndSuspend)
            .unwrap();
        source.run_until_stalled().unwrap();
        let snapshot: ActiveRunSnapshot = serde_json::from_str(
            &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
        )
        .unwrap();

        let mut restored = runtime();
        restored.restore(domain).unwrap();
        let runs = restored.restore_active_run(snapshot).unwrap();
        let run = RunHandle {
            runtime_id: restored.handle.runtime_id,
            entity: runs.0.get(pending.stable_id()).copied().unwrap(),
        };
        restored.handle().resume(run).unwrap();
        restored.run_until_stalled().unwrap();
        let approval = restored.effects().try_recv().unwrap().unwrap();
        restored
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
            .unwrap();
        restored.run_until_stalled().unwrap();
        let tool = restored.effects().try_recv().unwrap().unwrap();
        let input = tool.tool_input().unwrap();
        assert_eq!(input.decision.name, "lookup");
        assert_eq!(input.call_id, "invalid-call");
        assert_eq!(input.provider_result_id, "invalid-result");
        assert_eq!(input.provider_call_id.as_deref(), Some("provider-call"));
        assert_eq!(input.arguments, serde_json::json!({"query": "persist me"}));
    }

    #[test]
    fn active_run_snapshot_restores_memory_load_and_terminal_persistence() {
        let mut source = runtime();
        let model = source
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = source
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let store = source
            .spawn_store(
                id("memory"),
                tenant("a"),
                StoreCapability {
                    kind: "conversation-memory".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        source
            .grant_store(
                id("memory-grant"),
                tenant("a"),
                StoreGrant {
                    order: 0,
                    enabled: true,
                },
                agent,
                store,
            )
            .unwrap();
        let domain = source.snapshot().unwrap();
        let pending = source
            .handle()
            .prompt_in_conversation(agent, id("conversation"), "new question")
            .unwrap();
        source.run_until_stalled().unwrap();
        let load = source.effects().try_recv().unwrap().unwrap();
        assert!(matches!(
            load.store_input().unwrap().operation,
            StoreOperation::LoadConversation { .. }
        ));
        let run = source.resolve_run(&pending).unwrap();
        source
            .handle()
            .pause_with_mode(run, PauseMode::CancelAndSuspend)
            .unwrap();
        source.run_until_stalled().unwrap();
        let snapshot: ActiveRunSnapshot = serde_json::from_str(
            &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
        )
        .unwrap();

        let mut middle = runtime();
        middle.restore(domain).unwrap();
        let runs = middle.restore_active_run(snapshot).unwrap();
        let middle_run = RunHandle {
            runtime_id: middle.handle.runtime_id,
            entity: runs.0.get(pending.stable_id()).copied().unwrap(),
        };
        middle.handle().resume(middle_run).unwrap();
        middle.run_until_stalled().unwrap();
        let load = middle.effects().try_recv().unwrap().unwrap();
        middle
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: load.operation,
                generation: load.generation,
                result: Ok(EffectOutput::Store(StoreEffectOutput::LoadedConversation(
                    vec![
                        TranscriptEntry::User("old question".to_owned()),
                        TranscriptEntry::Assistant("old answer".to_owned()),
                    ],
                ))),
            })
            .unwrap();
        middle.run_until_stalled().unwrap();
        let model = middle.effects().try_recv().unwrap().unwrap();
        middle
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model.operation,
                generation: model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "new answer".to_owned(),
                    usage: Usage {
                        input_tokens: 5,
                        output_tokens: 2,
                    },
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        middle.run_until_stalled().unwrap();
        let persist = middle.effects().try_recv().unwrap().unwrap();
        assert!(matches!(
            persist.store_input().unwrap().operation,
            StoreOperation::PersistConversation { .. }
        ));
        middle
            .handle()
            .pause_with_mode(middle_run, PauseMode::CancelAndSuspend)
            .unwrap();
        middle.run_until_stalled().unwrap();
        let domain = middle.snapshot().unwrap();
        let snapshot: ActiveRunSnapshot = serde_json::from_str(
            &serde_json::to_string(&middle.snapshot_active_run(middle_run).unwrap()).unwrap(),
        )
        .unwrap();

        let mut restored = runtime();
        restored.restore(domain).unwrap();
        let runs = restored.restore_active_run(snapshot).unwrap();
        let run = RunHandle {
            runtime_id: restored.handle.runtime_id,
            entity: runs.0.get(pending.stable_id()).copied().unwrap(),
        };
        restored.handle().resume(run).unwrap();
        restored.run_until_stalled().unwrap();
        let persist = restored.effects().try_recv().unwrap().unwrap();
        let store_input = persist.store_input().unwrap();
        assert!(matches!(
            store_input.operation,
            StoreOperation::PersistConversation { .. }
        ));
        let StoreOperation::PersistConversation { entries, .. } = &store_input.operation else {
            return;
        };
        assert!(entries.contains(&TranscriptEntry::Assistant("new answer".to_owned())));
        restored
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: persist.operation,
                generation: persist.generation,
                result: Ok(EffectOutput::Store(StoreEffectOutput::Persisted)),
            })
            .unwrap();
        restored.run_until_stalled().unwrap();
        assert_eq!(
            restored.observe_run(run).unwrap(),
            Some(RunState::Completed(RunOutput {
                text: "new answer".to_owned(),
                usage: Usage {
                    input_tokens: 5,
                    output_tokens: 2,
                },
            }))
        );
    }

    #[test]
    fn active_run_snapshot_restores_child_topology_and_result_order() {
        let mut source = runtime();
        let model = source
            .spawn_model(
                id("model"),
                tenant("a"),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        let agent = source
            .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
            .unwrap();
        let domain = source.snapshot().unwrap();
        let parent_pending = source.handle().prompt(agent, "parent").unwrap();
        source.run_until_stalled().unwrap();
        let _ = source.effects().try_recv().unwrap().unwrap();
        let parent = source.resolve_run(&parent_pending).unwrap();
        let first_pending = source
            .handle()
            .spawn_child_run(parent, agent, "first child")
            .unwrap();
        let second_pending = source
            .handle()
            .spawn_child_run(parent, agent, "second child")
            .unwrap();
        source.run_until_stalled().unwrap();
        let _ = source.effects().try_recv().unwrap().unwrap();
        let _ = source.effects().try_recv().unwrap().unwrap();
        let first = source.resolve_run(&first_pending).unwrap();
        let second = source.resolve_run(&second_pending).unwrap();
        for run in [parent, first, second] {
            source
                .handle()
                .pause_with_mode(run, PauseMode::CancelAndSuspend)
                .unwrap();
        }
        source.run_until_stalled().unwrap();
        let snapshot: ActiveRunSnapshot = serde_json::from_str(
            &serde_json::to_string(&source.snapshot_active_run(parent).unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(snapshot.runs.len(), 3);

        let mut restored = runtime();
        restored.restore(domain).unwrap();
        let runs = restored.restore_active_run(snapshot).unwrap();
        let handle = |id: &StableId| RunHandle {
            runtime_id: restored.handle.runtime_id,
            entity: runs.0.get(id).copied().unwrap(),
        };
        let parent = handle(parent_pending.stable_id());
        let first = handle(first_pending.stable_id());
        let second = handle(second_pending.stable_id());
        assert_eq!(
            restored.world().get::<ParentRun>(first.entity()),
            Some(&ParentRun(parent.entity()))
        );
        assert_eq!(
            restored.world().get::<ParentRun>(second.entity()),
            Some(&ParentRun(parent.entity()))
        );
        for run in [parent, first, second] {
            restored.handle().resume(run).unwrap();
        }
        restored.run_until_stalled().unwrap();
        let requests = (0..3)
            .map(|_| restored.effects().try_recv().unwrap().unwrap())
            .collect::<Vec<_>>();
        let operation_for = |run: RunHandle| match restored.world().get::<RunState>(run.entity()) {
            Some(RunState::WaitingModel { operation }) => Some(*operation),
            _ => None,
        };
        let first_operation = operation_for(first).unwrap();
        let second_operation = operation_for(second).unwrap();
        let parent_operation = operation_for(parent).unwrap();
        let find_request = |operation| {
            requests
                .iter()
                .find(|request| request.operation == operation)
                .cloned()
                .unwrap()
        };
        for (request, text) in [
            (find_request(second_operation), "second output"),
            (find_request(first_operation), "first output"),
        ] {
            restored
                .effects()
                .completion_sender()
                .try_send(EffectCompletion {
                    operation: request.operation,
                    generation: request.generation,
                    result: Ok(EffectOutput::Model(ModelEffectOutput {
                        assistant_message: None,
                        text: text.to_owned(),
                        usage: Usage::default(),
                        tool_calls: Vec::new(),
                    })),
                })
                .unwrap();
            restored.run_until_stalled().unwrap();
        }
        let child_results = restored
            .world()
            .get::<RunRecord>(parent.entity())
            .unwrap()
            .transcript
            .iter()
            .filter_map(|entry| match entry {
                TranscriptEntry::ChildResult { result, .. } => result.as_ref().ok().cloned(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(child_results, vec!["first output", "second output"]);
        let parent_request = find_request(parent_operation);
        restored
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: parent_request.operation,
                generation: parent_request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: "parent output".to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        restored.run_until_stalled().unwrap();
        assert!(matches!(
            restored.observe_run(parent).unwrap(),
            Some(RunState::Completed(RunOutput { text, .. })) if text == "parent output"
        ));
    }
}
