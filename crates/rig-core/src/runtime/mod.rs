//! Bevy ECS-native runtime primitives.
//!
//! [`RigSchedule`] is world-resident and is the single progression engine used
//! by both [`Runtime`] and embedded worlds. External work leaves the world as an
//! owned [`EffectRequest`] and returns through [`EffectCompletion`]; neither
//! type can contain an ECS borrow.

pub mod adapters;

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

use crate::{completion::Message as CompletionMessage, message::UserContent};

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
pub struct Policy {
    /// Explicit composition order. Stable ID breaks ties.
    pub order: u32,
    /// Immutable policy revision recorded by accepted decisions.
    pub revision: u64,
    /// Data interpreted by the policy system.
    pub rule: PolicyRule,
}

/// Built-in policy data. Extensions can add components and systems in [`RigSet::Policy`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PolicyRule {
    /// Permit progression without modification.
    Allow,
    /// Deny prompts containing this substring.
    DenyPromptContains(String),
}

/// Immutable record of the ordered policy instances applied to an operation.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
pub struct AcceptedPolicies(pub Vec<AcceptedPolicy>);

/// One accepted policy revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedPolicy {
    /// Persistent policy identity.
    pub id: StableId,
    /// Exact revision applied.
    pub revision: u64,
}

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
enum PolicyStatus {
    Pending,
    Accepted,
}

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

/// A run's authoritative input and committed transcript.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
}

/// Terminal run output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunOutput {
    /// Final model text.
    pub text: String,
    /// Provider-reported usage.
    pub usage: Usage,
}

/// Canonical token usage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
}

/// Immutable decision accepted before model dispatch.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct ModelDecision {
    /// Entity identity is retained for in-memory correlation.
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolDecision {
    /// In-memory capability identity.
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoreDecision {
    /// In-memory store identity.
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
#[derive(Component, Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationOutcome {
    /// Successful typed effect output.
    Success(EffectOutput),
    /// Failed external operation.
    Failure(CanonicalError),
}

/// Canonical error safe for policy, telemetry, and persistence.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
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
            EffectInput::Tool(_) | EffectInput::Discovery(_) | EffectInput::Store(_) => None,
        }
    }

    /// Returns tool input when this is a tool operation.
    pub fn tool_input(&self) -> Option<&ToolEffectInput> {
        match &self.input {
            EffectInput::Model(_) | EffectInput::Discovery(_) | EffectInput::Store(_) => None,
            EffectInput::Tool(input) => Some(input),
        }
    }

    /// Returns discovery input when this is a refresh operation.
    pub fn discovery_input(&self) -> Option<&DiscoveryEffectInput> {
        match &self.input {
            EffectInput::Discovery(input) => Some(input),
            EffectInput::Model(_) | EffectInput::Tool(_) | EffectInput::Store(_) => None,
        }
    }

    /// Returns store input when this is a memory/store operation.
    pub fn store_input(&self) -> Option<&StoreEffectInput> {
        match &self.input {
            EffectInput::Store(input) => Some(input),
            EffectInput::Model(_) | EffectInput::Tool(_) | EffectInput::Discovery(_) => None,
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
}

/// Fully owned model effect input.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
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
#[derive(Component, Clone, Debug, Eq, PartialEq)]
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
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct StoreEffectInput {
    /// Exact store decision accepted for this run.
    pub decision: StoreDecision,
    /// Owned operation payload.
    pub operation: StoreOperation,
}

/// Provider-independent conversation store operation.
#[derive(Clone, Debug, Eq, PartialEq)]
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
    /// Incremental text value; deltas are not entities.
    pub text: String,
}

#[derive(Clone, Debug)]
enum EffectIngress {
    Completion(EffectCompletion),
    Delta(EffectDelta),
}

/// Typed provider-independent effect result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectOutput {
    /// Model operation result.
    Model(ModelEffectOutput),
    /// Tool operation result.
    Tool(ToolEffectOutput),
    /// Reconciled discovery payload.
    Discovery(DiscoveryEffectOutput),
    /// Store operation result.
    Store(StoreEffectOutput),
}

/// Provider-independent model output.
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryEffectOutput {
    /// Complete capability set observed for this generation.
    pub tools: Vec<DiscoveredTool>,
}

/// Provider-independent discovered tool definition.
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StoreEffectOutput {
    /// Loaded committed conversation history.
    LoadedConversation(Vec<TranscriptEntry>),
    /// Required transcript persistence completed.
    Persisted,
    /// Ordered retrieval results.
    Retrieved(Vec<RetrievedDocument>),
}

/// Operation kind used to reject type-confused completions.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
enum OperationKind {
    Model,
    Tool,
    Discovery,
    Store,
}

#[derive(Component)]
struct DiscoveryApplied;

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
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelStreamState {
    /// Next accepted sequence; duplicates and gaps are rejected.
    pub next_sequence: u64,
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
    Prompt {
        agent: Entity,
        run_id: StableId,
        prompt: PromptPayload,
        history: Vec<serde_json::Value>,
        output_schema: Option<serde_json::Value>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
        subscriber: Option<SyncSender<StreamItem>>,
    },
    Cancel {
        run: Entity,
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
        let entity = world.spawn((id, tenant, agent, UsesModel(model))).id();
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
        &UsesModel,
        Option<&OutputRequirement>,
        Option<&RetrievalRequirement>,
    )>();
    let agent_rows = agent_query
        .iter(world)
        .map(
            |(entity, id, tenant, agent, model, output_requirement, retrieval_requirement)| {
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

    let mut policy_query = world.query::<(&StableId, &TenantId, &Policy, &PolicyFor)>();
    let mut policies = policy_query
        .iter(world)
        .map(|(id, tenant, policy, relation)| {
            let agent_id = agent_ids
                .get(&relation.get())
                .cloned()
                .ok_or(PersistenceError::StaleRelationship(relation.get()))?;
            Ok(PersistedPolicy {
                id: id.clone(),
                tenant: tenant.clone(),
                policy: policy.clone(),
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
            UsesModel(model),
        ));
        if let Some(output_requirement) = record.output_requirement {
            entity.insert(output_requirement);
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
        (reconcile_discovery_operations, reconcile_stable_ids)
            .chain()
            .in_set(RigSet::Reconcile),
    );
    schedule.add_systems(
        (prepare_store_operations, prepare_model_operations)
            .chain()
            .in_set(RigSet::Prepare),
    );
    schedule.add_systems(apply_builtin_policy.in_set(RigSet::Policy));
    schedule.add_systems(
        (
            dispatch_model_operations,
            dispatch_tool_operations,
            dispatch_discovery_operations,
            dispatch_store_operations,
        )
            .chain()
            .in_set(RigSet::Dispatch),
    );
    schedule.add_systems(
        (apply_effect_ingress, expire_effects)
            .chain()
            .in_set(RigSet::Apply),
    );
    schedule.add_systems(
        (
            commit_store_operations,
            commit_model_operations,
            commit_tool_batches,
        )
            .chain()
            .in_set(RigSet::Commit),
    );
    schedule.add_systems(
        (
            propagate_cancellation,
            cleanup_observed_runs,
            cleanup_retired_tools,
            update_effect_messages,
        )
            .chain()
            .in_set(RigSet::Cleanup),
    );
    schedule.add_systems(
        (publish_terminal_streams, update_runtime_metrics)
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
        let entity = self.world.spawn((id, tenant, agent, UsesModel(model))).id();
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
            .spawn((id, tenant, policy, PolicyFor(agent.entity)))
            .id())
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
}

/// Entity construction error.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum SpawnError {
    /// Stable ID already exists in this world.
    #[error("stable id `{0}` already exists")]
    DuplicateStableId(String),
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
    agents: Query<(&TenantId, &Agent)>,
    mut runs: Query<(&StableId, &mut RunState, &mut RunRecord)>,
    mut sources: Query<(&StableId, &TenantId, &mut DiscoverySource)>,
    mut discovery_operations: Query<&mut OperationState, With<DiscoveryEffectInput>>,
    mut progress: ResMut<Progress>,
) {
    let Ok(receiver) = ingress.0.lock() else {
        return;
    };
    loop {
        match receiver.try_recv() {
            Ok(RuntimeCommand::Prompt {
                agent,
                run_id,
                prompt,
                history,
                output_schema,
                max_model_calls,
                conversation,
                subscriber,
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
                let record = RunRecord {
                    transcript,
                    prompt: prompt.message,
                    prompt_text: prompt.text,
                    output_schema,
                    max_model_calls,
                    observed: false,
                    usage: Usage::default(),
                    next_turn: 0,
                    conversation,
                    memory_loaded: false,
                    memory_store: None,
                    retrieval_loaded: false,
                    retrieval_store: None,
                    retrieved_documents: Vec::new(),
                    pending_output: None,
                };
                let run = match agents.get(agent) {
                    Ok((tenant, _)) => commands
                        .spawn((
                            run_id,
                            tenant.clone(),
                            RunOf(agent),
                            RunState::Queued,
                            record,
                        ))
                        .id(),
                    Err(_) => commands
                        .spawn((
                            run_id,
                            RunState::Failed(CanonicalError::StaleEntity("agent".to_owned())),
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
                mark_progress(&mut progress);
            }
            Ok(RuntimeCommand::Cancel { run }) => {
                if let Ok((_, mut state, _)) = runs.get_mut(run)
                    && !matches!(*state, RunState::Completed(_) | RunState::Failed(_))
                {
                    *state = RunState::Cancelled;
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
    validate_relation!(TurnOf, RunState, "TurnOf");
    validate_relation!(OperationOf, RunState, "OperationOf");
    validate_relation!(BatchOf, RunState, "BatchOf");
    validate_relation!(SubscriptionOf, RunState, "SubscriptionOf");

    world.resource_mut::<StableIdIndex>().0 = rebuilt;
    world.resource_mut::<InvariantViolations>().0 = found;
}

fn prepare_store_operations(
    mut commands: Commands,
    mut runs: Query<(Entity, &TenantId, &RunOf, &mut RunRecord, &mut RunState)>,
    agents: Query<(Option<&AgentStoreGrants>, Option<&RetrievalRequirement>)>,
    grants: Query<(&StableId, &TenantId, &StoreGrant, &StoreGrantForStore)>,
    stores: Query<(&StableId, &TenantId, &StoreCapability)>,
    mut progress: ResMut<Progress>,
) {
    for (run_entity, run_tenant, run_of, mut record, mut run_state) in &mut runs {
        if !matches!(*run_state, RunState::Queued) {
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
    mut runs: Query<(Entity, &TenantId, &RunOf, &RunRecord, &mut RunState)>,
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
    for (run_entity, run_tenant, run_of, record, mut run_state) in &mut runs {
        if !matches!(*run_state, RunState::Queued)
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
                ModelStreamState::default(),
                decision,
                ModelEffectInput {
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
                },
                AcceptedPolicies::default(),
                PolicyStatus::Pending,
            ))
            .id();
        *run_state = RunState::WaitingModel { operation };
        mark_progress(&mut progress);
    }
}

fn apply_builtin_policy(
    operations: Query<
        (
            &OperationOf,
            &ModelEffectInput,
            &mut OperationState,
            &mut AcceptedPolicies,
            &mut PolicyStatus,
        ),
        Without<Policy>,
    >,
    runs: Query<&RunOf>,
    agents: Query<Option<&AgentPolicies>>,
    policies: Query<(&StableId, &Policy)>,
    mut progress: ResMut<Progress>,
) {
    let mut operations = operations;
    for (operation_of, input, mut state, mut accepted, mut status) in &mut operations {
        if !matches!(state.phase, OperationPhase::Prepared)
            || !matches!(*status, PolicyStatus::Pending)
        {
            continue;
        }
        let Ok(run_of) = runs.get(operation_of.get()) else {
            continue;
        };
        let Ok(Some(agent_policies)) = agents.get(run_of.get()) else {
            *status = PolicyStatus::Accepted;
            mark_progress(&mut progress);
            continue;
        };
        let mut ordered = agent_policies
            .iter()
            .filter_map(|entity| policies.get(entity).ok())
            .collect::<Vec<_>>();
        ordered.sort_by(|(left_id, left), (right_id, right)| {
            left.order
                .cmp(&right.order)
                .then_with(|| left_id.cmp(right_id))
        });
        for (id, policy) in ordered {
            accepted.0.push(AcceptedPolicy {
                id: id.clone(),
                revision: policy.revision,
            });
            if let PolicyRule::DenyPromptContains(needle) = &policy.rule
                && serde_json::from_value::<CompletionMessage>(input.prompt.clone())
                    .ok()
                    .and_then(|message| message.rag_text())
                    .is_some_and(|text| text.contains(needle))
            {
                state.phase = OperationPhase::Settled(OperationOutcome::Failure(
                    CanonicalError::PolicyDenied {
                        policy: id.as_str().to_owned(),
                    },
                ));
                *status = PolicyStatus::Accepted;
                mark_progress(&mut progress);
                break;
            }
        }
        if matches!(state.phase, OperationPhase::Prepared) {
            *status = PolicyStatus::Accepted;
            mark_progress(&mut progress);
        }
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
        &PolicyStatus,
        &OperationState,
    ), (With<ModelEffectInput>, Without<ToolEffectInput>)>();
    let mut prepared = query
        .iter(world)
        .filter_map(|(entity, run, input, policy_status, state)| {
            if !matches!(state.phase, OperationPhase::Prepared)
                || !matches!(*policy_status, PolicyStatus::Accepted)
                || !matches!(
                    world.get::<RunState>(run.get()),
                    Some(RunState::WaitingModel { .. })
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
            apply_dispatch_phase(world, entity, generation, phase);
        }
    }
    if made_progress {
        mark_progress(&mut world.resource_mut::<Progress>());
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
                entity,
                state.generation,
                input.clone(),
            ))
        })
        .collect::<Vec<_>>();
    prepared.sort_by_key(|(batch, index, entity, _, _)| (*batch, *index, entity.to_bits()));
    let mut made_progress = false;
    for (_, _, entity, generation, input) in prepared {
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
            apply_dispatch_phase(world, entity, generation, phase);
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
    run_subscriptions: Query<&RunSubscriptions>,
    mut subscriptions: Query<(&StreamSink, &mut SubscriptionState)>,
    mut progress: ResMut<Progress>,
) {
    for EffectIngressMessage(message) in messages.read() {
        match message {
            EffectIngress::Completion(completion) => {
                let Ok((kind, _, mut state, _, _)) = operations.get_mut(completion.operation)
                else {
                    continue;
                };
                if state.generation != completion.generation
                    || !matches!(state.phase, OperationPhase::InFlight)
                {
                    continue;
                }
                state.phase = OperationPhase::Settled(match &completion.result {
                    Ok(output)
                        if matches!(
                            (kind, output),
                            (OperationKind::Model, EffectOutput::Model(_))
                                | (OperationKind::Tool, EffectOutput::Tool(_))
                                | (OperationKind::Discovery, EffectOutput::Discovery(_))
                                | (OperationKind::Store, EffectOutput::Store(_))
                        ) =>
                    {
                        OperationOutcome::Success(output.clone())
                    }
                    Ok(_) => OperationOutcome::Failure(CanonicalError::EffectKindMismatch),
                    Err(error) => OperationOutcome::Failure(error.clone()),
                });
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
                if let Ok(run_subscriptions) = run_subscriptions.get(operation_of.get()) {
                    for subscription in run_subscriptions.iter() {
                        let Ok((sink, mut subscription_state)) =
                            subscriptions.get_mut(subscription)
                        else {
                            continue;
                        };
                        if !matches!(*subscription_state, SubscriptionState::Active) {
                            continue;
                        }
                        if sink
                            .0
                            .try_send(StreamItem::Delta {
                                sequence: delta.sequence,
                                text: delta.text.clone(),
                            })
                            .is_err()
                        {
                            *subscription_state = SubscriptionState::DroppedSlowConsumer;
                            commands.entity(subscription).remove::<StreamSink>();
                        }
                    }
                }
                mark_progress(&mut progress);
            }
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

fn commit_store_operations(
    operations: Query<
        (Entity, &OperationOf, &StoreEffectInput, &OperationState),
        With<StoreEffectInput>,
    >,
    mut runs: Query<(&mut RunState, &mut RunRecord)>,
    mut progress: ResMut<Progress>,
) {
    for (operation_entity, operation_of, input, operation) in &operations {
        let OperationPhase::Settled(outcome) = &operation.phase else {
            continue;
        };
        let Ok((mut run_state, mut record)) = runs.get_mut(operation_of.get()) else {
            continue;
        };
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
        mark_progress(&mut progress);
    }
}

fn commit_model_operations(
    mut commands: Commands,
    operations: Query<
        (Entity, &OperationOf, &OperationState, &ModelEffectInput),
        With<ModelEffectInput>,
    >,
    mut runs: Query<(&mut RunState, &mut RunRecord)>,
    mut progress: ResMut<Progress>,
) {
    for (operation_entity, operation_of, operation, input) in &operations {
        let OperationPhase::Settled(outcome) = &operation.phase else {
            continue;
        };
        let Ok((mut run_state, mut record)) = runs.get_mut(operation_of.get()) else {
            continue;
        };
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
                commands.spawn((
                    TurnOf(operation_of.get()),
                    CommittedTurn {
                        index: record.next_turn,
                        output: output.clone(),
                    },
                ));
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
                    let mut failure = None;
                    for (index, call) in output.tool_calls.iter().enumerate() {
                        let Some(decision) = input.tools.iter().find(|tool| tool.name == call.name)
                        else {
                            failure = Some(CanonicalError::UnknownTool(call.name.clone()));
                            break;
                        };
                        prepared.push(ToolEffectInput {
                            decision: decision.clone(),
                            call_id: call.id.clone(),
                            provider_result_id: call.provider_result_id.clone(),
                            provider_call_id: call.provider_call_id.clone(),
                            arguments: call.arguments.clone(),
                            index: u32::try_from(index).unwrap_or(u32::MAX),
                        });
                    }
                    if let Some(error) = failure {
                        *run_state = RunState::Failed(error);
                    } else {
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
                            commands.spawn((
                                OperationOf(operation_of.get()),
                                OperationOfBatch(batch),
                                OperationKind::Tool,
                                OperationState {
                                    generation: 0,
                                    phase: OperationPhase::Prepared,
                                },
                                tool_input,
                            ));
                        }
                        *run_state = RunState::WaitingTools { batch };
                    }
                }
            }
            OperationOutcome::Success(
                EffectOutput::Tool(_) | EffectOutput::Discovery(_) | EffectOutput::Store(_),
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
    operations: Query<(&ToolEffectInput, &OperationState)>,
    source_models: Query<(&ModelEffectInput, &AcceptedPolicies)>,
    mut runs: Query<(&mut RunState, &mut RunRecord)>,
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
            let Ok((input, state)) = operations.get(operation_entity) else {
                all_settled = false;
                break;
            };
            let OperationPhase::Settled(outcome) = &state.phase else {
                all_settled = false;
                break;
            };
            settled.push((input.index, input, outcome));
        }
        if !all_settled {
            continue;
        }
        settled.sort_by_key(|(index, _, _)| *index);
        let Ok((mut run_state, mut record)) = runs.get_mut(batch_of.get()) else {
            continue;
        };
        if !matches!(*run_state, RunState::WaitingTools { batch } if batch == batch_entity) {
            continue;
        }
        if let Some((_, _, OperationOutcome::Failure(error))) = settled
            .iter()
            .find(|(_, _, outcome)| matches!(outcome, OperationOutcome::Failure(_)))
        {
            *run_state = RunState::Failed(error.clone());
            batch.committed = true;
            mark_progress(&mut progress);
            continue;
        }
        let mut results = Vec::with_capacity(settled.len());
        for (_, input, outcome) in settled {
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
            results.push(output.clone());
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
        let Ok((source_input, accepted_policies)) = source_models.get(batch.source_model_operation)
        else {
            *run_state = RunState::Failed(CanonicalError::StaleEntity(
                "source model operation".to_owned(),
            ));
            batch.committed = true;
            mark_progress(&mut progress);
            continue;
        };
        let mut next_input = source_input.clone();
        next_input.history = record.transcript.clone();
        next_input.tool_results = results;
        let next_operation = commands
            .spawn((
                OperationOf(batch_of.get()),
                OperationKind::Model,
                ModelStreamState::default(),
                OperationState {
                    generation: 0,
                    phase: OperationPhase::Prepared,
                },
                next_input.decision.clone(),
                next_input,
                accepted_policies.clone(),
                PolicyStatus::Accepted,
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
                text: "he".to_owned(),
            })
            .unwrap();
        deltas
            .try_send(EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence: 1,
                text: "llo".to_owned(),
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
                    text: text.to_owned(),
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
                text: "late".to_owned(),
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
                    text: sequence.to_string(),
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
        runtime
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
                revision: 4,
            }]))
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
        assert!(
            runtime
                .world()
                .get::<RunToolBatches>(run.entity())
                .is_none()
        );
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
}
