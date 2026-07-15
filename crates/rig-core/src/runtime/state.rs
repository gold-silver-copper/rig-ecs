//! Durable run, transcript, and operation state.

use super::*;

/// A run's authoritative input and committed transcript.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[require(RunPriority, ReadyAt)]
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
    /// Structured-output retries already consumed since the last real tool batch.
    #[serde(default)]
    pub structured_output_retries: u32,
    /// Immutable structured-output retry budget accepted when the run was admitted.
    #[serde(default = "default_structured_output_retries")]
    pub max_structured_output_retries: u32,
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
    /// Prevents repeated semantic tool selection.
    #[serde(default)]
    pub tool_retrieval_loaded: bool,
    /// Immutable tool-search store choice accepted for this run.
    #[serde(default)]
    pub tool_retrieval_store: Option<StoreDecision>,
    /// Provider-facing tool names selected for this run.
    #[serde(default)]
    pub retrieved_tool_names: Vec<String>,
    /// Output retained while required persistence settles.
    pub pending_output: Option<RunOutput>,
    /// Most recently committed logical batch forwarded to the next model operation.
    #[serde(default)]
    pub pending_tool_results: Vec<ToolEffectOutput>,
}

pub(super) const fn default_structured_output_retries() -> u32 {
    1
}

/// Agent-local lifecycle counters updated by observe-only ECS observers.
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LifecycleTelemetry {
    /// Immutable model requests finalized for dispatch.
    pub requests_prepared: u64,
    /// Model turns committed to run history.
    pub model_turns_committed: u64,
    /// Atomic tool batches committed to run history.
    pub tool_batches_committed: u64,
    /// Runs completing successfully.
    pub runs_completed: u64,
    /// Runs ending in failure.
    pub runs_failed: u64,
    /// Runs ending through cancellation.
    pub runs_cancelled: u64,
}

/// Bundle inserted on an agent to opt into built-in lifecycle counters.
#[derive(Bundle, Clone, Copy, Debug, Default)]
pub struct LifecycleTelemetryBundle {
    /// Queryable telemetry state owned by the agent entity.
    pub telemetry: LifecycleTelemetry,
}

/// Owned operation context resolved by [`RigOperationContext`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedOperationContext {
    /// Operation being resolved.
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Reusable agent definition.
    pub agent: Entity,
    /// Stable run identity.
    pub run_id: StableId,
    /// Stable agent identity.
    pub agent_id: StableId,
    /// Shared tenant boundary.
    pub tenant: TenantId,
    /// Human-readable agent name, when configured.
    pub agent_name: Option<String>,
}

/// Read-only system parameter resolving operation → run → agent ownership.
///
/// The helper returns owned metadata so extensions cannot retain ECS borrows
/// across asynchronous work.
#[derive(SystemParam)]
pub struct RigOperationContext<'w, 's> {
    operations: Query<'w, 's, &'static OperationOf>,
    runs: Query<'w, 's, (&'static RunOf, &'static StableId, &'static TenantId)>,
    agents: Query<'w, 's, (&'static Agent, &'static StableId, &'static TenantId)>,
}

impl RigOperationContext<'_, '_> {
    /// Resolves complete ownership context when every relationship is live and tenant-safe.
    pub fn resolve(&self, operation: Entity) -> Option<ResolvedOperationContext> {
        let run = self.operations.get(operation).ok()?.get();
        let (run_of, run_id, run_tenant) = self.runs.get(run).ok()?;
        let agent = run_of.get();
        let (agent_config, agent_id, agent_tenant) = self.agents.get(agent).ok()?;
        if run_tenant != agent_tenant {
            return None;
        }
        Some(ResolvedOperationContext {
            operation,
            run,
            agent,
            run_id: run_id.clone(),
            agent_id: agent_id.clone(),
            tenant: run_tenant.clone(),
            agent_name: agent_config.name.clone(),
        })
    }
}

/// Owned run ownership metadata resolved without exposing the world.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRunContext {
    /// Run being resolved.
    pub run: Entity,
    /// Reusable agent definition.
    pub agent: Entity,
    /// Stable run identity.
    pub run_id: StableId,
    /// Stable agent identity.
    pub agent_id: StableId,
    /// Shared tenant boundary.
    pub tenant: TenantId,
    /// Human-readable agent name, when configured.
    pub agent_name: Option<String>,
}

/// Restricted read-only system parameter for run → agent ownership.
#[derive(SystemParam)]
pub struct RigRunContext<'w, 's> {
    runs: Query<'w, 's, (&'static RunOf, &'static StableId, &'static TenantId)>,
    agents: Query<'w, 's, (&'static Agent, &'static StableId, &'static TenantId)>,
}

impl RigRunContext<'_, '_> {
    /// Resolves owned metadata only when the run and agent share a tenant.
    pub fn resolve(&self, run: Entity) -> Option<ResolvedRunContext> {
        let (run_of, run_id, run_tenant) = self.runs.get(run).ok()?;
        let agent = run_of.get();
        let (agent_config, agent_id, agent_tenant) = self.agents.get(agent).ok()?;
        if run_tenant != agent_tenant {
            return None;
        }
        Some(ResolvedRunContext {
            run,
            agent,
            run_id: run_id.clone(),
            agent_id: agent_id.clone(),
            tenant: run_tenant.clone(),
            agent_name: agent_config.name.clone(),
        })
    }
}

/// Owned policy binding metadata safe for extension inspection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedPolicyContext {
    /// Policy being resolved.
    pub policy: Entity,
    /// Stable policy identity.
    pub policy_id: StableId,
    /// Policy tenant boundary.
    pub tenant: TenantId,
    /// Exact installed policy revision.
    pub revision: u64,
    /// Deterministic policy order.
    pub order: u32,
    /// Agent owner for agent-scoped policy definitions.
    pub agent: Option<Entity>,
    /// Run owner for run-scoped policy definitions.
    pub run: Option<Entity>,
}

/// Restricted read-only system parameter for policy ownership and revision.
type PolicyContextQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static StableId,
        &'static TenantId,
        &'static PolicyMeta,
        Option<&'static PolicyFor>,
        Option<&'static PolicyForRun>,
    ),
>;

/// Restricted read-only system parameter for policy ownership and revision.
#[derive(SystemParam)]
pub struct RigPolicyContext<'w, 's> {
    policies: PolicyContextQuery<'w, 's>,
    agents: Query<'w, 's, &'static TenantId, With<Agent>>,
    runs: Query<'w, 's, &'static TenantId, With<RunRecord>>,
}

impl RigPolicyContext<'_, '_> {
    /// Resolves a policy only when every declared owner shares its tenant.
    pub fn resolve(&self, policy: Entity) -> Option<ResolvedPolicyContext> {
        let (id, tenant, meta, agent, run) = self.policies.get(policy).ok()?;
        let agent = agent.map(Relationship::get);
        let run = run.map(Relationship::get);
        if agent.is_some_and(|owner| self.agents.get(owner).ok() != Some(tenant))
            || run.is_some_and(|owner| self.runs.get(owner).ok() != Some(tenant))
        {
            return None;
        }
        Some(ResolvedPolicyContext {
            policy,
            policy_id: id.clone(),
            tenant: tenant.clone(),
            revision: meta.revision,
            order: meta.order,
            agent,
            run,
        })
    }
}

/// Tenant-filtering wrapper for extension queries.
///
/// The underlying [`Query`] is private, so callers must provide the tenant on
/// every lookup and cannot accidentally iterate another tenant's components.
#[derive(SystemParam)]
pub struct TenantScopedQuery<
    'w,
    's,
    D: bevy_ecs::query::ReadOnlyQueryData + 'static,
    F: bevy_ecs::query::QueryFilter + 'static = (),
> {
    query: Query<'w, 's, (&'static TenantId, D), F>,
}

impl<D, F> TenantScopedQuery<'_, '_, D, F>
where
    D: bevy_ecs::query::ReadOnlyQueryData + 'static,
    F: bevy_ecs::query::QueryFilter + 'static,
{
    /// Returns one item only when its entity belongs to `tenant`.
    pub fn get<'a>(&'a self, tenant: &TenantId, entity: Entity) -> Option<D::Item<'a, 'a>> {
        let (entity_tenant, item) = self.query.get(entity).ok()?;
        (entity_tenant == tenant).then_some(item)
    }

    /// Iterates only items belonging to `tenant`.
    pub fn iter<'a>(&'a self, tenant: &'a TenantId) -> impl Iterator<Item = D::Item<'a, 'a>> + 'a {
        self.query
            .iter()
            .filter_map(move |(entity_tenant, item)| (entity_tenant == tenant).then_some(item))
    }
}

/// Canonical transcript entry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
// Transcript entries are retained and cloned far less often than their content
// is inspected; boxing rich tool output would add allocations to every result.
#[allow(clippy::large_enum_variant)]
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
        raw: ToolOutput,
        /// Presentation returned to the next model call.
        content: ToolOutput,
        /// Whether policy requires the presentation to replace rich raw content.
        #[serde(default)]
        presentation_overrides_raw: bool,
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
#[require(OperationGeneration)]
pub struct OperationState {
    /// Mutually exclusive phase/outcome.
    pub phase: OperationPhase,
}

/// Immutable identity of one dispatchable operation generation.
///
/// Cancel-and-suspend and restoration explicitly replace this component before
/// redispatch so every generation transition crosses Bevy insertion lifecycle.
#[derive(Component, Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[component(immutable)]
pub struct OperationGeneration(pub u64);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct EffectDeadline {
    pub(super) expires_at: u64,
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
    /// A typed extension result was written to an extension-owned component.
    ExtensionSettled,
    /// Cancellation prevents a late result from mutating the run.
    Cancelled,
    /// A newer generation replaced this operation.
    Superseded,
}

/// Canonical terminal operation outcome.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
// Outcomes move between ECS phases once. Boxing the successful result would
// add an allocation to every completed external effect.
#[allow(clippy::large_enum_variant)]
pub enum OperationOutcome {
    /// Successful typed effect output.
    Success(EffectOutput),
    /// Failed external operation.
    Failure(CanonicalError),
}

/// Complete diagnostic context for a policy-caused terminal outcome.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PolicyTermination {
    /// Stable policy identity accepted by the operation.
    pub policy_id: StableId,
    /// Exact accepted policy revision.
    pub revision: u64,
    /// Lifecycle point where steering terminated progression.
    pub point: PolicyPoint,
    /// Policy-provided or fail-closed reason.
    pub reason: String,
    /// Stable run identity.
    pub run_id: StableId,
    /// Stable operation identity.
    pub operation_id: StableId,
    /// Complete transcript at the termination boundary.
    pub history: Vec<TranscriptEntry>,
}

impl fmt::Display for PolicyTermination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "policy `{}` revision {} terminated {:?}: {}",
            self.policy_id.as_str(),
            self.revision,
            self.point,
            self.reason
        )
    }
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
    /// Typed asynchronous approval failed outside the world.
    #[error("policy approval failure: {message}")]
    PolicyApproval {
        /// Operator-facing diagnostic.
        message: String,
        /// Whether policy may retry the failure.
        retryable: bool,
    },
    /// Ordered policy stopped or failed progression.
    #[error("{termination}")]
    PolicyTerminated {
        /// Complete durable diagnostic context.
        termination: Box<PolicyTermination>,
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
    /// Run-local policy specifications conflicted at command admission.
    #[error("invalid run-local policy: {0}")]
    InvalidRunPolicy(String),
}
