//! Owned effect-boundary protocol and operation result types.

use super::*;

/// A typed asynchronous effect installed by an ECS extension.
///
/// Core owns operation identity, generation, phase, cancellation, and tenant
/// correlation. The extension owns dispatch and the concrete result component,
/// so adding a new effect does not add variants to [`EffectInput`] or
/// [`EffectOutput`].
pub trait EcsEffect: Send + Sync + 'static {
    /// Fully owned input stored on the operation entity.
    type Input: Clone + Send + Sync + 'static;
    /// Successful extension-owned output.
    type Output: Send + Sync + 'static;
    /// Extension-owned failure value.
    type Error: Send + Sync + 'static;

    /// Stable diagnostic and persistence binding identifier.
    const KIND: &'static str;
}

/// Typed input for an extension effect operation.
#[derive(Component)]
#[component(immutable)]
pub struct ExtensionEffectInput<E: EcsEffect>(pub E::Input);

/// Typed terminal result for an extension effect operation.
#[derive(Component)]
#[component(immutable)]
pub struct ExtensionEffectResult<E: EcsEffect>(pub Result<E::Output, E::Error>);

/// Stable effect kind retained for diagnostics and extension rebinding.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct ExtensionEffectKind(pub &'static str);

/// Failure while applying the common extension-effect lifecycle.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ExtensionEffectLifecycleError {
    /// The target world has no Rig runtime schedule installed.
    #[error("target world has no installed runtime")]
    RuntimeNotInstalled,
    /// The owner is not a complete runtime run.
    #[error("extension effect owner is not a runtime run")]
    NotRun,
    /// The stable operation identity is already live.
    #[error("stable id `{0}` already exists")]
    DuplicateStableId(String),
    /// The entity is not an operation of the requested extension kind.
    #[error("operation is not extension effect kind `{0}`")]
    WrongKind(&'static str),
    /// A dispatch or completion attempted an invalid phase transition.
    #[error("invalid extension effect phase transition")]
    InvalidPhase,
    /// A late completion targeted an obsolete generation.
    #[error("stale extension effect generation: expected {expected}, received {received}")]
    StaleGeneration {
        /// Live operation generation.
        expected: u64,
        /// Completion generation.
        received: u64,
    },
}

/// Spawns a prepared typed extension operation owned by `run`.
///
/// Tenant scope is copied from the run rather than accepted from the caller.
pub fn spawn_extension_effect<E: EcsEffect>(
    world: &mut World,
    run: Entity,
    id: StableId,
    input: E::Input,
) -> Result<Entity, ExtensionEffectLifecycleError> {
    if !world.contains_resource::<RuntimeIdentity>() {
        return Err(ExtensionEffectLifecycleError::RuntimeNotInstalled);
    }
    let tenant = world
        .get::<TenantId>(run)
        .cloned()
        .filter(|_| {
            world.get::<RunOf>(run).is_some()
                && world.get::<RunState>(run).is_some()
                && world.get::<RunRecord>(run).is_some()
        })
        .ok_or(ExtensionEffectLifecycleError::NotRun)?;
    if world
        .query::<&StableId>()
        .iter(world)
        .any(|existing| existing == &id)
    {
        return Err(ExtensionEffectLifecycleError::DuplicateStableId(
            id.as_str().to_owned(),
        ));
    }
    let entity = world
        .spawn((
            id.clone(),
            tenant,
            OperationOf(run),
            OperationGeneration(0),
            OperationState {
                phase: OperationPhase::Prepared,
            },
            ExtensionEffectKind(E::KIND),
            ExtensionEffectInput::<E>(input),
        ))
        .id();
    if let Some(mut index) = world.get_resource_mut::<StableIdIndex>() {
        index.0.insert(id, entity);
    }
    Ok(entity)
}

/// Claims a prepared extension operation for application-owned dispatch.
///
/// This typed boundary does not use core's [`EffectRequest`] queue. Extension
/// dispatch systems therefore own their queue budget and deadline policy in
/// addition to the concrete transport.
pub fn dispatch_extension_effect<E: EcsEffect>(
    world: &mut World,
    operation: Entity,
) -> Result<(u64, E::Input), ExtensionEffectLifecycleError> {
    let kind = world
        .get::<ExtensionEffectKind>(operation)
        .ok_or(ExtensionEffectLifecycleError::WrongKind(E::KIND))?;
    if kind.0 != E::KIND {
        return Err(ExtensionEffectLifecycleError::WrongKind(E::KIND));
    }
    let generation = world
        .get::<OperationGeneration>(operation)
        .ok_or(ExtensionEffectLifecycleError::WrongKind(E::KIND))?
        .0;
    let input = world
        .get::<ExtensionEffectInput<E>>(operation)
        .ok_or(ExtensionEffectLifecycleError::WrongKind(E::KIND))?
        .0
        .clone();
    let mut state = world
        .get_mut::<OperationState>(operation)
        .ok_or(ExtensionEffectLifecycleError::WrongKind(E::KIND))?;
    if !matches!(state.phase, OperationPhase::Prepared) {
        return Err(ExtensionEffectLifecycleError::InvalidPhase);
    }
    state.phase = OperationPhase::InFlight;
    Ok((generation, input))
}

/// Correlates and commits one typed extension result exactly once.
pub fn settle_extension_effect<E: EcsEffect>(
    world: &mut World,
    operation: Entity,
    generation: u64,
    result: Result<E::Output, E::Error>,
) -> Result<(), ExtensionEffectLifecycleError> {
    let kind = world
        .get::<ExtensionEffectKind>(operation)
        .ok_or(ExtensionEffectLifecycleError::WrongKind(E::KIND))?;
    if kind.0 != E::KIND {
        return Err(ExtensionEffectLifecycleError::WrongKind(E::KIND));
    }
    let expected = world
        .get::<OperationGeneration>(operation)
        .ok_or(ExtensionEffectLifecycleError::WrongKind(E::KIND))?
        .0;
    if generation != expected {
        return Err(ExtensionEffectLifecycleError::StaleGeneration {
            expected,
            received: generation,
        });
    }
    let mut state = world
        .get_mut::<OperationState>(operation)
        .ok_or(ExtensionEffectLifecycleError::WrongKind(E::KIND))?;
    if !matches!(state.phase, OperationPhase::InFlight) {
        return Err(ExtensionEffectLifecycleError::InvalidPhase);
    }
    state.phase = OperationPhase::ExtensionSettled;
    world
        .entity_mut(operation)
        .insert(ExtensionEffectResult::<E>(result));
    Ok(())
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

/// Raw serialized provider response correlated to one model operation.
#[derive(Clone)]
pub struct ProviderDiagnosticsIngress {
    /// Model operation identity.
    pub operation: Entity,
    /// Operation generation copied at dispatch.
    pub generation: u64,
    /// Provider-specific response serialized by the typed adapter.
    pub diagnostics: serde_json::Value,
    pub(super) typed: Option<Arc<dyn ProviderDiagnosticsPayload>>,
}

pub(super) trait ProviderDiagnosticsPayload: Send + Sync {
    fn insert(&self, commands: &mut Commands<'_, '_>, operation: Entity);
}

struct ConcreteProviderDiagnostics<T>(Mutex<Option<T>>);

impl<T> ProviderDiagnosticsPayload for ConcreteProviderDiagnostics<T>
where
    T: Send + Sync + 'static,
{
    fn insert(&self, commands: &mut Commands<'_, '_>, operation: Entity) {
        let Ok(mut value) = self.0.lock() else {
            return;
        };
        if let Some(value) = value.take() {
            commands
                .entity(operation)
                .insert(TypedProviderResponseDiagnostics(value));
        }
    }
}

impl fmt::Debug for ProviderDiagnosticsIngress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderDiagnosticsIngress")
            .field("operation", &self.operation)
            .field("generation", &self.generation)
            .field("diagnostics", &self.diagnostics)
            .field("has_typed_payload", &self.typed.is_some())
            .finish()
    }
}

impl PartialEq for ProviderDiagnosticsIngress {
    fn eq(&self, other: &Self) -> bool {
        self.operation == other.operation
            && self.generation == other.generation
            && self.diagnostics == other.diagnostics
    }
}

impl Eq for ProviderDiagnosticsIngress {}

impl ProviderDiagnosticsIngress {
    /// Creates serialized diagnostics for generic telemetry and persistence.
    pub fn serialized(operation: Entity, generation: u64, diagnostics: serde_json::Value) -> Self {
        Self {
            operation,
            generation,
            diagnostics,
            typed: None,
        }
    }

    /// Adds the concrete provider response retained by a typed adapter.
    pub fn with_typed<T>(mut self, response: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.typed = Some(Arc::new(ConcreteProviderDiagnostics(Mutex::new(Some(
            response,
        )))));
        self
    }
}

/// Concrete provider response retained on a model operation.
///
/// Extensions query `TypedProviderResponseDiagnostics<M::Response>` when they
/// need provider-native fields. [`ProviderResponseDiagnostics`] remains the
/// serialized, provider-independent persistence and telemetry representation.
#[derive(Component, Debug)]
#[component(immutable)]
pub struct TypedProviderResponseDiagnostics<T>(pub T)
where
    T: Send + Sync + 'static;

#[derive(Clone, Debug)]
// Ingress values are transferred once from a bounded channel into the world.
#[allow(clippy::large_enum_variant)]
pub(super) enum EffectIngress {
    Completion(EffectCompletion),
    Delta(EffectDelta),
    ProviderDiagnostics(ProviderDiagnosticsIngress),
}

/// Typed provider-independent effect result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
// Tool output is intentionally rich; boxing it here would impose an allocation
// on every effect consumer and complicate the public matching surface.
#[allow(clippy::large_enum_variant)]
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

/// Immutable provider-specific response data retained for policy and telemetry queries.
///
/// Core progression never interprets this value. Typed provider adapters serialize
/// their raw response into this component before the correlated completion settles.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[component(immutable)]
pub struct ProviderResponseDiagnostics(pub serde_json::Value);

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
    pub raw: ToolOutput,
    /// Independently rewritable model-visible presentation.
    pub presentation: ToolOutput,
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
#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum OperationKind {
    Model,
    Tool,
    Discovery,
    Store,
    PolicyApproval,
}

#[derive(Component)]
pub(super) struct DiscoveryApplied;

#[derive(Component)]
pub(super) struct PolicyApprovalApplied;

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
