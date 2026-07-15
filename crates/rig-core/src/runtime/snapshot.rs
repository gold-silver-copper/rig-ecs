//! Stable-ID active-run checkpoints.
//!
//! Snapshot-local references are opaque strings allocated during capture. They
//! are not Bevy entity identifiers. Runtime-only observers, subscriptions,
//! channels, deadlines, and provider clients are deliberately reconstructed by
//! the installed runtime rather than serialized.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use bevy_ecs::{prelude::*, relationship::Relationship};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::*;

const ACTIVE_RUN_SNAPSHOT_VERSION: u32 = 6;
const ACTIVE_RUN_ENVELOPE_VERSION: u32 = 1;

/// Application-supplied encryption or reversible redaction boundary.
///
/// Core never stores keys. The protected bytes are checksummed after this hook,
/// and the inverse hook runs only after integrity validation.
pub trait ActiveRunSnapshotProtection {
    /// Protects canonical checkpoint JSON before it leaves the process.
    fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, String>;

    /// Recovers canonical checkpoint JSON after integrity validation.
    fn unprotect(&self, protected: &[u8]) -> Result<Vec<u8>, String>;
}

#[derive(Deserialize, Serialize)]
struct ActiveRunSnapshotEnvelope {
    version: u32,
    protected: bool,
    sha256: String,
    payload: Vec<u8>,
}

/// Append-only, hash-chained active-run checkpoint journal.
///
/// Entries are complete encoded envelopes. This keeps recovery deterministic
/// and permits later compaction without defining fragile field-level deltas.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ActiveRunCheckpointJournal {
    /// Journal schema version.
    pub version: u32,
    /// Stable root run shared by every appended checkpoint.
    pub root_run: StableId,
    /// Ordered hash-chained checkpoint entries.
    pub entries: Vec<ActiveRunCheckpointEntry>,
}

/// One encoded checkpoint in an append-only journal.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ActiveRunCheckpointEntry {
    /// Monotonic zero-based sequence.
    pub sequence: u64,
    /// SHA-256 of the preceding encoded envelope, or `None` for the first entry.
    pub previous_sha256: Option<String>,
    /// Integrity-checked envelope produced by [`encode_active_run_snapshot`].
    pub encoded: Vec<u8>,
}

impl ActiveRunCheckpointJournal {
    /// Creates an empty journal for `root_run`.
    pub fn new(root_run: StableId) -> Self {
        Self {
            version: 1,
            root_run,
            entries: Vec::new(),
        }
    }

    /// Appends an encoded checkpoint after verifying its root and envelope.
    pub fn append(
        &mut self,
        encoded: Vec<u8>,
        protection: Option<&dyn ActiveRunSnapshotProtection>,
        limits: ActiveRunSnapshotLimits,
    ) -> Result<(), ActiveRunSnapshotError> {
        let snapshot = decode_active_run_snapshot(&encoded, protection, limits)?;
        if snapshot.root_run != self.root_run {
            return Err(ActiveRunSnapshotError::InvalidSnapshot(
                "checkpoint journal root changed".to_owned(),
            ));
        }
        let sequence = u64::try_from(self.entries.len()).map_err(|_| {
            ActiveRunSnapshotError::InvalidSnapshot(
                "checkpoint journal sequence exhausted".to_owned(),
            )
        })?;
        let previous_sha256 = self.entries.last().map(|entry| sha256_hex(&entry.encoded));
        self.entries.push(ActiveRunCheckpointEntry {
            sequence,
            previous_sha256,
            encoded,
        });
        Ok(())
    }

    /// Verifies sequence and hash-chain integrity without decoding payloads.
    pub fn verify(&self) -> Result<(), ActiveRunSnapshotError> {
        if self.version != 1 {
            return Err(ActiveRunSnapshotError::Integrity);
        }
        let mut previous = None;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.sequence
                != u64::try_from(index).map_err(|_| ActiveRunSnapshotError::Integrity)?
                || entry.previous_sha256 != previous
            {
                return Err(ActiveRunSnapshotError::Integrity);
            }
            previous = Some(sha256_hex(&entry.encoded));
        }
        Ok(())
    }

    /// Returns the most recent encoded checkpoint after verifying the chain.
    pub fn latest(&self) -> Result<Option<&[u8]>, ActiveRunSnapshotError> {
        self.verify()?;
        Ok(self.entries.last().map(|entry| entry.encoded.as_slice()))
    }
}

/// Explicit codec for extension-owned active-run state.
///
/// Validation runs before core entities are created. `restore` is infallible by
/// design: codecs must reject malformed or unavailable state in `validate` so
/// restoration cannot fail after mutating the world.
pub trait ActiveRunSnapshotCodec: Send + Sync + 'static {
    /// Stable application or crate-defined binding identity.
    fn binding_id(&self) -> &'static str;

    /// Exact codec revision.
    fn revision(&self) -> u64;

    /// Whether restoration must fail when this codec is not rebound.
    fn required(&self) -> bool {
        true
    }

    /// Captures state related to the ordered run entities, if any.
    fn capture(&self, world: &World, runs: &[Entity]) -> Result<Option<serde_json::Value>, String>;

    /// Validates payload and external rebinding before world mutation.
    fn validate(&self, payload: &serde_json::Value) -> Result<(), String>;

    /// Applies a payload previously accepted by [`Self::validate`].
    fn restore(&self, world: &mut World, runs: &RestoredRuns, payload: &serde_json::Value);
}

#[derive(Resource, Default)]
pub(super) struct SnapshotExtensionCodecs(
    BTreeMap<String, std::sync::Arc<dyn ActiveRunSnapshotCodec>>,
);

/// Registers one stable extension codec before capture or restoration.
pub fn register_active_run_snapshot_codec(
    world: &mut World,
    codec: std::sync::Arc<dyn ActiveRunSnapshotCodec>,
) -> Result<(), ActiveRunSnapshotError> {
    if !world.contains_resource::<RuntimeIdentity>() {
        return Err(ActiveRunSnapshotError::InvalidSnapshot(
            "target world has no installed runtime".to_owned(),
        ));
    }
    let id = codec.binding_id();
    if id.trim().is_empty() {
        return Err(ActiveRunSnapshotError::ExtensionCodec {
            id: id.to_owned(),
            message: "binding ID must not be empty".to_owned(),
        });
    }
    let mut codecs = world.resource_mut::<SnapshotExtensionCodecs>();
    if codecs.0.contains_key(id) {
        return Err(ActiveRunSnapshotError::ExtensionCodec {
            id: id.to_owned(),
            message: "binding is already registered".to_owned(),
        });
    }
    codecs.0.insert(id.to_owned(), codec);
    Ok(())
}

/// Resource limits applied before an active-run checkpoint may be restored.
///
/// Limits are checked before domain rebinding or entity creation, so an
/// oversized untrusted checkpoint cannot partially mutate the target world.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveRunSnapshotLimits {
    /// Maximum canonical JSON size of the checkpoint.
    pub max_encoded_bytes: usize,
    /// Maximum number of runs, including descendants.
    pub max_runs: usize,
    /// Maximum parent-to-child nesting depth, where the root has depth zero.
    pub max_depth: usize,
    /// Maximum direct children owned by any one run.
    pub max_children_per_run: usize,
    /// Maximum external operations retained by the checkpoint.
    pub max_operations: usize,
    /// Maximum durable policy evaluations retained by the checkpoint.
    pub max_policy_evaluations: usize,
    /// Maximum transcript entries across all captured runs.
    pub max_transcript_entries: usize,
    /// Maximum extension-owned sections.
    pub max_extension_sections: usize,
    /// Maximum encoded JSON bytes across extension payloads.
    pub max_extension_bytes: usize,
}

impl Default for ActiveRunSnapshotLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 16 * 1024 * 1024,
            max_runs: 1_024,
            max_depth: 64,
            max_children_per_run: 256,
            max_operations: 16_384,
            max_policy_evaluations: 65_536,
            max_transcript_entries: 1_000_000,
            max_extension_sections: 256,
            max_extension_bytes: 4 * 1024 * 1024,
        }
    }
}

/// Structured, serializable inventory of an active-run checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ActiveRunSnapshotSummary {
    /// Checkpoint schema version.
    pub version: u32,
    /// Stable root-run identity.
    pub root_run: StableId,
    /// Canonical JSON size in bytes.
    pub encoded_bytes: usize,
    /// Number of captured runs.
    pub runs: usize,
    /// Deepest descendant level, where the root is zero.
    pub max_depth: usize,
    /// Largest direct child fan-out.
    pub max_children_per_run: usize,
    /// Number of run-scoped policy definitions.
    pub run_policies: usize,
    /// Number of external operations.
    pub operations: usize,
    /// Number of atomic tool batches.
    pub tool_batches: usize,
    /// Number of durable policy evaluations.
    pub policy_evaluations: usize,
    /// Number of committed turn audit records.
    pub turns: usize,
    /// Total transcript entries across captured runs.
    pub transcript_entries: usize,
    /// Number of extension-owned sections.
    pub extension_sections: usize,
    /// Encoded JSON bytes across extension payloads.
    pub extension_bytes: usize,
    /// Run-state counts keyed by stable diagnostic names.
    pub run_states: BTreeMap<String, usize>,
    /// Operation-kind counts keyed by stable diagnostic names.
    pub operation_kinds: BTreeMap<String, usize>,
}

/// Serializable checkpoint for one run and all of its descendant runs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ActiveRunSnapshot {
    /// Format version. Unknown versions are rejected before spawning entities.
    pub version: u32,
    /// Stable identity of the requested root run.
    pub root_run: StableId,
    /// Parent-before-child run records.
    pub runs: Vec<PersistedActiveRun>,
    /// Policy revisions owned by the captured runs.
    pub run_policies: Vec<PersistedRunPolicy>,
    /// Run-owned external operations.
    pub operations: Vec<PersistedRunOperation>,
    /// Atomic logical tool batches.
    pub tool_batches: Vec<PersistedToolBatch>,
    /// Durable ordered policy evaluations.
    pub policy_evaluations: Vec<PersistedPolicyEvaluation>,
    /// Committed model-turn audit entities.
    pub turns: Vec<PersistedCommittedTurn>,
    /// Explicit extension-owned state keyed by stable binding ID.
    #[serde(default)]
    pub extensions: Vec<SnapshotExtensionSection>,
}

/// Extension-owned active-run state with an explicit rebinding contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SnapshotExtensionSection {
    /// Stable application or crate-defined codec identity.
    pub binding_id: String,
    /// Exact codec revision required to interpret `payload`.
    pub revision: u64,
    /// Whether restoration must fail when the codec is unavailable.
    pub required: bool,
    /// Codec-owned canonical data. It must not contain runtime entities or secrets.
    pub payload: serde_json::Value,
}

/// Restored stable run identities and their new runtime-local entities.
#[derive(Clone, Debug, Default)]
pub struct RestoredRuns(pub HashMap<StableId, Entity>);

/// Snapshot or restoration failure that leaves the world unchanged.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ActiveRunSnapshotError {
    /// The requested entity is not a complete run.
    #[error("entity is not a complete runtime run")]
    NotRun,
    /// The checkpoint format is newer or otherwise unsupported.
    #[error("unsupported active-run snapshot version {0}")]
    UnsupportedVersion(u32),
    /// The checkpoint exceeds a caller-selected resource limit.
    #[error("active-run snapshot limit exceeded for {field}: {actual} > {limit}")]
    LimitExceeded {
        /// Stable limit name.
        field: &'static str,
        /// Observed value.
        actual: usize,
        /// Accepted maximum.
        limit: usize,
    },
    /// The checkpoint could not be encoded for size or integrity checks.
    #[error("active-run snapshot serialization failed: {0}")]
    Serialization(String),
    /// Encoded checkpoint integrity did not match its envelope.
    #[error("active-run snapshot integrity validation failed")]
    Integrity,
    /// Protected bytes were supplied without the corresponding application hook.
    #[error("active-run snapshot requires an application protection hook")]
    MissingProtection,
    /// A protection hook was supplied for an envelope that claims to be plaintext.
    #[error("active-run snapshot protection mode does not match the decoder configuration")]
    ProtectionModeMismatch,
    /// The application protection hook failed.
    #[error("active-run snapshot protection failed: {0}")]
    Protection(String),
    /// A required extension codec was not rebound before restoration.
    #[error("missing active-run snapshot extension binding `{0}`")]
    MissingExtensionBinding(String),
    /// A rebound extension codec has a different revision.
    #[error("extension binding `{id}` revision mismatch: expected {expected}, found {found}")]
    ExtensionRevisionMismatch {
        /// Stable binding identity.
        id: String,
        /// Revision recorded by the checkpoint.
        expected: u64,
        /// Revision installed in the target world.
        found: u64,
    },
    /// Extension capture or validation rejected its payload.
    #[error("active-run snapshot extension `{id}` failed: {message}")]
    ExtensionCodec {
        /// Stable binding identity.
        id: String,
        /// Codec-provided diagnostic.
        message: String,
    },
    /// A stable identity is duplicated or already live in the target world.
    #[error("conflicting stable id `{0}`")]
    ConflictingStableId(String),
    /// A snapshot-local relationship target is absent.
    #[error("missing snapshot reference `{0}`")]
    MissingSnapshotReference(String),
    /// A required domain capability or policy is absent.
    #[error("missing domain reference `{0}`")]
    MissingDomainReference(String),
    /// A referenced capability exists in another tenant.
    #[error("tenant mismatch for stable reference `{0}`")]
    TenantMismatch(String),
    /// The exact accepted capability or policy revision is unavailable.
    #[error("revision mismatch for stable reference `{id}`: expected {expected}")]
    RevisionMismatch {
        /// Stable domain identity.
        id: String,
        /// Revision accepted by the run.
        expected: u64,
    },
    /// A live effect cannot be duplicated safely by an ordinary checkpoint.
    #[error(
        "operation `{0}` is still in flight; use cancel-and-suspend and wait for its prepared checkpoint"
    )]
    UnsafeInFlightEffect(String),
    /// A typed extension operation cannot be represented by the core snapshot schema.
    #[error("extension operation `{id}` ({kind}) in phase {phase} is not checkpoint-safe")]
    UnsafeExtensionEffect {
        /// Stable operation identity, or an entity diagnostic for malformed extensions.
        id: String,
        /// Stable extension effect kind.
        kind: String,
        /// Live phase whose extension-owned payload would otherwise be lost.
        phase: &'static str,
    },
    /// A live ECS relationship is stale during capture.
    #[error("stale runtime relationship on entity {0:?}")]
    StaleRelationship(Entity),
    /// The snapshot encodes contradictory state.
    #[error("invalid active-run snapshot: {0}")]
    InvalidSnapshot(String),
}

/// Stable run record without raw entity identifiers.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedActiveRun {
    /// Persistent run identity.
    pub id: StableId,
    /// Tenant scope.
    pub tenant: TenantId,
    /// Stable agent identity.
    pub agent_id: StableId,
    /// Authoritative execution phase.
    pub state: PersistedRunState,
    /// Orthogonal suspension state.
    pub control: RunControl,
    /// Scheduling priority retained across restoration.
    #[serde(default)]
    pub priority: RunPriority,
    /// Original eligibility tick used for deterministic age ordering.
    #[serde(default)]
    pub ready_at: ReadyAt,
    /// Prompt, transcript, usage, budgets, and memory state.
    pub record: RunRecord,
    /// Stable parent identity, if this is a delegated run.
    pub parent_run: Option<StableId>,
    /// Deterministic child creation ordinal.
    pub child_ordinal: Option<u64>,
    /// Whether the result has already committed to its parent.
    pub child_result_committed: bool,
}

/// Stable definition of one policy scoped to a captured run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedRunPolicy {
    /// Persistent policy identity.
    pub id: StableId,
    /// Tenant scope shared with the target run.
    pub tenant: TenantId,
    /// Stable target run identity.
    pub run_id: StableId,
    /// Immutable ordered policy definition.
    pub policy: Policy,
    /// Admission status for future evaluations after restoration.
    pub status: PolicyStatus,
}

/// Run phase with snapshot-local operation and batch references.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PersistedRunState {
    /// Accepted but not prepared.
    Queued,
    /// Waiting for a model operation.
    WaitingModel { operation: String },
    /// Waiting for an atomic tool batch.
    WaitingTools { batch: String },
    /// Waiting for a store operation.
    WaitingStore { operation: String },
    /// Terminal success.
    Completed(RunOutput),
    /// Terminal failure.
    Failed(CanonicalError),
    /// Terminal cancellation.
    Cancelled,
}

/// Serializable operation kind.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PersistedOperationKind {
    /// Model completion.
    Model,
    /// Tool execution.
    Tool,
    /// Store or memory operation.
    Store,
    /// Asynchronous policy approval.
    PolicyApproval,
}

/// Complete run-owned operation state.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedRunOperation {
    /// Opaque snapshot-local reference.
    pub id: String,
    /// Stable owner run identity.
    pub run_id: StableId,
    /// Optional snapshot-local tool batch.
    pub batch: Option<String>,
    /// Narrow completion type.
    pub kind: PersistedOperationKind,
    /// Immutable generation identity.
    pub generation: u64,
    /// Mutually exclusive operation phase.
    pub state: OperationState,
    /// Accepted model decision stored independently for preparation queries.
    pub model_decision: Option<ModelDecision>,
    /// Model request before policy finalization.
    pub pending_model: Option<ModelEffectInput>,
    /// Final immutable model effect input.
    pub model_input: Option<ModelEffectInput>,
    /// Ordered streaming accumulation.
    pub model_stream: Option<ModelStreamState>,
    /// Tool call before policy finalization.
    pub pending_tool: Option<ToolEffectInput>,
    /// Final immutable tool effect input.
    pub tool_input: Option<ToolEffectInput>,
    /// Invalid tool call awaiting resolution.
    pub pending_invalid_tool: Option<PersistedPendingInvalidToolCall>,
    /// Store effect input.
    pub store_input: Option<StoreEffectInput>,
    /// Approval effect input.
    pub approval_input: Option<PolicyApprovalEffectInput>,
    /// Request policy revisions accepted for this operation.
    pub request_policies: Option<Vec<PersistedAcceptedPolicy>>,
    /// Tool-call policy revisions accepted for this operation.
    pub tool_call_policies: Option<Vec<PersistedAcceptedPolicy>>,
    /// Invalid-call policy revisions accepted for this operation.
    pub invalid_tool_policies: Option<Vec<PersistedAcceptedPolicy>>,
    /// Tool-result policy revisions accepted for this operation.
    pub tool_result_policies: Option<Vec<PersistedAcceptedPolicy>>,
    /// Completion-response policy revisions accepted for this operation.
    pub response_policies: Option<Vec<PersistedAcceptedPolicy>>,
    /// Effective tool presentation after result policy.
    pub effective_tool_output: Option<ToolEffectOutput>,
    /// Effective model output after response policy.
    pub effective_model_output: Option<ModelEffectOutput>,
    /// Provider-specific response data retained on the model operation.
    pub provider_diagnostics: Option<serde_json::Value>,
    /// Initialization and completion markers needed to resume exactly once.
    pub markers: PersistedOperationMarkers,
    /// Evaluation owning this approval operation, when applicable.
    pub approval_for_evaluation: Option<String>,
}

/// Private operation markers represented explicitly in the public format.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedOperationMarkers {
    /// Request evaluation was initialized.
    pub request_policy_initialized: bool,
    /// Tool-call evaluation was initialized.
    pub tool_call_policy_initialized: bool,
    /// Invalid-call evaluation was initialized.
    pub invalid_tool_policy_initialized: bool,
    /// Tool-result evaluation was initialized.
    pub tool_result_policy_initialized: bool,
    /// Tool-result evaluation completed.
    pub tool_result_policy_done: bool,
    /// Completion-response evaluation was initialized.
    pub response_policy_initialized: bool,
    /// Completion-response evaluation completed.
    pub response_policy_done: bool,
    /// Approval output was already reduced.
    pub approval_applied: bool,
}

/// Invalid-call context without runtime-local tool entities.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedPendingInvalidToolCall {
    /// Model-emitted call.
    pub call: ModelToolCall,
    /// Exact accepted executable tool revisions.
    pub available_tools: Vec<ToolDecision>,
    /// Logical batch position.
    pub index: u32,
    /// Snapshot-local producing model operation.
    pub source_model_operation: String,
    /// Model-call position that emitted the invalid call.
    pub turn: u32,
    /// Accepted tool-selection behavior.
    pub tool_choice: Option<ModelToolChoice>,
    /// Complete diagnostic history at detection.
    pub diagnostic_history: Vec<TranscriptEntry>,
    /// Whether the call originated from streamed provider data.
    pub streaming_origin: bool,
    /// Retry count before this resolution.
    pub retry_count: u32,
    /// Accepted invalid-call retry limit.
    pub max_retries: u32,
}

/// Accepted policy identity without a raw entity ID.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedAcceptedPolicy {
    /// Stable policy identity.
    pub id: StableId,
    /// Exact accepted revision.
    pub revision: u64,
    /// Explicit composition order accepted by the operation.
    pub order: u32,
    /// Exact lifecycle capability accepted by the operation.
    pub point: PolicyPoint,
}

/// Atomic tool batch and its explicit ordering relationships.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedToolBatch {
    /// Opaque snapshot-local reference.
    pub id: String,
    /// Stable owner run identity.
    pub run_id: StableId,
    /// Model operation that emitted the batch.
    pub source_model_operation: String,
    /// Number of expected calls.
    pub expected: u32,
    /// Whether logical commit already occurred.
    pub committed: bool,
}

/// One retained committed-turn entity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedCommittedTurn {
    /// Stable owner run identity.
    pub run_id: StableId,
    /// Explicit turn order.
    pub index: u32,
    /// Canonical committed output.
    pub output: ModelEffectOutput,
}

/// Durable policy state, including cursor and effective value.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PersistedPolicyEvaluation {
    /// Completion-request evaluation.
    Request(PersistedRequestPolicyEvaluation),
    /// Tool-call evaluation.
    ToolCall(PersistedToolCallPolicyEvaluation),
    /// Invalid-call resolution.
    InvalidToolCall(PersistedInvalidToolCallPolicyEvaluation),
    /// Tool-result evaluation.
    ToolResult(PersistedToolResultPolicyEvaluation),
    /// Completion-response evaluation.
    CompletionResponse(PersistedCompletionResponsePolicyEvaluation),
    /// Streaming text-delta evaluation.
    TextDelta(PersistedTextDeltaPolicyEvaluation),
    /// Streaming tool-call-delta evaluation.
    ToolCallDelta(PersistedToolCallDeltaPolicyEvaluation),
}

/// Request evaluation checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedRequestPolicyEvaluation {
    pub id: String,
    pub operation: String,
    pub run_id: StableId,
    pub policies: Vec<PersistedAcceptedPolicy>,
    pub cursor: usize,
    pub effective: ModelEffectInput,
    /// Deterministically reduced operation-local request patch.
    #[serde(default)]
    pub accumulated: RequestPatch,
    pub phase: PersistedRequestPolicyPhase,
}

/// Tool-call evaluation checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedToolCallPolicyEvaluation {
    pub id: String,
    pub operation: String,
    pub run_id: StableId,
    pub policies: Vec<PersistedAcceptedPolicy>,
    pub cursor: usize,
    pub effective: ToolEffectInput,
    pub phase: PersistedToolCallPolicyPhase,
}

/// Invalid-call evaluation checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedInvalidToolCallPolicyEvaluation {
    pub id: String,
    pub operation: String,
    pub run_id: StableId,
    pub policies: Vec<PersistedAcceptedPolicy>,
    pub cursor: usize,
    pub invalid: PersistedPendingInvalidToolCall,
    pub phase: PersistedInvalidToolCallPolicyPhase,
}

/// Tool-result evaluation checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedToolResultPolicyEvaluation {
    pub id: String,
    pub operation: String,
    pub run_id: StableId,
    pub policies: Vec<PersistedAcceptedPolicy>,
    pub cursor: usize,
    pub input: ToolEffectInput,
    pub effective: ToolEffectOutput,
    pub phase: PersistedToolResultPolicyPhase,
}

/// Completion-response evaluation checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedCompletionResponsePolicyEvaluation {
    pub id: String,
    pub operation: String,
    pub run_id: StableId,
    pub policies: Vec<PersistedAcceptedPolicy>,
    pub cursor: usize,
    pub request: ModelEffectInput,
    pub effective: ModelEffectOutput,
    pub phase: PersistedCompletionResponsePolicyPhase,
}

/// Text-delta evaluation checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedTextDeltaPolicyEvaluation {
    pub id: String,
    pub operation: String,
    pub run_id: StableId,
    pub policies: Vec<PersistedAcceptedPolicy>,
    pub cursor: usize,
    pub turn: u32,
    pub sequence: u64,
    pub delta: String,
    pub aggregated: String,
    pub phase: PersistedTextDeltaPolicyPhase,
}

/// Tool-call-delta evaluation checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PersistedToolCallDeltaPolicyEvaluation {
    pub id: String,
    pub operation: String,
    pub run_id: StableId,
    pub policies: Vec<PersistedAcceptedPolicy>,
    pub cursor: usize,
    pub turn: u32,
    pub sequence: u64,
    pub provider_correlation: Option<String>,
    pub call_id: String,
    pub internal_call_id: String,
    pub content: ToolCallDeltaContent,
    pub phase: PersistedToolCallDeltaPolicyPhase,
}

macro_rules! policy_phase {
    ($name:ident { $($variant:ident $(($payload:ty))?),* $(,)? }) => {
        #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
        pub enum $name {
            Evaluating,
            WaitingApproval { operation: String, policy: StableId },
            $($variant $(($payload))?,)*
        }
    };
}

policy_phase!(PersistedRequestPolicyPhase {
    Accepted,
    Rejected((StableId, String)),
});
policy_phase!(PersistedToolCallPolicyPhase {
    Accepted,
    Skipped(String),
    Rejected((StableId, String)),
});
policy_phase!(PersistedInvalidToolCallPolicyPhase {
    Resolved,
    Failed,
    Rejected((StableId, String)),
});
policy_phase!(PersistedToolResultPolicyPhase {
    Accepted,
    Rejected((StableId, String)),
});
policy_phase!(PersistedCompletionResponsePolicyPhase {
    Accepted,
    Rejected((StableId, String)),
});
policy_phase!(PersistedTextDeltaPolicyPhase {
    Published,
    Rejected((StableId, String)),
});
policy_phase!(PersistedToolCallDeltaPolicyPhase {
    Published,
    Rejected((StableId, String)),
});

/// Captures a run and all descendants at a resumable schedule boundary.
///
/// In-flight effects are rejected. Callers that need a durable boundary must
/// request [`PauseMode::CancelAndSuspend`], drive until the run is paused, and
/// then capture the resulting prepared generation. Settled completions retained
/// by freeze-after-ingress are safe and remain settled in the snapshot.
#[allow(clippy::type_complexity)]
pub fn snapshot_active_run(
    world: &mut World,
    root: Entity,
) -> Result<ActiveRunSnapshot, ActiveRunSnapshotError> {
    let mut run_rows = world.query::<(
        Entity,
        &StableId,
        &TenantId,
        &RunOf,
        &RunState,
        &RunControl,
        &RunRecord,
        Option<&ParentRun>,
        Option<&ChildOrdinal>,
        Option<&ChildResultCommitted>,
    )>();
    let all_runs = run_rows
        .iter(world)
        .map(
            |(entity, id, tenant, run_of, state, control, record, parent, ordinal, committed)| {
                (
                    entity,
                    id.clone(),
                    tenant.clone(),
                    run_of.get(),
                    state.clone(),
                    *control,
                    record.clone(),
                    parent.map(Relationship::get),
                    ordinal.map(|value| value.0),
                    committed.is_some(),
                )
            },
        )
        .collect::<Vec<_>>();
    let Some(root_row) = all_runs.iter().find(|row| row.0 == root) else {
        return Err(ActiveRunSnapshotError::NotRun);
    };
    let root_id = root_row.1.clone();

    let mut selected = HashSet::from([root]);
    let mut frontier = VecDeque::from([root]);
    while let Some(parent) = frontier.pop_front() {
        let mut children = all_runs
            .iter()
            .filter(|row| row.7 == Some(parent))
            .map(|row| (row.8.unwrap_or(u64::MAX), row.1.clone(), row.0))
            .collect::<Vec<_>>();
        children.sort();
        for (_, _, child) in children {
            if selected.insert(child) {
                frontier.push_back(child);
            }
        }
    }

    let run_ids = all_runs
        .iter()
        .filter(|row| selected.contains(&row.0))
        .map(|row| (row.0, row.1.clone()))
        .collect::<HashMap<_, _>>();
    let agent_entities = all_runs
        .iter()
        .filter(|row| selected.contains(&row.0))
        .map(|row| row.3)
        .collect::<HashSet<_>>();

    // Extension inputs and results are concrete application components, so core
    // cannot serialize them generically. Never omit a live extension operation:
    // the application must first consume/remove it or move it to a discard-safe
    // cancellation phase before taking a core checkpoint.
    let mut extension_operation_query = world.query::<(
        Entity,
        &OperationOf,
        &OperationState,
        &ExtensionEffectKind,
        Option<&StableId>,
    )>();
    for (entity, operation_of, state, kind, id) in extension_operation_query.iter(world) {
        if selected.contains(&operation_of.get())
            && !matches!(
                state.phase,
                OperationPhase::Cancelled | OperationPhase::Superseded
            )
        {
            return Err(ActiveRunSnapshotError::UnsafeExtensionEffect {
                id: id
                    .map(|id| id.as_str().to_owned())
                    .unwrap_or_else(|| format!("{entity:?}")),
                kind: kind.0.to_owned(),
                phase: match state.phase {
                    OperationPhase::Prepared => "prepared",
                    OperationPhase::InFlight => "in-flight",
                    OperationPhase::Settled(_) => "settled",
                    OperationPhase::ExtensionSettled => "extension-settled",
                    OperationPhase::Cancelled => "cancelled",
                    OperationPhase::Superseded => "superseded",
                },
            });
        }
    }

    let mut agent_query = world.query::<(Entity, &StableId, &TenantId)>();
    let agents = agent_query
        .iter(world)
        .filter(|(entity, _, _)| agent_entities.contains(entity))
        .map(|(entity, id, tenant)| (entity, (id.clone(), tenant.clone())))
        .collect::<HashMap<_, _>>();

    let mut operation_query = world.query::<(
        Entity,
        &OperationOf,
        &OperationKind,
        &OperationGeneration,
        &OperationState,
        Option<&OperationOfBatch>,
    )>();
    let operation_rows = operation_query
        .iter(world)
        .filter(|(_, operation_of, _, _, _, _)| selected.contains(&operation_of.get()))
        .map(|(entity, operation_of, kind, generation, state, batch)| {
            (
                entity,
                operation_of.get(),
                *kind,
                generation.0,
                state.clone(),
                batch.map(Relationship::get),
            )
        })
        .collect::<Vec<_>>();
    let mut operation_entities = operation_rows.iter().map(|row| row.0).collect::<Vec<_>>();
    operation_entities.sort();
    let operation_ids = operation_entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (*entity, format!("operation-{index}")))
        .collect::<HashMap<_, _>>();

    let mut batch_query = world.query::<(Entity, &BatchOf, &ToolBatchState)>();
    let mut batch_entities = batch_query
        .iter(world)
        .filter(|(_, batch_of, _)| selected.contains(&batch_of.get()))
        .map(|row| row.0)
        .collect::<Vec<_>>();
    batch_entities.sort();
    let batch_ids = batch_entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (*entity, format!("batch-{index}")))
        .collect::<HashMap<_, _>>();

    let mut evaluation_query = world.query::<(
        Entity,
        &EvaluationOfOperation,
        Option<&RequestPolicyEvaluation>,
        Option<&ToolCallPolicyEvaluation>,
        Option<&InvalidToolCallPolicyEvaluation>,
        Option<&ToolResultPolicyEvaluation>,
        Option<&CompletionResponsePolicyEvaluation>,
        Option<&TextDeltaPolicyEvaluation>,
        Option<&ToolCallDeltaPolicyEvaluation>,
    )>();
    let mut evaluation_entities = evaluation_query
        .iter(world)
        .filter(|(_, operation, _, _, _, _, _, _, _)| operation_ids.contains_key(&operation.get()))
        .map(|row| row.0)
        .collect::<Vec<_>>();
    evaluation_entities.sort();
    let evaluation_ids = evaluation_entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (*entity, format!("evaluation-{index}")))
        .collect::<HashMap<_, _>>();

    let mut runs = Vec::with_capacity(selected.len());
    let mut pending = VecDeque::from([root]);
    while let Some(entity) = pending.pop_front() {
        let row = all_runs
            .iter()
            .find(|row| row.0 == entity)
            .ok_or(ActiveRunSnapshotError::NotRun)?;
        let (agent_id, agent_tenant) = agents
            .get(&row.3)
            .ok_or(ActiveRunSnapshotError::StaleRelationship(row.3))?;
        if agent_tenant != &row.2 {
            return Err(ActiveRunSnapshotError::TenantMismatch(
                agent_id.as_str().to_owned(),
            ));
        }
        let parent_run = row
            .7
            .map(|parent| {
                run_ids
                    .get(&parent)
                    .cloned()
                    .ok_or(ActiveRunSnapshotError::StaleRelationship(parent))
            })
            .transpose()?;
        let state = persist_run_state(&row.4, &operation_ids, &batch_ids)?;
        let mut record = row.6.clone();
        sanitize_run_record(&mut record);
        runs.push(PersistedActiveRun {
            id: row.1.clone(),
            tenant: row.2.clone(),
            agent_id: agent_id.clone(),
            state,
            control: row.5,
            priority: world
                .get::<RunPriority>(entity)
                .copied()
                .unwrap_or_default(),
            ready_at: world.get::<ReadyAt>(entity).copied().unwrap_or_default(),
            record,
            parent_run,
            child_ordinal: row.8,
            child_result_committed: row.9,
        });
        let mut children = all_runs
            .iter()
            .filter(|child| child.7 == Some(entity) && selected.contains(&child.0))
            .map(|child| (child.8.unwrap_or(u64::MAX), child.1.clone(), child.0))
            .collect::<Vec<_>>();
        children.sort();
        pending.extend(children.into_iter().map(|(_, _, child)| child));
    }

    let mut run_policy_query =
        world.query::<(&StableId, &TenantId, &Policy, &PolicyStatus, &PolicyForRun)>();
    let mut run_policies = run_policy_query
        .iter(world)
        .filter_map(|(id, tenant, policy, status, policy_for)| {
            run_ids
                .get(&policy_for.get())
                .cloned()
                .map(|run_id| PersistedRunPolicy {
                    id: id.clone(),
                    tenant: tenant.clone(),
                    run_id,
                    policy: policy.clone(),
                    status: *status,
                })
        })
        .collect::<Vec<_>>();
    run_policies.sort_by(|left, right| left.id.cmp(&right.id));

    let mut operations = Vec::with_capacity(operation_entities.len());
    for row in &operation_rows {
        let entity = row.0;
        let Some(id) = operation_ids.get(&entity) else {
            continue;
        };
        if matches!(row.4.phase, OperationPhase::InFlight) {
            return Err(ActiveRunSnapshotError::UnsafeInFlightEffect(id.clone()));
        }
        let run_id = run_ids
            .get(&row.1)
            .cloned()
            .ok_or(ActiveRunSnapshotError::StaleRelationship(row.1))?;
        let kind = match row.2 {
            OperationKind::Model => PersistedOperationKind::Model,
            OperationKind::Tool => PersistedOperationKind::Tool,
            OperationKind::Store => PersistedOperationKind::Store,
            OperationKind::PolicyApproval => PersistedOperationKind::PolicyApproval,
            OperationKind::Discovery => {
                return Err(ActiveRunSnapshotError::InvalidSnapshot(
                    "run-owned discovery operation".to_owned(),
                ));
            }
        };
        let batch = row
            .5
            .map(|batch| {
                batch_ids
                    .get(&batch)
                    .cloned()
                    .ok_or(ActiveRunSnapshotError::StaleRelationship(batch))
            })
            .transpose()?;
        let mut model_decision = world.get::<ModelDecision>(entity).cloned();
        if let Some(decision) = &mut model_decision {
            decision.model_entity = Entity::PLACEHOLDER;
        }
        let mut pending_model = world
            .get::<PendingModelRequest>(entity)
            .map(|value| value.0.clone());
        let mut model_input = world.get::<ModelEffectInput>(entity).cloned();
        let mut pending_tool = world
            .get::<PendingToolCall>(entity)
            .map(|value| value.0.clone());
        let mut tool_input = world.get::<ToolEffectInput>(entity).cloned();
        let mut pending_invalid_tool = world
            .get::<PendingInvalidToolCall>(entity)
            .map(|value| persist_invalid_call(value, &operation_ids))
            .transpose()?;
        let mut store_input = world.get::<StoreEffectInput>(entity).cloned();
        for input in pending_model.iter_mut().chain(model_input.iter_mut()) {
            sanitize_model_input(input);
        }
        for input in pending_tool.iter_mut().chain(tool_input.iter_mut()) {
            sanitize_tool_input(input);
        }
        if let Some(invalid) = &mut pending_invalid_tool {
            sanitize_tool_decisions(&mut invalid.available_tools);
        }
        if let Some(input) = &mut store_input {
            input.decision.store_entity = Entity::PLACEHOLDER;
        }
        operations.push(PersistedRunOperation {
            id: id.clone(),
            run_id,
            batch,
            kind,
            generation: row.3,
            state: row.4.clone(),
            model_decision,
            pending_model,
            model_input,
            model_stream: world.get::<ModelStreamState>(entity).cloned(),
            pending_tool,
            tool_input,
            pending_invalid_tool,
            store_input,
            approval_input: world.get::<PolicyApprovalEffectInput>(entity).cloned(),
            request_policies: world
                .get::<AcceptedPolicies>(entity)
                .map(|value| persist_policies(&value.0)),
            tool_call_policies: world
                .get::<AcceptedToolCallPolicies>(entity)
                .map(|value| persist_policies(&value.0)),
            invalid_tool_policies: world
                .get::<AcceptedInvalidToolCallPolicies>(entity)
                .map(|value| persist_policies(&value.0)),
            tool_result_policies: world
                .get::<AcceptedToolResultPolicies>(entity)
                .map(|value| persist_policies(&value.0)),
            response_policies: world
                .get::<AcceptedCompletionResponsePolicies>(entity)
                .map(|value| persist_policies(&value.0)),
            effective_tool_output: world
                .get::<EffectiveToolOutput>(entity)
                .map(|value| value.0.clone()),
            effective_model_output: world
                .get::<EffectiveModelOutput>(entity)
                .map(|value| value.0.clone()),
            provider_diagnostics: world
                .get::<ProviderResponseDiagnostics>(entity)
                .map(|value| value.0.clone()),
            markers: PersistedOperationMarkers {
                request_policy_initialized: world.get::<RequestPolicyInitialized>(entity).is_some(),
                tool_call_policy_initialized: world
                    .get::<ToolCallPolicyInitialized>(entity)
                    .is_some(),
                invalid_tool_policy_initialized: world
                    .get::<InvalidToolCallPolicyInitialized>(entity)
                    .is_some(),
                tool_result_policy_initialized: world
                    .get::<ToolResultPolicyInitialized>(entity)
                    .is_some(),
                tool_result_policy_done: world.get::<ToolResultPolicyDone>(entity).is_some(),
                response_policy_initialized: world
                    .get::<CompletionResponsePolicyInitialized>(entity)
                    .is_some(),
                response_policy_done: world.get::<CompletionResponsePolicyDone>(entity).is_some(),
                approval_applied: world.get::<PolicyApprovalApplied>(entity).is_some(),
            },
            approval_for_evaluation: world
                .get::<ApprovalForEvaluation>(entity)
                .map(|relation| {
                    evaluation_ids
                        .get(&relation.get())
                        .cloned()
                        .ok_or(ActiveRunSnapshotError::StaleRelationship(relation.get()))
                })
                .transpose()?,
        });
    }
    operations.sort_by(|left, right| left.id.cmp(&right.id));

    let mut tool_batches = Vec::with_capacity(batch_entities.len());
    for (entity, batch_of, state) in batch_query.iter(world) {
        let Some(id) = batch_ids.get(&entity) else {
            continue;
        };
        tool_batches.push(PersistedToolBatch {
            id: id.clone(),
            run_id: run_ids
                .get(&batch_of.get())
                .cloned()
                .ok_or(ActiveRunSnapshotError::StaleRelationship(batch_of.get()))?,
            source_model_operation: operation_ids
                .get(&state.source_model_operation)
                .cloned()
                .ok_or(ActiveRunSnapshotError::StaleRelationship(
                    state.source_model_operation,
                ))?,
            expected: state.expected,
            committed: state.committed,
        });
    }
    tool_batches.sort_by(|left, right| left.id.cmp(&right.id));

    let mut policy_evaluations = Vec::with_capacity(evaluation_entities.len());
    for row in evaluation_query.iter(world) {
        let Some(id) = evaluation_ids.get(&row.0) else {
            continue;
        };
        let operation = operation_ids
            .get(&row.1.get())
            .cloned()
            .ok_or(ActiveRunSnapshotError::StaleRelationship(row.1.get()))?;
        policy_evaluations.push(persist_evaluation(
            id,
            &operation,
            row,
            &run_ids,
            &operation_ids,
        )?);
    }
    policy_evaluations.sort_by(|left, right| evaluation_id(left).cmp(evaluation_id(right)));

    let mut turn_query = world.query::<(&TurnOf, &CommittedTurn)>();
    let mut turns = turn_query
        .iter(world)
        .filter_map(|(turn_of, turn)| {
            run_ids
                .get(&turn_of.get())
                .cloned()
                .map(|run_id| PersistedCommittedTurn {
                    run_id,
                    index: turn.index,
                    output: turn.output.clone(),
                })
        })
        .collect::<Vec<_>>();
    turns.sort_by(|left, right| {
        left.run_id
            .cmp(&right.run_id)
            .then_with(|| left.index.cmp(&right.index))
    });

    let mut ordered_run_entities = run_ids
        .iter()
        .map(|(entity, id)| (id, *entity))
        .collect::<Vec<_>>();
    ordered_run_entities.sort_by(|left, right| left.0.cmp(right.0));
    let ordered_run_entities = ordered_run_entities
        .into_iter()
        .map(|(_, entity)| entity)
        .collect::<Vec<_>>();
    let codecs = world
        .resource::<SnapshotExtensionCodecs>()
        .0
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let mut extensions = Vec::new();
    for codec in codecs {
        if let Some(payload) = codec
            .capture(world, &ordered_run_entities)
            .map_err(|message| ActiveRunSnapshotError::ExtensionCodec {
                id: codec.binding_id().to_owned(),
                message,
            })?
        {
            extensions.push(SnapshotExtensionSection {
                binding_id: codec.binding_id().to_owned(),
                revision: codec.revision(),
                required: codec.required(),
                payload,
            });
        }
    }

    Ok(ActiveRunSnapshot {
        version: ACTIVE_RUN_SNAPSHOT_VERSION,
        root_run: root_id,
        runs,
        run_policies,
        operations,
        tool_batches,
        policy_evaluations,
        turns,
        extensions,
    })
}

fn persist_run_state(
    state: &RunState,
    operations: &HashMap<Entity, String>,
    batches: &HashMap<Entity, String>,
) -> Result<PersistedRunState, ActiveRunSnapshotError> {
    Ok(match state {
        RunState::Queued => PersistedRunState::Queued,
        RunState::WaitingModel { operation } => PersistedRunState::WaitingModel {
            operation: operations
                .get(operation)
                .cloned()
                .ok_or(ActiveRunSnapshotError::StaleRelationship(*operation))?,
        },
        RunState::WaitingTools { batch } => PersistedRunState::WaitingTools {
            batch: batches
                .get(batch)
                .cloned()
                .ok_or(ActiveRunSnapshotError::StaleRelationship(*batch))?,
        },
        RunState::WaitingStore { operation } => PersistedRunState::WaitingStore {
            operation: operations
                .get(operation)
                .cloned()
                .ok_or(ActiveRunSnapshotError::StaleRelationship(*operation))?,
        },
        RunState::Completed(output) => PersistedRunState::Completed(output.clone()),
        RunState::Failed(error) => PersistedRunState::Failed(error.clone()),
        RunState::Cancelled => PersistedRunState::Cancelled,
    })
}

fn persist_policies(policies: &[AcceptedPolicy]) -> Vec<PersistedAcceptedPolicy> {
    policies
        .iter()
        .map(|policy| PersistedAcceptedPolicy {
            id: policy.id.clone(),
            revision: policy.revision,
            order: policy.order,
            point: policy.point,
        })
        .collect()
}

fn persist_invalid_call(
    value: &PendingInvalidToolCall,
    operation_ids: &HashMap<Entity, String>,
) -> Result<PersistedPendingInvalidToolCall, ActiveRunSnapshotError> {
    Ok(PersistedPendingInvalidToolCall {
        call: value.call.clone(),
        available_tools: value.available_tools.clone(),
        index: value.index,
        source_model_operation: operation_ids
            .get(&value.source_model_operation)
            .cloned()
            .ok_or(ActiveRunSnapshotError::StaleRelationship(
                value.source_model_operation,
            ))?,
        turn: value.turn,
        tool_choice: value.tool_choice.clone(),
        diagnostic_history: value.diagnostic_history.clone(),
        streaming_origin: value.streaming_origin,
        retry_count: value.retry_count,
        max_retries: value.max_retries,
    })
}

fn sanitize_run_record(record: &mut RunRecord) {
    if let Some(decision) = &mut record.memory_store {
        decision.store_entity = Entity::PLACEHOLDER;
    }
    if let Some(decision) = &mut record.retrieval_store {
        decision.store_entity = Entity::PLACEHOLDER;
    }
}

fn sanitize_model_input(input: &mut ModelEffectInput) {
    input.decision.model_entity = Entity::PLACEHOLDER;
    sanitize_tool_decisions(&mut input.tools);
}

fn sanitize_tool_input(input: &mut ToolEffectInput) {
    input.decision.tool_entity = Entity::PLACEHOLDER;
}

fn sanitize_tool_decisions(tools: &mut [ToolDecision]) {
    for tool in tools {
        tool.tool_entity = Entity::PLACEHOLDER;
    }
}

#[allow(clippy::type_complexity)]
fn persist_evaluation(
    id: &str,
    operation: &str,
    row: (
        Entity,
        &EvaluationOfOperation,
        Option<&RequestPolicyEvaluation>,
        Option<&ToolCallPolicyEvaluation>,
        Option<&InvalidToolCallPolicyEvaluation>,
        Option<&ToolResultPolicyEvaluation>,
        Option<&CompletionResponsePolicyEvaluation>,
        Option<&TextDeltaPolicyEvaluation>,
        Option<&ToolCallDeltaPolicyEvaluation>,
    ),
    run_ids: &HashMap<Entity, StableId>,
    operation_ids: &HashMap<Entity, String>,
) -> Result<PersistedPolicyEvaluation, ActiveRunSnapshotError> {
    let run_id = |run| {
        run_ids
            .get(&run)
            .cloned()
            .ok_or(ActiveRunSnapshotError::StaleRelationship(run))
    };
    let operation_id = |entity| {
        operation_ids
            .get(&entity)
            .cloned()
            .ok_or(ActiveRunSnapshotError::StaleRelationship(entity))
    };
    let mut count = 0;
    count += usize::from(row.2.is_some());
    count += usize::from(row.3.is_some());
    count += usize::from(row.4.is_some());
    count += usize::from(row.5.is_some());
    count += usize::from(row.6.is_some());
    count += usize::from(row.7.is_some());
    count += usize::from(row.8.is_some());
    if count != 1 {
        return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
            "policy evaluation `{id}` has {count} evaluation kinds"
        )));
    }
    if let Some(value) = row.2 {
        let mut effective = value.effective.clone();
        sanitize_model_input(&mut effective);
        return Ok(PersistedPolicyEvaluation::Request(
            PersistedRequestPolicyEvaluation {
                id: id.to_owned(),
                operation: operation.to_owned(),
                run_id: run_id(value.run)?,
                policies: persist_policies(&value.policies),
                cursor: value.cursor,
                effective,
                accumulated: value.accumulated.clone(),
                phase: match &value.phase {
                    RequestPolicyEvaluationPhase::Evaluating => {
                        PersistedRequestPolicyPhase::Evaluating
                    }
                    RequestPolicyEvaluationPhase::WaitingApproval { operation, policy } => {
                        PersistedRequestPolicyPhase::WaitingApproval {
                            operation: operation_id(*operation)?,
                            policy: policy.clone(),
                        }
                    }
                    RequestPolicyEvaluationPhase::Accepted => PersistedRequestPolicyPhase::Accepted,
                    RequestPolicyEvaluationPhase::Rejected { policy, reason } => {
                        PersistedRequestPolicyPhase::Rejected((policy.clone(), reason.clone()))
                    }
                },
            },
        ));
    }
    if let Some(value) = row.3 {
        let mut effective = value.effective.clone();
        sanitize_tool_input(&mut effective);
        return Ok(PersistedPolicyEvaluation::ToolCall(
            PersistedToolCallPolicyEvaluation {
                id: id.to_owned(),
                operation: operation.to_owned(),
                run_id: run_id(value.run)?,
                policies: persist_policies(&value.policies),
                cursor: value.cursor,
                effective,
                phase: match &value.phase {
                    ToolCallPolicyEvaluationPhase::Evaluating => {
                        PersistedToolCallPolicyPhase::Evaluating
                    }
                    ToolCallPolicyEvaluationPhase::WaitingApproval { operation, policy } => {
                        PersistedToolCallPolicyPhase::WaitingApproval {
                            operation: operation_id(*operation)?,
                            policy: policy.clone(),
                        }
                    }
                    ToolCallPolicyEvaluationPhase::Accepted => {
                        PersistedToolCallPolicyPhase::Accepted
                    }
                    ToolCallPolicyEvaluationPhase::Skipped(reason) => {
                        PersistedToolCallPolicyPhase::Skipped(reason.clone())
                    }
                    ToolCallPolicyEvaluationPhase::Rejected { policy, reason } => {
                        PersistedToolCallPolicyPhase::Rejected((policy.clone(), reason.clone()))
                    }
                },
            },
        ));
    }
    if let Some(value) = row.4 {
        let mut invalid = persist_invalid_call(&value.invalid, operation_ids)?;
        sanitize_tool_decisions(&mut invalid.available_tools);
        return Ok(PersistedPolicyEvaluation::InvalidToolCall(
            PersistedInvalidToolCallPolicyEvaluation {
                id: id.to_owned(),
                operation: operation.to_owned(),
                run_id: run_id(value.run)?,
                policies: persist_policies(&value.policies),
                cursor: value.cursor,
                invalid,
                phase: match &value.phase {
                    InvalidToolCallPolicyEvaluationPhase::Evaluating => {
                        PersistedInvalidToolCallPolicyPhase::Evaluating
                    }
                    InvalidToolCallPolicyEvaluationPhase::WaitingApproval { operation, policy } => {
                        PersistedInvalidToolCallPolicyPhase::WaitingApproval {
                            operation: operation_id(*operation)?,
                            policy: policy.clone(),
                        }
                    }
                    InvalidToolCallPolicyEvaluationPhase::Resolved => {
                        PersistedInvalidToolCallPolicyPhase::Resolved
                    }
                    InvalidToolCallPolicyEvaluationPhase::Failed => {
                        PersistedInvalidToolCallPolicyPhase::Failed
                    }
                    InvalidToolCallPolicyEvaluationPhase::Rejected { policy, reason } => {
                        PersistedInvalidToolCallPolicyPhase::Rejected((
                            policy.clone(),
                            reason.clone(),
                        ))
                    }
                },
            },
        ));
    }
    if let Some(value) = row.5 {
        let mut input = value.input.clone();
        sanitize_tool_input(&mut input);
        return Ok(PersistedPolicyEvaluation::ToolResult(
            PersistedToolResultPolicyEvaluation {
                id: id.to_owned(),
                operation: operation.to_owned(),
                run_id: run_id(value.run)?,
                policies: persist_policies(&value.policies),
                cursor: value.cursor,
                input,
                effective: value.effective.clone(),
                phase: match &value.phase {
                    ToolResultPolicyEvaluationPhase::Evaluating => {
                        PersistedToolResultPolicyPhase::Evaluating
                    }
                    ToolResultPolicyEvaluationPhase::WaitingApproval { operation, policy } => {
                        PersistedToolResultPolicyPhase::WaitingApproval {
                            operation: operation_id(*operation)?,
                            policy: policy.clone(),
                        }
                    }
                    ToolResultPolicyEvaluationPhase::Accepted => {
                        PersistedToolResultPolicyPhase::Accepted
                    }
                    ToolResultPolicyEvaluationPhase::Rejected { policy, reason } => {
                        PersistedToolResultPolicyPhase::Rejected((policy.clone(), reason.clone()))
                    }
                },
            },
        ));
    }
    if let Some(value) = row.6 {
        let mut request = value.request.clone();
        sanitize_model_input(&mut request);
        return Ok(PersistedPolicyEvaluation::CompletionResponse(
            PersistedCompletionResponsePolicyEvaluation {
                id: id.to_owned(),
                operation: operation.to_owned(),
                run_id: run_id(value.run)?,
                policies: persist_policies(&value.policies),
                cursor: value.cursor,
                request,
                effective: value.effective.clone(),
                phase: match &value.phase {
                    CompletionResponsePolicyEvaluationPhase::Evaluating => {
                        PersistedCompletionResponsePolicyPhase::Evaluating
                    }
                    CompletionResponsePolicyEvaluationPhase::WaitingApproval {
                        operation,
                        policy,
                    } => PersistedCompletionResponsePolicyPhase::WaitingApproval {
                        operation: operation_id(*operation)?,
                        policy: policy.clone(),
                    },
                    CompletionResponsePolicyEvaluationPhase::Accepted => {
                        PersistedCompletionResponsePolicyPhase::Accepted
                    }
                    CompletionResponsePolicyEvaluationPhase::Rejected { policy, reason } => {
                        PersistedCompletionResponsePolicyPhase::Rejected((
                            policy.clone(),
                            reason.clone(),
                        ))
                    }
                },
            },
        ));
    }
    if let Some(value) = row.7 {
        return Ok(PersistedPolicyEvaluation::TextDelta(
            PersistedTextDeltaPolicyEvaluation {
                id: id.to_owned(),
                operation: operation.to_owned(),
                run_id: run_id(value.run)?,
                policies: persist_policies(&value.policies),
                cursor: value.cursor,
                turn: value.turn,
                sequence: value.sequence,
                delta: value.delta.clone(),
                aggregated: value.aggregated.clone(),
                phase: match &value.phase {
                    TextDeltaPolicyEvaluationPhase::Evaluating => {
                        PersistedTextDeltaPolicyPhase::Evaluating
                    }
                    TextDeltaPolicyEvaluationPhase::WaitingApproval { operation, policy } => {
                        PersistedTextDeltaPolicyPhase::WaitingApproval {
                            operation: operation_id(*operation)?,
                            policy: policy.clone(),
                        }
                    }
                    TextDeltaPolicyEvaluationPhase::Published => {
                        PersistedTextDeltaPolicyPhase::Published
                    }
                    TextDeltaPolicyEvaluationPhase::Rejected { policy, reason } => {
                        PersistedTextDeltaPolicyPhase::Rejected((policy.clone(), reason.clone()))
                    }
                },
            },
        ));
    }
    let value = row.8.ok_or_else(|| {
        ActiveRunSnapshotError::InvalidSnapshot(format!(
            "policy evaluation `{id}` has no evaluation component"
        ))
    })?;
    Ok(PersistedPolicyEvaluation::ToolCallDelta(
        PersistedToolCallDeltaPolicyEvaluation {
            id: id.to_owned(),
            operation: operation.to_owned(),
            run_id: run_id(value.run)?,
            policies: persist_policies(&value.policies),
            cursor: value.cursor,
            turn: value.turn,
            sequence: value.sequence,
            provider_correlation: value.provider_correlation.clone(),
            call_id: value.id.clone(),
            internal_call_id: value.internal_call_id.clone(),
            content: value.content.clone(),
            phase: match &value.phase {
                ToolCallDeltaPolicyEvaluationPhase::Evaluating => {
                    PersistedToolCallDeltaPolicyPhase::Evaluating
                }
                ToolCallDeltaPolicyEvaluationPhase::WaitingApproval { operation, policy } => {
                    PersistedToolCallDeltaPolicyPhase::WaitingApproval {
                        operation: operation_id(*operation)?,
                        policy: policy.clone(),
                    }
                }
                ToolCallDeltaPolicyEvaluationPhase::Published => {
                    PersistedToolCallDeltaPolicyPhase::Published
                }
                ToolCallDeltaPolicyEvaluationPhase::Rejected { policy, reason } => {
                    PersistedToolCallDeltaPolicyPhase::Rejected((policy.clone(), reason.clone()))
                }
            },
        },
    ))
}

fn evaluation_id(evaluation: &PersistedPolicyEvaluation) -> &str {
    match evaluation {
        PersistedPolicyEvaluation::Request(value) => &value.id,
        PersistedPolicyEvaluation::ToolCall(value) => &value.id,
        PersistedPolicyEvaluation::InvalidToolCall(value) => &value.id,
        PersistedPolicyEvaluation::ToolResult(value) => &value.id,
        PersistedPolicyEvaluation::CompletionResponse(value) => &value.id,
        PersistedPolicyEvaluation::TextDelta(value) => &value.id,
        PersistedPolicyEvaluation::ToolCallDelta(value) => &value.id,
    }
}

#[derive(Clone)]
struct DomainRef {
    entity: Entity,
    tenant: TenantId,
    revision: Option<u64>,
    policy_order: Option<u32>,
    policy_points: Option<Vec<PolicyPoint>>,
}

#[derive(Default)]
struct DomainRefs {
    agents: HashMap<StableId, DomainRef>,
    models: HashMap<StableId, DomainRef>,
    tools: HashMap<StableId, DomainRef>,
    stores: HashMap<StableId, DomainRef>,
    policies: HashMap<StableId, DomainRef>,
    existing_ids: HashSet<StableId>,
}

/// Restores a validated active-run graph into an installed runtime world.
///
/// Domain configuration must already be present. Exact accepted model, tool,
/// store, and policy revisions are checked before any run entity is spawned.
/// Migrates a supported historical checkpoint into the current schema.
///
/// Version 5 added tool-call-delta policy evaluation variants. Version 6 added
/// a default-empty extension section. Versions 4 and 5 therefore upgrade
/// without inventing runtime state. Older and future versions are rejected.
pub fn migrate_active_run_snapshot(
    mut snapshot: ActiveRunSnapshot,
) -> Result<ActiveRunSnapshot, ActiveRunSnapshotError> {
    match snapshot.version {
        4 | 5 => {
            snapshot.version = ACTIVE_RUN_SNAPSHOT_VERSION;
            Ok(snapshot)
        }
        ACTIVE_RUN_SNAPSHOT_VERSION => Ok(snapshot),
        version => Err(ActiveRunSnapshotError::UnsupportedVersion(version)),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Encodes a versioned, integrity-checked checkpoint envelope.
pub fn encode_active_run_snapshot(
    snapshot: &ActiveRunSnapshot,
    protection: Option<&dyn ActiveRunSnapshotProtection>,
) -> Result<Vec<u8>, ActiveRunSnapshotError> {
    let plaintext = serde_json::to_vec(snapshot)
        .map_err(|error| ActiveRunSnapshotError::Serialization(error.to_string()))?;
    let payload = match protection {
        Some(hook) => hook
            .protect(&plaintext)
            .map_err(ActiveRunSnapshotError::Protection)?,
        None => plaintext,
    };
    let envelope = ActiveRunSnapshotEnvelope {
        version: ACTIVE_RUN_ENVELOPE_VERSION,
        protected: protection.is_some(),
        sha256: sha256_hex(&payload),
        payload,
    };
    serde_json::to_vec(&envelope)
        .map_err(|error| ActiveRunSnapshotError::Serialization(error.to_string()))
}

/// Decodes, verifies, unprotects, migrates, and bounds-checks a checkpoint.
pub fn decode_active_run_snapshot(
    encoded: &[u8],
    protection: Option<&dyn ActiveRunSnapshotProtection>,
    limits: ActiveRunSnapshotLimits,
) -> Result<ActiveRunSnapshot, ActiveRunSnapshotError> {
    if encoded.len() > limits.max_encoded_bytes {
        return Err(ActiveRunSnapshotError::LimitExceeded {
            field: "encoded_bytes",
            actual: encoded.len(),
            limit: limits.max_encoded_bytes,
        });
    }
    let envelope: ActiveRunSnapshotEnvelope = serde_json::from_slice(encoded)
        .map_err(|error| ActiveRunSnapshotError::Serialization(error.to_string()))?;
    if envelope.version != ACTIVE_RUN_ENVELOPE_VERSION
        || sha256_hex(&envelope.payload) != envelope.sha256
    {
        return Err(ActiveRunSnapshotError::Integrity);
    }
    if envelope.protected && protection.is_none() {
        return Err(ActiveRunSnapshotError::MissingProtection);
    }
    if !envelope.protected && protection.is_some() {
        return Err(ActiveRunSnapshotError::ProtectionModeMismatch);
    }
    let plaintext = if envelope.protected {
        protection
            .ok_or(ActiveRunSnapshotError::MissingProtection)?
            .unprotect(&envelope.payload)
            .map_err(ActiveRunSnapshotError::Protection)?
    } else {
        envelope.payload
    };
    if plaintext.len() > limits.max_encoded_bytes {
        return Err(ActiveRunSnapshotError::LimitExceeded {
            field: "decoded_bytes",
            actual: plaintext.len(),
            limit: limits.max_encoded_bytes,
        });
    }
    let snapshot: ActiveRunSnapshot = serde_json::from_slice(&plaintext)
        .map_err(|error| ActiveRunSnapshotError::Serialization(error.to_string()))?;
    enforce_snapshot_limits(&snapshot, limits)?;
    migrate_active_run_snapshot(snapshot)
}

fn snapshot_topology(
    snapshot: &ActiveRunSnapshot,
) -> Result<(usize, usize), ActiveRunSnapshotError> {
    let mut children = HashMap::<&StableId, Vec<&StableId>>::new();
    let ids = snapshot
        .runs
        .iter()
        .map(|run| &run.id)
        .collect::<HashSet<_>>();
    if !ids.contains(&snapshot.root_run) {
        return Err(ActiveRunSnapshotError::MissingSnapshotReference(
            snapshot.root_run.as_str().to_owned(),
        ));
    }
    for run in &snapshot.runs {
        if let Some(parent) = &run.parent_run {
            if !ids.contains(parent) {
                return Err(ActiveRunSnapshotError::MissingSnapshotReference(
                    parent.as_str().to_owned(),
                ));
            }
            children.entry(parent).or_default().push(&run.id);
        }
    }
    let max_children = children.values().map(Vec::len).max().unwrap_or(0);
    let mut visited = HashSet::new();
    let mut frontier = VecDeque::from([(&snapshot.root_run, 0_usize)]);
    let mut max_depth = 0;
    while let Some((run, depth)) = frontier.pop_front() {
        if !visited.insert(run) {
            return Err(ActiveRunSnapshotError::InvalidSnapshot(
                "run graph contains a cycle".to_owned(),
            ));
        }
        max_depth = max_depth.max(depth);
        if let Some(descendants) = children.get(run) {
            frontier.extend(descendants.iter().map(|child| (*child, depth + 1)));
        }
    }
    if visited.len() != snapshot.runs.len() {
        return Err(ActiveRunSnapshotError::InvalidSnapshot(
            "run graph contains a cycle or disconnected run".to_owned(),
        ));
    }
    Ok((max_depth, max_children))
}

/// Builds a structured inventory without restoring the checkpoint.
pub fn summarize_active_run_snapshot(
    snapshot: &ActiveRunSnapshot,
) -> Result<ActiveRunSnapshotSummary, ActiveRunSnapshotError> {
    let encoded_bytes = serde_json::to_vec(snapshot)
        .map_err(|error| ActiveRunSnapshotError::Serialization(error.to_string()))?
        .len();
    let (max_depth, max_children_per_run) = snapshot_topology(snapshot)?;
    let mut run_states = BTreeMap::new();
    for run in &snapshot.runs {
        let name = match run.state {
            PersistedRunState::Queued => "queued",
            PersistedRunState::WaitingModel { .. } => "waiting_model",
            PersistedRunState::WaitingTools { .. } => "waiting_tools",
            PersistedRunState::WaitingStore { .. } => "waiting_store",
            PersistedRunState::Completed(_) => "completed",
            PersistedRunState::Failed(_) => "failed",
            PersistedRunState::Cancelled => "cancelled",
        };
        *run_states.entry(name.to_owned()).or_default() += 1;
    }
    let mut operation_kinds = BTreeMap::new();
    for operation in &snapshot.operations {
        let name = match operation.kind {
            PersistedOperationKind::Model => "model",
            PersistedOperationKind::Tool => "tool",
            PersistedOperationKind::Store => "store",
            PersistedOperationKind::PolicyApproval => "policy_approval",
        };
        *operation_kinds.entry(name.to_owned()).or_default() += 1;
    }
    let extension_bytes = snapshot
        .extensions
        .iter()
        .map(|section| {
            serde_json::to_vec(&section.payload)
                .map_err(|error| ActiveRunSnapshotError::Serialization(error.to_string()))
                .map(|value| value.len())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum();
    Ok(ActiveRunSnapshotSummary {
        version: snapshot.version,
        root_run: snapshot.root_run.clone(),
        encoded_bytes,
        runs: snapshot.runs.len(),
        max_depth,
        max_children_per_run,
        run_policies: snapshot.run_policies.len(),
        operations: snapshot.operations.len(),
        tool_batches: snapshot.tool_batches.len(),
        policy_evaluations: snapshot.policy_evaluations.len(),
        turns: snapshot.turns.len(),
        transcript_entries: snapshot
            .runs
            .iter()
            .map(|run| run.record.transcript.len())
            .sum(),
        extension_sections: snapshot.extensions.len(),
        extension_bytes,
        run_states,
        operation_kinds,
    })
}

fn enforce_snapshot_limits(
    snapshot: &ActiveRunSnapshot,
    limits: ActiveRunSnapshotLimits,
) -> Result<(), ActiveRunSnapshotError> {
    let summary = summarize_active_run_snapshot(snapshot)?;
    let checks = [
        (
            "encoded_bytes",
            summary.encoded_bytes,
            limits.max_encoded_bytes,
        ),
        ("runs", summary.runs, limits.max_runs),
        ("depth", summary.max_depth, limits.max_depth),
        (
            "children_per_run",
            summary.max_children_per_run,
            limits.max_children_per_run,
        ),
        ("operations", summary.operations, limits.max_operations),
        (
            "policy_evaluations",
            summary.policy_evaluations,
            limits.max_policy_evaluations,
        ),
        (
            "transcript_entries",
            summary.transcript_entries,
            limits.max_transcript_entries,
        ),
        (
            "extension_sections",
            summary.extension_sections,
            limits.max_extension_sections,
        ),
        (
            "extension_bytes",
            summary.extension_bytes,
            limits.max_extension_bytes,
        ),
    ];
    for (field, actual, limit) in checks {
        if actual > limit {
            return Err(ActiveRunSnapshotError::LimitExceeded {
                field,
                actual,
                limit,
            });
        }
    }
    Ok(())
}

/// Restores a checkpoint using caller-selected resource limits.
///
/// Subscriptions are runtime-local and must be reattached by the caller. All
/// migration, topology, size, and domain checks run before entity creation.
pub fn restore_active_run_with_limits(
    world: &mut World,
    snapshot: ActiveRunSnapshot,
    limits: ActiveRunSnapshotLimits,
) -> Result<RestoredRuns, ActiveRunSnapshotError> {
    enforce_snapshot_limits(&snapshot, limits)?;
    let snapshot = migrate_active_run_snapshot(snapshot)?;
    let mut domain = validate_active_snapshot(world, &snapshot)?;

    let mut captured_ready_order = snapshot
        .runs
        .iter()
        .map(|run| run.ready_at.0)
        .collect::<Vec<_>>();
    captured_ready_order.sort_unstable();
    captured_ready_order.dedup();
    let target_ready = i128::from(world.resource::<RuntimeClock>().tick);
    let rebased_ready = snapshot
        .runs
        .iter()
        .map(|run| {
            let rank = captured_ready_order
                .binary_search(&run.ready_at.0)
                .map_err(|_| {
                    ActiveRunSnapshotError::InvalidSnapshot(
                        "run readiness order is inconsistent".to_owned(),
                    )
                })?;
            let newer = captured_ready_order.len().saturating_sub(rank + 1);
            let newer = i128::try_from(newer).map_err(|_| {
                ActiveRunSnapshotError::InvalidSnapshot(
                    "run readiness order exceeds supported range".to_owned(),
                )
            })?;
            let ready = target_ready.checked_sub(newer).ok_or_else(|| {
                ActiveRunSnapshotError::InvalidSnapshot(
                    "run readiness order cannot be rebased".to_owned(),
                )
            })?;
            Ok((run.id.clone(), ReadyAt(ready)))
        })
        .collect::<Result<HashMap<_, _>, ActiveRunSnapshotError>>()?;

    let mut runs = HashMap::new();
    for persisted in &snapshot.runs {
        let agent = domain_ref(&domain.agents, &persisted.agent_id)?.entity;
        let mut record = persisted.record.clone();
        remap_run_record(&mut record, &domain, &persisted.tenant)?;
        let entity = world
            .spawn((
                persisted.id.clone(),
                persisted.tenant.clone(),
                RunOf(agent),
                RunState::Queued,
                persisted.control,
                persisted.priority,
                *rebased_ready.get(&persisted.id).ok_or_else(|| {
                    ActiveRunSnapshotError::InvalidSnapshot(
                        "run readiness order is missing".to_owned(),
                    )
                })?,
                record,
            ))
            .id();
        runs.insert(persisted.id.clone(), entity);
    }

    for persisted in &snapshot.runs {
        let run = local_run(&runs, &persisted.id)?;
        if let Some(parent_id) = &persisted.parent_run {
            let parent = local_run(&runs, parent_id)?;
            let ordinal = persisted.child_ordinal.ok_or_else(|| {
                ActiveRunSnapshotError::InvalidSnapshot(format!(
                    "child run `{}` has no child ordinal",
                    persisted.id.as_str()
                ))
            })?;
            world
                .entity_mut(run)
                .insert((ParentRun(parent), ChildOrdinal(ordinal)));
            if !persisted.child_result_committed {
                world.entity_mut(parent).insert(WaitingForChildren);
            }
        } else if persisted.child_ordinal.is_some() {
            return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
                "root run `{}` has a child ordinal",
                persisted.id.as_str()
            )));
        }
        if persisted.child_result_committed {
            world.entity_mut(run).insert(ChildResultCommitted);
        }
    }

    let mut restored_run_policies = Vec::with_capacity(snapshot.run_policies.len());
    for persisted in &snapshot.run_policies {
        let run = local_run(&runs, &persisted.run_id)?;
        let entity = world
            .spawn((
                persisted.id.clone(),
                persisted.tenant.clone(),
                persisted.policy.clone(),
                persisted.status,
                PolicyForRun(run),
            ))
            .id();
        domain.policies.insert(
            persisted.id.clone(),
            DomainRef {
                entity,
                tenant: persisted.tenant.clone(),
                revision: Some(persisted.policy.revision),
                policy_order: Some(persisted.policy.order),
                policy_points: Some(vec![persisted.policy.rule.point()]),
            },
        );
        restored_run_policies.push((persisted.id.clone(), entity));
    }

    let mut batches = HashMap::new();
    for persisted in &snapshot.tool_batches {
        let run = local_run(&runs, &persisted.run_id)?;
        let entity = world
            .spawn((
                BatchOf(run),
                ToolBatchState {
                    source_model_operation: Entity::PLACEHOLDER,
                    expected: persisted.expected,
                    committed: persisted.committed,
                },
            ))
            .id();
        batches.insert(persisted.id.clone(), entity);
    }

    let mut operations = HashMap::new();
    for persisted in &snapshot.operations {
        let run = local_run(&runs, &persisted.run_id)?;
        let batch = persisted
            .batch
            .as_ref()
            .map(|id| local_entity(&batches, id))
            .transpose()?;
        let kind = match persisted.kind {
            PersistedOperationKind::Model => OperationKind::Model,
            PersistedOperationKind::Tool => OperationKind::Tool,
            PersistedOperationKind::Store => OperationKind::Store,
            PersistedOperationKind::PolicyApproval => OperationKind::PolicyApproval,
        };
        let mut state = persisted.state.clone();
        if matches!(state.phase, OperationPhase::InFlight) {
            return Err(ActiveRunSnapshotError::UnsafeInFlightEffect(
                persisted.id.clone(),
            ));
        }
        let mut entity = world.spawn((
            OperationOf(run),
            kind,
            OperationGeneration(persisted.generation),
            state.clone(),
        ));
        if let Some(batch) = batch {
            entity.insert(OperationOfBatch(batch));
        }
        let entity = entity.id();
        operations.insert(persisted.id.clone(), entity);
        state.phase = persisted.state.phase.clone();
        world.entity_mut(entity).insert(state);
    }

    for persisted in &snapshot.operations {
        let operation = local_entity(&operations, &persisted.id)?;
        let run = local_run(&runs, &persisted.run_id)?;
        let tenant = world
            .get::<TenantId>(run)
            .cloned()
            .ok_or(ActiveRunSnapshotError::NotRun)?;
        restore_operation_components(world, operation, persisted, &domain, &tenant, &operations)?;
    }

    for persisted in &snapshot.tool_batches {
        let batch = local_entity(&batches, &persisted.id)?;
        let source = local_entity(&operations, &persisted.source_model_operation)?;
        let Some(mut state) = world.get_mut::<ToolBatchState>(batch) else {
            return Err(ActiveRunSnapshotError::MissingSnapshotReference(
                persisted.id.clone(),
            ));
        };
        state.source_model_operation = source;
    }

    let mut evaluations = HashMap::new();
    for persisted in &snapshot.policy_evaluations {
        let id = evaluation_id(persisted).to_owned();
        let operation_id = evaluation_operation(persisted);
        let operation = local_entity(&operations, operation_id)?;
        let entity = world.spawn(EvaluationOfOperation(operation)).id();
        evaluations.insert(id, entity);
    }
    for persisted in &snapshot.policy_evaluations {
        let entity = local_entity(&evaluations, evaluation_id(persisted))?;
        restore_evaluation_components(world, entity, persisted, &domain, &runs, &operations)?;
    }
    for persisted in &snapshot.operations {
        if let Some(evaluation_id) = &persisted.approval_for_evaluation {
            let operation = local_entity(&operations, &persisted.id)?;
            let evaluation = local_entity(&evaluations, evaluation_id)?;
            world
                .entity_mut(operation)
                .insert(ApprovalForEvaluation(evaluation));
        }
    }

    for persisted in &snapshot.runs {
        let run = local_run(&runs, &persisted.id)?;
        let state = restore_run_state(&persisted.state, &operations, &batches)?;
        world.entity_mut(run).insert(state);
    }
    for persisted in &snapshot.turns {
        let run = local_run(&runs, &persisted.run_id)?;
        world.spawn((
            TurnOf(run),
            CommittedTurn {
                index: persisted.index,
                output: persisted.output.clone(),
            },
        ));
    }

    if let Some(mut index) = world.get_resource_mut::<StableIdIndex>() {
        for (id, entity) in &runs {
            index.0.insert(id.clone(), *entity);
        }
        for (id, entity) in restored_run_policies {
            index.0.insert(id, entity);
        }
    }
    let restored = RestoredRuns(runs);
    let codecs = world.resource::<SnapshotExtensionCodecs>().0.clone();
    for section in &snapshot.extensions {
        if let Some(codec) = codecs.get(&section.binding_id) {
            codec.restore(world, &restored, &section.payload);
        }
    }
    Ok(restored)
}

/// Restores a checkpoint using conservative default resource limits.
pub fn restore_active_run(
    world: &mut World,
    snapshot: ActiveRunSnapshot,
) -> Result<RestoredRuns, ActiveRunSnapshotError> {
    restore_active_run_with_limits(world, snapshot, ActiveRunSnapshotLimits::default())
}

fn validate_active_snapshot(
    world: &mut World,
    snapshot: &ActiveRunSnapshot,
) -> Result<DomainRefs, ActiveRunSnapshotError> {
    if !world.contains_resource::<RuntimeIdentity>() {
        return Err(ActiveRunSnapshotError::InvalidSnapshot(
            "target world has no installed runtime".to_owned(),
        ));
    }
    if snapshot.version != ACTIVE_RUN_SNAPSHOT_VERSION {
        return Err(ActiveRunSnapshotError::UnsupportedVersion(snapshot.version));
    }
    let codecs = world.resource::<SnapshotExtensionCodecs>().0.clone();
    let mut extension_ids = HashSet::new();
    for section in &snapshot.extensions {
        if section.binding_id.trim().is_empty()
            || !extension_ids.insert(section.binding_id.as_str())
        {
            return Err(ActiveRunSnapshotError::InvalidSnapshot(
                "extension binding IDs must be non-empty and unique".to_owned(),
            ));
        }
        let Some(codec) = codecs.get(&section.binding_id) else {
            if section.required {
                return Err(ActiveRunSnapshotError::MissingExtensionBinding(
                    section.binding_id.clone(),
                ));
            }
            continue;
        };
        if codec.revision() != section.revision {
            return Err(ActiveRunSnapshotError::ExtensionRevisionMismatch {
                id: section.binding_id.clone(),
                expected: section.revision,
                found: codec.revision(),
            });
        }
        codec.validate(&section.payload).map_err(|message| {
            ActiveRunSnapshotError::ExtensionCodec {
                id: section.binding_id.clone(),
                message,
            }
        })?;
    }
    let mut domain = collect_domain_refs(world);
    let mut run_ids = HashSet::new();
    for run in &snapshot.runs {
        if domain.existing_ids.contains(&run.id) || !run_ids.insert(run.id.clone()) {
            return Err(ActiveRunSnapshotError::ConflictingStableId(
                run.id.as_str().to_owned(),
            ));
        }
        validate_domain_ref(&domain.agents, &run.agent_id, &run.tenant, None)?;
        let mut record = run.record.clone();
        remap_run_record(&mut record, &domain, &run.tenant)?;
    }
    if !run_ids.contains(&snapshot.root_run) {
        return Err(ActiveRunSnapshotError::MissingSnapshotReference(
            snapshot.root_run.as_str().to_owned(),
        ));
    }
    let mut run_policy_ids = HashSet::new();
    for policy in &snapshot.run_policies {
        if domain.existing_ids.contains(&policy.id)
            || run_ids.contains(&policy.id)
            || !run_policy_ids.insert(policy.id.clone())
        {
            return Err(ActiveRunSnapshotError::ConflictingStableId(
                policy.id.as_str().to_owned(),
            ));
        }
        let run = snapshot
            .runs
            .iter()
            .find(|run| run.id == policy.run_id)
            .ok_or_else(|| {
                ActiveRunSnapshotError::MissingSnapshotReference(policy.run_id.as_str().to_owned())
            })?;
        if run.tenant != policy.tenant {
            return Err(ActiveRunSnapshotError::TenantMismatch(
                policy.id.as_str().to_owned(),
            ));
        }
        domain.policies.insert(
            policy.id.clone(),
            DomainRef {
                entity: Entity::PLACEHOLDER,
                tenant: policy.tenant.clone(),
                revision: Some(policy.policy.revision),
                policy_order: Some(policy.policy.order),
                policy_points: Some(vec![policy.policy.rule.point()]),
            },
        );
        domain.existing_ids.insert(policy.id.clone());
    }
    let mut child_ordinals = HashSet::new();
    for run in &snapshot.runs {
        match (
            &run.parent_run,
            run.child_ordinal,
            run.id == snapshot.root_run,
        ) {
            (None, None, true) => {}
            (Some(parent), Some(ordinal), false) => {
                if !run_ids.contains(parent) {
                    return Err(ActiveRunSnapshotError::MissingSnapshotReference(
                        parent.as_str().to_owned(),
                    ));
                }
                if !child_ordinals.insert((parent.clone(), ordinal)) {
                    return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
                        "parent `{}` has duplicate child ordinal {ordinal}",
                        parent.as_str()
                    )));
                }
            }
            _ => {
                return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
                    "run `{}` has inconsistent parent topology",
                    run.id.as_str()
                )));
            }
        }
    }
    let mut reachable = HashSet::from([snapshot.root_run.clone()]);
    loop {
        let before = reachable.len();
        for run in &snapshot.runs {
            if run
                .parent_run
                .as_ref()
                .is_some_and(|parent| reachable.contains(parent))
            {
                reachable.insert(run.id.clone());
            }
        }
        if reachable.len() == before {
            break;
        }
    }
    if reachable != run_ids {
        return Err(ActiveRunSnapshotError::InvalidSnapshot(
            "run graph contains a cycle or disconnected run".to_owned(),
        ));
    }

    let operation_ids =
        unique_local_ids(snapshot.operations.iter().map(|value| value.id.as_str()))?;
    let batch_ids = unique_local_ids(snapshot.tool_batches.iter().map(|value| value.id.as_str()))?;
    let evaluation_ids = unique_local_ids(snapshot.policy_evaluations.iter().map(evaluation_id))?;
    for run in &snapshot.runs {
        validate_run_state_refs(&run.state, &operation_ids, &batch_ids)?;
    }
    for batch in &snapshot.tool_batches {
        require_run(&run_ids, &batch.run_id)?;
        require_local(&operation_ids, &batch.source_model_operation)?;
    }
    for operation in &snapshot.operations {
        require_run(&run_ids, &operation.run_id)?;
        if let Some(batch) = &operation.batch {
            require_local(&batch_ids, batch)?;
        }
        if matches!(operation.state.phase, OperationPhase::InFlight) {
            return Err(ActiveRunSnapshotError::UnsafeInFlightEffect(
                operation.id.clone(),
            ));
        }
        let run = snapshot
            .runs
            .iter()
            .find(|run| run.id == operation.run_id)
            .ok_or_else(|| {
                ActiveRunSnapshotError::MissingSnapshotReference(
                    operation.run_id.as_str().to_owned(),
                )
            })?;
        validate_operation(operation, &domain, &run.tenant, &operation_ids)?;
        if let Some(evaluation) = &operation.approval_for_evaluation {
            require_local(&evaluation_ids, evaluation)?;
        }
    }
    for evaluation in &snapshot.policy_evaluations {
        require_local(&operation_ids, evaluation_operation(evaluation))?;
        require_run(&run_ids, evaluation_run(evaluation))?;
        let run = snapshot
            .runs
            .iter()
            .find(|run| &run.id == evaluation_run(evaluation))
            .ok_or_else(|| {
                ActiveRunSnapshotError::MissingSnapshotReference(
                    evaluation_run(evaluation).as_str().to_owned(),
                )
            })?;
        validate_evaluation(evaluation, &domain, &run.tenant, &operation_ids)?;
    }
    for turn in &snapshot.turns {
        require_run(&run_ids, &turn.run_id)?;
    }
    Ok(domain)
}

fn collect_domain_refs(world: &mut World) -> DomainRefs {
    let mut refs = DomainRefs::default();
    let mut ids = world.query::<&StableId>();
    refs.existing_ids = ids.iter(world).cloned().collect();
    let mut agents = world.query::<(Entity, &StableId, &TenantId, &Agent)>();
    refs.agents = agents
        .iter(world)
        .map(|(entity, id, tenant, _)| {
            (
                id.clone(),
                DomainRef {
                    entity,
                    tenant: tenant.clone(),
                    revision: None,
                    policy_order: None,
                    policy_points: None,
                },
            )
        })
        .collect();
    let mut models = world.query::<(Entity, &StableId, &TenantId, &ModelCapability)>();
    refs.models = models
        .iter(world)
        .map(|(entity, id, tenant, value)| {
            (
                id.clone(),
                DomainRef {
                    entity,
                    tenant: tenant.clone(),
                    revision: Some(value.revision),
                    policy_order: None,
                    policy_points: None,
                },
            )
        })
        .collect();
    let mut tools = world.query::<(Entity, &StableId, &TenantId, &ToolCapability)>();
    refs.tools = tools
        .iter(world)
        .map(|(entity, id, tenant, value)| {
            (
                id.clone(),
                DomainRef {
                    entity,
                    tenant: tenant.clone(),
                    revision: Some(value.revision),
                    policy_order: None,
                    policy_points: None,
                },
            )
        })
        .collect();
    let mut stores = world.query::<(Entity, &StableId, &TenantId, &StoreCapability)>();
    refs.stores = stores
        .iter(world)
        .map(|(entity, id, tenant, value)| {
            (
                id.clone(),
                DomainRef {
                    entity,
                    tenant: tenant.clone(),
                    revision: Some(value.revision),
                    policy_order: None,
                    policy_points: None,
                },
            )
        })
        .collect();
    let mut policies = world.query::<(
        Entity,
        &StableId,
        &TenantId,
        &PolicyMeta,
        &PolicyCapabilities,
    )>();
    refs.policies = policies
        .iter(world)
        .map(|(entity, id, tenant, meta, capabilities)| {
            (
                id.clone(),
                DomainRef {
                    entity,
                    tenant: tenant.clone(),
                    revision: Some(meta.revision),
                    policy_order: Some(meta.order),
                    policy_points: Some(capabilities.points().to_vec()),
                },
            )
        })
        .collect();
    refs
}

fn unique_local_ids<'a>(
    ids: impl IntoIterator<Item = &'a str>,
) -> Result<HashSet<String>, ActiveRunSnapshotError> {
    let mut unique = HashSet::new();
    for id in ids {
        if id.is_empty() || !unique.insert(id.to_owned()) {
            return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
                "duplicate or empty snapshot reference `{id}`"
            )));
        }
    }
    Ok(unique)
}

fn require_run(runs: &HashSet<StableId>, id: &StableId) -> Result<(), ActiveRunSnapshotError> {
    if runs.contains(id) {
        Ok(())
    } else {
        Err(ActiveRunSnapshotError::MissingSnapshotReference(
            id.as_str().to_owned(),
        ))
    }
}

fn require_local(ids: &HashSet<String>, id: &str) -> Result<(), ActiveRunSnapshotError> {
    if ids.contains(id) {
        Ok(())
    } else {
        Err(ActiveRunSnapshotError::MissingSnapshotReference(
            id.to_owned(),
        ))
    }
}

fn domain_ref<'a>(
    values: &'a HashMap<StableId, DomainRef>,
    id: &StableId,
) -> Result<&'a DomainRef, ActiveRunSnapshotError> {
    values
        .get(id)
        .ok_or_else(|| ActiveRunSnapshotError::MissingDomainReference(id.as_str().to_owned()))
}

fn validate_domain_ref(
    values: &HashMap<StableId, DomainRef>,
    id: &StableId,
    tenant: &TenantId,
    revision: Option<u64>,
) -> Result<Entity, ActiveRunSnapshotError> {
    let value = domain_ref(values, id)?;
    if &value.tenant != tenant {
        return Err(ActiveRunSnapshotError::TenantMismatch(
            id.as_str().to_owned(),
        ));
    }
    if let Some(expected) = revision
        && value.revision != Some(expected)
    {
        return Err(ActiveRunSnapshotError::RevisionMismatch {
            id: id.as_str().to_owned(),
            expected,
        });
    }
    Ok(value.entity)
}

fn local_run(
    runs: &HashMap<StableId, Entity>,
    id: &StableId,
) -> Result<Entity, ActiveRunSnapshotError> {
    runs.get(id)
        .copied()
        .ok_or_else(|| ActiveRunSnapshotError::MissingSnapshotReference(id.as_str().to_owned()))
}

fn local_entity(
    entities: &HashMap<String, Entity>,
    id: &str,
) -> Result<Entity, ActiveRunSnapshotError> {
    entities
        .get(id)
        .copied()
        .ok_or_else(|| ActiveRunSnapshotError::MissingSnapshotReference(id.to_owned()))
}

fn validate_run_state_refs(
    state: &PersistedRunState,
    operations: &HashSet<String>,
    batches: &HashSet<String>,
) -> Result<(), ActiveRunSnapshotError> {
    match state {
        PersistedRunState::WaitingModel { operation }
        | PersistedRunState::WaitingStore { operation } => require_local(operations, operation),
        PersistedRunState::WaitingTools { batch } => require_local(batches, batch),
        PersistedRunState::Queued
        | PersistedRunState::Completed(_)
        | PersistedRunState::Failed(_)
        | PersistedRunState::Cancelled => Ok(()),
    }
}

fn restore_run_state(
    state: &PersistedRunState,
    operations: &HashMap<String, Entity>,
    batches: &HashMap<String, Entity>,
) -> Result<RunState, ActiveRunSnapshotError> {
    Ok(match state {
        PersistedRunState::Queued => RunState::Queued,
        PersistedRunState::WaitingModel { operation } => RunState::WaitingModel {
            operation: local_entity(operations, operation)?,
        },
        PersistedRunState::WaitingTools { batch } => RunState::WaitingTools {
            batch: local_entity(batches, batch)?,
        },
        PersistedRunState::WaitingStore { operation } => RunState::WaitingStore {
            operation: local_entity(operations, operation)?,
        },
        PersistedRunState::Completed(output) => RunState::Completed(output.clone()),
        PersistedRunState::Failed(error) => RunState::Failed(error.clone()),
        PersistedRunState::Cancelled => RunState::Cancelled,
    })
}

fn remap_run_record(
    record: &mut RunRecord,
    domain: &DomainRefs,
    tenant: &TenantId,
) -> Result<(), ActiveRunSnapshotError> {
    if let Some(decision) = &mut record.memory_store {
        remap_store_decision(decision, domain, tenant)?;
    }
    if let Some(decision) = &mut record.retrieval_store {
        remap_store_decision(decision, domain, tenant)?;
    }
    Ok(())
}

fn remap_model_input(
    input: &mut ModelEffectInput,
    domain: &DomainRefs,
    tenant: &TenantId,
) -> Result<(), ActiveRunSnapshotError> {
    input.decision.model_entity = validate_domain_ref(
        &domain.models,
        &input.decision.model_id,
        tenant,
        Some(input.decision.revision),
    )?;
    if &input.decision.tenant != tenant {
        return Err(ActiveRunSnapshotError::TenantMismatch(
            input.decision.model_id.as_str().to_owned(),
        ));
    }
    for tool in &mut input.tools {
        remap_tool_decision(tool, domain, tenant)?;
    }
    Ok(())
}

fn remap_model_decision(
    decision: &mut ModelDecision,
    domain: &DomainRefs,
    tenant: &TenantId,
) -> Result<(), ActiveRunSnapshotError> {
    decision.model_entity = validate_domain_ref(
        &domain.models,
        &decision.model_id,
        tenant,
        Some(decision.revision),
    )?;
    if &decision.tenant != tenant {
        return Err(ActiveRunSnapshotError::TenantMismatch(
            decision.model_id.as_str().to_owned(),
        ));
    }
    Ok(())
}

fn remap_tool_input(
    input: &mut ToolEffectInput,
    domain: &DomainRefs,
    tenant: &TenantId,
) -> Result<(), ActiveRunSnapshotError> {
    remap_tool_decision(&mut input.decision, domain, tenant)
}

fn remap_tool_decision(
    decision: &mut ToolDecision,
    domain: &DomainRefs,
    tenant: &TenantId,
) -> Result<(), ActiveRunSnapshotError> {
    decision.tool_entity = validate_domain_ref(
        &domain.tools,
        &decision.tool_id,
        tenant,
        Some(decision.revision),
    )?;
    Ok(())
}

fn remap_store_decision(
    decision: &mut StoreDecision,
    domain: &DomainRefs,
    tenant: &TenantId,
) -> Result<(), ActiveRunSnapshotError> {
    decision.store_entity = validate_domain_ref(
        &domain.stores,
        &decision.store_id,
        tenant,
        Some(decision.revision),
    )?;
    if &decision.tenant != tenant {
        return Err(ActiveRunSnapshotError::TenantMismatch(
            decision.store_id.as_str().to_owned(),
        ));
    }
    Ok(())
}

fn remap_policies(
    policies: &[PersistedAcceptedPolicy],
    expected_point: PolicyPoint,
    domain: &DomainRefs,
    tenant: &TenantId,
) -> Result<Vec<AcceptedPolicy>, ActiveRunSnapshotError> {
    let mut previous: Option<(u32, &str)> = None;
    for policy in policies {
        if policy.point != expected_point {
            return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
                "accepted policy `{}` has {:?} capability in a {:?} policy container",
                policy.id.as_str(),
                policy.point,
                expected_point
            )));
        }
        let key = (policy.order, policy.id.as_str());
        if previous.is_some_and(|previous| previous >= key) {
            return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
                "accepted {:?} policies are not uniquely ordered by (order, stable ID)",
                expected_point
            )));
        }
        previous = Some(key);
    }
    policies
        .iter()
        .map(|policy| {
            let current = domain_ref(&domain.policies, &policy.id)?;
            if current.policy_order != Some(policy.order)
                || current
                    .policy_points
                    .as_ref()
                    .is_none_or(|points| points.binary_search(&policy.point).is_err())
            {
                return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
                    "accepted policy `{}` no longer has order {} and {:?} capability",
                    policy.id.as_str(),
                    policy.order,
                    policy.point
                )));
            }
            Ok(AcceptedPolicy {
                id: policy.id.clone(),
                entity: validate_domain_ref(
                    &domain.policies,
                    &policy.id,
                    tenant,
                    Some(policy.revision),
                )?,
                revision: policy.revision,
                order: policy.order,
                point: policy.point,
            })
        })
        .collect()
}

fn validate_operation(
    operation: &PersistedRunOperation,
    domain: &DomainRefs,
    tenant: &TenantId,
    operation_ids: &HashSet<String>,
) -> Result<(), ActiveRunSnapshotError> {
    if operation.generation == u64::MAX {
        return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
            "operation `{}` has no successor generation",
            operation.id
        )));
    }
    let mut model_decision = operation.model_decision.clone();
    if let Some(value) = &mut model_decision {
        remap_model_decision(value, domain, tenant)?;
    }
    for value in operation
        .pending_model
        .iter()
        .chain(operation.model_input.iter())
    {
        let mut value = value.clone();
        remap_model_input(&mut value, domain, tenant)?;
    }
    for value in operation
        .pending_tool
        .iter()
        .chain(operation.tool_input.iter())
    {
        let mut value = value.clone();
        remap_tool_input(&mut value, domain, tenant)?;
    }
    if let Some(invalid) = &operation.pending_invalid_tool {
        require_local(operation_ids, &invalid.source_model_operation)?;
        for tool in &invalid.available_tools {
            let mut tool = tool.clone();
            remap_tool_decision(&mut tool, domain, tenant)?;
        }
    }
    if let Some(input) = &operation.store_input {
        let mut input = input.clone();
        remap_store_decision(&mut input.decision, domain, tenant)?;
    }
    for (policies, point) in [
        (operation.request_policies.as_deref(), PolicyPoint::Request),
        (
            operation.tool_call_policies.as_deref(),
            PolicyPoint::ToolCall,
        ),
        (
            operation.invalid_tool_policies.as_deref(),
            PolicyPoint::InvalidToolCall,
        ),
        (
            operation.tool_result_policies.as_deref(),
            PolicyPoint::ToolResult,
        ),
        (
            operation.response_policies.as_deref(),
            PolicyPoint::CompletionResponse,
        ),
    ] {
        if let Some(policies) = policies {
            remap_policies(policies, point, domain, tenant)?;
        }
    }
    let valid_kind = match operation.kind {
        PersistedOperationKind::Model => {
            operation.model_decision.is_some()
                && (operation.pending_model.is_some() || operation.model_input.is_some())
                && operation.pending_tool.is_none()
                && operation.tool_input.is_none()
                && operation.store_input.is_none()
                && operation.approval_input.is_none()
        }
        PersistedOperationKind::Tool => {
            operation.pending_model.is_none()
                && operation.model_input.is_none()
                && (operation.pending_tool.is_some()
                    || operation.tool_input.is_some()
                    || operation.pending_invalid_tool.is_some())
                && operation.store_input.is_none()
                && operation.approval_input.is_none()
        }
        PersistedOperationKind::Store => {
            operation.store_input.is_some()
                && operation.pending_model.is_none()
                && operation.model_input.is_none()
                && operation.pending_tool.is_none()
                && operation.tool_input.is_none()
                && operation.approval_input.is_none()
        }
        PersistedOperationKind::PolicyApproval => {
            operation.approval_input.is_some()
                && operation.pending_model.is_none()
                && operation.model_input.is_none()
                && operation.pending_tool.is_none()
                && operation.tool_input.is_none()
                && operation.store_input.is_none()
        }
    };
    if !valid_kind {
        return Err(ActiveRunSnapshotError::InvalidSnapshot(format!(
            "operation `{}` has components inconsistent with its kind",
            operation.id
        )));
    }
    Ok(())
}

fn restore_operation_components(
    world: &mut World,
    entity: Entity,
    persisted: &PersistedRunOperation,
    domain: &DomainRefs,
    tenant: &TenantId,
    operations: &HashMap<String, Entity>,
) -> Result<(), ActiveRunSnapshotError> {
    if let Some(mut value) = persisted.model_decision.clone() {
        remap_model_decision(&mut value, domain, tenant)?;
        world.entity_mut(entity).insert(value);
    }
    if let Some(mut value) = persisted.pending_model.clone() {
        remap_model_input(&mut value, domain, tenant)?;
        world.entity_mut(entity).insert(PendingModelRequest(value));
    }
    if let Some(mut value) = persisted.model_input.clone() {
        remap_model_input(&mut value, domain, tenant)?;
        world.entity_mut(entity).insert(value);
    }
    if let Some(value) = &persisted.model_stream {
        world.entity_mut(entity).insert(value.clone());
    }
    if let Some(mut value) = persisted.pending_tool.clone() {
        remap_tool_input(&mut value, domain, tenant)?;
        world.entity_mut(entity).insert(PendingToolCall(value));
    }
    if let Some(mut value) = persisted.tool_input.clone() {
        remap_tool_input(&mut value, domain, tenant)?;
        world.entity_mut(entity).insert(value);
    }
    if let Some(value) = &persisted.pending_invalid_tool {
        let mut available_tools = value.available_tools.clone();
        for tool in &mut available_tools {
            remap_tool_decision(tool, domain, tenant)?;
        }
        world.entity_mut(entity).insert(PendingInvalidToolCall {
            call: value.call.clone(),
            available_tools,
            index: value.index,
            source_model_operation: local_entity(operations, &value.source_model_operation)?,
            turn: value.turn,
            tool_choice: value.tool_choice.clone(),
            diagnostic_history: value.diagnostic_history.clone(),
            streaming_origin: value.streaming_origin,
            retry_count: value.retry_count,
            max_retries: value.max_retries,
        });
    }
    if let Some(mut value) = persisted.store_input.clone() {
        remap_store_decision(&mut value.decision, domain, tenant)?;
        world.entity_mut(entity).insert(value);
    }
    if let Some(value) = &persisted.approval_input {
        world.entity_mut(entity).insert(value.clone());
    }
    if let Some(value) = &persisted.request_policies {
        world
            .entity_mut(entity)
            .insert(AcceptedPolicies(remap_policies(
                value,
                PolicyPoint::Request,
                domain,
                tenant,
            )?));
    }
    if let Some(value) = &persisted.tool_call_policies {
        world
            .entity_mut(entity)
            .insert(AcceptedToolCallPolicies(remap_policies(
                value,
                PolicyPoint::ToolCall,
                domain,
                tenant,
            )?));
    }
    if let Some(value) = &persisted.invalid_tool_policies {
        world
            .entity_mut(entity)
            .insert(AcceptedInvalidToolCallPolicies(remap_policies(
                value,
                PolicyPoint::InvalidToolCall,
                domain,
                tenant,
            )?));
    }
    if let Some(value) = &persisted.tool_result_policies {
        world
            .entity_mut(entity)
            .insert(AcceptedToolResultPolicies(remap_policies(
                value,
                PolicyPoint::ToolResult,
                domain,
                tenant,
            )?));
    }
    if let Some(value) = &persisted.response_policies {
        world
            .entity_mut(entity)
            .insert(AcceptedCompletionResponsePolicies(remap_policies(
                value,
                PolicyPoint::CompletionResponse,
                domain,
                tenant,
            )?));
    }
    if let Some(value) = &persisted.effective_tool_output {
        world
            .entity_mut(entity)
            .insert(EffectiveToolOutput(value.clone()));
    }
    if let Some(value) = &persisted.effective_model_output {
        world
            .entity_mut(entity)
            .insert(EffectiveModelOutput(value.clone()));
    }
    if let Some(value) = &persisted.provider_diagnostics {
        world
            .entity_mut(entity)
            .insert(ProviderResponseDiagnostics(value.clone()));
    }
    let markers = persisted.markers;
    if markers.request_policy_initialized {
        world.entity_mut(entity).insert(RequestPolicyInitialized);
    }
    if markers.tool_call_policy_initialized {
        world.entity_mut(entity).insert(ToolCallPolicyInitialized);
    }
    if markers.invalid_tool_policy_initialized {
        world
            .entity_mut(entity)
            .insert(InvalidToolCallPolicyInitialized);
    }
    if markers.tool_result_policy_initialized {
        world.entity_mut(entity).insert(ToolResultPolicyInitialized);
    }
    if markers.tool_result_policy_done {
        world.entity_mut(entity).insert(ToolResultPolicyDone);
    }
    if markers.response_policy_initialized {
        world
            .entity_mut(entity)
            .insert(CompletionResponsePolicyInitialized);
    }
    if markers.response_policy_done {
        world
            .entity_mut(entity)
            .insert(CompletionResponsePolicyDone);
    }
    if markers.approval_applied {
        world.entity_mut(entity).insert(PolicyApprovalApplied);
    }
    Ok(())
}

fn evaluation_operation(evaluation: &PersistedPolicyEvaluation) -> &str {
    match evaluation {
        PersistedPolicyEvaluation::Request(value) => &value.operation,
        PersistedPolicyEvaluation::ToolCall(value) => &value.operation,
        PersistedPolicyEvaluation::InvalidToolCall(value) => &value.operation,
        PersistedPolicyEvaluation::ToolResult(value) => &value.operation,
        PersistedPolicyEvaluation::CompletionResponse(value) => &value.operation,
        PersistedPolicyEvaluation::TextDelta(value) => &value.operation,
        PersistedPolicyEvaluation::ToolCallDelta(value) => &value.operation,
    }
}

fn evaluation_run(evaluation: &PersistedPolicyEvaluation) -> &StableId {
    match evaluation {
        PersistedPolicyEvaluation::Request(value) => &value.run_id,
        PersistedPolicyEvaluation::ToolCall(value) => &value.run_id,
        PersistedPolicyEvaluation::InvalidToolCall(value) => &value.run_id,
        PersistedPolicyEvaluation::ToolResult(value) => &value.run_id,
        PersistedPolicyEvaluation::CompletionResponse(value) => &value.run_id,
        PersistedPolicyEvaluation::TextDelta(value) => &value.run_id,
        PersistedPolicyEvaluation::ToolCallDelta(value) => &value.run_id,
    }
}

fn validate_evaluation(
    evaluation: &PersistedPolicyEvaluation,
    domain: &DomainRefs,
    tenant: &TenantId,
    operations: &HashSet<String>,
) -> Result<(), ActiveRunSnapshotError> {
    match evaluation {
        PersistedPolicyEvaluation::Request(value) => {
            remap_policies(&value.policies, PolicyPoint::Request, domain, tenant)?;
            let mut effective = value.effective.clone();
            remap_model_input(&mut effective, domain, tenant)?;
        }
        PersistedPolicyEvaluation::ToolCall(value) => {
            remap_policies(&value.policies, PolicyPoint::ToolCall, domain, tenant)?;
            let mut effective = value.effective.clone();
            remap_tool_input(&mut effective, domain, tenant)?;
        }
        PersistedPolicyEvaluation::InvalidToolCall(value) => {
            remap_policies(
                &value.policies,
                PolicyPoint::InvalidToolCall,
                domain,
                tenant,
            )?;
            require_local(operations, &value.invalid.source_model_operation)?;
            for tool in &value.invalid.available_tools {
                let mut tool = tool.clone();
                remap_tool_decision(&mut tool, domain, tenant)?;
            }
        }
        PersistedPolicyEvaluation::ToolResult(value) => {
            remap_policies(&value.policies, PolicyPoint::ToolResult, domain, tenant)?;
            let mut input = value.input.clone();
            remap_tool_input(&mut input, domain, tenant)?;
        }
        PersistedPolicyEvaluation::CompletionResponse(value) => {
            remap_policies(
                &value.policies,
                PolicyPoint::CompletionResponse,
                domain,
                tenant,
            )?;
            let mut request = value.request.clone();
            remap_model_input(&mut request, domain, tenant)?;
        }
        PersistedPolicyEvaluation::TextDelta(value) => {
            remap_policies(&value.policies, PolicyPoint::TextDelta, domain, tenant)?;
        }
        PersistedPolicyEvaluation::ToolCallDelta(value) => {
            remap_policies(&value.policies, PolicyPoint::ToolCallDelta, domain, tenant)?;
        }
    }
    validate_evaluation_approval_ref(evaluation, operations)
}

fn validate_evaluation_approval_ref(
    evaluation: &PersistedPolicyEvaluation,
    operations: &HashSet<String>,
) -> Result<(), ActiveRunSnapshotError> {
    let waiting = match evaluation {
        PersistedPolicyEvaluation::Request(value) => match &value.phase {
            PersistedRequestPolicyPhase::WaitingApproval { operation, .. } => Some(operation),
            _ => None,
        },
        PersistedPolicyEvaluation::ToolCall(value) => match &value.phase {
            PersistedToolCallPolicyPhase::WaitingApproval { operation, .. } => Some(operation),
            _ => None,
        },
        PersistedPolicyEvaluation::InvalidToolCall(value) => match &value.phase {
            PersistedInvalidToolCallPolicyPhase::WaitingApproval { operation, .. } => {
                Some(operation)
            }
            _ => None,
        },
        PersistedPolicyEvaluation::ToolResult(value) => match &value.phase {
            PersistedToolResultPolicyPhase::WaitingApproval { operation, .. } => Some(operation),
            _ => None,
        },
        PersistedPolicyEvaluation::CompletionResponse(value) => match &value.phase {
            PersistedCompletionResponsePolicyPhase::WaitingApproval { operation, .. } => {
                Some(operation)
            }
            _ => None,
        },
        PersistedPolicyEvaluation::TextDelta(value) => match &value.phase {
            PersistedTextDeltaPolicyPhase::WaitingApproval { operation, .. } => Some(operation),
            _ => None,
        },
        PersistedPolicyEvaluation::ToolCallDelta(value) => match &value.phase {
            PersistedToolCallDeltaPolicyPhase::WaitingApproval { operation, .. } => Some(operation),
            _ => None,
        },
    };
    if let Some(operation) = waiting {
        require_local(operations, operation)?;
    }
    Ok(())
}

fn restore_evaluation_components(
    world: &mut World,
    entity: Entity,
    persisted: &PersistedPolicyEvaluation,
    domain: &DomainRefs,
    runs: &HashMap<StableId, Entity>,
    operations: &HashMap<String, Entity>,
) -> Result<(), ActiveRunSnapshotError> {
    let run_id = evaluation_run(persisted);
    let run = local_run(runs, run_id)?;
    let tenant = world
        .get::<TenantId>(run)
        .cloned()
        .ok_or(ActiveRunSnapshotError::NotRun)?;
    match persisted {
        PersistedPolicyEvaluation::Request(value) => {
            let mut effective = value.effective.clone();
            remap_model_input(&mut effective, domain, &tenant)?;
            world.entity_mut(entity).insert(RequestPolicyEvaluation {
                run,
                policies: remap_policies(&value.policies, PolicyPoint::Request, domain, &tenant)?,
                cursor: value.cursor,
                effective,
                accumulated: value.accumulated.clone(),
                phase: match &value.phase {
                    PersistedRequestPolicyPhase::Evaluating => {
                        RequestPolicyEvaluationPhase::Evaluating
                    }
                    PersistedRequestPolicyPhase::WaitingApproval { operation, policy } => {
                        RequestPolicyEvaluationPhase::WaitingApproval {
                            operation: local_entity(operations, operation)?,
                            policy: policy.clone(),
                        }
                    }
                    PersistedRequestPolicyPhase::Accepted => RequestPolicyEvaluationPhase::Accepted,
                    PersistedRequestPolicyPhase::Rejected((policy, reason)) => {
                        RequestPolicyEvaluationPhase::Rejected {
                            policy: policy.clone(),
                            reason: reason.clone(),
                        }
                    }
                },
            });
        }
        PersistedPolicyEvaluation::ToolCall(value) => {
            let mut effective = value.effective.clone();
            remap_tool_input(&mut effective, domain, &tenant)?;
            world.entity_mut(entity).insert(ToolCallPolicyEvaluation {
                run,
                policies: remap_policies(&value.policies, PolicyPoint::ToolCall, domain, &tenant)?,
                cursor: value.cursor,
                effective,
                phase: match &value.phase {
                    PersistedToolCallPolicyPhase::Evaluating => {
                        ToolCallPolicyEvaluationPhase::Evaluating
                    }
                    PersistedToolCallPolicyPhase::WaitingApproval { operation, policy } => {
                        ToolCallPolicyEvaluationPhase::WaitingApproval {
                            operation: local_entity(operations, operation)?,
                            policy: policy.clone(),
                        }
                    }
                    PersistedToolCallPolicyPhase::Accepted => {
                        ToolCallPolicyEvaluationPhase::Accepted
                    }
                    PersistedToolCallPolicyPhase::Skipped(reason) => {
                        ToolCallPolicyEvaluationPhase::Skipped(reason.clone())
                    }
                    PersistedToolCallPolicyPhase::Rejected((policy, reason)) => {
                        ToolCallPolicyEvaluationPhase::Rejected {
                            policy: policy.clone(),
                            reason: reason.clone(),
                        }
                    }
                },
            });
        }
        PersistedPolicyEvaluation::InvalidToolCall(value) => {
            let mut tools = value.invalid.available_tools.clone();
            for tool in &mut tools {
                remap_tool_decision(tool, domain, &tenant)?;
            }
            world
                .entity_mut(entity)
                .insert(InvalidToolCallPolicyEvaluation {
                    run,
                    policies: remap_policies(
                        &value.policies,
                        PolicyPoint::InvalidToolCall,
                        domain,
                        &tenant,
                    )?,
                    cursor: value.cursor,
                    invalid: PendingInvalidToolCall {
                        call: value.invalid.call.clone(),
                        available_tools: tools,
                        index: value.invalid.index,
                        source_model_operation: local_entity(
                            operations,
                            &value.invalid.source_model_operation,
                        )?,
                        turn: value.invalid.turn,
                        tool_choice: value.invalid.tool_choice.clone(),
                        diagnostic_history: value.invalid.diagnostic_history.clone(),
                        streaming_origin: value.invalid.streaming_origin,
                        retry_count: value.invalid.retry_count,
                        max_retries: value.invalid.max_retries,
                    },
                    phase: match &value.phase {
                        PersistedInvalidToolCallPolicyPhase::Evaluating => {
                            InvalidToolCallPolicyEvaluationPhase::Evaluating
                        }
                        PersistedInvalidToolCallPolicyPhase::WaitingApproval {
                            operation,
                            policy,
                        } => InvalidToolCallPolicyEvaluationPhase::WaitingApproval {
                            operation: local_entity(operations, operation)?,
                            policy: policy.clone(),
                        },
                        PersistedInvalidToolCallPolicyPhase::Resolved => {
                            InvalidToolCallPolicyEvaluationPhase::Resolved
                        }
                        PersistedInvalidToolCallPolicyPhase::Failed => {
                            InvalidToolCallPolicyEvaluationPhase::Failed
                        }
                        PersistedInvalidToolCallPolicyPhase::Rejected((policy, reason)) => {
                            InvalidToolCallPolicyEvaluationPhase::Rejected {
                                policy: policy.clone(),
                                reason: reason.clone(),
                            }
                        }
                    },
                });
        }
        PersistedPolicyEvaluation::ToolResult(value) => {
            let mut input = value.input.clone();
            remap_tool_input(&mut input, domain, &tenant)?;
            world.entity_mut(entity).insert(ToolResultPolicyEvaluation {
                run,
                policies: remap_policies(
                    &value.policies,
                    PolicyPoint::ToolResult,
                    domain,
                    &tenant,
                )?,
                cursor: value.cursor,
                input,
                effective: value.effective.clone(),
                phase: match &value.phase {
                    PersistedToolResultPolicyPhase::Evaluating => {
                        ToolResultPolicyEvaluationPhase::Evaluating
                    }
                    PersistedToolResultPolicyPhase::WaitingApproval { operation, policy } => {
                        ToolResultPolicyEvaluationPhase::WaitingApproval {
                            operation: local_entity(operations, operation)?,
                            policy: policy.clone(),
                        }
                    }
                    PersistedToolResultPolicyPhase::Accepted => {
                        ToolResultPolicyEvaluationPhase::Accepted
                    }
                    PersistedToolResultPolicyPhase::Rejected((policy, reason)) => {
                        ToolResultPolicyEvaluationPhase::Rejected {
                            policy: policy.clone(),
                            reason: reason.clone(),
                        }
                    }
                },
            });
        }
        PersistedPolicyEvaluation::CompletionResponse(value) => {
            let mut request = value.request.clone();
            remap_model_input(&mut request, domain, &tenant)?;
            world
                .entity_mut(entity)
                .insert(CompletionResponsePolicyEvaluation {
                    run,
                    policies: remap_policies(
                        &value.policies,
                        PolicyPoint::CompletionResponse,
                        domain,
                        &tenant,
                    )?,
                    cursor: value.cursor,
                    request,
                    effective: value.effective.clone(),
                    phase: match &value.phase {
                        PersistedCompletionResponsePolicyPhase::Evaluating => {
                            CompletionResponsePolicyEvaluationPhase::Evaluating
                        }
                        PersistedCompletionResponsePolicyPhase::WaitingApproval {
                            operation,
                            policy,
                        } => CompletionResponsePolicyEvaluationPhase::WaitingApproval {
                            operation: local_entity(operations, operation)?,
                            policy: policy.clone(),
                        },
                        PersistedCompletionResponsePolicyPhase::Accepted => {
                            CompletionResponsePolicyEvaluationPhase::Accepted
                        }
                        PersistedCompletionResponsePolicyPhase::Rejected((policy, reason)) => {
                            CompletionResponsePolicyEvaluationPhase::Rejected {
                                policy: policy.clone(),
                                reason: reason.clone(),
                            }
                        }
                    },
                });
        }
        PersistedPolicyEvaluation::TextDelta(value) => {
            let policies =
                remap_policies(&value.policies, PolicyPoint::TextDelta, domain, &tenant)?;
            world.entity_mut(entity).insert((
                AcceptedTextDeltaPolicies(policies.clone()),
                TextDeltaPolicyEvaluation {
                    run,
                    operation: local_entity(operations, &value.operation)?,
                    policies,
                    cursor: value.cursor,
                    turn: value.turn,
                    sequence: value.sequence,
                    delta: value.delta.clone(),
                    aggregated: value.aggregated.clone(),
                    phase: match &value.phase {
                        PersistedTextDeltaPolicyPhase::Evaluating => {
                            TextDeltaPolicyEvaluationPhase::Evaluating
                        }
                        PersistedTextDeltaPolicyPhase::WaitingApproval { operation, policy } => {
                            TextDeltaPolicyEvaluationPhase::WaitingApproval {
                                operation: local_entity(operations, operation)?,
                                policy: policy.clone(),
                            }
                        }
                        PersistedTextDeltaPolicyPhase::Published => {
                            TextDeltaPolicyEvaluationPhase::Published
                        }
                        PersistedTextDeltaPolicyPhase::Rejected((policy, reason)) => {
                            TextDeltaPolicyEvaluationPhase::Rejected {
                                policy: policy.clone(),
                                reason: reason.clone(),
                            }
                        }
                    },
                },
            ));
        }
        PersistedPolicyEvaluation::ToolCallDelta(value) => {
            let policies =
                remap_policies(&value.policies, PolicyPoint::ToolCallDelta, domain, &tenant)?;
            world.entity_mut(entity).insert((
                AcceptedToolCallDeltaPolicies(policies.clone()),
                ToolCallDeltaPolicyEvaluation {
                    run,
                    operation: local_entity(operations, &value.operation)?,
                    policies,
                    cursor: value.cursor,
                    turn: value.turn,
                    sequence: value.sequence,
                    provider_correlation: value.provider_correlation.clone(),
                    id: value.call_id.clone(),
                    internal_call_id: value.internal_call_id.clone(),
                    content: value.content.clone(),
                    phase: match &value.phase {
                        PersistedToolCallDeltaPolicyPhase::Evaluating => {
                            ToolCallDeltaPolicyEvaluationPhase::Evaluating
                        }
                        PersistedToolCallDeltaPolicyPhase::WaitingApproval {
                            operation,
                            policy,
                        } => ToolCallDeltaPolicyEvaluationPhase::WaitingApproval {
                            operation: local_entity(operations, operation)?,
                            policy: policy.clone(),
                        },
                        PersistedToolCallDeltaPolicyPhase::Published => {
                            ToolCallDeltaPolicyEvaluationPhase::Published
                        }
                        PersistedToolCallDeltaPolicyPhase::Rejected((policy, reason)) => {
                            ToolCallDeltaPolicyEvaluationPhase::Rejected {
                                policy: policy.clone(),
                                reason: reason.clone(),
                            }
                        }
                    },
                },
            ));
        }
    }
    Ok(())
}
