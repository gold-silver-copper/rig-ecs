//! Run and agent control components.

use super::*;

/// Explicit scheduling priority; larger values dispatch first.
#[derive(
    Component, Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct RunPriority(pub i32);

/// Monotonic logical order in which a run became eligible for work.
///
/// Live admissions use the runtime tick. Snapshot restoration rebases captured
/// ordering around the target clock, so negative values are valid.
#[derive(
    Component, Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct ReadyAt(pub i128);

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

/// Per-agent structured-output retry budget snapshotted onto admitted runs.
#[derive(Component, Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructuredOutputRetryBudget {
    /// Maximum corrective model retries after schema validation fails.
    pub max_retries: u32,
}

impl Default for StructuredOutputRetryBudget {
    fn default() -> Self {
        Self { max_retries: 1 }
    }
}
