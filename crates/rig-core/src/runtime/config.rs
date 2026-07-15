//! Runtime-wide bounded queue and progress configuration.

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
    /// Maximum schedule passes in [`super::Runtime::run_until_stalled`].
    pub progress_limit: usize,
    /// Maximum schedule ticks a core-dispatched external effect may remain in flight.
    pub effect_timeout_ticks: u64,
    /// Maximum new core effects dispatched during one schedule pass.
    pub max_effect_dispatches_per_pass: usize,
    /// Maximum simultaneously in-flight core effects owned by one run.
    pub per_run_effect_limit: usize,
    /// Maximum simultaneously in-flight core effects owned by one agent.
    pub per_agent_effect_limit: usize,
    /// Maximum simultaneously in-flight core effects owned by one tenant.
    pub per_tenant_effect_limit: usize,
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
            max_effect_dispatches_per_pass: 256,
            per_run_effect_limit: 64,
            per_agent_effect_limit: 256,
            per_tenant_effect_limit: 1_024,
        }
    }
}
