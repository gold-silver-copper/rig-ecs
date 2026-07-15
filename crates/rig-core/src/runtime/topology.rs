//! Agent, capability, grant, and runtime relationship topology.
//!
//! Relationship source components are authoritative. Target collections keep
//! their storage private while remaining queryable through Bevy's
//! [`RelationshipTarget`](bevy_ecs::relationship::RelationshipTarget) API.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use super::{CanonicalError, RetrievedDocument};

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

/// Per-run semantic selection of a subset of otherwise executable tool entities.
#[derive(Component, Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolRetrievalRequirement {
    /// Maximum tool names requested from the configured tool-search store.
    pub limit: usize,
    /// Provider-facing names eligible for semantic selection.
    pub candidates: Vec<String>,
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

/// Explicit execution dependency that prevents a parent from advancing while
/// one or more delegated child results are still uncommitted.
///
/// The related [`ChildRuns`] collection is the source of truth for the actual
/// dependency set; this marker makes the parent's readiness directly queryable
/// without copying entity identifiers into another component.
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WaitingForChildren;

/// Deterministic creation order for child-result reduction.
#[derive(Component, Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ChildOrdinal(pub u64);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ChildResultCommitted;

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

/// Relationship from a run-local policy instance to the run it governs.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = RunPolicies)]
pub struct PolicyForRun(pub Entity);

/// Policy instances that apply only to one run and its future operations.
#[derive(Component, Debug)]
#[relationship_target(relationship = PolicyForRun, linked_spawn)]
pub struct RunPolicies(Vec<Entity>);

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
