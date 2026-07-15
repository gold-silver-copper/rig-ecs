//! Typed policy facts, durable evaluations, steering decisions, and lifecycle events.
//!
//! Serialized [`PolicyRule`] values are installation input only. Runtime
//! selection and enforcement use typed components and exact registered-system
//! bindings so observer registration order cannot affect steering.

use super::*;

/// Authoritative ordering and revision facts for an ECS policy entity.
#[derive(Component, Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[component(immutable)]
pub struct PolicyMeta {
    /// Explicit composition order. Stable ID breaks ties.
    pub order: u32,
    /// Immutable policy revision recorded by accepted decisions.
    pub revision: u64,
}

/// Ergonomic and serialized policy installation input.
///
/// The add lifecycle materializes [`PolicyMeta`], [`PolicyCapabilities`], and
/// the matching typed policy component. Core selection and steering never read
/// this component after installation.
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

/// Policy definition installed atomically with one newly admitted run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunPolicySpec {
    /// Persistent identity unique within the runtime world.
    pub id: StableId,
    /// Immutable ordered policy definition.
    pub policy: Policy,
}

impl RunPolicySpec {
    /// Creates a run-local policy specification.
    pub fn new(id: StableId, policy: Policy) -> Self {
        Self { id, policy }
    }
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

/// Relationship from one explicit steering-responder binding to its policy.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[relationship(relationship_target = PolicyResponders)]
pub struct PolicyResponderFor(pub Entity);

/// Explicit steering-responder bindings owned by a policy entity.
#[derive(Component, Debug)]
#[relationship_target(relationship = PolicyResponderFor)]
pub struct PolicyResponders(Vec<Entity>);

/// Lifecycle point handled by a responder binding entity.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct ResponderPoint(pub PolicyPoint);

/// Stable identity used to reconstruct an extension-owned responder binding.
#[derive(Component, Clone, Debug, Eq, Hash, PartialEq)]
#[component(immutable)]
pub struct PolicyResponderId(String);

impl PolicyResponderId {
    pub(super) fn generated(value: String) -> Self {
        debug_assert!(!value.is_empty());
        Self(value)
    }

    /// Creates a stable responder-binding identity.
    pub fn new(value: impl Into<String>) -> Result<Self, IdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(IdentityError::Empty);
        }
        Ok(Self(value))
    }

    /// Returns the serialized binding identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque Bevy registered-system entity invoked for one policy point.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[component(immutable)]
pub(super) struct RegisteredPolicyResponder(pub(super) Entity);

pub(super) fn accepts_new_policy_evaluations(status: Option<&PolicyStatus>) -> bool {
    !matches!(status, Some(PolicyStatus::Retired))
}

pub(super) fn policy_entities<'a>(
    agent: Option<&'a AgentPolicies>,
    run: Option<&'a RunPolicies>,
) -> impl Iterator<Item = Entity> + 'a {
    agent
        .into_iter()
        .flat_map(AgentPolicies::iter)
        .chain(run.into_iter().flat_map(RunPolicies::iter))
}

/// Ergonomic and serialized policy input.
///
/// Installation materializes this value into typed ECS policy components and
/// [`PolicyCapabilities`]. Runtime selection and steering query those typed
/// facts rather than dispatching on this enum.
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
        presentation: ToolOutput,
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
    /// Stop a streaming run when a tool-call delta contains a substring.
    StopToolCallDeltaContains {
        /// Substring matched against a tool name or argument fragment.
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

/// Lifecycle capabilities materialized on a policy entity.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct PolicyCapabilities(Vec<PolicyPoint>);

impl PolicyCapabilities {
    /// Creates a deterministic, duplicate-free capability set.
    pub fn new(points: impl IntoIterator<Item = PolicyPoint>) -> Self {
        let mut points = points.into_iter().collect::<Vec<_>>();
        points.sort();
        points.dedup();
        Self(points)
    }

    /// Returns whether this policy participates at `point`.
    pub fn contains(&self, point: PolicyPoint) -> bool {
        self.0.binary_search(&point).is_ok()
    }

    /// Borrows the deterministic lifecycle-point set.
    pub fn points(&self) -> &[PolicyPoint] {
        &self.0
    }
}

/// Unconditional request progression.
#[derive(Component, Clone, Copy, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AllowRequestPolicy;

/// Request denial based on canonical prompt text.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct DenyPromptContainsPolicy(pub String);

/// Operation-local request patch contribution.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct RequestPatchPolicy(pub RequestPatch);

/// Typed tool-argument rewrite policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct RewriteToolArgumentsPolicy {
    /// Optional tool name; `None` applies to every call.
    pub tool: Option<String>,
    /// Replacement arguments.
    pub arguments: serde_json::Value,
}

/// Typed tool-call skip policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct SkipToolCallPolicy {
    /// Optional tool name; `None` applies to every call.
    pub tool: Option<String>,
    /// Model-visible feedback.
    pub reason: String,
}

/// Typed invalid-tool repair policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct RepairInvalidToolPolicy {
    /// Optional emitted name; `None` applies to every invalid call.
    pub from: Option<String>,
    /// Replacement advertised tool name.
    pub to: String,
}

/// Typed invalid-tool retry policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct RetryInvalidToolPolicy {
    /// Optional emitted name; `None` applies to every invalid call.
    pub tool: Option<String>,
    /// Corrective model feedback.
    pub feedback: String,
}

/// Typed invalid-tool skip policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct SkipInvalidToolPolicy {
    /// Optional emitted name; `None` applies to every invalid call.
    pub tool: Option<String>,
    /// Model-visible feedback.
    pub reason: String,
}

/// Typed tool-result presentation rewrite policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct RewriteToolResultPolicy {
    /// Optional tool name; `None` applies to every result.
    pub tool: Option<String>,
    /// Replacement typed presentation.
    pub presentation: ToolOutput,
}

/// Typed tool-result stop policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct StopToolResultPolicy {
    /// Optional tool name; `None` applies to every result.
    pub tool: Option<String>,
    /// Audit reason.
    pub reason: String,
}

/// Typed completion-text rewrite policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct RewriteCompletionTextPolicy(pub String);

/// Typed completion-response stop policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct StopCompletionContainsPolicy {
    /// Text matched against normalized completion output.
    pub needle: String,
    /// Audit reason.
    pub reason: String,
}

/// Typed streaming text-delta stop policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct StopTextDeltaContainsPolicy {
    /// Text matched against the new delta.
    pub needle: String,
    /// Audit reason.
    pub reason: String,
}

/// Typed streaming tool-call-delta stop policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct StopToolCallDeltaContainsPolicy {
    /// Text matched against a name or argument fragment.
    pub needle: String,
    /// Audit reason.
    pub reason: String,
}

/// Typed asynchronous approval policy.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct RequireApprovalPolicy {
    /// Lifecycle point awaiting approval.
    pub point: PolicyPoint,
    /// Approver-facing prompt.
    pub prompt: String,
}

/// Marks a lifecycle point whose steering responder is extension-owned.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
#[component(immutable)]
pub struct CustomPolicy(pub PolicyPoint);

/// Lifecycle point governed by a custom policy entity.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
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
    /// Before an accepted tool-call delta is published to subscribers.
    ToolCallDelta,
}

impl PolicyPoint {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::ToolCall => "tool-call",
            Self::InvalidToolCall => "invalid-tool-call",
            Self::ToolResult => "tool-result",
            Self::CompletionResponse => "completion-response",
            Self::TextDelta => "text-delta",
            Self::ToolCallDelta => "tool-call-delta",
        }
    }
}

impl PolicyRule {
    pub(super) fn point(&self) -> PolicyPoint {
        match self {
            Self::Allow | Self::DenyPromptContains(_) | Self::PatchRequest(_) => {
                PolicyPoint::Request
            }
            Self::RewriteToolArguments { .. } | Self::SkipToolCall { .. } => PolicyPoint::ToolCall,
            Self::RepairInvalidTool { .. }
            | Self::RetryInvalidTool { .. }
            | Self::SkipInvalidTool { .. } => PolicyPoint::InvalidToolCall,
            Self::RewriteToolResult { .. } | Self::StopToolResult { .. } => PolicyPoint::ToolResult,
            Self::RewriteCompletionText { .. } | Self::StopCompletionContains { .. } => {
                PolicyPoint::CompletionResponse
            }
            Self::StopTextDeltaContains { .. } => PolicyPoint::TextDelta,
            Self::StopToolCallDeltaContains { .. } => PolicyPoint::ToolCallDelta,
            Self::RequireApproval { point, .. } | Self::Custom(point) => *point,
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
    /// Replacement canonical history preceding the current prompt.
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

    /// Replaces canonical prior history while retaining the current prompt.
    pub fn history<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = TranscriptEntry>,
    {
        self.history = Some(values.into_iter().collect());
        self
    }
}

/// Native bundle shared by the ergonomic policy bundles below.
#[derive(Bundle, Clone, Debug)]
pub struct PolicyEntityBundle {
    /// Persistent policy identity.
    pub id: StableId,
    /// Tenant boundary shared with the governed agent.
    pub tenant: TenantId,
    /// Immutable ordered policy data.
    pub policy: Policy,
    /// Admission state for future evaluations.
    pub status: PolicyStatus,
    /// Agent governed by this policy entity.
    pub policy_for: PolicyFor,
}

impl PolicyEntityBundle {
    fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        rule: PolicyRule,
    ) -> Self {
        Self {
            id,
            tenant,
            policy: Policy {
                order,
                revision,
                rule,
            },
            status: PolicyStatus::Enabled,
            policy_for: PolicyFor(agent),
        }
    }
}

macro_rules! policy_bundle {
    ($name:ident, $docs:literal) => {
        #[doc = $docs]
        #[derive(Bundle, Clone, Debug)]
        pub struct $name {
            /// Native policy entity components.
            pub policy: PolicyEntityBundle,
        }

        impl RigExtension for $name {
            fn install(&self, world: &mut World) -> Result<(), ExtensionInstallError> {
                install_policy_extension(world, self.policy.clone())
            }
        }
    };
}

policy_bundle!(
    RequestPatchPolicyBundle,
    "An ordered, non-sticky completion-request patch policy."
);

impl RequestPatchPolicyBundle {
    /// Creates a request-patch policy related to `agent`.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        patch: RequestPatch,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::PatchRequest(patch),
            ),
        }
    }
}

policy_bundle!(
    ToolApprovalPolicyBundle,
    "An asynchronous approval policy evaluated before tool dispatch."
);

impl ToolApprovalPolicyBundle {
    /// Creates a tool-call approval policy related to `agent`.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        prompt: impl Into<String>,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::RequireApproval {
                    point: PolicyPoint::ToolCall,
                    prompt: prompt.into(),
                },
            ),
        }
    }
}

policy_bundle!(
    ToolArgumentRewritePolicyBundle,
    "An ordered tool-argument rewrite policy."
);

impl ToolArgumentRewritePolicyBundle {
    /// Creates an argument rewrite for one named tool or every tool when `tool` is `None`.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        tool: Option<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::RewriteToolArguments { tool, arguments },
            ),
        }
    }
}

policy_bundle!(
    ToolSkipPolicyBundle,
    "A policy that skips matching tool calls."
);

impl ToolSkipPolicyBundle {
    /// Creates a model-visible skip policy for one named tool or every tool.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        tool: Option<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::SkipToolCall {
                    tool,
                    reason: reason.into(),
                },
            ),
        }
    }
}

policy_bundle!(
    ToolResultRedactionPolicyBundle,
    "A policy that rewrites model-visible tool output while preserving raw audit data."
);

impl ToolResultRedactionPolicyBundle {
    /// Creates a result-presentation rewrite for one named tool or every tool.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        tool: Option<String>,
        presentation: impl Into<ToolOutput>,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::RewriteToolResult {
                    tool,
                    presentation: presentation.into(),
                },
            ),
        }
    }
}

policy_bundle!(
    TextDeltaStopPolicyBundle,
    "A policy that stops streaming before publishing matching text deltas."
);

impl TextDeltaStopPolicyBundle {
    /// Creates a text-delta guard for one agent.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        needle: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::StopTextDeltaContains {
                    needle: needle.into(),
                    reason: reason.into(),
                },
            ),
        }
    }
}

policy_bundle!(
    ToolCallDeltaStopPolicyBundle,
    "A policy that stops streaming before publishing matching tool-call deltas."
);

impl ToolCallDeltaStopPolicyBundle {
    /// Creates a tool-call-delta guard for one agent.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        needle: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::StopToolCallDeltaContains {
                    needle: needle.into(),
                    reason: reason.into(),
                },
            ),
        }
    }
}

policy_bundle!(
    InvalidToolRepairPolicyBundle,
    "A policy that repairs a model-emitted tool name against the accepted snapshot."
);

impl InvalidToolRepairPolicyBundle {
    /// Creates an invalid-tool repair policy.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        from: Option<String>,
        to: impl Into<String>,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::RepairInvalidTool {
                    from,
                    to: to.into(),
                },
            ),
        }
    }
}

policy_bundle!(
    InvalidToolRetryPolicyBundle,
    "A policy that feeds an invalid call back to the model under explicit budgets."
);

impl InvalidToolRetryPolicyBundle {
    /// Creates an invalid-tool retry policy.
    pub fn new(
        id: StableId,
        tenant: TenantId,
        agent: Entity,
        order: u32,
        revision: u64,
        tool: Option<String>,
        feedback: impl Into<String>,
    ) -> Self {
        Self {
            policy: PolicyEntityBundle::new(
                id,
                tenant,
                agent,
                order,
                revision,
                PolicyRule::RetryInvalidTool {
                    tool,
                    feedback: feedback.into(),
                },
            ),
        }
    }
}

/// Extension installation failure that leaves the existing world unchanged.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ExtensionInstallError {
    /// The world has not had [`install_runtime`] applied.
    #[error("Rig runtime is not installed in this world")]
    RuntimeNotInstalled,
    /// The policy stable identity already exists.
    #[error("duplicate stable id `{0}`")]
    DuplicateStableId(String),
    /// The target is not a live agent entity.
    #[error("policy target {0:?} is not a live agent")]
    StaleAgent(Entity),
    /// Policy and agent tenant scopes differ.
    #[error("policy and target agent cross tenant scope")]
    TenantMismatch,
}

/// Thin installer for extensions composed exclusively from Bevy ECS primitives.
pub trait RigExtension {
    /// Installs components, bundles, observers, systems, resources, or schedule configuration.
    fn install(&self, world: &mut World) -> Result<(), ExtensionInstallError>;
}

pub(super) fn install_policy_extension(
    world: &mut World,
    bundle: PolicyEntityBundle,
) -> Result<(), ExtensionInstallError> {
    if !world.contains_resource::<RuntimeIdentity>() {
        return Err(ExtensionInstallError::RuntimeNotInstalled);
    }
    let mut identities = world.query::<&StableId>();
    if identities.iter(world).any(|id| id == &bundle.id) {
        return Err(ExtensionInstallError::DuplicateStableId(
            bundle.id.as_str().to_owned(),
        ));
    }
    let agent = bundle.policy_for.get();
    let Some(agent_tenant) = world.get::<TenantId>(agent) else {
        return Err(ExtensionInstallError::StaleAgent(agent));
    };
    if world.get::<Agent>(agent).is_none() {
        return Err(ExtensionInstallError::StaleAgent(agent));
    }
    if agent_tenant != &bundle.tenant {
        return Err(ExtensionInstallError::TenantMismatch);
    }
    world.spawn(bundle);
    Ok(())
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
/// The runtime passes this value only to the responder explicitly bound to the
/// selected policy. Audit observers consume the events published after reduction.
#[derive(Clone, Debug)]
pub struct RequestPolicyInvocation {
    /// Policy entity selected for this invocation.
    pub policy: Entity,
    /// Durable evaluation entity owning the cursor and effective request.
    pub evaluation: Entity,
    /// Run being evaluated.
    pub run: Entity,
    /// Model operation awaiting a finalized request.
    pub operation: Entity,
    /// Current effective request, including all earlier rewrites.
    pub request: ModelEffectInput,
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
    /// Explicit composition order accepted for this evaluation.
    pub order: u32,
    /// Exact lifecycle capability accepted for this evaluation.
    pub point: PolicyPoint,
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
    /// Deterministically reduced operation-local patch retained for audit and resume.
    pub accumulated: RequestPatch,
    /// Authoritative evaluation phase.
    pub phase: RequestPolicyEvaluationPhase,
}

#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub(super) struct PendingModelRequest(pub(super) ModelEffectInput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RequestPolicyInitialized;

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

/// Typed input for one tool call and its explicitly bound policy responder.
#[derive(Clone, Debug)]
pub struct ToolCallPolicyInvocation {
    /// Policy entity selected by the core evaluator.
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Run containing the logical tool batch.
    pub run: Entity,
    /// Tool operation awaiting finalized arguments.
    pub operation: Entity,
    /// Effective call including all earlier rewrites.
    pub call: ToolEffectInput,
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
pub(super) struct PendingToolCall(pub(super) ToolEffectInput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ToolCallPolicyInitialized;

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

/// Typed input for one invalid call and its explicitly bound policy responder.
#[derive(Clone, Debug)]
pub struct InvalidToolCallPolicyInvocation {
    /// Policy entity selected by the core evaluator.
    pub policy: Entity,
    /// Durable evaluation identity.
    pub evaluation: Entity,
    /// Run containing the invalid call.
    pub run: Entity,
    /// Operation retaining the invalid-call context.
    pub operation: Entity,
    /// Immutable invalid-call context.
    pub invalid: PendingInvalidToolCall,
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
pub(super) struct InvalidToolCallPolicyInitialized;

pub(super) enum PreparedToolOperation {
    Valid(ToolEffectInput),
    Invalid(PendingInvalidToolCall),
}

/// Typed result returned by a tool-result policy observer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolResultPolicyDecision {
    /// Preserve the current effective presentation.
    Keep,
    /// Replace only the model-visible presentation.
    Rewrite(ToolOutput),
    /// Stop the run without publishing raw result content.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Typed input for one settled tool result and its explicitly bound responder.
#[derive(Clone, Debug)]
pub struct ToolResultPolicyInvocation {
    /// Policy entity selected by the core evaluator.
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
pub(super) struct ToolResultPolicyInitialized;

#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub(super) struct EffectiveToolOutput(pub(super) ToolEffectOutput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ToolResultPolicyDone;

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

/// Typed input for one settled model response and its explicitly bound responder.
#[derive(Clone, Debug)]
pub struct CompletionResponsePolicyInvocation {
    /// Policy entity selected by the core evaluator.
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
pub(super) struct CompletionResponsePolicyInitialized;

#[derive(Component, Clone, Debug, Eq, PartialEq)]
#[component(immutable)]
pub(super) struct EffectiveModelOutput(pub(super) ModelEffectOutput);

#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompletionResponsePolicyDone;

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
    /// Content-free execution status safe for telemetry before result policy.
    pub status: ToolExecutionStatus,
}

/// Content-free tool settlement metadata safe for telemetry publication.
///
/// The immutable [`ToolEffectOutput`] remains on the operation for explicitly
/// authorized audit and policy queries. Keeping it out of observation events
/// prevents stopped or redacted results from being copied into telemetry sinks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolExecutionStatus {
    /// Whether the tool returned without any execution failure metadata.
    pub succeeded: bool,
    /// Stable error classification, when one is available.
    pub failure_kind: Option<crate::tool::ToolErrorKind>,
    /// Retry classification without an operator-facing diagnostic string.
    pub retryable: Option<bool>,
    /// Whether the tool intentionally refused the call.
    pub refusal: bool,
}

/// Observe-only notification after tool-result presentation policy finalizes.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolResultPresentationFinalized {
    /// Tool operation.
    #[event_target]
    pub operation: Entity,
    /// Owning run.
    pub run: Entity,
    /// Content-free execution status retained alongside the presentation.
    pub status: ToolExecutionStatus,
    /// Final policy-approved model and telemetry presentation.
    pub presentation: String,
}

/// Policy-approved tool result safe to copy into lifecycle telemetry.
///
/// Raw result content and operator-facing failure messages remain immutable
/// operation audit state and are intentionally absent from this value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedToolResult {
    /// Provider correlation identifier copied from the call.
    pub call_id: String,
    /// Provider-facing result identifier.
    pub provider_result_id: String,
    /// Separate provider call identifier, when present.
    pub provider_call_id: Option<String>,
    /// Provider-facing tool name.
    pub name: String,
    /// Final policy-approved presentation.
    pub presentation: String,
    /// Content-free execution status.
    pub status: ToolExecutionStatus,
}

/// Observe-only notification after an atomic tool batch commits successfully.
#[derive(EntityEvent, Clone, Debug)]
pub struct ToolBatchCommitted {
    /// Tool-batch entity.
    #[event_target]
    pub batch: Entity,
    /// Owning run.
    pub run: Entity,
    /// Policy-approved, content-safe results in logical call order.
    pub results: Vec<PublishedToolResult>,
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

/// Typed input for one accepted, not-yet-published text delta.
#[derive(Clone, Debug)]
pub struct TextDeltaPolicyInvocation {
    /// Policy entity selected by the core evaluator.
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

/// Typed steering result for a streamed tool-call delta.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolCallDeltaPolicyDecision {
    /// Publish the accepted delta.
    Continue,
    /// Stop the run before publishing this delta.
    Stop(String),
    /// Suspend evaluation on an external approval operation.
    AwaitApproval(String),
}

/// Typed input for one accepted tool-call delta.
#[derive(Clone, Debug)]
pub struct ToolCallDeltaPolicyInvocation {
    /// Policy entity selected by the evaluator.
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
    /// Provider correlation retained for audit and policy.
    pub provider_correlation: Option<String>,
    /// Provider-facing tool-call identifier.
    pub id: String,
    /// Rig correlation identifier.
    pub internal_call_id: String,
    /// Tool name or argument fragment.
    pub content: ToolCallDeltaContent,
}

/// Authoritative tool-call-delta evaluation phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolCallDeltaPolicyEvaluationPhase {
    /// The policy at `cursor` is ready to run.
    Evaluating,
    /// Waiting for a correlated approval operation.
    WaitingApproval { operation: Entity, policy: StableId },
    /// The delta was published once.
    Published,
    /// A policy stopped the run.
    Rejected { policy: StableId, reason: String },
}

/// Durable ordered tool-call-delta policy state.
#[derive(Component, Clone, Debug, Eq, PartialEq)]
pub struct ToolCallDeltaPolicyEvaluation {
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
    /// Provider correlation retained for audit and policy.
    pub provider_correlation: Option<String>,
    /// Provider-facing tool-call identifier.
    pub id: String,
    /// Rig correlation identifier.
    pub internal_call_id: String,
    /// Tool name or argument fragment.
    pub content: ToolCallDeltaContent,
    /// Authoritative evaluation phase.
    pub phase: ToolCallDeltaPolicyEvaluationPhase,
}

/// Immutable tool-call-delta policy snapshot accepted for one delta.
#[derive(Component, Clone, Debug, Default, Eq, PartialEq)]
#[component(immutable)]
pub struct AcceptedToolCallDeltaPolicies(pub Vec<AcceptedPolicy>);
