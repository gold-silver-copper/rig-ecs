# Rig as a Bevy ECS-Native Agent Runtime

**Status:** Target architecture and cutover specification  
**Scope:** `rig-core`  
**Migration policy:** Direct replacement; no compatibility layer, dual runtime, or staged public API transition

## Executive decision

Rig will make [`bevy_ecs`](https://github.com/bevyengine/bevy/tree/493a2f477b3bcc1b52d45e2359fc7e26d7037b27/crates/bevy_ecs) a required dependency of `rig-core` and adopt its model as the foundation of the runtime:

- entities provide identity;
- components hold runtime data and capabilities;
- systems contain behavior;
- schedules define ordering and safe parallelism;
- resources hold singleton runtime services;
- relationships connect agents, models, tools, stores, runs, and calls;
- buffered messages move asynchronous results between schedule boundaries;
- deferred commands apply structural changes atomically;
- change detection drives incremental recomputation;
- bundles and plugins are the construction and extension mechanisms.

The ECS world will be the single authoritative runtime state. The convenient `agent.prompt(...)` experience will remain possible, but it will be a facade over entities and schedules—not a second execution engine.

The migration deliberately prioritizes the ideal architecture over source compatibility. Existing APIs that duplicate ECS responsibilities will be removed rather than deprecated or mirrored.

## Why ECS belongs in `rig-core`

Rig is evolving from a collection of typed provider clients into a runtime that coordinates:

- heterogeneous models;
- dynamic tools and MCP servers;
- databases and memory stores;
- retrieval and policy;
- concurrent agent runs;
- tool-call batches;
- cancellation and retries;
- telemetry and auditing;
- long-lived, dynamically changing infrastructure.

These are identity- and lifecycle-heavy concerns. Treating them as nested builder fields and registries creates ownership, synchronization, discovery, and extensibility problems. ECS makes identity, composition, lifecycle, querying, mutation, and scheduling explicit.

Bevy ECS is suitable as a standalone crate, not only as part of the Bevy game engine. Its core concepts and standalone intent are documented in the [Bevy ECS README](https://github.com/bevyengine/bevy/blob/493a2f477b3bcc1b52d45e2359fc7e26d7037b27/crates/bevy_ecs/README.md#L10-L117).

## Goals

1. Make `World` the sole source of truth for live runtime state.
2. Represent agents, models, tools, stores, runs, model calls, and tool calls as entities.
3. Represent capabilities and configuration as independently queryable components.
4. Express runtime behavior as systems in explicitly ordered schedules.
5. Use one execution engine for local, hosted, blocking, and streaming APIs.
6. Preserve typed authoring while using private erasure at heterogeneous runtime boundaries.
7. Never borrow the ECS world across `.await`.
8. Preserve deterministic transcript and tool ordering under concurrency.
9. Make extensions normal systems and plugins rather than privileged hook traits.
10. Keep secrets and global services in resources, not ordinary components.
11. Make dynamic infrastructure updates observable through change detection.
12. Support native embedding into an existing Bevy ECS world.

## Non-goals

1. Preserving the current `Agent<M>`, `AgentBuilder`, `AgentRunner`, `ToolSet`, `ToolServer`, or `HookStack` APIs.
2. Providing an `ecs` Cargo feature that leaves the old runtime available.
3. Running two registries or synchronizing a classic object graph with an ECS world.
4. Turning every token, message part, JSON value, or provider chunk into an entity.
5. Giving asynchronous tools or models `&mut World` access.
6. Persisting raw Bevy `Entity` identifiers.
7. Using ECS archetype performance as the primary justification; the primary benefits are composition, lifecycle correctness, extensibility, and safe scheduling.
8. Depending on `bevy_app` in `rig-core`. A separate integration can embed Rig schedules into a full Bevy `App`.

## Hard architectural rules

### One world

Every live runtime object exists in one `World`. There is no parallel `ToolSet`, model registry, agent registry, memory registry, or runner-owned copy of authoritative state.

### One execution engine

All user-facing paths drive the same schedules:

- local prompt;
- hosted runtime prompt;
- blocking completion;
- streaming completion;
- manual stepping;
- tests;
- embedded Bevy application.

Convenience APIs may hide the world, but may not implement execution separately.

### No world borrow across `.await`

Systems snapshot owned inputs, submit effects to an executor resource, mark entities in flight, and return. Effect completions re-enter through a thread-safe inbox and are applied by later systems.

### Determinism is explicit

ECS query order is never used as semantic order. Tool registration order, model-call order, and tool-call order are represented by explicit ordinal components and sorted before provider presentation or transcript commit.

### Runtime entities are not persistent identities

Every persistable entity has a stable application ID such as a UUID. Raw Bevy `Entity` values are process-local handles and never cross persistence, network, or public protocol boundaries.

### Components are cohesive units

ECS-native does not mean one component per scalar field. Data that shares invariants or mutation patterns stays together. Components are split when systems need independent access, querying, replacement, or change detection.

## Dependency and re-export policy

`bevy_ecs` is a normal, required dependency of `rig-core`.

Rig re-exports its exact version so extension crates use compatible ECS types:

```rust
pub mod ecs {
    pub use bevy_ecs::*;
}

pub mod ecs_prelude {
    pub use bevy_ecs::prelude::*;
    pub use crate::bundles::*;
    pub use crate::components::*;
    pub use crate::relationships::*;
    pub use crate::runtime::{RigApp, RigPlugin, RigSet};
}
```

Rig plugins should import ECS types through `rig::ecs` or `rig_core::ecs`, preventing accidental duplicate Bevy versions and incompatible `Entity`/`Component` identities.

`bevy_app` remains outside `rig-core`. A future `rig-bevy` crate can install Rig schedules and resources into an existing Bevy application.

## Core runtime object

```rust
pub struct RigApp {
    world: World,
    schedule: Schedule,
    plugins: PluginRegistry,
}
```

`RigApp` exposes:

```rust
impl RigApp {
    pub fn new() -> Self;
    pub fn world(&self) -> &World;
    pub fn world_mut(&mut self) -> &mut World;
    pub fn add_plugins<P: RigPlugins>(&mut self, plugins: P) -> &mut Self;
    pub fn update(&mut self) -> UpdateStatus;
    pub async fn run_until(&mut self, run: RunId) -> Result<PromptResponse, RunError>;
}
```

`update` drains external commands and effect completions, executes the Rig schedule until quiescent, and reports whether the world has immediate work or is waiting for external effects.

A hosted `RigRuntime` owns a `RigApp` on a dedicated runtime task. Cloneable handles communicate through a command inbox:

```rust
pub struct RigRuntimeHandle { /* command sender + wake handle */ }
pub struct AgentHandle { runtime: RigRuntimeHandle, entity: Entity }
pub struct RunHandle { runtime: RigRuntimeHandle, stable_id: StableRunId }
```

The local and hosted forms differ only in world ownership. Both execute the same systems.

## Entity taxonomy

The initial entity categories are:

```rust
#[derive(Component)] pub struct Agent;
#[derive(Component)] pub struct Model;
#[derive(Component)] pub struct Tool;
#[derive(Component)] pub struct Store;
#[derive(Component)] pub struct McpServer;
#[derive(Component)] pub struct Run;
#[derive(Component)] pub struct Turn;
#[derive(Component)] pub struct ModelCall;
#[derive(Component)] pub struct ToolCall;
#[derive(Component)] pub struct ToolGrant;
```

Marker components identify broad categories. Capabilities and relationships provide the meaningful behavior.

## Stable identity

```rust
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StableId(pub Uuid);

pub struct AgentId(pub Entity);
pub struct ModelId(pub Entity);
pub struct ToolId(pub Entity);
pub struct StoreId(pub Entity);
pub struct RunId(pub Entity);
```

Typed entity wrappers are ergonomic domain identifiers. Components and relationships may store raw `Entity` internally where required by Bevy APIs. Serialization uses `StableId`, never `Entity`.

Loading persisted state is a two-pass process:

1. spawn entities and build `StableId -> Entity`;
2. restore relationships using that map.

## Agent entities

An agent is configuration plus relationships, not a generic object containing every dependency.

```rust
#[derive(Component)]
pub struct AgentIdentity {
    pub name: Option<String>,
    pub description: Option<String>,
}

#[derive(Component)]
pub struct Instructions {
    pub preamble: Option<String>,
    pub static_context: Vec<Document>,
}

#[derive(Component)]
pub struct SamplingConfig {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<serde_json::Value>,
}

#[derive(Component)]
pub struct AgentBudget {
    pub max_turns: usize,
    pub max_invalid_tool_call_retries: usize,
}

#[derive(Component)]
pub struct StructuredOutputConfig {
    pub schema: Option<schemars::Schema>,
    pub mode: OutputMode,
}
```

Relationships attach the model and stores:

```rust
#[derive(Component)]
#[relationship(relationship_target = AgentsUsingModel)]
pub struct UsesModel(pub Entity);

#[derive(Component)]
#[relationship_target(relationship = UsesModel)]
pub struct AgentsUsingModel(Vec<Entity>);

#[derive(Component)]
#[relationship(relationship_target = AgentsUsingMemory)]
pub struct UsesMemory(pub Entity);
```

Common agent construction uses a bundle:

```rust
#[derive(Bundle)]
pub struct AgentBundle {
    pub marker: Agent,
    pub stable_id: StableId,
    pub identity: AgentIdentity,
    pub instructions: Instructions,
    pub sampling: SamplingConfig,
    pub budget: AgentBudget,
    pub output: StructuredOutputConfig,
    pub model: UsesModel,
}
```

`Agent<M>` is removed. The model type belongs to the model entity, not the agent type.

## Model entities

Typed models are preserved at insertion boundaries:

```rust
let model = app.spawn_model(openai_client.completion_model("gpt-5"));
```

Internally, each model entity contains a private erased driver:

```rust
#[derive(Component)]
struct ModelDriver(Arc<dyn ErasedCompletionModel>);

#[derive(Component)]
pub struct ModelIdentity {
    pub provider: String,
    pub model: String,
}

#[derive(Component)]
pub struct ModelCapabilities {
    pub streaming: bool,
    pub tools: bool,
    pub native_structured_output: bool,
    pub multimodal: bool,
}

#[derive(Component)]
pub struct ModelHealth {
    pub status: HealthStatus,
    pub last_checked: Option<Instant>,
}
```

The private erased driver accepts canonical requests and starts effects that return canonical model results. Typed provider responses are preserved as typed effect metadata where possible and serialized metadata where persistence is required.

The public `CompletionModel` trait remains a typed authoring and direct-provider interface. A blanket adapter converts it into a model bundle.

## Tool entities

Rig's tool architecture provides the correct typed-authoring/private-erasure boundary. ECS registration makes the erased executor an entity component.

```rust
#[derive(Component)]
pub struct ToolIdentity {
    pub name: String,
    pub registration_order: u64,
}

#[derive(Component, Clone)]
pub struct ToolSpec {
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Component)]
struct ToolExecutor(Arc<dyn ErasedTool>);

#[derive(Component)]
pub struct ToolStatus {
    pub enabled: bool,
    pub health: HealthStatus,
}

#[derive(Component)]
pub struct ToolRevision(pub u64);

#[derive(Bundle)]
pub struct ToolBundle {
    pub marker: Tool,
    pub stable_id: StableId,
    pub identity: ToolIdentity,
    pub spec: ToolSpec,
    executor: ToolExecutor,
    pub status: ToolStatus,
    pub revision: ToolRevision,
}
```

A typed tool is spawned directly:

```rust
let search = app.spawn_tool(SearchTool::new(client));
```

The `Tool` trait remains an ergonomic typed authoring interface. Registration consumes a typed tool and creates a `ToolBundle`. `ToolSet` and `ToolServerHandle` are removed.

### Runtime-defined tools

`DynamicTool` remains a construction type, but registration immediately spawns a normal tool entity. Static, dynamic, retrieved, and MCP tools share the same entity representation after insertion.

### MCP ownership

```rust
#[derive(Component)]
#[relationship(relationship_target = HostedTools)]
pub struct HostedBy(pub Entity);

#[derive(Component)]
#[relationship_target(relationship = HostedBy, linked_spawn)]
pub struct HostedTools(Vec<Entity>);
```

MCP discovery and refresh mutate tool entities through deferred commands. Ownership, generation, liveness, and server ordering are components. A refresh computes changes outside the world, then applies one atomic command batch.

## Tool grants and policy

Agent-to-tool access is many-to-many, so it is represented by grant entities rather than a vector on the agent.

```rust
#[derive(Component)] pub struct GrantAgent(pub Entity);
#[derive(Component)] pub struct GrantTool(pub Entity);

#[derive(Component)]
pub struct GrantPolicy {
    pub priority: i32,
    pub expires_at: Option<SystemTime>,
    pub tenant: Option<TenantId>,
    pub permission: PermissionPolicy,
}

#[derive(Bundle)]
pub struct ToolGrantBundle {
    pub marker: ToolGrant,
    pub stable_id: StableId,
    pub agent: GrantAgent,
    pub tool: GrantTool,
    pub policy: GrantPolicy,
}
```

This makes access control, expiration, auditing, tenant scoping, and ordering independently queryable and change-detectable.

## Store and database entities

A database is not represented by a backend enum. A store entity carries one or more capability components:

```rust
#[derive(Component)]
pub struct ConversationMemoryDriver(Arc<dyn ConversationMemory>);

#[derive(Component)]
pub struct VectorSearchDriver(Arc<dyn VectorSearch>);

#[derive(Component)]
pub struct DocumentStoreDriver(Arc<dyn DocumentStore>);

#[derive(Component)]
pub struct SqlExecutorDriver(Arc<dyn SqlExecutor>);

#[derive(Component)]
pub struct StoreHealth {
    pub status: HealthStatus,
    pub latency: Option<Duration>,
}
```

A MongoDB entity may provide document storage and conversation memory. A Postgres entity may provide SQL execution, vector search, and memory. Systems query capabilities rather than matching concrete backend variants.

Singleton pools may be resources when there is exactly one process-wide instance. Addressable stores with identity, policy, health, or relationships are entities.

Secrets are never ordinary components. Store components carry opaque references into a `SecretStore` resource.

## Run entities

A prompt creates a run entity. Runs are the central lifecycle objects.

```rust
#[derive(Component)]
pub struct ForAgent(pub Entity);

#[derive(Component)]
pub struct Transcript(pub Vec<Message>);

#[derive(Component, Default)]
pub struct RunUsage(pub Usage);

#[derive(Component)]
pub struct RemainingBudget {
    pub turns: usize,
    pub invalid_tool_call_retries: usize,
}

#[derive(Component)]
pub struct RunScratchpad(TypeMap);

#[derive(Component)]
pub struct CancellationToken(/* runtime-neutral cancellation primitive */);
```

Initially, the existing sans-I/O `AgentRun` may be stored as one cohesive component:

```rust
#[derive(Component)]
struct RunState(AgentRun);
```

This is not a compatibility runtime. It is reuse of the canonical state machine as ECS-owned data. It can later be split only where independent component access is valuable and invariants remain explicit.

## Run phases

Lifecycle phases are marker components:

```rust
#[derive(Component)] pub struct ReadyForModel;
#[derive(Component)] pub struct WaitingForModel;
#[derive(Component)] pub struct ApplyingModelResult;
#[derive(Component)] pub struct ReadyForTools;
#[derive(Component)] pub struct WaitingForTools;
#[derive(Component)] pub struct ReadyToCommit;
#[derive(Component)] pub struct Persisting;
#[derive(Component)] pub struct RunCompleted;
#[derive(Component)] pub struct RunFailed;
#[derive(Component)] pub struct RunCancelled;
```

Systems query exactly the phase they process. Deferred commands replace markers at schedule barriers.

Only one primary phase marker may exist on a run. Debug validation systems assert this invariant.

## Turn entities and exact snapshots

Each accepted model turn gets a turn entity:

```rust
#[derive(Component)] pub struct ForRun(pub Entity);
#[derive(Component)] pub struct TurnIndex(pub u32);

#[derive(Component)]
pub struct ToolSnapshot {
    pub registry_revision: u64,
    pub tools: Vec<Entity>,
    pub definitions: Vec<crate::completion::ToolDefinition>,
}
```

The snapshot is built from enabled grants and tools, sorted by explicit registration/grant order. It captures both provider definitions and exact executable entities.

A tool mutation during an in-flight turn never changes that turn. It affects only snapshots built for later turns.

## Model-call entities

```rust
#[derive(Component)] pub struct ForTurn(pub Entity);
#[derive(Component)] pub struct ModelRequest(pub CompletionRequest);
#[derive(Component)] pub struct ModelCallIndex(pub u32);
#[derive(Component)] pub struct ModelCallInFlight;
#[derive(Component)] pub struct ModelCallResponse(pub ModelTurn);
#[derive(Component)] pub struct ModelCallFailure(pub CompletionError);
```

A dispatch system snapshots `ModelRequest` and `ModelDriver`, submits an async effect, marks the call in flight, and returns.

## Tool-call entities

```rust
#[derive(Component)] pub struct CallsTool(pub Entity);
#[derive(Component)] pub struct ToolCallIndex(pub u32);
#[derive(Component)] pub struct ToolArguments(pub serde_json::Value);
#[derive(Component)] pub struct InternalCallId(pub String);
#[derive(Component)] pub struct ToolCallInFlight;
#[derive(Component)] pub struct ToolCallResult(pub ToolResult);
#[derive(Component)] pub struct ToolDispatchMetadata(pub ToolContext);
```

Every model-emitted tool call has its own entity. Correlation never depends on a single pending slot or event arrival order.

Parallel execution may complete in any order. Commit systems wait until the complete batch is terminal, sort by `ToolCallIndex`, and commit atomically.

## Schedule

Rig defines one core schedule with ordered sets:

```rust
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RigSet {
    Ingress,
    Resolve,
    PolicyBeforeModel,
    PrepareModel,
    DispatchModel,
    ApplyModel,
    PolicyAfterModel,
    PrepareTools,
    PolicyBeforeTool,
    DispatchTools,
    ApplyTools,
    PolicyAfterTool,
    Commit,
    Persist,
    Observe,
    Cleanup,
    Flush,
}
```

Core sets are chained in semantic order. Systems inside a set run in parallel when their declared data access permits it. Extensions place systems in documented sets and use explicit ordering when composition order matters.

The final `Flush` barrier applies deferred structural changes and publishes outbound notifications.

## Core systems

The target runtime includes at least these systems:

```text
drain_runtime_commands
drain_effect_completions
validate_entity_invariants
resolve_agent_configuration
build_tool_snapshot
create_model_call
apply_before_model_policy
dispatch_model_calls
apply_model_completions
resolve_invalid_tool_calls
create_tool_calls
apply_before_tool_policy
dispatch_tool_calls
apply_tool_completions
apply_after_tool_policy
commit_tool_batches
advance_run_state
persist_transcripts
complete_runs
cancel_runs
cleanup_terminal_calls
cleanup_terminal_runs
publish_observations
```

Systems are small and capability-oriented. No single system recreates the old monolithic runner.

## Async effect boundary

Bevy systems are synchronous. Models, tools, MCP, and stores are asynchronous. Rig bridges them with an explicit effect layer.

```rust
#[derive(Resource)]
pub struct EffectExecutor {
    commands: EffectCommandSender,
}

pub enum EffectCommand {
    Model(ModelEffect),
    Tool(ToolEffect),
    Store(StoreEffect),
}

pub enum EffectCompletion {
    Model(ModelEffectCompletion),
    Tool(ToolEffectCompletion),
    Store(StoreEffectCompletion),
}
```

An effect contains only owned data, stable correlation IDs, and cloneable driver handles. It contains no `World`, `Query`, `Commands`, component reference, or system parameter.

Execution flow:

1. schedule system clones required inputs;
2. system submits an effect;
3. system inserts an in-flight marker;
4. executor performs asynchronous work;
5. executor sends an effect completion;
6. runtime wakes and drains completions;
7. apply system verifies correlation and entity generation;
8. apply system writes result components;
9. later systems commit state.

The executor is runtime-neutral. Native builds may use Tokio or another executor. WASM may use a local executor. Tests use a deterministic fake executor.

## Resources

Singleton runtime services are resources:

```rust
#[derive(Resource)] pub struct EffectExecutor(...);
#[derive(Resource)] pub struct RuntimeInbox(...);
#[derive(Resource)] pub struct RuntimeWake(...);
#[derive(Resource)] pub struct Clock(...);
#[derive(Resource)] pub struct IdGenerator(...);
#[derive(Resource)] pub struct SecretStore(...);
#[derive(Resource)] pub struct TelemetrySink(...);
#[derive(Resource)] pub struct RuntimeConfig(...);
#[derive(Resource)] pub struct RegistryRevision(pub u64);
```

Resources do not represent addressable domain objects. If users need multiple independently configured instances, those instances are entities.

## ToolContext in an ECS-native core

`ToolContext` remains a per-dispatch snapshot. Tools never receive `&mut World`.

Before dispatch, systems populate context with owned or cloneable values:

```rust
context.insert(RunEntity(run));
context.insert(AgentEntity(agent));
context.insert(TurnEntity(turn));
context.insert(TenantContext(...));
context.insert(AuthContext(...));
```

Tool-authored result metadata remains isolated in the returned dispatch context. The apply system attaches relevant metadata to the tool-call entity or makes it available to policy systems.

This preserves async safety, dispatch isolation, and the raw-versus-model-presentation separation of the tool runtime.

## Policy and extension systems

`AgentHook` and `HookStack` are removed.

Policy is represented by components that systems inspect or mutate:

```rust
#[derive(Component)] pub struct RequestPatch(...);
#[derive(Component)] pub struct ToolCallDecision(...);
#[derive(Component)] pub struct ToolPresentation(...);
#[derive(Component)] pub struct InvalidToolCallDecision(...);
#[derive(Component)] pub struct RunDecision(...);
```

Extensions register systems in policy sets:

```rust
app.add_systems(
    RigSet::PolicyBeforeTool,
    (tenant_guard, approval_policy, argument_normalizer).chain(),
);
```

Rewrites compose by mutating the current decision/presentation component in explicit system order. Stops and failures are terminal states validated by later systems.

Core correctness never depends on immediate observer ordering.

## Messages and observers

Buffered messages drive control and integration boundaries:

```rust
#[derive(Message)] pub struct RunRequested(...);
#[derive(Message)] pub struct ModelEffectFinished(...);
#[derive(Message)] pub struct ToolEffectFinished(...);
#[derive(Message)] pub struct StoreEffectFinished(...);
#[derive(Message)] pub struct RunFinished(...);
#[derive(Message)] pub struct RunFailedMessage(...);
```

Messages are consumed at fixed schedule points. This gives deterministic batching and ordering.

Observers are reserved for reactive notifications that do not own core state transitions:

- UI updates;
- metrics;
- tracing;
- audit sinks;
- debugging;
- external notifications.

An observer may enqueue a command or message, but does not directly mutate transcript or call-phase invariants.

## Change detection

Change detection replaces manual invalidation plumbing.

Examples:

- `Changed<ToolSpec>` increments registry revision;
- `Changed<ToolStatus>` invalidates future agent snapshots;
- added/removed `ToolGrant` entities update visibility indexes;
- `Changed<Instructions>` recompiles derived prompt context;
- `Changed<ModelHealth>` changes routing eligibility;
- `Changed<StoreHealth>` activates failover systems;
- `Changed<McpServerConfig>` schedules a refresh effect.

In-flight snapshots remain immutable. Change detection affects only future preparation systems.

## Plugin model

```rust
pub trait RigPlugin: Send + Sync + 'static {
    fn build(&self, app: &mut RigApp);
    fn ready(&self, _world: &World) -> bool { true }
    fn finish(&self, _app: &mut RigApp) {}
    fn cleanup(&self, _app: &mut RigApp) {}
}
```

Plugins may:

- initialize resources;
- register systems and ordering;
- register messages;
- install observers;
- spawn model/store/server entities;
- add capability components;
- install policy systems.

Example plugins:

```text
CoreAgentPlugin
ToolRuntimePlugin
MemoryPlugin
McpPlugin
OpenAiPlugin
AnthropicPlugin
GeminiPlugin
TelemetryPlugin
PersistencePlugin
CortexPlugin
```

Provider clients are not required to be plugins merely to create a model entity. Plugins are for runtime integration and reusable system packages.

## Public API

The normal API remains concise while creating real entities:

```rust
use rig::ecs_prelude::*;
use rig::providers::openai;

let mut app = RigApp::new();
app.add_plugins(DefaultRigPlugins);

let model = app.spawn_model(openai::Client::from_env()?.completion_model("gpt-5"));
let memory = app.spawn_store(PostgresMemory::connect(url).await?);
let search = app.spawn_tool(SearchTool::new(search_client));

let agent = app.spawn_agent(
    AgentBundle::builder(model)
        .name("researcher")
        .preamble("Research carefully and cite sources.")
        .memory(memory)
        .max_turns(16)
        .build(),
);

app.grant_tool(agent, search, GrantPolicy::default());

let run = app.spawn_run(agent, "Compare the two implementations");
let response = app.run_until(run).await?;
```

Hosted usage:

```rust
let runtime = RigRuntime::spawn(app);
let response = runtime.agent(agent).prompt("Hello").await?;
```

Advanced users add native systems:

```rust
fn deny_unhealthy_tools(
    tools: Query<&ToolStatus, With<Tool>>,
    mut calls: Query<(&CallsTool, &mut ToolCallDecision), With<ToolCall>>,
) {
    // policy
}

app.add_systems(RigSet::PolicyBeforeTool, deny_unhealthy_tools);
```

## Streaming

Streaming chunks remain values delivered through a stream channel; they do not become entities.

The model-call entity owns stream lifecycle and accumulated state. Stream events carry stable run/call IDs. Systems apply canonical deltas or completion summaries at controlled boundaries.

The same model-call entity and systems serve blocking and streaming modes. Blocking consumers ignore incremental output; streaming consumers subscribe to it. There is no separate blocking runner.

## Error model

Typed model/tool/store errors are preserved at direct authoring boundaries and normalized before entering heterogeneous ECS state.

Entities carry canonical failures:

```rust
#[derive(Component)] pub struct RunFailure(pub PromptError);
#[derive(Component)] pub struct ModelCallFailure(pub CompletionError);
#[derive(Component)] pub struct ToolCallFailure(pub ToolExecutionError);
#[derive(Component)] pub struct StoreCallFailure(pub StoreError);
```

Operator diagnostics, model-visible presentation, retryability, refusal, and concrete typed sources remain distinct.

A failed entity has exactly one terminal outcome. Validation systems reject contradictory success/failure components.

## Cancellation and cleanup

Cancellation is entity-scoped:

- cancelling a run marks it `RunCancelled`;
- dispatch systems stop creating new effects;
- effect executor receives cancellation requests for in-flight call IDs;
- late completions are recognized and discarded or recorded as late telemetry;
- child call entities become terminal;
- cleanup runs only after all required cancellation bookkeeping settles.

Despawn is not cancellation. Entities are despawned only after terminal state is externally observable and persistence/telemetry obligations complete.

## Multi-tenancy and security

Recommended isolation is one `RigApp`/`World` per strong tenant or trust boundary.

If multiple tenants share a world:

- every agent, tool, store, run, grant, and call carries tenant scope;
- all core queries include tenant-compatible relationships or filters;
- tool snapshots validate tenant scope before provider exposure;
- secrets remain in a resource that resolves tenant-scoped opaque keys;
- debug formatting never exposes secret resource values;
- cross-tenant invariant tests are mandatory.

A broad query must never be sufficient to grant access. Access comes from explicit grant entities and validated snapshots.

## Determinism and concurrency invariants

1. Provider tool definitions are sorted by explicit registration/grant order.
2. Every turn stores the exact tool entity snapshot advertised to the model.
3. The entity executed for a call is the one captured in that snapshot.
4. Parallel tool completion order never determines transcript order.
5. Tool batches commit atomically in `ToolCallIndex` order.
6. System ordering required for semantics is explicit.
7. Query iteration order is never persisted or externally observable.
8. Effect completions include stable correlation IDs and are idempotently applied.
9. Late or duplicate completions cannot mutate a newer run/call generation.
10. Blocking and streaming consumers observe the same committed history and terminal outcome.

## Persistence

Persistence operates on explicit snapshots, not raw world serialization.

Persistable records use stable IDs and canonical data:

```text
AgentRecord
ModelReferenceRecord
StoreReferenceRecord
ToolRecord
ToolGrantRecord
RunRecord
TurnRecord
CallRecord
```

Private drivers, live clients, task handles, channels, and raw `Entity` values are not serialized. Plugins reconstruct runtime-only components from persisted configuration.

## Observability

Entity identity provides correlation across the runtime:

```text
stable_run_id
stable_turn_id
stable_call_id
agent_stable_id
model_stable_id
tool_stable_id
store_stable_id
```

Telemetry systems query terminal or changed call/run components. Raw execution result and model-facing presentation remain separate components, preventing presentation rewrites from corrupting policy or audit data.

## Module layout

```text
rig-core/src/
├── app.rs
├── ecs.rs
├── components/
│   ├── agent.rs
│   ├── model.rs
│   ├── tool.rs
│   ├── store.rs
│   ├── run.rs
│   ├── turn.rs
│   └── call.rs
├── bundles/
├── relationships/
├── messages/
├── schedules/
├── systems/
│   ├── ingress.rs
│   ├── resolve.rs
│   ├── model.rs
│   ├── tool.rs
│   ├── policy.rs
│   ├── commit.rs
│   ├── persistence.rs
│   └── cleanup.rs
├── effects/
│   ├── executor.rs
│   ├── model.rs
│   ├── tool.rs
│   └── store.rs
├── plugins/
├── runtime/
│   ├── local.rs
│   └── hosted.rs
├── provider/          # canonical provider traits and request/response types
├── tool/              # typed authoring and private erasure
├── memory/
└── prelude.rs
```

Provider implementations may remain in their current modules initially, but registration and execution enter through model entities and effects.

## APIs removed at cutover

| Removed API/concept | ECS-native replacement |
| --- | --- |
| `Agent<M>` as runtime object | `Agent` entity + components + `UsesModel` |
| `AgentBuilder` returning an owned agent | `AgentBundleBuilder` + spawn into `World` |
| `AgentRunner` as execution engine | Rig schedules and run entities |
| runner-owned configuration copies | queried agent/run components and turn snapshots |
| `ToolSet` | tool entities + grants + snapshot queries |
| `ToolServer` / `ToolServerHandle` | world commands and tool entity lookup |
| `HookStack` / `AgentHook` | systems in policy/observation sets |
| agent-owned memory handle | `UsesMemory` relationship to store entity |
| dynamic context vectors | retrieval store relationships and preparation systems |
| mutable registry callbacks | deferred commands and change detection |
| separate blocking/streaming loops | one call entity lifecycle with optional delta subscription |
| persisted runtime IDs | stable UUID components and remapping |

Typed `Tool`, `CompletionModel`, canonical request/response/message types, and provider clients remain as authoring and wire boundaries, but no longer own runtime orchestration.

## Migration strategy: one architectural cutover

This migration is not delivered as a sequence of compatibility-preserving runtime modes. Implementation may use parallel internal workstreams, but the merged result is one coherent ECS-native runtime.

### Explicitly forbidden migration techniques

- no `ecs` feature that selects a second implementation;
- no old and new agent runners living side-by-side in a release;
- no `ToolSet` mirrored into tool entities;
- no compatibility registry synchronizing strings and entity IDs;
- no deprecated wrappers that retain old ownership semantics;
- no adapter that executes outside the world and merely reports results into ECS;
- no public API that sometimes uses ECS and sometimes bypasses it;
- no long-lived conversion layer between `Agent<M>` and agent entities;
- no temporary persistence of raw Bevy entity IDs;
- no silent fallback to old hook behavior.

### Development branch policy

A dedicated rewrite branch may be temporarily non-releasable. Work is organized by internal subsystem, not by backwards-compatible release phases:

1. establish components, bundles, relationships, schedule sets, and invariant tests;
2. implement effect executor and completion inbox;
3. move model execution into model-call entities;
4. move tool execution into tool-call entities;
5. move transcript advancement and batch commit into systems;
6. move memory/store execution into store entities and effects;
7. replace hooks with extension systems;
8. route every public execution API through `RigApp`;
9. delete old registries, runner control flow, and duplicated state;
10. merge only when cutover criteria are satisfied.

These are implementation workstreams, not compatibility stages. The target branch contains no dual runtime when merged.

### Release policy

The cutover ships as an explicitly breaking release. Migration documentation teaches the new entity/bundle/system model rather than presenting mechanical aliases for removed APIs.

## Cutover criteria

The ECS-native rewrite is ready only when all criteria pass.

### Architecture

- `bevy_ecs` is required by `rig-core`.
- every live agent, model, tool, store, run, turn, and call has entity identity.
- the world is the only authoritative registry.
- all execution APIs use the same schedule.
- no world borrow crosses `.await`.
- old runner and registry implementations are deleted.
- extension behavior is implemented through systems/plugins.

### Correctness

- exact advertised/executed tool snapshot tests pass;
- deterministic parallel tool ordering tests pass;
- atomic batch commit tests pass;
- cancellation and late-completion tests pass;
- blocking/streaming transcript parity tests pass;
- invalid-tool repair/skip/failure tests pass;
- structured-output tests pass;
- memory persistence tests pass;
- MCP ownership/refresh tests pass;
- tenant isolation tests pass;
- stable-ID persistence/remapping tests pass;
- change-detection invalidation tests pass.

### Providers and platforms

- all provider completion and streaming suites pass;
- tool and multimodal result suites pass;
- native and WASM checks pass;
- no provider needs direct world access;
- fake executor tests are deterministic without network or sleeps.

### Quality

- no schedule ambiguity exists in core sets;
- component invariant validator passes in debug/test builds;
- ECS query order is absent from externally visible semantics;
- benchmarks cover one-shot prompts, many concurrent runs, large tool registries, and tool-batch execution;
- public examples demonstrate both simple facade usage and native ECS extension.

## Testing architecture

### Deterministic fake effects

Tests use a fake `EffectExecutor` that records effects and allows explicit completion order. This makes concurrency, cancellation, retries, and late completion deterministic.

### Schedule-level tests

Tests spawn entities, run schedules, inspect components, and assert transitions. They do not test core behavior exclusively through facade APIs.

### Facade conformance

Every facade operation is checked against direct world/schedule operation to prove it is only a convenience layer.

### Invariant systems

Debug/test builds run validation systems for:

- exactly one primary run phase;
- exactly one terminal call outcome;
- valid relationships;
- stable IDs present on persistable entities;
- no cross-tenant grants;
- snapshot definitions aligned with executable entities;
- no commit before complete batch settlement;
- no despawn with outstanding required effects.

## Risks and mitigations

### Bevy version coupling

**Risk:** ECS plugins compiled against different Bevy versions have incompatible types.  
**Mitigation:** re-export the pinned version through Rig and require plugins to import through Rig.

### Compile-time increase

**Risk:** `bevy_ecs` increases compilation cost for simple provider users.  
**Mitigation:** accept this as the cost of making `rig-core` ECS-native; keep optional provider features narrow and avoid `bevy_app`/rendering dependencies.

### Async impedance mismatch

**Risk:** systems are synchronous while AI infrastructure is I/O-heavy.  
**Mitigation:** enforce the effect boundary and prohibit world borrows in futures.

### Excessive component fragmentation

**Risk:** related invariants become distributed across too many components.  
**Mitigation:** use cohesive components, bundles, required components, and invariant systems.

### Hidden nondeterminism

**Risk:** query and parallel completion order leaks into provider requests or transcripts.  
**Mitigation:** explicit ordinal components, snapshots, ordered commit systems, and deterministic fake-executor tests.

### World as service locator

**Risk:** systems query broad global state and weaken boundaries.  
**Mitigation:** capability components, explicit relationships, narrow queries, resources only for true singletons, and tenant isolation rules.

### Entity lifecycle leaks

**Risk:** abandoned call/run entities accumulate.  
**Mitigation:** terminal markers, cleanup systems, lifecycle metrics, and invariant checks for aged in-flight entities.

## Definition of the ideal end state

The migration is complete when the following description is literally true:

> Rig is an asynchronous agent ECS built on `bevy_ecs`. Models, tools, stores, agents, runs, turns, and calls are entities. Their data and capabilities are components. Their connections are relationships. Agent execution is a schedule of systems. Network and tool I/O cross an explicit owned effect boundary. Deferred commands make lifecycle transitions atomic. Buffered messages deliver completions predictably. Change detection drives dynamic updates. Plugins extend the runtime by adding components, systems, resources, and observers. The ergonomic prompt API is only a handle over this world, and no second runtime exists.
