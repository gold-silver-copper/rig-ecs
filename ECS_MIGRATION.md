# Rig as a Bevy ECS-Native Agent Runtime

**Status:** Implemented architecture, migration guide, and verification contract

**Scope:** `rig-core`

**Migration policy:** Breaking replacement; no compatibility runtime, dual registry, or staged public API transition

## How to read this document

This document defines architectural constraints, runtime semantics, and acceptance criteria. It intentionally does **not** prescribe a complete set of Rust types, exact system names, module paths, or trait signatures.

The implementation agent should use the most idiomatic Bevy 0.19 design that satisfies the invariants here. Illustrative concepts are not requirements to reproduce today's Rig objects as components. Exact component granularity, system decomposition, private type erasure, executor wiring, and facade naming should be decided while implementing and validating vertical slices.

The target is an ECS-native runtime, not the existing runtime relocated into a `World`.

## Executive decision

Rig will make [`bevy_ecs` 0.19](https://github.com/bevyengine/bevy/tree/v0.19.0/crates/bevy_ecs) a required dependency of `rig-core` and use Bevy ECS as the foundation of agent execution.

- entities provide runtime identity;
- components hold domain state, configuration, capabilities, and outcomes;
- relationships express durable runtime structure;
- systems implement behavior and state transitions;
- schedules and system sets define ordering and safe parallelism;
- resources hold true runtime-wide services;
- messages carry buffered work across schedule boundaries;
- commands apply structural changes at explicit synchronization points;
- observers provide reactive integration where immediate ordering is not part of correctness;
- change detection drives incremental work;
- bundles and extension installers provide ergonomic construction and composition.

Each runtime has one authoritative `World`. Convenient APIs such as prompting an agent are facades over entities and the same schedules used by hosted, manually stepped, tested, and embedded execution.

The migration prioritizes the ideal architecture over source compatibility. Existing abstractions that duplicate ECS responsibilities are removed rather than wrapped, mirrored, or preserved inside components.

## Pinned foundation

### Bevy version

`rig-core` targets the Bevy ECS 0.19 release line. The workspace dependency is `bevy_ecs = "0.19"`, the lockfile pins the selected patch release, and no git or development revision is used. Rig must intentionally review any future Bevy upgrade rather than accepting an architectural upgrade incidentally.

Bevy ECS 0.19 requires Rust 1.95, so the workspace toolchain and MSRV move to Rust 1.95 or newer as part of the cutover.

The dependency should begin with default features disabled and enable only features justified by the runtime. `rig-core` must not acquire rendering or game-engine dependencies. `bevy_app` remains outside `rig-core`; integration with a full Bevy `App` belongs in a separate crate or adapter.

### Re-export policy

Rig re-exports its selected `bevy_ecs` dependency so extension crates can use the exact same `Entity`, `World`, `Component`, `Resource`, message, relationship, and schedule types. Rig may provide a focused prelude, but it should not hide standard Bevy concepts behind parallel Rig-specific versions.

### Standard Rust bounds; no WASM compatibility traits

The `wasm_compat` abstraction is removed. In particular, the migration removes custom conditional marker traits and aliases such as:

- `WasmCompatSend`;
- `WasmCompatSync`;
- `WasmCompatSendStream`;
- `WasmBoxedFuture` and equivalent compatibility aliases where they exist only to vary thread-safety semantics by target.

Runtime-facing traits use ordinary Rust bounds with the same semantic meaning on every target. Values stored as Bevy components or ordinary resources obey Bevy's normal `Send + Sync + 'static` requirements. Public trait contracts must not silently mean something different on WASM.

A Bevy non-send resource may be used narrowly for a genuinely thread-affine platform service at the effect boundary. It must not become a registry for addressable domain objects or a replacement compatibility mechanism.

WASM remains a supported target. Target-specific HTTP, task spawning, and browser integration may exist privately at the asynchronous effect boundary, but platform differences must not leak into ECS identity, component validity, policy semantics, or public marker traits. Native and `wasm32-unknown-unknown` compilation are mandatory cutover gates.

## Why ECS belongs in `rig-core`

Rig coordinates heterogeneous models, tools, MCP servers, stores, policies, concurrent runs, streaming calls, cancellation, telemetry, persistence, and dynamically changing infrastructure. These concerns have identity, lifecycle, composition, visibility, and scheduling semantics.

Nested builders, shared registries, generic runner objects, callback stacks, and ad hoc synchronization obscure those semantics. ECS makes them queryable and explicit. The benefit is not primarily archetype performance; it is a coherent runtime model with clear ownership and extension boundaries.

## Goals

1. Make one Bevy `World` the sole authority for each live Rig runtime.
2. Give agents, executable capabilities, stores, runs, turns, and asynchronous operations entity identity where lifecycle or relationships matter.
3. Represent runtime state and capability through components rather than nested runtime object graphs.
4. Express orchestration, policy, dispatch, commit, persistence, and cleanup as systems.
5. Use one schedule-driven execution engine for local, hosted, blocking, streaming, test, and embedded use.
6. Preserve ergonomic typed authoring without allowing generic types to define runtime topology.
7. Keep all world access synchronous; no ECS borrow may cross `.await`.
8. Make externally visible ordering deterministic under system and effect concurrency.
9. Make normal Bevy systems, system sets, messages, relationships, observers, and change detection the extension model.
10. Keep secrets and process-wide services out of ordinary domain components.
11. Support embedding Rig schedules in an existing Bevy ECS world.
12. Keep native and WASM behavior aligned without custom compatibility traits.

## Non-goals

1. Preserving the current `Agent<M>`, `AgentBuilder`, `AgentRunner`, `ToolSet`, `ToolServer`, `HookStack`, or `AgentRun` architecture.
2. Providing an optional ECS mode alongside the old runtime.
3. Converting the old object graph one-for-one into component wrappers.
4. Keeping a classic registry synchronized with ECS entities.
5. Turning every message part, token, provider chunk, JSON value, or transient calculation into an entity.
6. Giving asynchronous providers, tools, stores, or user futures access to `World`, `Commands`, `Query`, or component references.
7. Persisting or transmitting raw Bevy `Entity` values.
8. Requiring `bevy_app` in `rig-core`.
9. Freezing exact component layouts or public API names before vertical slices prove them.

## Non-negotiable architectural rules

### One authoritative world per runtime

Every live ECS-managed agent, model, tool, store, policy, run, turn, and call belongs to one authoritative world. There is no parallel runtime registry or runner-owned copy of authoritative state.

Multiple runtime worlds are valid and are the preferred boundary for strong tenant or trust isolation. “One world” means one source of truth within a runtime, not one global process singleton.

### ECS-native structures, not wrapped legacy structures

The implementation must decompose behavior into systems and runtime state into components. It must not make the migration appear complete by inserting the old runner, hook stack, tool registry, or state machine into a component and advancing it from one system.

Private adapters at external typed boundaries are acceptable where heterogeneous I/O requires them. They are not acceptable as a way to preserve the old orchestration model.

When choosing among designs, prefer Bevy-native mechanisms:

- typed components queried by capability systems;
- relationships rather than copied ownership vectors;
- messages for buffered cross-boundary work;
- registered systems or provider-specific systems where dynamic behavior fits them;
- `SystemParam` for coherent system access;
- commands and explicit deferred boundaries for structural mutation;
- change detection and removal detection rather than manual invalidation callbacks;
- observers for reaction and integration, not hidden ownership of the core state machine.

Trait objects or private erasure may still be the right boundary for some provider, tool, or store calls. Their use should be justified by heterogeneous execution, kept private, and prevented from becoming a second runtime architecture.

### One execution engine

All agent execution paths drive the same world state and schedules. A facade may own a world, a hosted task may own it, or an external Bevy host may run its schedule, but none may implement agent progression independently.

Low-level provider clients may continue to expose direct transport APIs. They are authoring and transport boundaries, not an alternative agent runtime.

### No world borrow across `.await`

Systems synchronously inspect world state and create owned asynchronous work. External work completes without a world borrow and re-enters through a controlled inbox, message, or command boundary. Only later systems validate and apply the result.

No future may capture a `World`, `Query`, `Commands`, `Res`, `Mut`, or borrowed entity/component data. Copying an `Entity` handle for correlation is permitted, but completion application must still validate its generation and expected state.

### One source of truth for every fact

Run phase, transcript, usage, budgets, call outcomes, cancellation, and persistence status each have one authoritative representation. Derived components or indexes are permitted only when they are clearly derived, invalidated, and rebuildable.

Mutually exclusive phases and outcomes must be represented so contradictory states are difficult or impossible to construct. The implementation may use cohesive enum state, component presence, required components, or another idiomatic Bevy design, but must not rely on a large set of independent marker components plus routine repair.

### Determinism is explicit

Query order, hash-map order, system scheduling order, and effect completion order are never semantic order. Provider-facing tools, policies, calls, transcript entries, and persisted records use explicit ordering data and deterministic tie-breaking.

### Runtime identity is not persistent identity

Bevy `Entity` is the canonical in-memory handle. Anything crossing process, persistence, network, or public protocol boundaries uses stable domain identity. Persistence reconstructs entities and relationships through stable-ID remapping.

Stable-ID uniqueness and stale-entity behavior must be enforced rather than assumed.

### In-flight work sees immutable decisions

A turn or call must execute against the exact model, tool definitions, grants, policy decisions, and capability revisions accepted for it. Later world mutations affect future preparation, not already dispatched work.

The implementation may guarantee this with retained version entities, owned immutable snapshots, leases, revisioned capabilities, or another ECS-native design. Storing only an entity ID is insufficient if that entity can be replaced or despawned before the operation completes.

## Runtime shape

### World and schedules

Rig installs one or more labeled Bevy schedules into the world's schedule resources. The exact labels and number of schedules are implementation choices, but the core runtime must have a documented update entry point suitable for:

- local `run_until` execution;
- hosted wake-driven execution;
- deterministic tests;
- manual stepping;
- installation into an existing Bevy world.

A convenience owner may wrap `World`, but schedules should remain world-resident rather than becoming a private second scheduling abstraction.

Core system ordering must be expressed with Bevy system sets and explicit dependencies. Required command-application boundaries must be intentional. Core tests should treat schedule ambiguities as failures rather than relying on Bevy's permissive defaults.

### Entity categories

The following concepts normally deserve entity identity:

- agents;
- model capabilities or configured model endpoints;
- tools and dynamically discovered executable capabilities;
- addressable stores and infrastructure capabilities;
- MCP or other discovery sources;
- policy, grant, or approval instances when they have independent lifecycle;
- runs;
- committed turns;
- model, tool, store, and persistence operations;
- subscriptions or approvals when they have cancellation or audit semantics.

This list describes the domain topology, not mandatory marker component names. The implementation may combine or refine categories when invariants and query patterns justify it.

Small immutable values, message parts, tool arguments, deltas, usage counters, and provider payload fragments remain values unless independent identity or lifecycle is useful.

### Components

Components should represent cohesive state, configuration, capability, ordering, relationship metadata, or outcome. They should be split when systems need independent querying, change detection, replacement, access, or lifecycle—not merely because a field can be separated.

Good component design should make these facts easy to query:

- what an entity is capable of;
- what it is related to;
- whether it is eligible for new work;
- what immutable decision an in-flight operation uses;
- what phase or outcome it currently has;
- who may observe or mutate it;
- whether it is persistent, retired, cancelled, or ready for cleanup.

Components that contain clients or executable handles must satisfy normal Bevy component bounds on every platform. Components must not contain borrowed world data.

### Relationships

Use Bevy relationships for structural links whose consistency benefits from relationship hooks and target queries, such as agent-to-model, run-to-agent, turn-to-run, call-to-turn, capability-to-host, and store usage.

Many-to-many access with independent metadata generally deserves a relationship or grant entity rather than a vector copied onto an agent.

Cascade despawn is appropriate only when the target can never outlive its owner semantically. Dynamically discovered tools, snapshots, and in-flight calls require retirement semantics; they must not disappear merely because a discovery source refreshes or disconnects.

### Resources

Resources are reserved for runtime-wide services and coordination, such as:

- external effect submission and completion ingress;
- wakeup coordination;
- clocks and ID generation;
- secret resolution;
- runtime configuration;
- telemetry sinks;
- indexes that are explicitly derived from world state;
- progress or quiescence tracking.

Addressable models, tools, stores, policies, pools, and tenant-scoped capabilities remain entities even if only one currently exists. Cardinality alone does not turn a domain object into a resource.

### Messages, commands, observers, and change detection

Use Bevy messages for ordered, buffered work crossing schedule stages or runtime boundaries. Use commands for deferred structural mutation. Use change and removal detection for incremental reconciliation.

Observers are useful for telemetry, UI, audit, integration notifications, and localized reactions. Core transcript, phase, authorization, or commit correctness must not depend on undocumented observer timing.

## Domain architecture

### Agents

An agent is an entity composed from configuration and relationships. Its identity does not encode a model type. Models, stores, policies, and tools are associated through world structure and resolved by systems.

Agent configuration should be separated according to real mutation and query patterns: identity, instructions, model selection or routing, sampling, budgets, output requirements, memory relationships, and policy relationships need not form one permanent object.

Construction should be ergonomic through bundles, helper functions, or builders that spawn into a world. A builder is acceptable as a construction aid; an owned agent object that becomes a second runtime is not.

### Models

A configured model is an entity with provider identity, capabilities, configuration, health, routing metadata, and whatever executable integration the chosen ECS design requires.

Do not assume the current generic `CompletionModel` object must be stored behind an erased component. Prefer typed provider components and systems, generic system registration, registered systems, or other Bevy-native dispatch where practical. Private erasure remains available when it is genuinely the cleanest heterogeneous I/O boundary.

Typed provider authoring remains desirable, but current trait signatures and associated types are not migration constraints. They may be redesigned to use ordinary Rust bounds and to integrate cleanly with Bevy.

Provider-specific raw data must not become necessary for core progression. Preserve it for diagnostics or specialized integrations without making canonical runtime state provider-dependent.

### Tools

A tool is an entity with stable identity, provider-facing definition, ordering, status, provenance, revision, policy visibility, and executable capability.

Static, dynamically created, retrieved, and MCP-discovered tools converge on the same runtime representation. Their construction paths may differ; dispatch and policy semantics must not.

The old `ToolSet` and `ToolServer` are not retained as hidden registries. Tool discovery and visibility are world queries over capabilities, relationships, grants, status, and policy.

Tool-name collisions, replacement, discovery ordering, and duplicate suppression must have explicit deterministic semantics. The exact ordering representation is an implementation decision, but current externally valuable behavior should be covered by tests before old registries are removed.

Per-call context should be represented by call state, relationships, and owned effect input assembled by systems. The existing `ToolContext` type map must not survive as an alternate orchestration or dependency-injection world. A smaller typed value may remain at a tool authoring boundary if useful.

### Dynamic discovery and MCP

Discovery sources are entities with configuration, generation, liveness, ownership, and refresh state. Discovery work occurs outside world borrows and applies a reconciled change set at a schedule boundary.

Refresh must distinguish retirement from immediate destruction. Existing snapshots and calls continue to reference the exact capability version they accepted. New snapshots exclude retired versions. Cleanup occurs only when no in-flight work needs them.

Stale refresh completions must not overwrite newer generations.

### Stores and infrastructure

Stores are capability entities rather than variants in a central backend enum. A single entity may support conversation memory, vector search, document storage, SQL execution, or several capabilities.

Systems query the capability they need. Provider-specific installation adds the corresponding data and systems without changing a central dispatch match.

Secrets are resolved through runtime services using opaque references. Secret material is not a normal component, is not persisted with domain records, and is not exposed through debug output.

### Runs, turns, and operations

A prompt creates a run entity. The run is authoritative ECS state, not a wrapper around the old `AgentRun` state machine.

Systems own progression. State should be decomposed around actual invariants and access patterns while keeping one source of truth for phase, transcript, budget, usage, output state, and terminal outcome.

A committed turn is distinct from an in-flight model operation. Model calls, tool calls, store calls, and persistence work should have entity identity when they need correlation, cancellation, retries, policy, telemetry, or independent lifecycle.

Call entities make concurrency explicit. They allow multiple operations to exist simultaneously without a single pending slot and let results be correlated independently of arrival order.

Terminal outcomes should be write-once from the perspective of a call generation. Success, failure, cancellation, timeout, and supersession must not coexist ambiguously.

### Immutable turn decisions and tool snapshots

Before provider dispatch, systems resolve the material facts needed by the turn. The accepted decision must preserve:

- selected model and capability revision;
- ordered provider-facing tool definitions;
- the exact executable tool capability behind each definition;
- grants, tenant scope, and policy decisions relevant to execution;
- ordering and correlation information;
- any infrastructure choice that must not drift mid-turn.

A mutation after this boundary applies to later turns. It must not alter what the provider saw or which implementation executes a returned tool call.

The snapshot representation should be designed for lifecycle correctness rather than as a copy of the current `ToolRegistrySnapshot`. It may use ECS version entities, leases, immutable owned values, or another mechanism proven by replacement and retirement tests.

### Policy and extensions

Policy is ordinary ECS data interpreted or transformed by systems in documented sets. Per-agent or per-run policy instances have entity/component state and relationships rather than a privileged callback stack.

Extensions may inspect and transform prepared requests, approve or deny calls, normalize arguments, shape model-visible results, react to invalid calls, select routing, or stop a run. Composition order is explicit where order affects semantics.

`AgentHook` and `HookStack` are removed. The implementation must not recreate them as an erased callback list in a component. Where users need dynamic policy instances, represent instance state in ECS and use registered or typed systems, observers, messages, or a narrowly scoped private dispatch boundary that remains subordinate to the schedule.

Core state transitions remain owned by core systems. Observation extensions do not mutate transcript or lifecycle invariants out of band.

### Streaming

Streaming chunks are values, not entities. The corresponding model-call entity owns stream lifecycle, cancellation, accumulation, correlation, and final outcome.

Blocking and streaming consumers observe the same committed run state. Streaming adds incremental delivery; it does not use a separate runner.

The implementation must define bounded buffering, slow-consumer behavior, dropped subscribers, cancellation, and final usage handling. Policies that inspect deltas should receive them at controlled schedule boundaries rather than borrowing the world from a stream task.

### Errors

Direct typed authoring APIs may preserve concrete provider errors. Heterogeneous ECS state uses canonical outcomes suitable for policy, retry, telemetry, persistence, and model-visible presentation.

Concrete source diagnostics, retry classification, operator detail, raw tool output, and model-facing presentation are distinct concerns. Rewriting presentation must not corrupt audit or retry data.

### Cancellation, retirement, and cleanup

Cancellation is a state transition, not despawn. It prevents new dispatch, requests cancellation of external work where supported, and defines how late completions are handled.

Retirement prevents new references while preserving old ones. Despawn occurs only after in-flight effects, persistence, telemetry, subscriptions, and result observation no longer require the entity.

The implementation must define result retention so a facade cannot lose a completed run to cleanup before observing it.

## Scheduling and progression

The exact system-set enum is intentionally unspecified. The schedule must nevertheless expose clear semantic stages equivalent to:

1. ingest external commands and effect completions;
2. reconcile changed world structure and resolve configuration;
3. prepare immutable run, model, tool, store, and policy decisions;
4. apply ordered policy;
5. dispatch owned external effects;
6. validate and apply completions;
7. commit deterministic turn and batch outcomes;
8. persist required state;
9. publish observations;
10. cancel, retire, and clean up terminal entities.

Systems within a stage should run in parallel when their Bevy access permits it. Ordering between stages and order-sensitive extensions must be explicit.

Structural changes made through `Commands` become visible only at intentional synchronization points. The implementation must choose and test its `ApplyDeferred` boundaries instead of assuming a final flush is sufficient.

### Quiescence and wakeup

A local or hosted driver repeatedly runs the schedule while immediate progress is possible. It stops when the requested result is observable, the runtime is waiting exclusively on external work, or a bounded progress guard detects a livelock.

The implementation should track progress explicitly rather than infer it from arbitrary component scans. External command or effect arrival wakes a hosted runtime. Tests must be able to drive the same progression deterministically without sleeps.

## Asynchronous effect boundary

Bevy systems are synchronous; provider, tool, store, persistence, and discovery work is asynchronous. The effect boundary is the only place where execution leaves the world.

The design must guarantee:

1. systems create an operation entity or otherwise establish authoritative correlation state;
2. all effect input is owned;
3. no ECS borrow enters the future;
4. submission records the operation generation;
5. completion re-enters through a thread-safe or target-appropriate ingress;
6. an apply system validates identity, generation, cancellation, and expected phase;
7. duplicate, stale, and late completions are idempotently rejected or recorded;
8. only ECS systems transition authoritative runtime state.

The exact executor, task type, channel, message, and erasure strategy are implementation choices. The design should support native and WASM executors without conditional public trait semantics.

Queues and streams must have explicit capacity and backpressure behavior. Shutdown, task panic, timeout ownership, cancellation, and wakeup semantics are part of the implementation contract, not incidental executor behavior.

Tests use controllable fake effects that can complete out of order, fail, duplicate, arrive late, or remain pending.

## Extension and plugin model

An extension is a package that installs Bevy-native pieces into a world and its schedules:

- components and bundles;
- resources;
- relationships;
- messages;
- systems and system sets;
- observers;
- provider/tool/store installation helpers;
- policy and telemetry behavior.

Because `rig-core` does not depend on `bevy_app`, it may expose a small installation abstraction or functions for configuring a `World` and its schedules. That abstraction must remain a thin composition aid, not a competing scheduler or service container.

Extensions must declare ordering when composition semantics require it. Duplicate schedule labels, incompatible resources, stable-ID conflicts, and ambiguous mandatory ordering should fail clearly.

A separate full-Bevy integration can translate the same installers into `bevy_app::Plugin` implementations.

## Public API principles

The public API should be concise without obscuring entity ownership:

- creating an agent, model, tool, store, or run spawns or configures world entities;
- handles identify runtime/world ownership and reject stale or foreign entities;
- prompting through a handle submits a command and observes the resulting run;
- local and hosted handles drive the same schedules;
- advanced users can query the world, install systems, and run schedules directly at documented safe points;
- typed provider and tool construction remains ergonomic even if runtime storage is heterogeneous;
- APIs do not expose private erasure solely because the runtime needs it internally.

Direct mutable world access is safe during construction and controlled schedule execution. A hosted runtime should accept commands or execute user closures at a safe point rather than expose concurrent `&mut World` access.

Exact facade names, handle types, builders, and return types should emerge from the implementation rather than be fixed by this document.

## Determinism and concurrency invariants

1. Provider-facing definitions use explicit deterministic ordering.
2. Name collisions and replacement have documented deterministic resolution.
3. A turn executes the exact capabilities and policy decisions advertised for that turn.
4. Tool or model replacement cannot mutate an in-flight decision.
5. Parallel completion order does not determine transcript order.
6. A logical tool batch commits in model-call order, not arrival order.
7. Query iteration order is never persisted or externally observable.
8. Required system and policy ordering is explicit.
9. Completion application is correlated and idempotent.
10. Late work cannot mutate a newer entity generation.
11. Blocking and streaming consumers observe the same committed history and terminal outcome.
12. Cancellation and retirement are deterministic under concurrent completion.

## Multi-tenancy and security

A separate world per strong tenant or trust boundary is preferred.

When tenants share a world, tenant scope must participate in agent, capability, grant, run, call, and secret resolution. Broad discovery never grants access by itself. Authorization comes from explicit relationships and policy evaluated before immutable turn decisions are accepted.

Core tests must attempt cross-tenant model, tool, store, snapshot, and stable-ID misuse.

## Persistence

Persistence operates on explicit domain records or snapshots, never raw world serialization.

Persisted data uses stable IDs and canonical values. Runtime-only clients, executable handles, tasks, channels, schedules, private erasure, and raw `Entity` values are reconstructed rather than serialized.

Loading is a remapping operation: establish entities and stable-ID uniqueness first, then restore relationships and capability state. Missing or conflicting references produce explicit errors.

Persistence itself participates in ECS scheduling and call lifecycle where its completion affects run correctness.

## Observability

Stable agent, run, turn, and operation identities provide correlation. Telemetry and audit systems observe changed or terminal ECS state and effect metadata.

Observability must distinguish:

- raw external response;
- canonical runtime outcome;
- policy decision;
- model-visible presentation;
- retry and cancellation history;
- persistence status.

Instrumentation should arise from systems and effect boundaries rather than hidden callbacks embedded throughout provider code.

## APIs and concepts removed at cutover

| Removed concept | ECS-native direction |
| --- | --- |
| `Agent<M>` as a runtime object | agent entity composed from components and relationships |
| runner-owning `AgentBuilder` | construction helper that spawns into a world |
| `AgentRunner` and parallel streaming loops | schedule-driven run, turn, and operation entities |
| `AgentRun` as the orchestration state machine | ECS-owned run state transitioned by systems |
| runner-owned configuration copies | resolved ECS state and immutable in-flight decisions |
| `ToolSet` | tool entities, capability queries, grants, and snapshots |
| `ToolServer` / `ToolServerHandle` | world mutation and discovery/reconciliation systems |
| `HookStack` / `AgentHook` | policy components, systems, sets, messages, and observers |
| `ToolContext` as a general type-map runtime bus | typed call state and owned effect input |
| agent-owned memory/store handles | relationships to store capability entities |
| mutable registry callbacks | commands, messages, relationships, and change detection |
| separate blocking and streaming runners | one operation lifecycle with optional delta delivery |
| persisted runtime handles | stable IDs and remapping |
| `WasmCompat*` traits and aliases | ordinary Rust bounds plus private target-specific effect integration |

Typed provider clients, request/response/message values, and ergonomic typed tool/model authoring may remain, but their existing signatures are not compatibility requirements and they do not own agent orchestration.

## Migration strategy: one architectural cutover

The migration is developed on a rewrite branch that may temporarily be non-releasable. Work can proceed in dependency order, but the merged result contains one ECS runtime and no compatibility mode.

### Implementation sequence

1. Pin Bevy ECS 0.19, move the toolchain to Rust 1.95, establish the re-export, and compile minimal native/WASM ECS fixtures.
2. Remove `wasm_compat` traits and aliases; convert public and private contracts to ordinary Rust bounds and isolate unavoidable target-specific execution details.
3. Install world-resident schedules, resources, messages, progress tracking, and invariant tests.
4. Establish the ECS domain topology and stable-identity rules without embedding old registries or `AgentRun`.
5. Implement one end-to-end model-call vertical slice through the owned effect boundary.
6. Implement tool discovery, immutable turn decisions, execution, replacement, retirement, and deterministic batch commit.
7. Implement policy as systems and ECS state, including per-agent or per-run dynamic policy data.
8. Add stores, memory, retrieval, persistence, MCP reconciliation, streaming, cancellation, and observability through the same operation model.
9. Route every agent facade through the installed schedules and validate local, hosted, test, and embedded driving.
10. Delete old runners, registries, hook infrastructure, duplicated state, compatibility bounds, and dead adapters.
11. Merge only after the complete cutover criteria pass.

These are implementation dependencies, not compatibility phases. No released or merged target state contains two agent runtimes.

### Implementation freedom

The executing agent is expected to revise details when code, Bevy APIs, provider constraints, or tests reveal a better ECS-native design. It may choose different component names, cohesive state layouts, system boundaries, registered-system strategies, private adapters, or facade shapes.

It must preserve the architectural rules and observable invariants in this document. Significant deviations should update this document with rationale rather than silently preserve legacy architecture.

### Explicitly forbidden migration techniques

- an `ecs` feature that selects a second runtime;
- old and new runners living side by side in the completed branch;
- mirroring `ToolSet`, model registries, or store registries into entities;
- storing the old `AgentRun` or hook stack in a component and driving it from ECS;
- a compatibility registry synchronizing strings and entities;
- deprecated wrappers that retain old ownership semantics;
- public APIs that sometimes bypass ECS orchestration;
- a long-lived conversion layer between generic agents and agent entities;
- raw Bevy entity IDs in persistence or protocols;
- custom `WasmCompat*` bounds reintroduced under new names;
- effect tasks that borrow or mutate the world;
- silent fallback to old hook or runner behavior.

## Cutover criteria

### Dependency and platform

- `bevy_ecs` 0.19 is a required `rig-core` dependency and is re-exported by Rig.
- the workspace toolchain satisfies Bevy 0.19's Rust requirement.
- `wasm_compat` traits, aliases, imports, and target-dependent public bounds are gone.
- native and `wasm32-unknown-unknown` checks pass.
- Bevy features beyond the minimal runtime set have explicit justification.

### Architecture

- each runtime has one authoritative world;
- agents, executable capabilities, stores, runs, turns, and lifecycle-bearing operations have ECS identity;
- core behavior is implemented by systems and schedules rather than an embedded legacy runner;
- addressable domain objects are not hidden singleton resources;
- all agent execution APIs drive the same schedules;
- no world borrow crosses `.await`;
- old runners, registries, hook stacks, and duplicated orchestration state are deleted;
- extension behavior uses Bevy-native composition.

### Correctness

- advertised and executed capability identity remains exact across replacement and retirement;
- duplicate-name and ordering tests are deterministic;
- parallel tool completion commits in model-call order;
- logical batches commit atomically;
- cancellation, timeout, duplicate, stale, and late-completion tests pass;
- blocking and streaming histories and terminal outcomes agree;
- policy ordering and stop/failure semantics are deterministic;
- structured-output behavior passes;
- memory and persistence behavior passes;
- MCP refresh generation and retirement behavior passes;
- tenant isolation and stable-ID remapping tests pass;
- cleanup never destroys unobserved required results or referenced capability versions.

### Providers and effects

- representative provider completion, streaming, tool, and multimodal suites pass before broad provider conversion;
- all supported providers pass before cutover;
- no provider, tool, or store future receives ECS access;
- fake effect tests control completion order without network calls or sleeps;
- queue capacity, backpressure, wakeup, shutdown, panic, timeout, and cancellation behavior are tested.

### ECS quality

- mandatory schedule ambiguities fail tests;
- required deferred-command visibility is covered by schedule tests;
- mutually exclusive phases and outcomes cannot silently coexist;
- stable-ID and relationship invariants are validated;
- query order is absent from observable semantics;
- removal and retirement paths are tested;
- facade tests are matched by direct world/schedule tests;
- examples show both concise facade usage and native ECS extension.

### Performance and operability

Benchmarks cover one-shot prompts, many concurrent runs, large capability sets, streaming, and parallel tool batches. Lifecycle metrics make stuck, retired, cancelled, and cleanup-eligible entities observable.

## Testing architecture

### Vertical-slice tests

Prefer schedule-level vertical slices over unit tests of copied legacy helpers. Spawn representative world state, run schedules, control effects, and inspect state transitions and outcomes.

### Deterministic fake effects

A fake effect boundary records owned work and allows tests to choose success, failure, ordering, duplication, lateness, cancellation, and delay. Core concurrency tests use no wall-clock sleeps.

### Invariant tests

Debug and test configurations validate at least:

- stable-ID uniqueness;
- valid and tenant-compatible relationships;
- coherent run and operation phase/outcome state;
- exact immutable turn decisions;
- no commit before a complete logical batch settles;
- no stale or duplicate completion mutation;
- no despawn while required references or effects remain;
- no secret material in persisted or debug-visible state.

### Platform tests

Compile and run the appropriate deterministic suites on native and WASM. Platform differences are confined to effect integration and do not change ECS semantics.

## Risks and architectural responses

### Bevy version coupling

ECS extensions compiled against different Bevy releases have incompatible types. Rig therefore pins and re-exports Bevy ECS 0.19 and treats upgrades as deliberate migrations.

### Compile-time cost

A required Bevy ECS dependency increases compile time for simple users. This is accepted as the cost of one coherent `rig-core` runtime; optional provider features should remain narrow and rendering/full-engine dependencies stay out of core.

### Async impedance mismatch

AI infrastructure is I/O-heavy while systems are synchronous. The owned effect boundary, explicit operation entities, completion ingress, and wake-driven scheduling make that boundary visible and testable.

### Recreating the old runtime inside ECS

The easiest migration path would wrap legacy traits, registries, and `AgentRun` in components. The cutover explicitly forbids that. Vertical slices should be reviewed for ECS-native ownership and behavior before broad conversion.

### Excessive fragmentation

Too many components can distribute invariants and create marker-state contradictions. Use cohesive state, required components, relationships, bundles, and carefully chosen outcome representations.

### Service-locator world

Broad queries can weaken boundaries. Systems should query narrow capabilities and explicit relationships; access should come from grants and resolved immutable decisions rather than discovery alone.

### Hidden nondeterminism

System parallelism, query order, and external completion order can leak into provider requests or transcripts. Explicit ordering, immutable decisions, atomic commit, and adversarial fake-effect tests prevent that.

### Lifecycle leaks

Long-lived runtimes can accumulate calls, retired tools, subscriptions, and completed runs. Retirement and cleanup are explicit scheduled lifecycles with observability and reference-aware tests.

## Definition of the ideal end state

The migration is complete when this statement is literally true:

> Rig is an asynchronous agent runtime built directly on Bevy ECS 0.19. Agents, executable capabilities, stores, policies, runs, turns, and lifecycle-bearing operations are entities composed from components and relationships. Systems own orchestration and policy. World-resident schedules own progression and deterministic commit. Asynchronous I/O crosses an owned, testable effect boundary and never borrows ECS state. Messages, commands, observers, and change detection are used according to Bevy semantics. Native and WASM use ordinary Rust bounds without compatibility marker traits. The ergonomic prompt API is only a facade over this world, and no legacy runner, mirrored registry, or second runtime exists.

## Implemented capability matrix

The table below is the audit map for the pre-ECS runtime. Test names refer to
`runtime::tests` unless a provider or adapter suite is named explicitly.

| Pre-ECS capability | ECS primitive and authoritative state | Public surface | Principal evidence | Migrated example |
| --- | --- | --- | --- | --- |
| blocking prompt loop | run/model-operation entities progressed by `RigSchedule` | `AgentFacade::prompt`, `RuntimeHandle::prompt` | `model_effect_round_trip_uses_world_resident_schedule` | `agent` |
| streaming prompt loop | ordered `EffectIngressMessage` deltas plus run subscription entities | `prompt_stream`, `RunStream` | streaming sequence, parity, slow-consumer, and adapter suites | `agent_stream_chat` |
| completion-call hook | durable request-policy evaluation entity and targeted invocation event | `RequestPatchPolicyBundle`, `RequestPolicyInvocation` | request patch/order/non-sticky tests | `request_hook` |
| completion-response hook | response-policy evaluation before model commit | `CompletionResponsePolicyInvocation` | completion rewrite/stop tests | `tool_result_outcomes` |
| model-turn-finished hook | observe-only targeted entity event | `ModelTurnFinished` | `completion_response_policy_runs_before_commit_and_emits_turn_event` | `agent_with_tools_otel` |
| invalid-tool hook | pending-invalid component plus ordered durable evaluation | repair/retry/skip bundles and invocation event | invalid-tool action, budget, streaming, and snapshot tests | `gemini_default_api_recovery` |
| tool-call hook | per-operation policy evaluation with immutable tool decision | rewrite/approval/skip bundles | rewrite, approval, skip, and snapshot tests | `agent_with_approval_policy` |
| tool-result hook | immutable raw effect plus mutable presentation evaluation | `ToolResultRedactionPolicyBundle` | raw/presentation separation and stop tests | `tool_result_outcomes` |
| hook scratchpad/context | extension-owned typed components and relationship queries | `RigOperationContext`, ordinary `Component` | extension/SystemParam test | `ecs_extension` |
| asynchronous hooks | approval operation entities and owned effect I/O | `PolicyRule::RequireApproval` | request/tool/result/invalid approval snapshot tests | `agent_with_durable_approval` |
| tools and dynamic tools | capability and grant entities with revisioned immutable snapshots | agent builder tools, `spawn_tool`, `grant_tool` | collision, retirement, batch, and provider suites | `agent_with_tools`, `rag_dynamic_tools` |
| MCP/tool-server refresh | discovery-source and discovered-capability relationships | discovery commands and RMCP adapter | generation/retirement and RMCP tests | `rmcp` |
| memory and retrieval | store capability/grant/operation entities | builder `memory` and `dynamic_context` | memory and vector-retrieval tests | `agent_with_memory`, `rag` |
| structured output | `OutputRequirement` plus run-local retry counters | output schema/mode/retry builder methods | validation, retry, snapshot, provider extraction tests | `extractor` |
| cancellation and suspension | `RunControl` orthogonal to `RunState` | pause modes, resume, cancel commands | drain/freeze/cancel-and-suspend race tests | `agent_run_stepping` |
| active `AgentRun` serialization | stable-ID `ActiveRunSnapshot` | `snapshot_active_run`, `restore_active_run` | every waiting-phase restoration test | `agent_with_durable_approval` |
| child-agent delegation | `ParentRun`/`ChildRuns`, `WaitingForChildren`, explicit ordinal | `spawn_child_run` | deterministic result and cancellation tests | `agent_with_agent_tool` |
| telemetry hooks | observe-only entity events and optional typed counters | `LifecycleTelemetryBundle` | lifecycle event ordering and telemetry tests | `agent_with_tools_otel` |
| provider diagnostics | canonical effect outcome plus provider-owned typed data | effect adapters | provider and adapter suites | provider examples |
| WASM | identical ECS state with target-specific effect transport only | normal Rust bounds | WASM compile gate | browser-capable core consumers |

## Hook-to-ECS migration guide

Old hooks no longer form a callback stack. Steering is represented by policy
entities sorted by `(Policy.order, StableId)`. An operation snapshots the exact
policy revisions, creates a durable evaluation, and targets one policy entity at
a time. The next policy therefore sees the effective value produced by every
earlier policy. Observer registration order has no semantic role.

| Old event | Steering boundary | Observe-only boundary |
| --- | --- | --- |
| completion call | `RequestPolicyInvocation` | `CompletionRequestPrepared`, `ModelDispatched` |
| completion response | `CompletionResponsePolicyInvocation` | `ModelSettled`, `CompletionResponseApplied` |
| model turn finished | none after commit | `ModelTurnFinished` |
| invalid tool call | `InvalidToolCallPolicyInvocation` | `InvalidToolCallDetected` |
| tool call | `ToolCallPolicyInvocation` | `ToolCallPrepared`, `ToolExecutionStarted` |
| tool result | `ToolResultPolicyInvocation` | `ToolExecutionSettled`, `ToolResultPresentationFinalized` |
| stream text delta | `TextDeltaPolicyInvocation` | `TextDeltaObserved`, `StreamResponseFinished` |

`RequestPatch` is operation-local. The evaluation retains the baseline request,
the reduced `accumulated` patch, and the current effective request separately.
Context appends, provider parameters shallow-merge, active tools intersect, and
scalar/history fields use last-writer-wins. Repeated last-writer fields emit a
structured warning containing the field and later policy ID. The next turn is
always rebuilt from agent/run state, never from the prior patched input.

Asynchronous policy behavior creates a `PolicyApproval` operation related to
the evaluation. The evaluation remains at its cursor while the owned request is
outside the world. Completion ingress validates operation identity, generation,
phase, cancellation, and output kind before advancing the cursor.

## Public extension API

`RigExtension` is intentionally only an installer over `&mut World`. The common
policy helpers—`RequestPatchPolicyBundle`, `ToolApprovalPolicyBundle`,
`ToolArgumentRewritePolicyBundle`, `ToolSkipPolicyBundle`,
`ToolResultRedactionPolicyBundle`, `InvalidToolRepairPolicyBundle`, and
`InvalidToolRetryPolicyBundle`—are ordinary Bevy `Bundle`s and also implement
the installer. `LifecycleTelemetryBundle` is inserted on an agent to opt into
queryable observe-only counters. `RigOperationContext` is a read-only
`SystemParam` resolving operation → run → agent metadata without allowing a
borrow to escape into asynchronous work.

Custom behavior may add components and systems at the public `RigSet`
boundaries or attach one targeted steering observer to a `PolicyRule::Custom`
entity. Additional audit observers consume the separate observation events.

## Schedule and lifecycle diagrams

The public schedule is one ordered progression engine for every agent and run:

```text
IngestCommands -> IngestControlCommands -> IngestEffects -> Reconcile
 -> ReconcileAgentControl -> ApplyRunControl -> SpawnDynamicAgentsAndRuns
 -> PrepareRun -> PrepareModel -> BeginRequestPolicy -> InvokeRequestPolicy
 -> ReduceRequestPolicy -> FinalizeRequest -> DispatchModel
 -> ApplyModelCompletion -> BeginResponsePolicy -> CommitModelTurn
 -> ResolveInvalidTools -> PrepareToolBatch -> BeginToolCallPolicy
 -> DispatchTools -> ApplyToolCompletions -> BeginToolResultPolicy
 -> CommitToolBatch -> Persist -> Publish -> Cancel -> Retire -> Cleanup
 -> MaintainMessages
```

Mandatory ambiguity detection is enabled at `Error` level and the core schedule
graph is asserted conflict-free. Deferred structural commands are visible at
the intentional chained system boundaries. Component lifecycle tests cover
`Add -> Insert`, replacement `Discard -> Insert`, explicit removal
`Discard -> Remove`, required-component insertion, observer commands, and the
despawn observation/removal sequence. Business phases remain ordinary
queryable components rather than lifecycle-hook control flow.

Parent/child progression is explicit:

```text
parent model operation active
 -> SpawnChildRun command
 -> ParentRun/ChildRuns relationship + WaitingForChildren marker
 -> child progresses in the same RigSchedule
 -> child terminal outcome retained
 -> results reduced by ChildOrdinal (not completion order)
 -> marker removed only after every child result commits
 -> parent resumes its preserved RunState
```

Every ready evaluation is visited once per schedule pass, ordered by stable run
identity and cursor. The runtime does not cap a pass after one run, so an
immediately-ready policy-heavy run cannot exclude a ready sibling. Waiting and
paused runs do not report progress. Separate worlds remain the isolation and
scheduling-shard boundary for distinct trust domains.

## Active-run snapshot format

`ActiveRunSnapshot` is versioned and contains only stable domain IDs plus opaque
snapshot-local references. It records:

- root and descendant runs, parent identity, child ordinal, and committed-child status;
- authoritative run phase and orthogonal pause mode;
- prompt, transcript, usage, turn/model budgets, invalid-tool retries, and structured-output retries;
- memory/retrieval decisions, conversation state, pending output, and persistence state;
- operation generation/phase, immutable model/tool/store decisions, stream sequence, and settled output;
- tool batches and explicit call order;
- accepted policy IDs/revisions, evaluation kind/cursor, accumulated request patch, effective arguments/presentation, and pending approvals;
- committed-turn audit records.

Raw entity IDs, observers, registered systems, clients, secrets, channels, and
task handles are never serialized. Restoration validates tenant and revision
compatibility, remaps stable references, reconstructs relationships and the
`WaitingForChildren` dependency, and then resumes through `RigSchedule`.
In-flight effects are rejected unless cancel-and-suspend first produced a
redispatchable prepared generation. Snapshots contain prompts, transcripts,
tool data, policy decisions, and provider content and must therefore be handled
as application-sensitive data. Runtime-only custom observers/systems must be
reinstalled after domain restoration before runs resume.

## Restored example inventory

`.github/example-inventory.txt` is the CI-enforced merge-base inventory. The
complete restored root package set is:

`agent_autonomous`, `agent_evaluator_optimizer`, `agent_orchestrator`,
`agent_parallelization`, `agent_prompt_chaining`, `agent_routing`,
`agent_run_stepping`, `agent_stream_chat`, `agent_with_agent_tool`,
`agent_with_approval_policy`, `agent_with_context`,
`agent_with_default_max_turns`, `agent_with_durable_approval`,
`agent_with_echochambers`, `agent_with_human_in_the_loop`,
`agent_with_loaders`, `agent_with_memory_streaming`, `agent_with_memory`,
`agent_with_tools_otel`, `agent_with_tools`, `agent`, `calculator_chatbot`,
`chain`, `complex_agentic_loop_claude`, `custom_vector_store`, `debate`,
`discord_bot`, `enum_dispatch`, `extractor`, `force_tool_first_turn`,
`gemini_deep_research`, `gemini_default_api_recovery`,
`gemini_extractor_with_rag`, `gemini_nanobanana_image_generation`,
`gemini_stream_kill_token_count`, `gemini_video_understanding`,
`manual_tool_calls`, `multi_agent`, `multi_extract`,
`multi_turn_agent_extended`, `multi_turn_agent`,
`openai_agent_completions_api_otel`, `openai_streaming_per_call_usage`,
`openai_streaming_with_tools_otel`, `pdf_agent`,
`rag_dynamic_tools_multi_turn`, `rag_dynamic_tools`, `rag_ollama`, `rag`,
`reasoning_loop`, `request_hook`, `reqwest_middleware`, `rmcp`,
`sentiment_classifier`, `tool_result_outcomes`, `transcription`,
`vector_search_cohere`, `vector_search_ollama`, and `vector_search`.

Core-native `ecs_runtime` and `ecs_extension` examples additionally demonstrate
standalone and embedded-world execution. The feature-to-example map is:

| Runtime feature | Example |
| --- | --- |
| targeted lifecycle observation / embedded world | `crates/rig-core/examples/ecs_extension.rs` |
| request patch / deterministic ordering | `request_hook` |
| tool rewrite, skip, result redaction | `tool_result_outcomes`, `force_tool_first_turn` |
| approval / durable approval | `agent_with_approval_policy`, `agent_with_durable_approval` |
| invalid repair/retry | `gemini_default_api_recovery` |
| streaming delta observation/cancellation | `agent_stream_chat`, `gemini_stream_kill_token_count` |
| checkpoint/resume and all pause modes | `agent_run_stepping`, `agent_with_durable_approval` |
| shared world, sibling progress, child delegation | `multi_agent`, `agent_with_agent_tool` |

## Benchmark matrix

`cargo bench -p rig-core --bench ecs_runtime` measures all required operational
shapes with deterministic fake effects: one-shot prompts, 100 concurrent runs
on one agent, 100 concurrent runs across eight agents, freeze/resume beside an
active sibling, dynamic child orchestration, 100-policy evaluation, 100 stream
deltas, and a 16-call parallel logical tool batch. Results are machine-specific;
the final PR verification record captures the exact run used for review rather
than presenting these smoke timings as stable performance guarantees. The
2026-07-15 verification run on the PR workstation reported:

| Shape | Result |
| --- | ---: |
| one-shot prompt | 2,088,351.7 ns/op |
| 100 runs, one agent | 30,498.3 ns/op |
| 100 runs, eight agents | 29,802.5 ns/op |
| freeze/resume beside active | 532,625.0 ns/op |
| dynamic child orchestration | 935,812.5 ns/op |
| 100-policy run | 298,031.7 ns/op |
| 100 streaming deltas | 5,725.4 ns/op |
| 16-call tool batch | 48,224.0 ns/op |
