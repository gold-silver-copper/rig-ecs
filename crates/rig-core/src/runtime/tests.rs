use super::*;
use sha2::{Digest, Sha256};

#[derive(Component)]
struct InspectRequestPolicy {
    expected_instructions: String,
    replacement_instructions: String,
}

#[derive(Component)]
struct InspectToolPolicy {
    expected_arguments: serde_json::Value,
    replacement_arguments: serde_json::Value,
}

#[derive(Component)]
struct InspectToolResultPolicy {
    expected_presentation: String,
    replacement_presentation: String,
}

#[derive(Component)]
struct TerminalToolPolicy {
    expected_arguments: serde_json::Value,
    stop: bool,
}

#[derive(Resource, Default)]
struct FinishedTurns(Vec<ModelTurnFinished>);

#[derive(Resource, Default)]
struct ObservedStreamDeltas {
    text: Vec<TextDeltaObserved>,
    tool: Vec<ToolCallDeltaObserved>,
    finished: Vec<StreamResponseFinished>,
}

#[derive(Resource, Default)]
struct FinalizedToolResults(Vec<ToolResultPresentationFinalized>);

#[derive(Resource, Default)]
struct LifecycleLog(Vec<&'static str>);

#[derive(Resource, Default)]
struct CapturedOperationContext(Option<ResolvedOperationContext>);

#[derive(Component)]
#[require(LifecycleRequired)]
struct LifecycleProbe(u32);

#[derive(Component, Default)]
struct LifecycleRequired;

#[derive(Component)]
struct DeferredLifecycleMarker;

#[derive(Resource, Default)]
struct ComponentLifecycleLog(Vec<&'static str>);

#[derive(Resource, Default)]
struct OperationGenerationLifecycleLog(Vec<&'static str>);

#[derive(Resource)]
struct ObservationEnabled(bool);

#[derive(Resource, Default)]
struct ConditionalObservationCount(u32);

#[derive(Resource, Default)]
struct TenantAgentCounts((usize, usize));

fn count_tenant_agents(
    agents: TenantScopedQuery<'_, '_, &'static Agent>,
    mut counts: ResMut<TenantAgentCounts>,
) {
    let a = TenantId::new("a").unwrap();
    let b = TenantId::new("b").unwrap();
    counts.0 = (agents.iter(&a).count(), agents.iter(&b).count());
}

#[derive(EntityEvent)]
struct ConditionalProbe {
    #[event_target]
    target: Entity,
}

#[derive(Component, Clone, Debug, Eq, PartialEq)]
struct SnapshotNote(String);

struct SnapshotNoteCodec {
    revision: u64,
}

struct XorSnapshotProtection;

impl ActiveRunSnapshotProtection for XorSnapshotProtection {
    fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        Ok(plaintext.iter().map(|byte| byte ^ 0xa5).collect())
    }

    fn unprotect(&self, protected: &[u8]) -> Result<Vec<u8>, String> {
        self.protect(protected)
    }
}

impl ActiveRunSnapshotCodec for SnapshotNoteCodec {
    fn binding_id(&self) -> &'static str {
        "tests.snapshot-note"
    }

    fn revision(&self) -> u64 {
        self.revision
    }

    fn capture(&self, world: &World, runs: &[Entity]) -> Result<Option<serde_json::Value>, String> {
        Ok(runs.iter().find_map(|run| {
            let note = world.get::<SnapshotNote>(*run)?;
            let id = world.get::<StableId>(*run)?;
            Some(serde_json::json!({"run_id": id.as_str(), "note": note.0}))
        }))
    }

    fn validate(&self, payload: &serde_json::Value) -> Result<(), String> {
        if payload
            .get("run_id")
            .and_then(serde_json::Value::as_str)
            .is_none()
            || payload
                .get("note")
                .and_then(serde_json::Value::as_str)
                .is_none()
        {
            return Err("run_id and note strings are required".to_owned());
        }
        Ok(())
    }

    fn restore(&self, world: &mut World, runs: &RestoredRuns, payload: &serde_json::Value) {
        let Some(run_id) = payload.get("run_id").and_then(serde_json::Value::as_str) else {
            return;
        };
        let Some(note) = payload.get("note").and_then(serde_json::Value::as_str) else {
            return;
        };
        let Ok(run_id) = StableId::new(run_id) else {
            return;
        };
        if let Some(entity) = runs.0.get(&run_id) {
            world
                .entity_mut(*entity)
                .insert(SnapshotNote(note.to_owned()));
        }
    }
}

fn record_conditional_probe(
    _: On<ConditionalProbe>,
    mut count: ResMut<ConditionalObservationCount>,
) {
    count.0 = count.0.saturating_add(1);
}

fn capture_operation_context(
    context: RigOperationContext<'_, '_>,
    operations: Query<Entity, With<ModelEffectInput>>,
    mut captured: ResMut<CapturedOperationContext>,
) {
    captured.0 = operations
        .iter()
        .next()
        .and_then(|operation| context.resolve(operation));
}

macro_rules! record_lifecycle_event {
    ($function:ident, $event:ty, $name:literal) => {
        fn $function(_: On<$event>, mut log: ResMut<LifecycleLog>) {
            log.0.push($name);
        }
    };
}

record_lifecycle_event!(
    record_request_prepared,
    CompletionRequestPrepared,
    "request-prepared"
);
record_lifecycle_event!(record_model_dispatched, ModelDispatched, "model-dispatched");
record_lifecycle_event!(record_model_settled, ModelSettled, "model-settled");
record_lifecycle_event!(
    record_response_applied,
    CompletionResponseApplied,
    "response-applied"
);
record_lifecycle_event!(
    record_invalid_detected,
    InvalidToolCallDetected,
    "invalid-detected"
);
record_lifecycle_event!(record_tool_prepared, ToolCallPrepared, "tool-prepared");
record_lifecycle_event!(record_tool_started, ToolExecutionStarted, "tool-started");
record_lifecycle_event!(record_tool_settled, ToolExecutionSettled, "tool-settled");
record_lifecycle_event!(
    record_result_finalized,
    ToolResultPresentationFinalized,
    "result-finalized"
);
record_lifecycle_event!(
    record_batch_committed,
    ToolBatchCommitted,
    "batch-committed"
);
record_lifecycle_event!(
    record_persistence_settled,
    PersistenceSettled,
    "persistence-settled"
);
record_lifecycle_event!(record_run_completed, RunCompleted, "run-completed");
record_lifecycle_event!(record_run_failed, RunFailed, "run-failed");
record_lifecycle_event!(record_run_cancelled, RunCancelled, "run-cancelled");

fn inspect_request_policy(
    In(event): In<RequestPolicyInvocation>,
    policies: Query<&InspectRequestPolicy>,
) -> Option<RequestPolicyDecision> {
    let Ok(policy) = policies.get(event.policy) else {
        return None;
    };
    Some(
        if event.request.instructions == policy.expected_instructions {
            RequestPolicyDecision::Patch(
                RequestPatch::new().instructions(policy.replacement_instructions.clone()),
            )
        } else {
            RequestPolicyDecision::Stop("earlier rewrite was not visible".to_owned())
        },
    )
}

fn allow_custom_request(_: In<RequestPolicyInvocation>) -> Option<RequestPolicyDecision> {
    Some(RequestPolicyDecision::Continue)
}

fn stop_custom_request(_: In<RequestPolicyInvocation>) -> Option<RequestPolicyDecision> {
    Some(RequestPolicyDecision::Stop("ambiguous".to_owned()))
}

#[derive(Component)]
struct MultiPointPolicy {
    instructions: String,
    completion: String,
}

fn apply_multi_point_request(
    In(event): In<RequestPolicyInvocation>,
    policies: Query<&MultiPointPolicy>,
) -> Option<RequestPolicyDecision> {
    policies.get(event.policy).ok().map(|policy| {
        RequestPolicyDecision::Patch(RequestPatch::new().instructions(policy.instructions.clone()))
    })
}

fn apply_multi_point_completion(
    In(event): In<CompletionResponsePolicyInvocation>,
    policies: Query<&MultiPointPolicy>,
) -> Option<CompletionResponsePolicyDecision> {
    policies
        .get(event.policy)
        .ok()
        .map(|policy| CompletionResponsePolicyDecision::RewriteText(policy.completion.clone()))
}

fn inspect_provider_diagnostics(
    In(event): In<CompletionResponsePolicyInvocation>,
    diagnostics: Query<&ProviderResponseDiagnostics>,
) -> Option<CompletionResponsePolicyDecision> {
    Some(
        if diagnostics
            .get(event.operation)
            .is_ok_and(|diagnostics| diagnostics.0["provider_request_id"] == "provider-response")
        {
            CompletionResponsePolicyDecision::RewriteText("diagnostics observed".to_owned())
        } else {
            CompletionResponsePolicyDecision::Stop("provider diagnostics missing".to_owned())
        },
    )
}

fn inspect_tool_policy(
    In(event): In<ToolCallPolicyInvocation>,
    policies: Query<&InspectToolPolicy>,
) -> Option<ToolCallPolicyDecision> {
    let Ok(policy) = policies.get(event.policy) else {
        return None;
    };
    Some(if event.call.arguments == policy.expected_arguments {
        ToolCallPolicyDecision::Rewrite(policy.replacement_arguments.clone())
    } else {
        ToolCallPolicyDecision::Stop("earlier argument rewrite was not visible".to_owned())
    })
}

fn finish_tool_policy(
    In(event): In<ToolCallPolicyInvocation>,
    policies: Query<&TerminalToolPolicy>,
) -> Option<ToolCallPolicyDecision> {
    let Ok(policy) = policies.get(event.policy) else {
        return None;
    };
    Some(if event.call.arguments != policy.expected_arguments {
        ToolCallPolicyDecision::Stop("earlier rewrite was not retained".to_owned())
    } else if policy.stop {
        ToolCallPolicyDecision::Stop("operator stopped dispatch".to_owned())
    } else {
        ToolCallPolicyDecision::Skip("operator skipped dispatch".to_owned())
    })
}

fn isolate_tool_call_policy(
    In(event): In<ToolCallPolicyInvocation>,
) -> Option<ToolCallPolicyDecision> {
    let Some(value) = event.call.arguments["value"].as_i64() else {
        return Some(ToolCallPolicyDecision::Stop(
            "missing isolated input".to_owned(),
        ));
    };
    Some(ToolCallPolicyDecision::Rewrite(
        serde_json::json!({"value": value + 10}),
    ))
}

fn stop_second_tool_call(
    In(event): In<ToolCallPolicyInvocation>,
) -> Option<ToolCallPolicyDecision> {
    Some(if event.call.call_id == "second" {
        ToolCallPolicyDecision::Stop("second call terminates the batch".to_owned())
    } else {
        ToolCallPolicyDecision::Run
    })
}

fn inspect_tool_result_policy(
    In(event): In<ToolResultPolicyInvocation>,
    policies: Query<&InspectToolResultPolicy>,
) -> Option<ToolResultPolicyDecision> {
    let Ok(policy) = policies.get(event.policy) else {
        return None;
    };
    Some(
        if event.result.presentation == policy.expected_presentation {
            ToolResultPolicyDecision::Rewrite(policy.replacement_presentation.clone().into())
        } else {
            ToolResultPolicyDecision::Stop("earlier result rewrite was not visible".to_owned())
        },
    )
}

fn fail_invalid_tool(
    _: In<InvalidToolCallPolicyInvocation>,
) -> Option<InvalidToolCallPolicyDecision> {
    Some(InvalidToolCallPolicyDecision::Fail)
}

fn skip_invalid_tool(
    _: In<InvalidToolCallPolicyInvocation>,
) -> Option<InvalidToolCallPolicyDecision> {
    Some(InvalidToolCallPolicyDecision::Skip(
        "synthetic skip".to_owned(),
    ))
}

fn stop_invalid_tool(
    _: In<InvalidToolCallPolicyInvocation>,
) -> Option<InvalidToolCallPolicyDecision> {
    Some(InvalidToolCallPolicyDecision::Stop(
        "operator stopped the run".to_owned(),
    ))
}

fn record_finished_turn(event: On<ModelTurnFinished>, mut turns: ResMut<FinishedTurns>) {
    turns.0.push(event.event().clone());
}

fn record_text_delta(event: On<TextDeltaObserved>, mut observed: ResMut<ObservedStreamDeltas>) {
    observed.text.push(event.event().clone());
}

fn record_tool_delta(event: On<ToolCallDeltaObserved>, mut observed: ResMut<ObservedStreamDeltas>) {
    observed.tool.push(event.event().clone());
}

fn record_stream_finished(
    event: On<StreamResponseFinished>,
    mut observed: ResMut<ObservedStreamDeltas>,
) {
    observed.finished.push(event.event().clone());
}

fn record_finalized_tool_result(
    event: On<ToolResultPresentationFinalized>,
    mut finalized: ResMut<FinalizedToolResults>,
) {
    finalized.0.push(event.event().clone());
}

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn tenant(value: &str) -> TenantId {
    TenantId::new(value).unwrap()
}

fn runtime() -> Runtime {
    Runtime::new(RuntimeConfig::default()).unwrap()
}

fn policy_termination<'a>(
    runtime: &'a Runtime,
    run: Entity,
    expected_policy: &str,
) -> &'a PolicyTermination {
    let Some(RunState::Failed(CanonicalError::PolicyTerminated { termination })) =
        runtime.world().get::<RunState>(run)
    else {
        panic!("expected policy termination");
    };
    assert_eq!(termination.policy_id.as_str(), expected_policy);
    termination
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

fn advance_to_lookup_tool(
    runtime: &mut Runtime,
    agent: AgentHandle,
) -> (PendingRunHandle, EffectRequest) {
    let pending = runtime.handle().prompt(agent, "use lookup").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let tool = runtime.effects().try_recv().unwrap().unwrap();
    (pending, tool)
}

fn submit_invalid_tool_call(
    runtime: &mut Runtime,
    agent: AgentHandle,
    name: &str,
) -> PendingRunHandle {
    let pending = runtime
        .handle()
        .prompt(agent, "use a missing tool")
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "invalid-call".to_owned(),
                    provider_result_id: "provider-result".to_owned(),
                    provider_call_id: Some("provider-call".to_owned()),
                    name: name.to_owned(),
                    arguments: serde_json::json!({"query": "weather"}),
                }],
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    pending
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
fn lifecycle_observations_cover_model_tool_batch_and_terminal_boundaries() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime.world_mut().insert_resource(LifecycleLog::default());
    runtime.world_mut().add_observer(record_request_prepared);
    runtime.world_mut().add_observer(record_model_dispatched);
    runtime.world_mut().add_observer(record_model_settled);
    runtime.world_mut().add_observer(record_response_applied);
    runtime.world_mut().add_observer(record_invalid_detected);
    runtime.world_mut().add_observer(record_tool_prepared);
    runtime.world_mut().add_observer(record_tool_started);
    runtime.world_mut().add_observer(record_tool_settled);
    runtime.world_mut().add_observer(record_result_finalized);
    runtime.world_mut().add_observer(record_batch_committed);
    runtime.world_mut().add_observer(record_persistence_settled);
    runtime.world_mut().add_observer(record_run_completed);
    runtime.world_mut().add_observer(record_run_failed);
    runtime.world_mut().add_observer(record_run_cancelled);

    let (_, tool) = advance_to_lookup_tool(&mut runtime, agent);
    let input = tool.tool_input().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: input.call_id.clone(),
                provider_result_id: input.provider_result_id.clone(),
                provider_call_id: input.provider_call_id.clone(),
                name: input.decision.name.clone(),
                raw: serde_json::json!({"answer": 42}).into(),
                presentation: "42".into(),
                failure: None,
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "done".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();

    assert_eq!(
        runtime.world().resource::<LifecycleLog>().0,
        vec![
            "request-prepared",
            "model-dispatched",
            "model-settled",
            "response-applied",
            "tool-prepared",
            "tool-started",
            "tool-settled",
            "result-finalized",
            "batch-committed",
            "request-prepared",
            "model-dispatched",
            "model-settled",
            "response-applied",
            "run-completed",
        ]
    );
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
            provider_correlation: None,
            kind: EffectDeltaKind::Text("he".to_owned()),
        })
        .unwrap();
    deltas
        .try_send(EffectDelta {
            operation: request.operation,
            generation: request.generation,
            sequence: 1,
            provider_correlation: None,
            kind: EffectDeltaKind::Text("llo".to_owned()),
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
fn streaming_cancellation_publishes_terminal_and_rejects_late_completion() {
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
    let (pending, stream) = runtime.handle().prompt_stream(agent, "cancel").unwrap();
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
                text: "late".to_owned(),
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
        stream.try_recv().unwrap(),
        Some(StreamItem::Finished(StreamTerminal::Cancelled))
    );
    assert_eq!(stream.try_recv().unwrap(), None);
    runtime.run_until_stalled().unwrap();
    assert!(runtime.world().get_entity(run.entity()).is_err());
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
    let retry = runtime.effects().try_recv().unwrap().unwrap();
    assert_ne!(retry.operation, second.operation);
    let retry_input = retry.model_input().unwrap();
    assert!(retry_input.history.iter().any(|entry| matches!(
        entry,
        TranscriptEntry::User(feedback)
            if feedback.contains("did not satisfy the required output schema")
    )));
    let retry_record = runtime
        .world()
        .get::<RunRecord>(second_run.entity())
        .unwrap();
    assert_eq!(retry_record.structured_output_retries, 1);
    assert_eq!(retry_record.max_structured_output_retries, 1);
    runtime
        .handle()
        .pause_with_mode(second_run, PauseMode::CancelAndSuspend)
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let checkpoint = runtime.snapshot_active_run(second_run).unwrap();
    let checkpoint_record = &checkpoint
        .runs
        .iter()
        .find(|run| run.id == *second_pending.stable_id())
        .unwrap()
        .record;
    assert_eq!(checkpoint_record.structured_output_retries, 1);
    assert_eq!(checkpoint_record.max_structured_output_retries, 1);
    runtime.handle().resume(second_run).unwrap();
    runtime.run_until_stalled().unwrap();
    let retry = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: retry.operation,
            generation: retry.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: r#"{"old":"still rejected"}"#.to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
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
fn semantic_tool_retrieval_filters_and_refreshes_each_model_operation() {
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
    for (order, (tool_id, name)) in [
        ("static", "static"),
        ("candidate-alpha", "alpha"),
        ("candidate-beta", "beta"),
    ]
    .into_iter()
    .enumerate()
    {
        let tool = runtime
            .spawn_tool(
                id(tool_id),
                tenant("a"),
                ToolCapability {
                    name: name.to_owned(),
                    description: name.to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: u32::try_from(order).unwrap(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_tool(
                id(&format!("grant-{name}")),
                tenant("a"),
                ToolGrant {
                    order: u32::try_from(order).unwrap(),
                    enabled: true,
                },
                agent,
                tool,
            )
            .unwrap();
    }
    let store = runtime
        .spawn_store(
            id("tool-index"),
            tenant("a"),
            StoreCapability {
                kind: "tool-vector-search".to_owned(),
                revision: 7,
                retired: false,
            },
        )
        .unwrap();
    runtime
        .grant_store(
            id("tool-index-grant"),
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
        .set_tool_retrieval_requirement(
            agent,
            ToolRetrievalRequirement {
                limit: 1,
                candidates: vec!["alpha".to_owned(), "beta".to_owned()],
            },
        )
        .unwrap();

    runtime.handle().prompt(agent, "select beta").unwrap();
    runtime.run_until_stalled().unwrap();
    let retrieval = runtime.effects().try_recv().unwrap().unwrap();
    let retrieval_input = retrieval.store_input().unwrap();
    assert_eq!(retrieval_input.decision.kind, "tool-vector-search");
    assert_eq!(retrieval_input.decision.revision, 7);
    assert!(matches!(
        &retrieval_input.operation,
        StoreOperation::Retrieve { query, limit } if query == "select beta" && *limit == 1
    ));
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: retrieval.operation,
            generation: retrieval.generation,
            result: Ok(EffectOutput::Store(StoreEffectOutput::Retrieved(vec![
                RetrievedDocument {
                    id: "beta".to_owned(),
                    text: String::new(),
                    metadata: BTreeMap::new(),
                },
            ]))),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let model_request = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        model_request
            .model_input()
            .unwrap()
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        vec!["beta", "static"]
    );

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
                tool_calls: vec![ModelToolCall {
                    id: "beta-call".to_owned(),
                    provider_result_id: "beta-call".to_owned(),
                    provider_call_id: None,
                    name: "beta".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let tool_request = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool_request.operation,
            generation: tool_request.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: "beta-call".to_owned(),
                provider_result_id: "beta-call".to_owned(),
                provider_call_id: None,
                name: "beta".to_owned(),
                raw: serde_json::json!("ok").into(),
                presentation: "ok".into(),
                failure: None,
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let next_retrieval = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        next_retrieval.store_input().unwrap().decision.kind,
        "tool-vector-search"
    );
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
                provider_correlation: None,
                kind: EffectDeltaKind::Text(text.to_owned()),
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
            provider_correlation: None,
            kind: EffectDeltaKind::Text("late".to_owned()),
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
fn streaming_ingress_reports_backpressure_at_the_configured_bound() {
    let runtime = Runtime::new(RuntimeConfig {
        completion_capacity: 1,
        ..RuntimeConfig::default()
    })
    .unwrap();
    let deltas = runtime.effects().delta_sender();
    let delta = |sequence| EffectDelta {
        operation: Entity::PLACEHOLDER,
        generation: 0,
        sequence,
        provider_correlation: None,
        kind: EffectDeltaKind::Text(format!("delta-{sequence}")),
    };

    assert_eq!(deltas.try_send(delta(0)), Ok(()));
    assert_eq!(deltas.try_send(delta(1)), Err(EffectIoError::Backpressure));
}

#[test]
fn text_and_tool_call_deltas_share_sequence_and_observation_pipeline() {
    let mut runtime = runtime();
    runtime
        .world_mut()
        .insert_resource(ObservedStreamDeltas::default());
    runtime.world_mut().add_observer(record_text_delta);
    runtime.world_mut().add_observer(record_stream_finished);
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
    runtime
        .world_mut()
        .entity_mut(request.operation)
        .observe(record_tool_delta);
    let deltas = runtime.effects().delta_sender();
    for delta in [
        EffectDelta {
            operation: request.operation,
            generation: request.generation,
            sequence: 0,
            provider_correlation: Some("message-1".to_owned()),
            kind: EffectDeltaKind::Text("hello".to_owned()),
        },
        EffectDelta {
            operation: request.operation,
            generation: request.generation,
            sequence: 1,
            provider_correlation: Some("call-1".to_owned()),
            kind: EffectDeltaKind::ToolCall {
                id: "call-1".to_owned(),
                internal_call_id: "internal-1".to_owned(),
                content: ToolCallDeltaContent::Name("lookup".to_owned()),
            },
        },
        EffectDelta {
            operation: request.operation,
            generation: request.generation,
            sequence: 2,
            provider_correlation: Some("call-1".to_owned()),
            kind: EffectDeltaKind::ToolCall {
                id: "call-1".to_owned(),
                internal_call_id: "internal-1".to_owned(),
                content: ToolCallDeltaContent::Delta("{\"q\":".to_owned()),
            },
        },
    ] {
        deltas.try_send(delta).unwrap();
    }
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "hello".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();

    assert_eq!(
        stream.try_recv().unwrap(),
        Some(StreamItem::Delta {
            sequence: 0,
            text: "hello".to_owned(),
        })
    );
    assert!(matches!(
        stream.try_recv().unwrap(),
        Some(StreamItem::ToolCallDelta {
            sequence: 1,
            id,
            internal_call_id,
            content: ToolCallDeltaContent::Name(name),
        }) if id == "call-1" && internal_call_id == "internal-1" && name == "lookup"
    ));
    assert!(matches!(
        stream.try_recv().unwrap(),
        Some(StreamItem::ToolCallDelta {
            sequence: 2,
            content: ToolCallDeltaContent::Delta(fragment),
            ..
        }) if fragment == "{\"q\":"
    ));
    let observed = runtime.world().resource::<ObservedStreamDeltas>();
    assert_eq!(observed.text.len(), 1);
    assert_eq!(observed.tool.len(), 2);
    assert_eq!(observed.finished.len(), 1);
    assert_eq!(
        observed
            .text
            .first()
            .and_then(|event| event.provider_correlation.as_deref()),
        Some("message-1")
    );
    assert_eq!(observed.tool.first().map(|event| event.sequence), Some(1));
    assert_eq!(observed.tool.get(1).map(|event| event.sequence), Some(2));
    let mut evaluations = runtime
        .world_mut()
        .query_filtered::<Entity, With<TextDeltaPolicyEvaluation>>();
    assert_eq!(evaluations.iter(runtime.world()).count(), 0);
    let mut tool_evaluations = runtime
        .world_mut()
        .query_filtered::<Entity, With<ToolCallDeltaPolicyEvaluation>>();
    assert_eq!(tool_evaluations.iter(runtime.world()).count(), 0);
}

#[test]
fn text_delta_policy_preserves_sequence_and_stops_before_publication() {
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
            id("a-pass-first"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::StopTextDeltaContains {
                    needle: "never present".to_owned(),
                    reason: "must not stop".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    runtime
        .spawn_policy(
            id("stream-guard"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 2,
                rule: PolicyRule::StopTextDeltaContains {
                    needle: "blocked".to_owned(),
                    reason: "unsafe stream".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let (_, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    let deltas = runtime.effects().delta_sender();
    for (sequence, text) in [(0, "safe"), (1, "blocked")] {
        deltas
            .try_send(EffectDelta {
                operation: request.operation,
                generation: request.generation,
                sequence,
                provider_correlation: None,
                kind: EffectDeltaKind::Text(text.to_owned()),
            })
            .unwrap();
    }

    runtime.run_until_stalled().unwrap();
    assert_eq!(
        stream.try_recv().unwrap(),
        Some(StreamItem::Delta {
            sequence: 0,
            text: "safe".to_owned(),
        })
    );
    assert!(matches!(
        stream.try_recv().unwrap(),
        Some(StreamItem::Finished(StreamTerminal::Failed(
            CanonicalError::PolicyTerminated { termination }
        ))) if termination.policy_id.as_str() == "stream-guard"
    ));
    assert_eq!(stream.try_recv().unwrap(), None);
    let mut evaluations = runtime
        .world_mut()
        .query::<(&AcceptedTextDeltaPolicies, &TextDeltaPolicyEvaluation)>();
    let accepted = evaluations
        .iter(runtime.world())
        .find(|(_, evaluation)| evaluation.sequence == 1)
        .map(|(accepted, _)| {
            accepted
                .0
                .iter()
                .map(|policy| policy.id.as_str())
                .collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(accepted, ["a-pass-first", "stream-guard"]);
}

#[test]
fn tool_call_delta_policy_preserves_correlation_and_stops_before_publication() {
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
            id("tool-delta-guard"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 3,
                rule: PolicyRule::StopToolCallDeltaContains {
                    needle: "secret".to_owned(),
                    reason: "unsafe tool arguments".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let (_, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .delta_sender()
        .try_send(EffectDelta {
            operation: request.operation,
            generation: request.generation,
            sequence: 0,
            provider_correlation: Some("provider-call-7".to_owned()),
            kind: EffectDeltaKind::ToolCall {
                id: "call-7".to_owned(),
                internal_call_id: "internal-7".to_owned(),
                content: ToolCallDeltaContent::Delta("{\"secret\":".to_owned()),
            },
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    assert!(matches!(
        stream.try_recv().unwrap(),
        Some(StreamItem::Finished(StreamTerminal::Failed(
            CanonicalError::PolicyTerminated { termination }
        ))) if termination.policy_id.as_str() == "tool-delta-guard"
    ));
    assert_eq!(stream.try_recv().unwrap(), None);
    let mut evaluations = runtime.world_mut().query::<(
        &AcceptedToolCallDeltaPolicies,
        &ToolCallDeltaPolicyEvaluation,
    )>();
    let (accepted, evaluation) = evaluations.iter(runtime.world()).next().unwrap();
    assert_eq!(accepted.0[0].point, PolicyPoint::ToolCallDelta);
    assert_eq!(evaluation.sequence, 0);
    assert_eq!(
        evaluation.provider_correlation.as_deref(),
        Some("provider-call-7")
    );
    assert_eq!(evaluation.id, "call-7");
    assert_eq!(evaluation.internal_call_id, "internal-7");
}

#[test]
fn tool_call_delta_policy_awaits_correlated_approval() {
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
            id("tool-delta-approval"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 4,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::ToolCallDelta,
                    prompt: "approve streamed tool arguments".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let (_, stream) = runtime.handle().prompt_stream(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .delta_sender()
        .try_send(EffectDelta {
            operation: request.operation,
            generation: request.generation,
            sequence: 0,
            provider_correlation: Some("provider-call-9".to_owned()),
            kind: EffectDeltaKind::ToolCall {
                id: "call-9".to_owned(),
                internal_call_id: "internal-9".to_owned(),
                content: ToolCallDeltaContent::Name("lookup".to_owned()),
            },
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    assert_eq!(stream.try_recv().unwrap(), None);
    let approval = runtime.effects().try_recv().unwrap().unwrap();
    let approval_input = approval.policy_approval_input().unwrap();
    assert_eq!(approval_input.point, PolicyPoint::ToolCallDelta);
    assert_eq!(approval_input.revision, 4);
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: Some("reviewed".to_owned()),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();

    assert!(matches!(
        stream.try_recv().unwrap(),
        Some(StreamItem::ToolCallDelta {
            sequence: 0,
            id,
            internal_call_id,
            content: ToolCallDeltaContent::Name(name),
        }) if id == "call-9" && internal_call_id == "internal-9" && name == "lookup"
    ));
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
                provider_correlation: None,
                kind: EffectDeltaKind::Text(sequence.to_string()),
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
fn tenant_scoped_query_cannot_iterate_another_tenants_components() {
    let mut runtime = runtime();
    for name in ["a", "b"] {
        let scope = tenant(name);
        let model = runtime
            .spawn_model(
                StableId::new(format!("model-{name}")).unwrap(),
                scope.clone(),
                ModelCapability {
                    provider: "fake".to_owned(),
                    model: "test".to_owned(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .spawn_agent(
                StableId::new(format!("agent-{name}")).unwrap(),
                scope,
                Agent::default(),
                model,
            )
            .unwrap();
    }
    runtime
        .world_mut()
        .insert_resource(TenantAgentCounts::default());
    let mut schedule = Schedule::default();
    schedule.add_systems(count_tenant_agents);
    schedule.run(runtime.world_mut());
    assert_eq!(runtime.world().resource::<TenantAgentCounts>().0, (1, 1));
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
    let policy = runtime
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
    let termination = policy_termination(&runtime, run.entity(), "deny-secret");
    assert_eq!(termination.revision, 4);
    assert_eq!(termination.point, PolicyPoint::Request);
    assert_eq!(termination.reason, "prompt contains `secret`");
    assert_eq!(termination.run_id, *pending.stable_id());
    assert!(!termination.history.is_empty());
    let operation = runtime
        .world()
        .get::<RunOperations>(run.entity())
        .unwrap()
        .iter()
        .next()
        .unwrap();
    assert!(termination.operation_id.as_str().starts_with("operation-"));
    assert_eq!(
        runtime.world().get::<AcceptedPolicies>(operation),
        Some(&AcceptedPolicies(vec![AcceptedPolicy {
            id: id("deny-secret"),
            entity: policy,
            revision: 4,
            order: 10,
            point: PolicyPoint::Request,
        }]))
    );
}

#[test]
fn request_patches_reduce_by_order_and_remain_operation_local() {
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
        .spawn_agent(
            id("agent"),
            tenant("a"),
            Agent {
                instructions: "baseline".to_owned(),
                additional_params: Some(serde_json::json!({"baseline": true, "winner": 0})),
                ..Agent::default()
            },
            model,
        )
        .unwrap();
    runtime
        .spawn_policy(
            id("z-last"),
            tenant("a"),
            Policy {
                order: 10,
                revision: 2,
                rule: PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .instructions("last")
                        .history([TranscriptEntry::Assistant("prior".to_owned())])
                        .additional_params(serde_json::json!({"winner": 2, "z": true})),
                ),
            },
            agent,
        )
        .unwrap();
    runtime
        .spawn_policy(
            id("a-first"),
            tenant("a"),
            Policy {
                order: 10,
                revision: 1,
                rule: PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .instructions("first")
                        .additional_params(serde_json::json!({"winner": 1, "a": true})),
                ),
            },
            agent,
        )
        .unwrap();

    runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let effect = runtime.effects().try_recv().unwrap().unwrap();
    let input = effect.model_input().unwrap();
    assert_eq!(input.instructions, "last");
    assert_eq!(
        input.history,
        vec![
            TranscriptEntry::Assistant("prior".to_owned()),
            TranscriptEntry::Message(input.prompt.clone()),
        ]
    );
    assert_eq!(
        input.additional_params,
        Some(serde_json::json!({
            "a": true,
            "baseline": true,
            "winner": 2,
            "z": true
        }))
    );
    let evaluation = runtime
        .world()
        .get::<OperationPolicyEvaluations>(effect.operation)
        .unwrap()
        .iter()
        .find_map(|entity| runtime.world().get::<RequestPolicyEvaluation>(entity))
        .unwrap();
    assert_eq!(evaluation.accumulated.instructions.as_deref(), Some("last"));
    assert_eq!(
        evaluation.accumulated.history,
        Some(vec![TranscriptEntry::Assistant("prior".to_owned())])
    );
    assert_eq!(
        evaluation.accumulated.additional_params,
        Some(serde_json::json!({"a": true, "winner": 2, "z": true}))
    );
    assert_eq!(
        runtime
            .world()
            .get::<Agent>(agent.entity())
            .unwrap()
            .instructions,
        "baseline"
    );
}

#[test]
fn request_patch_covers_every_field_without_accumulating_on_later_turns() {
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
    let baseline_document = RetrievedDocument {
        id: "baseline".to_owned(),
        text: "baseline context".to_owned(),
        metadata: BTreeMap::new(),
    };
    let agent = runtime
        .spawn_agent(
            id("agent"),
            tenant("a"),
            Agent {
                instructions: "baseline instructions".to_owned(),
                temperature_bits: Some(0.9_f64.to_bits()),
                max_tokens: Some(1_000),
                tool_choice: Some(ModelToolChoice::Auto),
                additional_params: Some(serde_json::json!({
                    "baseline": true,
                    "nested": {"baseline": true}
                })),
                documents: vec![baseline_document.clone()],
                ..Agent::default()
            },
            model,
        )
        .unwrap();
    for (order, name) in ["alpha", "beta", "gamma"].into_iter().enumerate() {
        let tool = runtime
            .spawn_tool(
                id(&format!("{name}-tool")),
                tenant("a"),
                ToolCapability {
                    name: name.to_owned(),
                    description: name.to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: u32::try_from(order).unwrap(),
                    revision: 1,
                    retired: false,
                },
            )
            .unwrap();
        runtime
            .grant_tool(
                id(&format!("{name}-grant")),
                tenant("a"),
                ToolGrant {
                    order: u32::try_from(order).unwrap(),
                    enabled: true,
                },
                agent,
                tool,
            )
            .unwrap();
    }
    let first_document = RetrievedDocument {
        id: "first".to_owned(),
        text: "first context".to_owned(),
        metadata: BTreeMap::new(),
    };
    let last_document = RetrievedDocument {
        id: "last".to_owned(),
        text: "last context".to_owned(),
        metadata: BTreeMap::new(),
    };
    runtime
        .spawn_policy(
            id("z-last"),
            tenant("a"),
            Policy {
                order: 5,
                revision: 2,
                rule: PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .instructions("last instructions")
                        .temperature(0.25)
                        .max_tokens(64)
                        .tool_choice(ModelToolChoice::Specific(vec!["beta".to_owned()]))
                        .active_tools(["beta", "gamma"])
                        .additional_params(serde_json::json!({
                            "last": true,
                            "nested": {"last": true}
                        }))
                        .extra_context([last_document.clone()])
                        .history([TranscriptEntry::Assistant("last history".to_owned())]),
                ),
            },
            agent,
        )
        .unwrap();
    runtime
        .spawn_policy(
            id("a-first"),
            tenant("a"),
            Policy {
                order: 5,
                revision: 1,
                rule: PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .instructions("first instructions")
                        .temperature(0.5)
                        .max_tokens(128)
                        .tool_choice(ModelToolChoice::Required)
                        .active_tools(["alpha", "beta"])
                        .additional_params(serde_json::json!({
                            "first": true,
                            "nested": {"first": true}
                        }))
                        .extra_context([first_document.clone()])
                        .history([TranscriptEntry::Assistant("first history".to_owned())]),
                ),
            },
            agent,
        )
        .unwrap();

    let pending = runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let first_model = runtime.effects().try_recv().unwrap().unwrap();
    let assert_effective = |input: &ModelEffectInput| {
        assert_eq!(input.instructions, "last instructions");
        assert_eq!(input.temperature_bits, Some(0.25_f64.to_bits()));
        assert_eq!(input.max_tokens, Some(64));
        assert_eq!(
            input.tool_choice,
            Some(ModelToolChoice::Specific(vec!["beta".to_owned()]))
        );
        assert_eq!(
            input
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["beta"]
        );
        assert_eq!(
            input.additional_params,
            Some(serde_json::json!({
                "baseline": true,
                "first": true,
                "last": true,
                "nested": {"last": true}
            }))
        );
        assert_eq!(
            input.documents,
            vec![
                baseline_document.clone(),
                first_document.clone(),
                last_document.clone(),
            ]
        );
        assert_eq!(
            input.history,
            vec![
                TranscriptEntry::Assistant("last history".to_owned()),
                TranscriptEntry::Message(input.prompt.clone()),
            ]
        );
    };
    assert_effective(first_model.model_input().unwrap());
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: first_model.operation,
            generation: first_model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "beta-call".to_owned(),
                    provider_result_id: "beta-call".to_owned(),
                    provider_call_id: None,
                    name: "beta".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let tool = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: "beta-call".to_owned(),
                provider_result_id: "beta-call".to_owned(),
                provider_call_id: None,
                name: "beta".to_owned(),
                raw: serde_json::json!("ok").into(),
                presentation: "ok".into(),
                failure: None,
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let second_model = runtime.effects().try_recv().unwrap().unwrap();
    assert_effective(second_model.model_input().unwrap());

    let baseline = runtime.world().get::<Agent>(agent.entity()).unwrap();
    assert_eq!(baseline.instructions, "baseline instructions");
    assert_eq!(baseline.temperature_bits, Some(0.9_f64.to_bits()));
    assert_eq!(baseline.max_tokens, Some(1_000));
    assert_eq!(baseline.tool_choice, Some(ModelToolChoice::Auto));
    assert_eq!(baseline.documents, vec![baseline_document]);
    assert_eq!(
        baseline.additional_params,
        Some(serde_json::json!({
            "baseline": true,
            "nested": {"baseline": true}
        }))
    );
    let run = runtime.resolve_run(&pending).unwrap();
    assert_eq!(
        runtime
            .world()
            .get::<RunRecord>(run.entity())
            .unwrap()
            .next_turn,
        1
    );
}

#[test]
fn request_patch_non_object_provider_params_replace_prior_values() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime.handle().prompt(agent, "params").unwrap();
    runtime.run_until_stalled().unwrap();
    let mut input = runtime
        .effects()
        .try_recv()
        .unwrap()
        .unwrap()
        .model_input()
        .unwrap()
        .clone();
    let mut accumulated = RequestPatch::default();
    reduce_request_patch(
        &mut input,
        &mut accumulated,
        RequestPatch::new().additional_params(serde_json::json!({"first": true})),
        &id("first"),
    );
    reduce_request_patch(
        &mut input,
        &mut accumulated,
        RequestPatch::new().additional_params(serde_json::json!("replacement")),
        &id("second"),
    );
    assert_eq!(
        input.additional_params,
        Some(serde_json::json!("replacement"))
    );
    assert_eq!(
        accumulated.additional_params,
        Some(serde_json::json!("replacement"))
    );
}

#[test]
fn extension_bundle_telemetry_and_system_param_use_native_ecs_state() {
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
        .spawn_agent(
            id("agent"),
            tenant("a"),
            Agent {
                name: Some("audited".to_owned()),
                ..Agent::default()
            },
            model,
        )
        .unwrap();
    runtime
        .world_mut()
        .entity_mut(agent.entity())
        .insert(LifecycleTelemetryBundle::default());
    runtime
        .install_extension(&RequestPatchPolicyBundle::new(
            id("request-patch"),
            tenant("a"),
            agent.entity(),
            0,
            1,
            RequestPatch::new().instructions("extension-owned"),
        ))
        .unwrap();

    let pending = runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        request.model_input().unwrap().instructions,
        "extension-owned"
    );

    runtime
        .world_mut()
        .insert_resource(CapturedOperationContext::default());
    let mut context_schedule = Schedule::default();
    context_schedule.add_systems(capture_operation_context);
    context_schedule.run(runtime.world_mut());
    let context = runtime
        .world()
        .resource::<CapturedOperationContext>()
        .0
        .as_ref()
        .unwrap();
    assert_eq!(context.operation, request.operation);
    assert_eq!(context.agent, agent.entity());
    assert_eq!(context.agent_name.as_deref(), Some("audited"));

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
    assert_eq!(
        runtime
            .world()
            .get::<LifecycleTelemetry>(agent.entity())
            .copied(),
        Some(LifecycleTelemetry {
            requests_prepared: 1,
            model_turns_committed: 1,
            tool_batches_committed: 0,
            runs_completed: 1,
            runs_failed: 0,
            runs_cancelled: 0,
        })
    );
}

#[test]
fn typed_extension_effect_reuses_generation_tenant_and_cancellation_lifecycle() {
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct ModerationInput(String);

    #[derive(Debug, Eq, PartialEq)]
    struct ModerationOutput {
        allowed: bool,
    }

    struct Moderation;

    impl EcsEffect for Moderation {
        type Input = ModerationInput;
        type Output = ModerationOutput;
        type Error = String;

        const KIND: &'static str = "example.moderation.v1";
    }

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
    let pending = runtime.handle().prompt(agent, "screen this").unwrap();
    runtime.run_until_stalled().unwrap();
    let _model_request = runtime.effects().try_recv().unwrap().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();

    let operation = spawn_extension_effect::<Moderation>(
        runtime.world_mut(),
        run.entity(),
        id("moderation-1"),
        ModerationInput("screen this".to_owned()),
    )
    .unwrap();
    assert_eq!(
        runtime.world().get::<TenantId>(operation),
        Some(&tenant("a"))
    );
    assert_eq!(
        runtime
            .world()
            .get::<OperationOf>(operation)
            .map(|owner| owner.get()),
        Some(run.entity())
    );
    let (generation, input) =
        dispatch_extension_effect::<Moderation>(runtime.world_mut(), operation).unwrap();
    assert_eq!(input, ModerationInput("screen this".to_owned()));
    assert_eq!(
        settle_extension_effect::<Moderation>(
            runtime.world_mut(),
            operation,
            generation + 1,
            Ok(ModerationOutput { allowed: true }),
        ),
        Err(ExtensionEffectLifecycleError::StaleGeneration {
            expected: generation,
            received: generation + 1,
        })
    );
    settle_extension_effect::<Moderation>(
        runtime.world_mut(),
        operation,
        generation,
        Ok(ModerationOutput { allowed: true }),
    )
    .unwrap();
    assert!(matches!(
        runtime.world().get::<OperationState>(operation),
        Some(OperationState {
            phase: OperationPhase::ExtensionSettled,
        })
    ));
    assert_eq!(
        runtime
            .world()
            .get::<ExtensionEffectResult<Moderation>>(operation)
            .and_then(|result| result.0.as_ref().ok()),
        Some(&ModerationOutput { allowed: true })
    );

    let cancelled = spawn_extension_effect::<Moderation>(
        runtime.world_mut(),
        run.entity(),
        id("moderation-2"),
        ModerationInput("cancel me".to_owned()),
    )
    .unwrap();
    let (cancelled_generation, _) =
        dispatch_extension_effect::<Moderation>(runtime.world_mut(), cancelled).unwrap();
    runtime.handle().cancel(run).unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(matches!(
        runtime.world().get::<OperationState>(cancelled),
        Some(OperationState {
            phase: OperationPhase::Cancelled,
        })
    ));
    assert_eq!(
        settle_extension_effect::<Moderation>(
            runtime.world_mut(),
            cancelled,
            cancelled_generation,
            Ok(ModerationOutput { allowed: false }),
        ),
        Err(ExtensionEffectLifecycleError::InvalidPhase)
    );
}

#[test]
fn serialized_policy_input_materializes_typed_runtime_facts() {
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
    let policy = runtime
        .spawn_policy(
            id("typed-rewrite"),
            tenant("a"),
            Policy {
                order: 3,
                revision: 7,
                rule: PolicyRule::RewriteToolArguments {
                    tool: Some("lookup".to_owned()),
                    arguments: serde_json::json!({"query": "typed"}),
                },
            },
            agent,
        )
        .unwrap();

    assert_eq!(
        runtime.world().get::<PolicyCapabilities>(policy),
        Some(&PolicyCapabilities::new([PolicyPoint::ToolCall]))
    );
    assert_eq!(
        runtime.world().get::<RewriteToolArgumentsPolicy>(policy),
        Some(&RewriteToolArgumentsPolicy {
            tool: Some("lookup".to_owned()),
            arguments: serde_json::json!({"query": "typed"}),
        })
    );
    assert!(runtime.world().get::<RequestPatchPolicy>(policy).is_none());
}

#[test]
fn extension_owned_typed_policy_can_serve_multiple_points() {
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
    let policy = runtime
        .spawn_typed_policy(
            id("multi-point"),
            tenant("a"),
            PolicyMeta {
                order: 2,
                revision: 9,
            },
            PolicyCapabilities::new([PolicyPoint::Request, PolicyPoint::CompletionResponse]),
            MultiPointPolicy {
                instructions: "typed request".to_owned(),
                completion: "typed completion".to_owned(),
            },
            agent,
        )
        .unwrap();
    runtime
        .register_request_policy_responder(
            policy,
            PolicyResponderId::new("multi-point:request").unwrap(),
            apply_multi_point_request,
        )
        .unwrap();
    runtime
        .register_completion_response_policy_responder(
            policy,
            PolicyResponderId::new("multi-point:completion").unwrap(),
            apply_multi_point_completion,
        )
        .unwrap();
    assert_eq!(
        runtime.register_tool_call_policy_responder(
            policy,
            PolicyResponderId::new("multi-point:wrong").unwrap(),
            isolate_tool_call_policy,
        ),
        Err(PolicyResponderRegistrationError::CapabilityMismatch(
            PolicyPoint::ToolCall
        ))
    );

    let pending = runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(request.model_input().unwrap().instructions, "typed request");
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "provider completion".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();

    let run = runtime.resolve_run(&pending).unwrap();
    assert_eq!(
        runtime.world().get::<RunState>(run.entity()),
        Some(&RunState::Completed(RunOutput {
            text: "typed completion".to_owned(),
            usage: Usage::default(),
        }))
    );
}

#[test]
fn targeted_custom_policy_observes_earlier_effective_request() {
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
            id("first"),
            tenant("a"),
            Policy {
                order: 1,
                revision: 1,
                rule: PolicyRule::PatchRequest(RequestPatch::new().instructions("rewritten once")),
            },
            agent,
        )
        .unwrap();
    let custom = runtime
        .spawn_policy(
            id("second"),
            tenant("a"),
            Policy {
                order: 2,
                revision: 7,
                rule: PolicyRule::Custom(PolicyPoint::Request),
            },
            agent,
        )
        .unwrap();
    runtime
        .world_mut()
        .entity_mut(custom)
        .insert(InspectRequestPolicy {
            expected_instructions: "rewritten once".to_owned(),
            replacement_instructions: "rewritten twice".to_owned(),
        });
    runtime
        .register_request_policy_responder(
            custom,
            PolicyResponderId::new("inspect-request").unwrap(),
            inspect_request_policy,
        )
        .unwrap();

    runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let effect = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        effect.model_input().unwrap().instructions,
        "rewritten twice"
    );
}

#[test]
fn custom_policy_without_a_responder_fails_closed() {
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
            id("unbound"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Custom(PolicyPoint::Request),
            },
            agent,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "hello").unwrap();

    runtime.run_until_stalled().unwrap();
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
    let run = runtime.resolve_run(&pending).unwrap();
    policy_termination(&runtime, run.entity(), "unbound");
}

#[test]
fn conflicting_explicit_responder_registration_is_rejected() {
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
    let policy = runtime
        .spawn_policy(
            id("ambiguous-policy"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Custom(PolicyPoint::Request),
            },
            agent,
        )
        .unwrap();
    runtime
        .register_request_policy_responder(
            policy,
            PolicyResponderId::new("allow").unwrap(),
            allow_custom_request,
        )
        .unwrap();
    assert_eq!(
        runtime.register_request_policy_responder(
            policy,
            PolicyResponderId::new("stop").unwrap(),
            stop_custom_request,
        ),
        Err(PolicyResponderRegistrationError::ConflictingResponder(
            PolicyPoint::Request
        ))
    );

    runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(runtime.effects().try_recv().unwrap().is_some());
}

#[test]
fn policy_observer_invocation_advances_only_one_cursor_per_schedule_pass() {
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
    for order in 0..3 {
        runtime
            .spawn_policy(
                id(&format!("policy-{order}")),
                tenant("a"),
                Policy {
                    order,
                    revision: 1,
                    rule: PolicyRule::Allow,
                },
                agent,
            )
            .unwrap();
    }
    runtime
        .handle()
        .prompt(agent, "one cursor per pass")
        .unwrap();

    let cursor = (0..4)
        .find_map(|_| {
            runtime.update();
            runtime
                .world_mut()
                .query::<&RequestPolicyEvaluation>()
                .iter(runtime.world())
                .next()
                .map(|evaluation| evaluation.cursor)
        })
        .expect("request evaluation must be initialized within bounded schedule passes");
    assert_eq!(cursor, 1);
    assert_eq!(runtime.effects().try_recv().unwrap(), None);

    runtime.update();
    let cursor = runtime
        .world_mut()
        .query::<&RequestPolicyEvaluation>()
        .single(runtime.world())
        .unwrap()
        .cursor;
    assert_eq!(cursor, 2);
    assert_eq!(runtime.effects().try_recv().unwrap(), None);

    runtime.run_until_stalled().unwrap();
    assert!(runtime.effects().try_recv().unwrap().is_some());
}

#[test]
fn run_scoped_policy_affects_only_its_target_run_and_cleans_up_with_it() {
    let (mut runtime, agent) = runtime_with_tool();
    let first_pending = runtime.handle().prompt(agent, "first").unwrap();
    let second_pending = runtime.handle().prompt(agent, "second").unwrap();
    runtime.run_until_stalled().unwrap();
    let model_requests = [
        runtime.effects().try_recv().unwrap().unwrap(),
        runtime.effects().try_recv().unwrap().unwrap(),
    ];
    let first = runtime.resolve_run(&first_pending).unwrap();
    let second = runtime.resolve_run(&second_pending).unwrap();
    let run_policy = runtime
        .spawn_run_policy(
            id("first-run-only"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 5,
                rule: PolicyRule::SkipToolCall {
                    tool: Some("lookup".to_owned()),
                    reason: "first run skipped".to_owned(),
                },
            },
            first,
        )
        .unwrap();
    let run_policy_binding = runtime
        .world()
        .get::<PolicyResponders>(run_policy)
        .and_then(|bindings| bindings.iter().next())
        .expect("run-local policy must bind an explicit responder");
    let run_policy_responder = runtime
        .world()
        .get::<RegisteredPolicyResponder>(run_policy_binding)
        .expect("binding must reference a registered system")
        .0;

    for request in model_requests {
        let run = runtime
            .world()
            .get::<OperationOf>(request.operation)
            .unwrap()
            .get();
        let call_id = if run == first.entity() {
            "first-call"
        } else {
            assert_eq!(run, second.entity());
            "second-call"
        };
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
                    tool_calls: vec![ModelToolCall {
                        id: call_id.to_owned(),
                        provider_result_id: call_id.to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({"run": call_id}),
                    }],
                })),
            })
            .unwrap();
    }
    runtime.run_until_stalled().unwrap();
    let effects = [
        runtime.effects().try_recv().unwrap().unwrap(),
        runtime.effects().try_recv().unwrap().unwrap(),
    ];
    let first_next_model = effects
        .iter()
        .find_map(EffectRequest::model_input)
        .expect("the scoped skip must re-enter only the first run's model flow");
    assert_eq!(
        first_next_model.tool_results[0].presentation,
        "first run skipped"
    );
    let second_tool = effects
        .iter()
        .find_map(EffectRequest::tool_input)
        .expect("the unrelated sibling must still dispatch its tool");
    assert_eq!(second_tool.call_id, "second-call");

    runtime.world_mut().despawn(first.entity());
    assert!(runtime.world().get_entity(run_policy).is_err());
    assert!(runtime.world().get_entity(run_policy_binding).is_err());
    assert!(runtime.world().get_entity(run_policy_responder).is_err());
}

#[test]
fn request_approval_operation_suspends_and_resumes_policy_cursor() {
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
            id("approval"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 9,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::Request,
                    prompt: "approve model call".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let approval = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        approval.policy_approval_input(),
        Some(&PolicyApprovalEffectInput {
            policy_id: id("approval"),
            revision: 9,
            point: PolicyPoint::Request,
            prompt: "approve model call".to_owned(),
        })
    );
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: None,
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    assert!(
        runtime
            .effects()
            .try_recv()
            .unwrap()
            .unwrap()
            .model_input()
            .is_some()
    );
}

#[test]
fn retired_policy_is_retained_for_accepted_work_and_excluded_from_future_snapshots() {
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
    let policy = runtime
        .spawn_policy(
            id("approval"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 7,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::Request,
                    prompt: "approve accepted request".to_owned(),
                },
            },
            agent,
        )
        .unwrap();

    runtime.handle().prompt(agent, "first").unwrap();
    runtime.run_until_stalled().unwrap();
    let approval = runtime.effects().try_recv().unwrap().unwrap();
    assert!(approval.policy_approval_input().is_some());

    runtime.retire_policy(policy).unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: None,
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(
        runtime
            .effects()
            .try_recv()
            .unwrap()
            .unwrap()
            .model_input()
            .is_some()
    );

    runtime.handle().prompt(agent, "second").unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(
        runtime
            .effects()
            .try_recv()
            .unwrap()
            .unwrap()
            .model_input()
            .is_some()
    );

    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot.policies[0].status, PolicyStatus::Retired);
    let mut restored = Runtime::new(RuntimeConfig::default()).unwrap();
    let restored_entities = restored.restore(snapshot).unwrap();
    let restored_agent = AgentHandle {
        runtime_id: restored.handle.runtime_id,
        entity: restored_entities.0[&id("agent")],
    };
    let restored_policy = restored_entities.0[&id("approval")];
    assert_eq!(
        restored.world().get::<PolicyStatus>(restored_policy),
        Some(&PolicyStatus::Retired)
    );
    restored
        .handle()
        .prompt(restored_agent, "after restore")
        .unwrap();
    restored.run_until_stalled().unwrap();
    assert!(
        restored
            .effects()
            .try_recv()
            .unwrap()
            .unwrap()
            .model_input()
            .is_some()
    );
}

#[test]
fn denied_tool_approval_prevents_tool_dispatch() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .spawn_policy(
            id("tool-approval"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::ToolCall,
                    prompt: "approve tool".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "use lookup").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let approval = runtime.effects().try_recv().unwrap().unwrap();
    assert!(approval.policy_approval_input().is_some());
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: false,
                reason: Some("operator denied".to_owned()),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
    let run = runtime.resolve_run(&pending).unwrap();
    let termination = policy_termination(&runtime, run.entity(), "tool-approval");
    assert_eq!(termination.point, PolicyPoint::ToolCall);
    assert_eq!(termination.reason, "operator denied");
}

#[test]
fn cancelling_tool_approval_cancels_effect_and_rejects_late_completion() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .spawn_policy(
            id("approve-tool"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 3,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::ToolCall,
                    prompt: "approve lookup".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let (pending, approval) = advance_to_lookup_tool(&mut runtime, agent);
    assert!(approval.policy_approval_input().is_some());
    let run = runtime.resolve_run(&pending).unwrap();

    runtime.handle().cancel(run).unwrap();
    runtime.run_until_stalled().unwrap();
    assert_eq!(
        runtime.effects().try_recv_cancellation().unwrap(),
        Some(EffectCancellation {
            operation: approval.operation,
            generation: approval.generation,
        })
    );
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: Some("late".to_owned()),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();

    assert_eq!(
        runtime.world().get::<RunState>(run.entity()),
        Some(&RunState::Cancelled)
    );
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
}

#[test]
fn completion_response_policy_runs_before_commit_and_emits_turn_event() {
    let mut runtime = runtime();
    runtime
        .world_mut()
        .insert_resource(FinishedTurns::default());
    runtime.world_mut().add_observer(record_finished_turn);
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
            id("rewrite-response"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 5,
                rule: PolicyRule::RewriteCompletionText {
                    text: "rewritten".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "hello").unwrap();
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
                text: "original".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();
    assert!(matches!(
        runtime.world().get::<RunState>(run.entity()),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "rewritten"
    ));
    let turns = &runtime.world().resource::<FinishedTurns>().0;
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].output.text, "rewritten");
    assert_eq!(turns[0].response_policies[0].revision, 5);
}

#[test]
fn completion_response_policy_can_query_provider_diagnostics() {
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
    let policy = runtime
        .spawn_policy(
            id("inspect-provider"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Custom(PolicyPoint::CompletionResponse),
            },
            agent,
        )
        .unwrap();
    runtime
        .register_completion_response_policy_responder(
            policy,
            PolicyResponderId::new("inspect-provider").unwrap(),
            inspect_provider_diagnostics,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "hello").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    let sender = runtime.effects().completion_sender();
    sender
        .try_send_provider_diagnostics(ProviderDiagnosticsIngress::serialized(
            request.operation,
            request.generation,
            serde_json::json!({
                "provider_request_id": "provider-response"
            }),
        ))
        .unwrap();
    sender
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "original".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();
    assert!(matches!(
        runtime.world().get::<RunState>(run.entity()),
        Some(RunState::Completed(RunOutput { text, .. }))
            if text == "diagnostics observed"
    ));
}

#[test]
fn stopped_completion_response_never_commits_a_turn() {
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
            id("stop-response"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::StopCompletionContains {
                    needle: "secret".to_owned(),
                    reason: "sensitive completion".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "hello").unwrap();
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
                text: "a secret".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();
    policy_termination(&runtime, run.entity(), "stop-response");
    assert!(runtime.run_committed_turns(run).is_empty());
}

#[test]
fn tool_argument_rewrites_chain_in_policy_order() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .spawn_policy(
            id("first-rewrite"),
            tenant("a"),
            Policy {
                order: 1,
                revision: 1,
                rule: PolicyRule::RewriteToolArguments {
                    tool: Some("lookup".to_owned()),
                    arguments: serde_json::json!({"step": 1}),
                },
            },
            agent,
        )
        .unwrap();
    let second = runtime
        .spawn_policy(
            id("second-rewrite"),
            tenant("a"),
            Policy {
                order: 2,
                revision: 1,
                rule: PolicyRule::Custom(PolicyPoint::ToolCall),
            },
            agent,
        )
        .unwrap();
    runtime
        .world_mut()
        .entity_mut(second)
        .insert(InspectToolPolicy {
            expected_arguments: serde_json::json!({"step": 1}),
            replacement_arguments: serde_json::json!({"step": 2}),
        });
    runtime
        .register_tool_call_policy_responder(
            second,
            PolicyResponderId::new("inspect-tool").unwrap(),
            inspect_tool_policy,
        )
        .unwrap();

    runtime.handle().prompt(agent, "use lookup").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    arguments: serde_json::json!({"step": 0}),
                }],
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let tool = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        tool.tool_input().unwrap().arguments,
        serde_json::json!({"step": 2})
    );
}

#[test]
fn tool_skip_and_stop_after_rewrite_retain_effective_arguments() {
    for (stop, terminal_id) in [(false, "skip-after-rewrite"), (true, "stop-after-rewrite")] {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .spawn_policy(
                id("rewrite-first"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::RewriteToolArguments {
                        tool: Some("lookup".to_owned()),
                        arguments: serde_json::json!({"rewritten": true}),
                    },
                },
                agent,
            )
            .unwrap();
        let terminal = runtime
            .spawn_policy(
                id(terminal_id),
                tenant("a"),
                Policy {
                    order: 1,
                    revision: 2,
                    rule: PolicyRule::Custom(PolicyPoint::ToolCall),
                },
                agent,
            )
            .unwrap();
        runtime
            .world_mut()
            .entity_mut(terminal)
            .insert(TerminalToolPolicy {
                expected_arguments: serde_json::json!({"rewritten": true}),
                stop,
            });
        runtime
            .register_tool_call_policy_responder(
                terminal,
                PolicyResponderId::new(format!("{terminal_id}-responder")).unwrap(),
                finish_tool_policy,
            )
            .unwrap();

        let pending = runtime.handle().prompt(agent, "use lookup").unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model.operation,
                generation: model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: String::new(),
                    usage: Usage::default(),
                    tool_calls: vec![ModelToolCall {
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({"original": true}),
                    }],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        let mut evaluations = runtime.world_mut().query::<&ToolCallPolicyEvaluation>();
        let evaluation = evaluations.iter(runtime.world()).next().unwrap();
        assert_eq!(
            evaluation.effective.arguments,
            serde_json::json!({"rewritten": true})
        );

        if stop {
            assert_eq!(runtime.effects().try_recv().unwrap(), None);
            policy_termination(&runtime, run.entity(), terminal_id);
        } else {
            let next_model = runtime.effects().try_recv().unwrap().unwrap();
            let result = &next_model.model_input().unwrap().tool_results[0];
            assert_eq!(result.presentation, "operator skipped dispatch");
            assert_eq!(
                result.raw,
                serde_json::json!({
                    "skipped": true,
                    "reason": "operator skipped dispatch"
                })
            );
        }
    }
}

#[test]
fn concurrent_tool_call_policy_evaluations_are_isolated() {
    let (mut runtime, agent) = runtime_with_tool();
    let policy = runtime
        .spawn_policy(
            id("isolated-rewrite"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Custom(PolicyPoint::ToolCall),
            },
            agent,
        )
        .unwrap();
    runtime
        .register_tool_call_policy_responder(
            policy,
            PolicyResponderId::new("isolated-rewrite").unwrap(),
            isolate_tool_call_policy,
        )
        .unwrap();
    runtime.handle().prompt(agent, "two lookups").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
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
                        arguments: serde_json::json!({"value": 1}),
                    },
                    ModelToolCall {
                        id: "second".to_owned(),
                        provider_result_id: "second".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({"value": 2}),
                    },
                ],
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();

    let first = runtime.effects().try_recv().unwrap().unwrap();
    let second = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(first.tool_input().unwrap().call_id, "first");
    assert_eq!(
        first.tool_input().unwrap().arguments,
        serde_json::json!({"value": 11})
    );
    assert_eq!(second.tool_input().unwrap().call_id, "second");
    assert_eq!(
        second.tool_input().unwrap().arguments,
        serde_json::json!({"value": 12})
    );
}

#[test]
fn terminating_tool_call_prevents_not_yet_dispatched_batch_siblings() {
    let (mut runtime, agent) = runtime_with_tool();
    let policy = runtime
        .spawn_policy(
            id("stop-second"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Custom(PolicyPoint::ToolCall),
            },
            agent,
        )
        .unwrap();
    runtime
        .register_tool_call_policy_responder(
            policy,
            PolicyResponderId::new("stop-second").unwrap(),
            stop_second_tool_call,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "two lookups").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
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
                        arguments: serde_json::json!({"value": 1}),
                    },
                    ModelToolCall {
                        id: "second".to_owned(),
                        provider_result_id: "second".to_owned(),
                        provider_call_id: None,
                        name: "lookup".to_owned(),
                        arguments: serde_json::json!({"value": 2}),
                    },
                ],
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
    let run = runtime.resolve_run(&pending).unwrap();
    policy_termination(&runtime, run.entity(), "stop-second");
    let mut tools = runtime
        .world_mut()
        .query_filtered::<&OperationState, With<ToolEffectInput>>();
    assert_eq!(
        tools
            .iter(runtime.world())
            .filter(|state| matches!(state.phase, OperationPhase::InFlight))
            .count(),
        0
    );
}

#[test]
fn skipped_tool_call_never_dispatches_and_reenters_model_flow() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .spawn_policy(
            id("skip"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::SkipToolCall {
                    tool: Some("lookup".to_owned()),
                    reason: "approval denied".to_owned(),
                },
            },
            agent,
        )
        .unwrap();

    runtime.handle().prompt(agent, "use lookup").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let next_model = runtime.effects().try_recv().unwrap().unwrap();
    let input = next_model.model_input().unwrap();
    assert_eq!(input.tool_results.len(), 1);
    assert_eq!(input.tool_results[0].presentation, "approval denied");
    assert_eq!(
        input.tool_results[0].raw,
        serde_json::json!({"skipped": true, "reason": "approval denied"})
    );
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
}

#[test]
fn tool_result_rewrites_chain_without_mutating_raw_audit_data() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .world_mut()
        .insert_resource(FinalizedToolResults::default());
    runtime
        .world_mut()
        .add_observer(record_finalized_tool_result);
    runtime
        .spawn_policy(
            id("first-result"),
            tenant("a"),
            Policy {
                order: 1,
                revision: 1,
                rule: PolicyRule::RewriteToolResult {
                    tool: Some("lookup".to_owned()),
                    presentation: "redacted once".into(),
                },
            },
            agent,
        )
        .unwrap();
    let second = runtime
        .spawn_policy(
            id("second-result"),
            tenant("a"),
            Policy {
                order: 2,
                revision: 1,
                rule: PolicyRule::Custom(PolicyPoint::ToolResult),
            },
            agent,
        )
        .unwrap();
    runtime
        .world_mut()
        .entity_mut(second)
        .insert(InspectToolResultPolicy {
            expected_presentation: "redacted once".to_owned(),
            replacement_presentation: "redacted twice".to_owned(),
        });
    runtime
        .register_tool_result_policy_responder(
            second,
            PolicyResponderId::new("inspect-result").unwrap(),
            inspect_tool_result_policy,
        )
        .unwrap();
    let (pending, tool) = advance_to_lookup_tool(&mut runtime, agent);
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: "call".to_owned(),
                provider_result_id: "call".to_owned(),
                provider_call_id: None,
                name: "lookup".to_owned(),
                raw: serde_json::json!({"secret": 42}).into(),
                presentation: "unredacted".into(),
                failure: Some(ToolEffectFailure {
                    message: "operator-only refusal detail".to_owned(),
                    retryable: Some(false),
                    kind: crate::tool::ToolErrorKind::PermissionDenied,
                    refusal: true,
                }),
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let next_model = runtime.effects().try_recv().unwrap().unwrap();
    let result = &next_model.model_input().unwrap().tool_results[0];
    assert_eq!(result.raw, serde_json::json!({"secret": 42}));
    assert_eq!(result.presentation, "redacted twice");
    assert_eq!(
        result.failure,
        Some(ToolEffectFailure {
            message: "operator-only refusal detail".to_owned(),
            retryable: Some(false),
            kind: crate::tool::ToolErrorKind::PermissionDenied,
            refusal: true,
        })
    );
    let run = runtime.resolve_run(&pending).unwrap();
    assert!(
        runtime
            .world()
            .get::<RunRecord>(run.entity())
            .unwrap()
            .transcript
            .iter()
            .any(|entry| matches!(
                entry,
                TranscriptEntry::ToolResult { raw, content, .. }
                    if raw == &serde_json::json!({"secret": 42}) && content == "redacted twice"
            ))
    );
    let finalized = &runtime.world().resource::<FinalizedToolResults>().0;
    assert_eq!(finalized.len(), 1);
    assert_eq!(finalized[0].presentation, "redacted twice");
    assert_eq!(
        finalized[0].status,
        ToolExecutionStatus {
            succeeded: false,
            failure_kind: Some(crate::tool::ToolErrorKind::PermissionDenied),
            retryable: Some(false),
            refusal: true,
        }
    );
}

#[test]
fn tool_result_policy_preserves_structured_multimodal_presentation()
-> Result<(), Box<dyn std::error::Error>> {
    use crate::message::{DocumentSourceKind, Image, ImageMediaType, ToolResultContent};

    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .world_mut()
        .insert_resource(FinalizedToolResults::default());
    runtime
        .world_mut()
        .add_observer(record_finalized_tool_result);
    let presentation = ToolOutput::content(crate::OneOrMany::many([
        ToolResultContent::json(serde_json::json!({"status": "redacted"})),
        ToolResultContent::Image(Image {
            data: DocumentSourceKind::url("https://example.invalid/redacted.png"),
            media_type: Some(ImageMediaType::PNG),
            detail: None,
            additional_params: None,
        }),
    ])?);
    runtime.spawn_policy(
        id("rich-result"),
        tenant("a"),
        Policy {
            order: 1,
            revision: 1,
            rule: PolicyRule::RewriteToolResult {
                tool: Some("lookup".to_owned()),
                presentation: presentation.clone(),
            },
        },
        agent,
    )?;
    let (_, tool) = advance_to_lookup_tool(&mut runtime, agent);
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: "call".to_owned(),
                provider_result_id: "call".to_owned(),
                provider_call_id: None,
                name: "lookup".to_owned(),
                raw: serde_json::json!({"secret": 42}).into(),
                presentation: "unredacted".into(),
                failure: None,
            })),
        })?;

    runtime.run_until_stalled()?;
    let next_model = runtime
        .effects()
        .try_recv()?
        .ok_or("missing model request")?;
    let result = &next_model
        .model_input()
        .ok_or("expected model effect")?
        .tool_results[0];
    assert_eq!(result.raw, serde_json::json!({"secret": 42}));
    assert_eq!(result.presentation, presentation);
    let finalized = &runtime.world().resource::<FinalizedToolResults>().0;
    assert_eq!(finalized[0].presentation, "<typed tool output: 2 parts>");
    assert!(!finalized[0].presentation.contains("redacted.png"));

    Ok(())
}

#[test]
fn stopped_tool_result_is_not_committed_or_redispatched() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .world_mut()
        .insert_resource(FinalizedToolResults::default());
    runtime
        .world_mut()
        .add_observer(record_finalized_tool_result);
    runtime
        .spawn_policy(
            id("stop-result"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::StopToolResult {
                    tool: None,
                    reason: "sensitive result".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let (pending, tool) = advance_to_lookup_tool(&mut runtime, agent);
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: "call".to_owned(),
                provider_result_id: "call".to_owned(),
                provider_call_id: None,
                name: "lookup".to_owned(),
                raw: serde_json::json!({"secret": 42}).into(),
                presentation: "secret".into(),
                failure: None,
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
    let run = runtime.resolve_run(&pending).unwrap();
    policy_termination(&runtime, run.entity(), "stop-result");
    assert!(
        !runtime
            .world()
            .get::<RunRecord>(run.entity())
            .unwrap()
            .transcript
            .iter()
            .any(|entry| matches!(entry, TranscriptEntry::ToolResult { .. }))
    );
    assert!(
        runtime
            .world()
            .resource::<FinalizedToolResults>()
            .0
            .is_empty(),
        "stopped results must never cross the finalized telemetry boundary"
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
                raw: serde_json::json!({"result": 1}).into(),
                presentation: "second result".into(),
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
                raw: serde_json::json!({"result": 0}).into(),
                presentation: "first result".into(),
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
                raw: serde_json::json!({"result": 0}).into(),
                presentation: "first result".into(),
                failure: None,
            },
            ToolEffectOutput {
                call_id: "call-1".to_owned(),
                provider_result_id: "call-1".to_owned(),
                provider_call_id: None,
                name: "second".to_owned(),
                raw: serde_json::json!({"result": 1}).into(),
                presentation: "second result".into(),
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
                raw: serde_json::json!({"result": 0}).into(),
                content: "first result".into(),
                presentation_overrides_raw: true,
            },
            TranscriptEntry::ToolResult {
                call_id: "call-1".to_owned(),
                provider_result_id: "call-1".to_owned(),
                provider_call_id: None,
                name: "second".to_owned(),
                raw: serde_json::json!({"result": 1}).into(),
                content: "second result".into(),
                presentation_overrides_raw: true,
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
    let batch = runtime
        .world()
        .get::<RunToolBatches>(run.entity())
        .unwrap()
        .iter()
        .next()
        .unwrap();
    assert_eq!(
        runtime
            .world()
            .get::<ToolBatchState>(batch)
            .unwrap()
            .expected,
        2
    );
    let invalid_operation = runtime
        .world()
        .get::<BatchOperations>(batch)
        .unwrap()
        .iter()
        .find(|operation| {
            runtime
                .world()
                .get::<PendingInvalidToolCall>(*operation)
                .is_some()
        })
        .unwrap();
    assert_eq!(
        runtime
            .world()
            .get::<AcceptedInvalidToolCallPolicies>(invalid_operation),
        Some(&AcceptedInvalidToolCallPolicies::default())
    );
    assert!(
        runtime
            .world()
            .get::<OperationPolicyEvaluations>(invalid_operation)
            .is_none(),
        "the no-policy path must not allocate an evaluation entity"
    );
}

#[test]
fn invalid_tool_repair_resolves_against_the_advertised_snapshot() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .spawn_policy(
            id("repair"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 3,
                rule: PolicyRule::RepairInvalidTool {
                    from: Some("lookpu".to_owned()),
                    to: "lookup".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    runtime.handle().prompt(agent, "use lookup").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookpu".to_owned(),
                    arguments: serde_json::json!({"query": "x"}),
                }],
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let tool = runtime.effects().try_recv().unwrap().unwrap();
    let input = tool.tool_input().unwrap();
    assert_eq!(input.decision.name, "lookup");
    assert_eq!(input.arguments, serde_json::json!({"query": "x"}));
}

#[test]
fn invalid_tool_repair_rejects_target_disallowed_by_tool_choice() {
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
        .spawn_agent(
            id("agent"),
            tenant("a"),
            Agent {
                tool_choice: Some(ModelToolChoice::None),
                ..Agent::default()
            },
            model,
        )
        .unwrap();
    let tool = runtime
        .spawn_tool(
            id("tool"),
            tenant("a"),
            ToolCapability {
                name: "lookup".to_owned(),
                description: "lookup".to_owned(),
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
    runtime
        .spawn_policy(
            id("repair"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RepairInvalidTool {
                    from: None,
                    to: "lookup".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let pending = runtime.handle().prompt(agent, "malicious call").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "missing".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();
    assert_eq!(
        runtime.world().get::<RunState>(run.entity()),
        Some(&RunState::Failed(CanonicalError::UnknownTool(
            "lookup".to_owned()
        )))
    );
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
}

#[test]
fn invalid_tool_fail_skip_and_stop_decisions_have_distinct_semantics() {
    enum Expected {
        Fail,
        Skip,
        Stop,
    }

    type InvalidPolicyCase = (
        fn(In<InvalidToolCallPolicyInvocation>) -> Option<InvalidToolCallPolicyDecision>,
        Expected,
        &'static str,
    );
    let cases: [InvalidPolicyCase; 3] = [
        (fail_invalid_tool, Expected::Fail, "fail-invalid"),
        (skip_invalid_tool, Expected::Skip, "skip-invalid"),
        (stop_invalid_tool, Expected::Stop, "stop-invalid"),
    ];

    for (responder, expected, policy_id) in cases {
        let (mut runtime, agent) = runtime_with_tool();
        let policy = runtime
            .spawn_policy(
                id(policy_id),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 7,
                    rule: PolicyRule::Custom(PolicyPoint::InvalidToolCall),
                },
                agent,
            )
            .unwrap();
        runtime
            .register_invalid_tool_call_policy_responder(
                policy,
                PolicyResponderId::new(format!("{policy_id}-responder")).unwrap(),
                responder,
            )
            .unwrap();
        let pending = submit_invalid_tool_call(&mut runtime, agent, "missing");
        let run = runtime.resolve_run(&pending).unwrap();

        match expected {
            Expected::Fail => assert_eq!(
                runtime.world().get::<RunState>(run.entity()),
                Some(&RunState::Failed(CanonicalError::UnknownTool(
                    "missing".to_owned()
                )))
            ),
            Expected::Stop => {
                policy_termination(&runtime, run.entity(), policy_id);
            }
            Expected::Skip => {
                let next_model = runtime.effects().try_recv().unwrap().unwrap();
                let input = next_model.model_input().unwrap();
                assert_eq!(input.tool_results.len(), 1);
                assert_eq!(input.tool_results[0].presentation, "synthetic skip");
                assert_eq!(
                    input.tool_results[0].raw,
                    serde_json::json!({
                        "invalid_tool_call": "skip",
                        "feedback": "synthetic skip"
                    })
                );
            }
        }
    }
}

#[test]
fn invalid_tool_approval_resumes_at_the_next_policy_cursor() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime
        .spawn_policy(
            id("approve-invalid"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 4,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::InvalidToolCall,
                    prompt: "approve repair".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    runtime
        .spawn_policy(
            id("repair-after-approval"),
            tenant("a"),
            Policy {
                order: 1,
                revision: 9,
                rule: PolicyRule::RepairInvalidTool {
                    from: Some("lookpu".to_owned()),
                    to: "lookup".to_owned(),
                },
            },
            agent,
        )
        .unwrap();

    submit_invalid_tool_call(&mut runtime, agent, "lookpu");
    let approval = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        approval.policy_approval_input(),
        Some(&PolicyApprovalEffectInput {
            policy_id: id("approve-invalid"),
            revision: 4,
            point: PolicyPoint::InvalidToolCall,
            prompt: "approve repair".to_owned(),
        })
    );
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: Some("operator approved".to_owned()),
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let repaired = runtime.effects().try_recv().unwrap().unwrap();
    let input = repaired.tool_input().unwrap();
    assert_eq!(input.decision.name, "lookup");
    assert_eq!(input.call_id, "invalid-call");
    assert_eq!(input.provider_result_id, "provider-result");
    assert_eq!(input.provider_call_id.as_deref(), Some("provider-call"));
    assert_eq!(input.arguments, serde_json::json!({"query": "weather"}));
}

#[test]
fn streamed_invalid_tool_fragments_survive_repair_with_blocking_parity() {
    let (mut streaming, streaming_agent) = runtime_with_tool();
    streaming
        .spawn_policy(
            id("stream-repair"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RepairInvalidTool {
                    from: Some("lookpu".to_owned()),
                    to: "lookup".to_owned(),
                },
            },
            streaming_agent,
        )
        .unwrap();
    let (_, stream) = streaming
        .handle()
        .prompt_stream(streaming_agent, "use lookup")
        .unwrap();
    streaming.run_until_stalled().unwrap();
    let model = streaming.effects().try_recv().unwrap().unwrap();
    for (sequence, content) in [
        (0, ToolCallDeltaContent::Name("look".to_owned())),
        (
            1,
            ToolCallDeltaContent::Delta("{\"query\":\"weather\"}".to_owned()),
        ),
    ] {
        streaming
            .effects()
            .delta_sender()
            .try_send(EffectDelta {
                operation: model.operation,
                generation: model.generation,
                sequence,
                provider_correlation: Some("wire-call".to_owned()),
                kind: EffectDeltaKind::ToolCall {
                    id: "wire-call".to_owned(),
                    internal_call_id: "invalid-call".to_owned(),
                    content,
                },
            })
            .unwrap();
    }
    streaming
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "invalid-call".to_owned(),
                    provider_result_id: "provider-result".to_owned(),
                    provider_call_id: Some("provider-call".to_owned()),
                    name: "lookpu".to_owned(),
                    arguments: serde_json::json!({"query": "weather"}),
                }],
            })),
        })
        .unwrap();
    streaming.run_until_stalled().unwrap();
    let streamed_tool = streaming.effects().try_recv().unwrap().unwrap();
    let streamed_input = streamed_tool.tool_input().unwrap();
    assert_eq!(streamed_input.decision.name, "lookup");
    assert_eq!(streamed_input.call_id, "invalid-call");
    assert_eq!(streamed_input.provider_result_id, "provider-result");
    assert_eq!(
        streamed_input.provider_call_id.as_deref(),
        Some("provider-call")
    );
    assert_eq!(
        streamed_input.arguments,
        serde_json::json!({"query": "weather"})
    );
    let mut evaluations = streaming
        .world_mut()
        .query::<&InvalidToolCallPolicyEvaluation>();
    assert!(
        evaluations
            .iter(streaming.world())
            .all(|evaluation| evaluation.invalid.streaming_origin)
    );
    assert!(matches!(
        stream.try_recv().unwrap(),
        Some(StreamItem::ToolCallDelta {
            sequence: 0,
            id,
            internal_call_id,
            content: ToolCallDeltaContent::Name(name),
        }) if id == "wire-call" && internal_call_id == "invalid-call" && name == "look"
    ));
    assert!(matches!(
        stream.try_recv().unwrap(),
        Some(StreamItem::ToolCallDelta {
            sequence: 1,
            internal_call_id,
            content: ToolCallDeltaContent::Delta(arguments),
            ..
        }) if internal_call_id == "invalid-call"
            && arguments == "{\"query\":\"weather\"}"
    ));

    let (mut blocking, blocking_agent) = runtime_with_tool();
    blocking
        .spawn_policy(
            id("blocking-repair"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RepairInvalidTool {
                    from: Some("lookpu".to_owned()),
                    to: "lookup".to_owned(),
                },
            },
            blocking_agent,
        )
        .unwrap();
    submit_invalid_tool_call(&mut blocking, blocking_agent, "lookpu");
    let blocking_tool = blocking.effects().try_recv().unwrap().unwrap();
    let blocking_input = blocking_tool.tool_input().unwrap();
    assert_eq!(blocking_input.decision.name, streamed_input.decision.name);
    assert_eq!(blocking_input.call_id, streamed_input.call_id);
    assert_eq!(
        blocking_input.provider_result_id,
        streamed_input.provider_result_id
    );
    assert_eq!(
        blocking_input.provider_call_id,
        streamed_input.provider_call_id
    );
    assert_eq!(blocking_input.arguments, streamed_input.arguments);
}

#[test]
fn invalid_tool_retry_returns_feedback_without_tool_dispatch() {
    let (mut runtime, agent) = runtime_with_tool();
    runtime.set_invalid_tool_call_budget(agent, 1).unwrap();
    runtime
        .spawn_policy(
            id("retry"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RetryInvalidTool {
                    tool: None,
                    feedback: "choose an advertised tool".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    runtime.handle().prompt(agent, "use lookup").unwrap();
    runtime.run_until_stalled().unwrap();
    let model = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "missing".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    let next_model = runtime.effects().try_recv().unwrap().unwrap();
    let input = next_model.model_input().unwrap();
    assert_eq!(input.tool_results.len(), 1);
    assert_eq!(
        input.tool_results[0].presentation,
        "choose an advertised tool"
    );
    assert_eq!(runtime.effects().try_recv().unwrap(), None);
}

#[test]
fn invalid_tool_retry_enforces_invalid_and_total_model_budgets() {
    for (invalid_budget, max_model_calls, expected) in [
        (0, 4, CanonicalError::UnknownTool("missing".to_owned())),
        (1, 1, CanonicalError::ModelCallBudget { limit: 1 }),
    ] {
        let (mut runtime, agent) = runtime_with_tool();
        runtime
            .set_invalid_tool_call_budget(agent, invalid_budget)
            .unwrap();
        runtime
            .spawn_policy(
                id("retry"),
                tenant("a"),
                Policy {
                    order: 0,
                    revision: 1,
                    rule: PolicyRule::RetryInvalidTool {
                        tool: None,
                        feedback: "retry".to_owned(),
                    },
                },
                agent,
            )
            .unwrap();
        let pending = runtime
            .handle()
            .prompt_configured(
                agent,
                "invalid",
                Vec::<CompletionMessage>::new(),
                None,
                Some(max_model_calls),
                None,
            )
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let model = runtime.effects().try_recv().unwrap().unwrap();
        runtime
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: model.operation,
                generation: model.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: String::new(),
                    usage: Usage::default(),
                    tool_calls: vec![ModelToolCall {
                        id: "call".to_owned(),
                        provider_result_id: "call".to_owned(),
                        provider_call_id: None,
                        name: "missing".to_owned(),
                        arguments: serde_json::json!({}),
                    }],
                })),
            })
            .unwrap();
        runtime.run_until_stalled().unwrap();
        let run = runtime.resolve_run(&pending).unwrap();
        assert_eq!(
            runtime.world().get::<RunState>(run.entity()),
            Some(&RunState::Failed(expected))
        );
        assert_eq!(runtime.effects().try_recv().unwrap(), None);
    }
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
fn cross_tenant_policy_store_and_snapshot_bindings_fail_before_mutation() {
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
    assert_eq!(
        runtime.spawn_policy(
            id("foreign-policy"),
            tenant("b"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::Allow,
            },
            agent,
        ),
        Err(SpawnError::TenantMismatch)
    );
    let store = runtime
        .spawn_store(
            id("foreign-store"),
            tenant("b"),
            StoreCapability {
                kind: "vector-search".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .unwrap();
    assert_eq!(
        runtime.grant_store(
            id("foreign-store-grant"),
            tenant("a"),
            StoreGrant {
                order: 0,
                enabled: true,
            },
            agent,
            store,
        ),
        Err(SpawnError::TenantMismatch)
    );

    let domain = runtime.snapshot().unwrap();
    let pending = runtime.handle().prompt(agent, "snapshot tenant").unwrap();
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
    let run = runtime.resolve_run(&pending).unwrap();
    let mut snapshot = runtime.snapshot_active_run(run).unwrap();
    snapshot.runs[0].tenant = tenant("b");

    let mut target = self::runtime();
    target.restore(domain).unwrap();
    let entities_before = target.world().iter_entities().count();
    assert!(matches!(
        target.restore_active_run(snapshot),
        Err(ActiveRunSnapshotError::TenantMismatch(_))
    ));
    assert_eq!(target.world().iter_entities().count(), entities_before);
}

#[test]
fn wrong_kind_completion_fails_closed_without_committing_output() {
    let (mut runtime, agent) = runtime_with_tool();
    let pending = runtime.handle().prompt(agent, "wrong kind").unwrap();
    runtime.run_until_stalled().unwrap();
    let request = runtime.effects().try_recv().unwrap().unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: "wrong".to_owned(),
                provider_result_id: "wrong".to_owned(),
                provider_call_id: None,
                name: "lookup".to_owned(),
                raw: "wrong".into(),
                presentation: "wrong".into(),
                failure: None,
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let run = runtime.resolve_run(&pending).unwrap();
    assert_eq!(
        runtime.world().get::<RunState>(run.entity()),
        Some(&RunState::Failed(CanonicalError::EffectKindMismatch))
    );
    assert_eq!(
        runtime.run_transcript(run),
        Some(vec![TranscriptEntry::User("wrong kind".to_owned())])
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
    let mut tools = world.query::<(&DiscoveryKey, &ToolCapability, Option<&RetiredCapability>)>();
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
            phase: OperationPhase::Cancelled,
        })
    );
    assert_eq!(
        runtime
            .world()
            .get::<OperationGeneration>(request.operation),
        Some(&OperationGeneration(request.generation))
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
fn freeze_after_ingress_retains_completion_until_resume() {
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
    runtime
        .handle()
        .pause_with_mode(run, PauseMode::FreezeAfterIngress)
        .unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "retained".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    assert_eq!(
        runtime.world().get::<RunControl>(run.entity()),
        Some(&RunControl::Paused(PauseMode::FreezeAfterIngress))
    );
    assert!(matches!(
        runtime.world().get::<RunState>(run.entity()),
        Some(RunState::WaitingModel { .. })
    ));
    assert!(runtime.run_committed_turns(run).is_empty());

    runtime.handle().resume(run).unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(matches!(
        runtime.world().get::<RunState>(run.entity()),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "retained"
    ));
}

#[test]
fn cancel_and_suspend_rejects_late_completion_and_redispatches_new_generation() {
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
    runtime
        .world_mut()
        .init_resource::<OperationGenerationLifecycleLog>();
    runtime.world_mut().add_observer(
        |_: On<Discard, OperationGeneration>, mut log: ResMut<OperationGenerationLifecycleLog>| {
            log.0.push("discard");
        },
    );
    runtime.world_mut().add_observer(
        |_: On<Insert, OperationGeneration>, mut log: ResMut<OperationGenerationLifecycleLog>| {
            log.0.push("insert");
        },
    );
    runtime
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert_eq!(
        runtime.effects().try_recv_cancellation().unwrap(),
        Some(EffectCancellation {
            operation: request.operation,
            generation: request.generation,
        })
    );
    assert_eq!(
        runtime.world().get::<RunControl>(run.entity()),
        Some(&RunControl::Paused(PauseMode::CancelAndSuspend))
    );
    assert_eq!(
        runtime
            .world()
            .get::<OperationGeneration>(request.operation),
        Some(&OperationGeneration(request.generation + 1))
    );
    assert_eq!(
        runtime
            .world()
            .resource::<OperationGenerationLifecycleLog>()
            .0,
        vec!["discard", "insert"]
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

    runtime.handle().resume(run).unwrap();
    runtime.run_until_stalled().unwrap();
    let retried = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(retried.operation, request.operation);
    assert_eq!(retried.generation, request.generation + 1);
    assert_eq!(
        runtime
            .world()
            .get::<OperationGeneration>(request.operation),
        Some(&OperationGeneration(retried.generation))
    );
}

#[test]
fn cancel_and_suspend_fails_closed_when_generation_is_exhausted() {
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
    runtime
        .world_mut()
        .entity_mut(request.operation)
        .insert(OperationGeneration(u64::MAX));

    runtime
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    runtime.run_until_stalled().unwrap();

    assert_eq!(
        runtime.effects().try_recv_cancellation().unwrap(),
        Some(EffectCancellation {
            operation: request.operation,
            generation: u64::MAX,
        })
    );
    assert_eq!(
        runtime.world().get::<RunState>(run.entity()),
        Some(&RunState::Cancelled)
    );
    assert_eq!(
        runtime.world().get::<OperationState>(request.operation),
        Some(&OperationState {
            phase: OperationPhase::Cancelled,
        })
    );
    runtime.handle().resume(run).unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(runtime.effects().try_recv().unwrap().is_none());
}

#[test]
fn drain_paused_run_does_not_block_sibling_progress() {
    let (mut runtime, agent) = runtime_with_tool();
    let first_pending = runtime.handle().prompt(agent, "first").unwrap();
    let second_pending = runtime.handle().prompt(agent, "second").unwrap();
    runtime.run_until_stalled().unwrap();
    let first_request = runtime.effects().try_recv().unwrap().unwrap();
    let second_request = runtime.effects().try_recv().unwrap().unwrap();
    let first = runtime.resolve_run(&first_pending).unwrap();
    let second = runtime.resolve_run(&second_pending).unwrap();
    runtime.handle().pause(first).unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: first_request.operation,
            generation: first_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    arguments: serde_json::json!({}),
                }],
            })),
        })
        .unwrap();
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: second_request.operation,
            generation: second_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "sibling completed".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();

    runtime.run_until_stalled().unwrap();
    assert_eq!(
        runtime.world().get::<RunControl>(first.entity()),
        Some(&RunControl::Paused(PauseMode::Drain))
    );
    assert!(matches!(
        runtime.world().get::<RunState>(first.entity()),
        Some(RunState::WaitingTools { .. })
    ));
    assert!(matches!(
        runtime.world().get::<RunState>(second.entity()),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "sibling completed"
    ));
    assert_eq!(runtime.effects().try_recv().unwrap(), None);

    runtime.handle().resume(first).unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(
        runtime
            .effects()
            .try_recv()
            .unwrap()
            .unwrap()
            .tool_input()
            .is_some()
    );
}

#[test]
fn hosted_commands_spawn_agent_and_named_run_while_another_run_is_active() {
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
    let parent_agent = runtime
        .spawn_agent(id("parent-agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let parent_pending = runtime
        .handle()
        .prompt(parent_agent, "parent stays active")
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let parent_request = runtime.effects().try_recv().unwrap().unwrap();
    let parent = runtime.resolve_run(&parent_pending).unwrap();
    assert!(matches!(
        runtime.observe_run(parent).unwrap(),
        Some(RunState::WaitingModel { .. })
    ));

    let pending_agent = runtime
        .handle()
        .spawn_agent(
            id("dynamic-agent"),
            tenant("a"),
            Agent {
                instructions: "spawned during execution".to_owned(),
                ..Agent::default()
            },
            id("model"),
        )
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let dynamic_agent = pending_agent.try_resolve().unwrap().unwrap().unwrap();
    let dynamic_pending = runtime
        .handle()
        .spawn_run(id("dynamic-run"), dynamic_agent, "independent")
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let dynamic_request = runtime.effects().try_recv().unwrap().unwrap();
    let dynamic_run = runtime.resolve_run(&dynamic_pending).unwrap();
    assert_eq!(
        runtime.world().get::<StableId>(dynamic_run.entity()),
        Some(&id("dynamic-run"))
    );

    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: dynamic_request.operation,
            generation: dynamic_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "dynamic completed".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(matches!(
        runtime.observe_run(dynamic_run).unwrap(),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "dynamic completed"
    ));
    assert!(matches!(
        runtime.observe_run(parent).unwrap(),
        Some(RunState::WaitingModel { .. })
    ));
    assert_eq!(
        runtime
            .world()
            .get::<OperationState>(parent_request.operation),
        Some(&OperationState {
            phase: OperationPhase::InFlight,
        })
    );
    assert_eq!(
        runtime
            .world()
            .get::<OperationGeneration>(parent_request.operation),
        Some(&OperationGeneration(parent_request.generation))
    );
}

#[test]
fn child_results_commit_in_creation_order_not_completion_order() {
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
    let parent_pending = runtime.handle().prompt(agent, "parent").unwrap();
    runtime.run_until_stalled().unwrap();
    let parent_request = runtime.effects().try_recv().unwrap().unwrap();
    let parent = runtime.resolve_run(&parent_pending).unwrap();
    let first_pending = runtime
        .handle()
        .spawn_child_run(parent, agent, "first child")
        .unwrap();
    let second_pending = runtime
        .handle()
        .spawn_child_run(parent, agent, "second child")
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let first_request = runtime.effects().try_recv().unwrap().unwrap();
    let second_request = runtime.effects().try_recv().unwrap().unwrap();
    let first = runtime.resolve_run(&first_pending).unwrap();
    let second = runtime.resolve_run(&second_pending).unwrap();
    assert_eq!(
        runtime.world().get::<ParentRun>(first.entity()),
        Some(&ParentRun(parent.entity()))
    );
    assert_eq!(
        runtime.world().get::<ParentRun>(second.entity()),
        Some(&ParentRun(parent.entity()))
    );
    assert_eq!(
        runtime.world().get::<WaitingForChildren>(parent.entity()),
        Some(&WaitingForChildren)
    );

    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: parent_request.operation,
            generation: parent_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "parent output".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(matches!(
        runtime.world().get::<RunState>(parent.entity()),
        Some(RunState::WaitingModel { .. })
    ));
    assert!(matches!(
        runtime
            .world()
            .get::<OperationState>(parent_request.operation),
        Some(OperationState {
            phase: OperationPhase::Settled(_),
            ..
        })
    ));

    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: second_request.operation,
            generation: second_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "second output".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(
        !runtime
            .world()
            .get::<RunRecord>(parent.entity())
            .unwrap()
            .transcript
            .iter()
            .any(|entry| matches!(entry, TranscriptEntry::ChildResult { .. }))
    );
    assert_eq!(
        runtime.world().get::<WaitingForChildren>(parent.entity()),
        Some(&WaitingForChildren)
    );

    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: first_request.operation,
            generation: first_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "first output".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let results = runtime
        .world()
        .get::<RunRecord>(parent.entity())
        .unwrap()
        .transcript
        .iter()
        .filter_map(|entry| match entry {
            TranscriptEntry::ChildResult { result, .. } => result.as_ref().ok().cloned(),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(results, vec!["first output", "second output"]);
    assert!(
        runtime
            .world()
            .get::<WaitingForChildren>(parent.entity())
            .is_none()
    );
    assert!(matches!(
        runtime.world().get::<RunState>(parent.entity()),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "parent output"
    ));
}

#[test]
fn cancelling_parent_cancels_active_children() {
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
    let parent_pending = runtime.handle().prompt(agent, "parent").unwrap();
    runtime.run_until_stalled().unwrap();
    let _ = runtime.effects().try_recv().unwrap().unwrap();
    let parent = runtime.resolve_run(&parent_pending).unwrap();
    let child_pending = runtime
        .handle()
        .spawn_child_run(parent, agent, "child")
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let _ = runtime.effects().try_recv().unwrap().unwrap();
    let child = runtime.resolve_run(&child_pending).unwrap();

    runtime.handle().cancel(parent).unwrap();
    runtime.run_until_stalled().unwrap();
    assert_eq!(
        runtime.world().get::<RunState>(child.entity()),
        Some(&RunState::Cancelled)
    );
}

#[test]
fn agent_control_rejects_admission_and_bulk_resumes_runs() {
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
    let active_pending = runtime.handle().prompt(agent, "active").unwrap();
    runtime.run_until_stalled().unwrap();
    let _ = runtime.effects().try_recv().unwrap().unwrap();
    let active = runtime.resolve_run(&active_pending).unwrap();

    runtime
        .handle()
        .pause_agent(agent, PauseMode::FreezeAfterIngress)
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert_eq!(
        runtime.world().get::<RunControl>(active.entity()),
        Some(&RunControl::Paused(PauseMode::FreezeAfterIngress))
    );
    let rejected_pending = runtime.handle().prompt(agent, "rejected").unwrap();
    runtime.run_until_stalled().unwrap();
    let rejected = runtime.resolve_run(&rejected_pending).unwrap();
    assert_eq!(
        runtime.world().get::<RunState>(rejected.entity()),
        Some(&RunState::Failed(CanonicalError::AgentAdmissionDenied))
    );

    runtime.handle().resume_agent(agent).unwrap();
    runtime.run_until_stalled().unwrap();
    assert_eq!(
        runtime.world().get::<RunControl>(active.entity()),
        Some(&RunControl::Running)
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
            control: AgentControl::default(),
            invalid_tool_call_budget: None,
            structured_output_retry_budget: None,
            model_id: id("missing"),
            output_requirement: None,
            retrieval_requirement: None,
            tool_retrieval_requirement: None,
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
fn rig_schedule_has_no_mandatory_system_ambiguities() {
    let mut runtime = runtime();
    runtime
        .world_mut()
        .schedule_scope(RigSchedule, |world, schedule| {
            schedule
                .initialize(world)
                .expect("the core schedule must build without mandatory ambiguities");
            assert!(
                schedule.graph().conflicting_systems().is_empty(),
                "every conflicting core system pair must have explicit ordering"
            );
        });
}

#[test]
fn component_lifecycle_order_required_components_and_deferred_commands_are_explicit() {
    let mut world = World::new();
    world.init_resource::<ComponentLifecycleLog>();
    world.add_observer(
        |event: On<Add, LifecycleProbe>,
         mut log: ResMut<ComponentLifecycleLog>,
         mut commands: Commands| {
            log.0.push("add");
            commands
                .entity(event.entity)
                .insert(DeferredLifecycleMarker);
        },
    );
    world.add_observer(
        |_: On<Insert, LifecycleProbe>, mut log: ResMut<ComponentLifecycleLog>| {
            log.0.push("insert");
        },
    );
    world.add_observer(
        |_: On<Discard, LifecycleProbe>, mut log: ResMut<ComponentLifecycleLog>| {
            log.0.push("discard");
        },
    );
    world.add_observer(
        |_: On<Remove, LifecycleProbe>, mut log: ResMut<ComponentLifecycleLog>| {
            log.0.push("remove");
        },
    );
    world.add_observer(
        |_: On<Despawn, LifecycleProbe>, mut log: ResMut<ComponentLifecycleLog>| {
            log.0.push("despawn");
        },
    );

    let entity = world.spawn(LifecycleProbe(1)).id();
    assert!(world.get::<LifecycleRequired>(entity).is_some());
    assert!(world.get::<DeferredLifecycleMarker>(entity).is_some());
    assert_eq!(world.get::<LifecycleProbe>(entity).unwrap().0, 1);
    world.entity_mut(entity).insert(LifecycleProbe(2));
    assert_eq!(world.get::<LifecycleProbe>(entity).unwrap().0, 2);
    world.entity_mut(entity).remove::<LifecycleProbe>();
    assert!(world.get::<LifecycleProbe>(entity).is_none());

    let despawned = world.spawn(LifecycleProbe(3)).id();
    world.despawn(despawned);
    assert_eq!(
        world.resource::<ComponentLifecycleLog>().0,
        vec![
            "add", "insert", "discard", "insert", "discard", "remove", "add", "insert", "despawn",
            "discard", "remove"
        ]
    );
}

#[test]
fn entity_observer_conditions_gate_observation_without_order_dependence() {
    let mut world = World::new();
    world.insert_resource(ObservationEnabled(false));
    world.insert_resource(ConditionalObservationCount::default());
    let target = world.spawn_empty().id();
    world
        .entity_mut(target)
        .observe(record_conditional_probe.run_if(|enabled: Res<ObservationEnabled>| enabled.0));

    world.trigger(ConditionalProbe { target });
    assert_eq!(world.resource::<ConditionalObservationCount>().0, 0);
    world.resource_mut::<ObservationEnabled>().0 = true;
    world.trigger(ConditionalProbe { target });
    assert_eq!(world.resource::<ConditionalObservationCount>().0, 1);
}

#[test]
fn built_in_policy_responders_are_removed_with_policy_lifecycle() {
    let (mut runtime, agent) = runtime_with_tool();
    let removed_policy = runtime
        .spawn_policy(
            id("removed-policy"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::PatchRequest(RequestPatch::new().instructions("temporary")),
            },
            agent,
        )
        .unwrap();
    let removed_binding = runtime
        .world()
        .get::<PolicyResponders>(removed_policy)
        .and_then(|bindings| bindings.iter().next())
        .expect("On<Add, Policy> must bind an explicit responder");
    let removed_responder = runtime
        .world()
        .get::<RegisteredPolicyResponder>(removed_binding)
        .unwrap()
        .0;
    assert!(runtime.world().get_entity(removed_responder).is_ok());

    runtime
        .world_mut()
        .entity_mut(removed_policy)
        .remove::<Policy>();
    assert!(runtime.world().get_entity(removed_binding).is_err());
    assert!(runtime.world().get_entity(removed_responder).is_err());
    assert!(
        runtime
            .world()
            .get::<PolicyResponders>(removed_policy)
            .is_none()
    );

    let despawned_policy = runtime
        .spawn_policy(
            id("despawned-policy"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::SkipToolCall {
                    tool: None,
                    reason: "temporary".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let despawned_binding = runtime
        .world()
        .get::<PolicyResponders>(despawned_policy)
        .and_then(|bindings| bindings.iter().next())
        .unwrap();
    let despawned_responder = runtime
        .world()
        .get::<RegisteredPolicyResponder>(despawned_binding)
        .unwrap()
        .0;
    runtime.world_mut().despawn(despawned_policy);
    assert!(runtime.world().get_entity(despawned_binding).is_err());
    assert!(runtime.world().get_entity(despawned_responder).is_err());

    let typed_policy = runtime
        .spawn_typed_policy(
            id("typed-lifecycle"),
            tenant("a"),
            PolicyMeta {
                order: 0,
                revision: 1,
            },
            PolicyCapabilities::new([PolicyPoint::Request]),
            (),
            agent,
        )
        .unwrap();
    let typed_binding = runtime
        .register_request_policy_responder(
            typed_policy,
            PolicyResponderId::new("typed-lifecycle:request").unwrap(),
            allow_custom_request,
        )
        .unwrap();
    let typed_responder = runtime
        .world()
        .get::<RegisteredPolicyResponder>(typed_binding)
        .unwrap()
        .0;
    runtime.retire_policy(typed_policy).unwrap();
    assert_eq!(
        runtime.world().get::<PolicyStatus>(typed_policy),
        Some(&PolicyStatus::Retired)
    );
    runtime.world_mut().despawn(typed_policy);
    assert!(runtime.world().get_entity(typed_binding).is_err());
    assert!(runtime.world().get_entity(typed_responder).is_err());
}

#[test]
fn dispatch_budget_and_tenant_quota_defer_work_without_starving_it() {
    let mut runtime = Runtime::new(RuntimeConfig {
        per_tenant_effect_limit: 1,
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
    runtime.handle().prompt(agent, "first").unwrap();
    runtime.handle().prompt(agent, "second").unwrap();

    runtime.run_until_stalled().unwrap();
    let first = runtime.effects().try_recv().unwrap().unwrap();
    assert!(runtime.effects().try_recv().unwrap().is_none());
    runtime.run_until_stalled().unwrap();
    assert!(runtime.effects().try_recv().unwrap().is_none());

    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: first.operation,
            generation: first.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "done".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    runtime.run_until_stalled().unwrap();
    assert!(runtime.effects().try_recv().unwrap().is_some());
}

#[test]
fn per_pass_budget_round_robins_hostile_tenants() {
    let mut runtime = Runtime::new(RuntimeConfig {
        max_effect_dispatches_per_pass: 1,
        ..RuntimeConfig::default()
    })
    .unwrap();
    let attacker_model = runtime
        .spawn_model(
            id("attacker-model"),
            tenant("attacker"),
            ModelCapability {
                provider: "fake".to_owned(),
                model: "test".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .unwrap();
    let victim_model = runtime
        .spawn_model(
            id("victim-model"),
            tenant("victim"),
            ModelCapability {
                provider: "fake".to_owned(),
                model: "test".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .unwrap();
    let attacker = runtime
        .spawn_agent(
            id("attacker-agent"),
            tenant("attacker"),
            Agent::default(),
            attacker_model,
        )
        .unwrap();
    let victim = runtime
        .spawn_agent(
            id("victim-agent"),
            tenant("victim"),
            Agent::default(),
            victim_model,
        )
        .unwrap();
    for index in 0..8 {
        runtime
            .handle()
            .prompt(attacker, format!("attacker-{index}"))
            .unwrap();
    }
    runtime.handle().prompt(victim, "victim").unwrap();

    runtime.run_until_stalled().unwrap();
    let first_two = [
        runtime.effects().try_recv().unwrap().unwrap(),
        runtime.effects().try_recv().unwrap().unwrap(),
    ];
    let dispatched_tenants = first_two
        .iter()
        .map(|request| {
            let run = runtime
                .world()
                .get::<OperationOf>(request.operation)
                .unwrap()
                .get();
            runtime.world().get::<TenantId>(run).unwrap().clone()
        })
        .collect::<HashSet<_>>();
    assert_eq!(
        dispatched_tenants,
        HashSet::from([tenant("attacker"), tenant("victim")])
    );
}

#[test]
fn quota_saturated_tenant_does_not_consume_another_tenants_allowance() {
    let mut runtime = Runtime::new(RuntimeConfig {
        max_effect_dispatches_per_pass: 1,
        per_tenant_effect_limit: 1,
        ..RuntimeConfig::default()
    })
    .unwrap();
    let attacker_model = runtime
        .spawn_model(
            id("attacker-model"),
            tenant("attacker"),
            ModelCapability {
                provider: "fake".to_owned(),
                model: "test".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .unwrap();
    let victim_model = runtime
        .spawn_model(
            id("victim-model"),
            tenant("victim"),
            ModelCapability {
                provider: "fake".to_owned(),
                model: "test".to_owned(),
                revision: 1,
                retired: false,
            },
        )
        .unwrap();
    let attacker = runtime
        .spawn_agent(
            id("attacker-agent"),
            tenant("attacker"),
            Agent::default(),
            attacker_model,
        )
        .unwrap();
    let victim = runtime
        .spawn_agent(
            id("victim-agent"),
            tenant("victim"),
            Agent::default(),
            victim_model,
        )
        .unwrap();
    runtime.handle().prompt(attacker, "occupy quota").unwrap();
    runtime.run_until_stalled().unwrap();
    let occupied = runtime.effects().try_recv().unwrap().unwrap();
    runtime.handle().prompt(attacker, "still blocked").unwrap();
    runtime.handle().prompt(victim, "must progress").unwrap();

    runtime.run_until_stalled().unwrap();
    let dispatched = runtime.effects().try_recv().unwrap().unwrap();
    let run = runtime
        .world()
        .get::<OperationOf>(dispatched.operation)
        .unwrap()
        .get();
    assert_eq!(
        runtime.world().get::<TenantId>(run),
        Some(&tenant("victim"))
    );
    assert_ne!(dispatched.operation, occupied.operation);
}

#[test]
fn run_priority_orders_same_tenant_dispatch_and_survives_snapshot() {
    let mut runtime = Runtime::new(RuntimeConfig {
        max_effect_dispatches_per_pass: 1,
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
    let domain = runtime.snapshot().unwrap();
    let low = runtime.handle().prompt(agent, "low").unwrap();
    let high = runtime.handle().prompt(agent, "high").unwrap();
    runtime.world_mut().run_schedule(RigSchedule);
    let low = runtime.resolve_run(&low).unwrap();
    let high = runtime.resolve_run(&high).unwrap();
    runtime
        .world_mut()
        .entity_mut(low.entity())
        .insert(RunPriority(-1));
    runtime
        .world_mut()
        .entity_mut(high.entity())
        .insert(RunPriority(10));

    runtime.run_until_stalled().unwrap();
    let first = runtime.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        runtime
            .world()
            .get::<OperationOf>(first.operation)
            .map(Relationship::get),
        Some(high.entity())
    );
    runtime
        .handle()
        .pause_with_mode(low, PauseMode::CancelAndSuspend)
        .unwrap();
    runtime.run_until_stalled().unwrap();
    let low_id = runtime
        .world()
        .get::<StableId>(low.entity())
        .unwrap()
        .clone();
    let mut snapshot = runtime.snapshot_active_run(low).unwrap();
    assert_eq!(snapshot.runs[0].priority, RunPriority(-1));
    assert!(snapshot.runs[0].ready_at.0 > 0);

    snapshot.runs[0].priority = RunPriority::default();
    let mut restored = Runtime::new(RuntimeConfig {
        max_effect_dispatches_per_pass: 1,
        ..RuntimeConfig::default()
    })
    .unwrap();
    let restored_domain = restored.restore(domain).unwrap();
    let restored_agent = AgentHandle {
        runtime_id: restored.handle().runtime_id,
        entity: restored_domain.0[&id("agent")],
    };
    let restored_runs = restored.restore_active_run(snapshot).unwrap();
    let restored_low_entity = restored_runs.0[&low_id];
    let restored_low = RunHandle {
        runtime_id: restored.handle().runtime_id,
        entity: restored_low_entity,
    };
    restored.handle().resume(restored_low).unwrap();
    restored.handle().prompt(restored_agent, "newer").unwrap();
    restored.run_until_stalled().unwrap();
    let first = restored.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        restored
            .world()
            .get::<OperationOf>(first.operation)
            .map(Relationship::get),
        Some(restored_low_entity)
    );
}

#[test]
fn policy_heavy_run_does_not_starve_ready_sibling() {
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
    let heavy_agent = runtime
        .spawn_agent(id("heavy-agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let sibling_agent = runtime
        .spawn_agent(id("sibling-agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    for index in 0..64 {
        runtime
            .spawn_policy(
                id(&format!("policy-{index:02}")),
                tenant("a"),
                Policy {
                    order: index,
                    revision: 1,
                    rule: PolicyRule::Allow,
                },
                heavy_agent,
            )
            .unwrap();
    }
    let heavy = runtime.handle().prompt(heavy_agent, "heavy").unwrap();
    let sibling = runtime.handle().prompt(sibling_agent, "sibling").unwrap();

    runtime.run_until_stalled().unwrap();
    let requests = [
        runtime.effects().try_recv().unwrap().unwrap(),
        runtime.effects().try_recv().unwrap().unwrap(),
    ];
    let heavy_run = runtime.resolve_run(&heavy).unwrap().entity();
    let sibling_run = runtime.resolve_run(&sibling).unwrap().entity();
    let dispatched_runs = requests
        .iter()
        .filter_map(|request| runtime.world().get::<OperationOf>(request.operation))
        .map(Relationship::get)
        .collect::<HashSet<_>>();
    assert_eq!(dispatched_runs, HashSet::from([heavy_run, sibling_run]));
}

#[test]
fn hosted_ingress_notifies_the_supplied_waker() {
    let wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed_wakes = wakes.clone();
    let mut world = World::new();
    let installed = install_runtime_with_waker(&mut world, RuntimeConfig::default(), move || {
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

#[test]
fn active_run_snapshot_rejects_live_effect_and_resumes_cancelled_checkpoint() {
    struct ExpandingProtection;

    impl ActiveRunSnapshotProtection for ExpandingProtection {
        fn protect(&self, _plaintext: &[u8]) -> Result<Vec<u8>, String> {
            Ok(vec![0])
        }

        fn unprotect(&self, _protected: &[u8]) -> Result<Vec<u8>, String> {
            Ok(vec![0; 4_097])
        }
    }

    let mut source = runtime();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let domain = source.snapshot().unwrap();
    let pending = source.handle().prompt(agent, "checkpoint me").unwrap();
    source.run_until_stalled().unwrap();
    let original = source.effects().try_recv().unwrap().unwrap();
    let run = source.resolve_run(&pending).unwrap();
    assert!(matches!(
        source.snapshot_active_run(run),
        Err(ActiveRunSnapshotError::UnsafeInFlightEffect(_))
    ));

    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    assert!(matches!(
        source.world().get::<RunControl>(run.entity()),
        Some(RunControl::Paused(PauseMode::CancelAndSuspend))
    ));
    let snapshot = source.snapshot_active_run(run).unwrap();
    let encoded = serde_json::to_string(&snapshot).unwrap();
    assert!(!encoded.contains("model_entity"));
    assert!(!encoded.contains("tool_entity"));
    assert!(!encoded.contains("store_entity"));
    let snapshot: ActiveRunSnapshot = serde_json::from_str(&encoded).unwrap();

    let protected = encode_active_run_snapshot(&snapshot, Some(&XorSnapshotProtection)).unwrap();
    assert_eq!(
        decode_active_run_snapshot(
            &protected,
            Some(&XorSnapshotProtection),
            ActiveRunSnapshotLimits::default(),
        )
        .unwrap(),
        snapshot
    );
    assert_eq!(
        decode_active_run_snapshot(&protected, None, ActiveRunSnapshotLimits::default(),)
            .unwrap_err(),
        ActiveRunSnapshotError::MissingProtection
    );
    let mut downgraded: serde_json::Value = serde_json::from_slice(&protected).unwrap();
    let plaintext = serde_json::to_vec(&snapshot).unwrap();
    downgraded["protected"] = serde_json::json!(false);
    let digest = Sha256::digest(&plaintext)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    downgraded["sha256"] = serde_json::json!(digest);
    downgraded["payload"] = serde_json::to_value(plaintext).unwrap();
    assert_eq!(
        decode_active_run_snapshot(
            &serde_json::to_vec(&downgraded).unwrap(),
            Some(&XorSnapshotProtection),
            ActiveRunSnapshotLimits::default(),
        )
        .unwrap_err(),
        ActiveRunSnapshotError::ProtectionModeMismatch
    );
    let expanding = encode_active_run_snapshot(&snapshot, Some(&ExpandingProtection)).unwrap();
    assert_eq!(
        decode_active_run_snapshot(
            &expanding,
            Some(&ExpandingProtection),
            ActiveRunSnapshotLimits {
                max_encoded_bytes: 4_096,
                ..ActiveRunSnapshotLimits::default()
            },
        )
        .unwrap_err(),
        ActiveRunSnapshotError::LimitExceeded {
            field: "decoded_bytes",
            actual: 4_097,
            limit: 4_096,
        }
    );
    let mut tampered: serde_json::Value = serde_json::from_slice(&protected).unwrap();
    let payload = tampered["payload"].as_array_mut().unwrap();
    payload[0] = serde_json::json!(payload[0].as_u64().unwrap() ^ 1);
    assert_eq!(
        decode_active_run_snapshot(
            &serde_json::to_vec(&tampered).unwrap(),
            Some(&XorSnapshotProtection),
            ActiveRunSnapshotLimits::default(),
        )
        .unwrap_err(),
        ActiveRunSnapshotError::Integrity
    );
    let mut journal = ActiveRunCheckpointJournal::new(snapshot.root_run.clone());
    journal
        .append(
            protected.clone(),
            Some(&XorSnapshotProtection),
            ActiveRunSnapshotLimits::default(),
        )
        .unwrap();
    journal
        .append(
            protected,
            Some(&XorSnapshotProtection),
            ActiveRunSnapshotLimits::default(),
        )
        .unwrap();
    assert!(journal.latest().unwrap().is_some());
    journal.entries[1].previous_sha256 = Some("tampered".to_owned());
    assert_eq!(journal.verify(), Err(ActiveRunSnapshotError::Integrity));

    let summary = source.snapshot_summary(run).unwrap();
    assert_eq!(summary.version, 6);
    assert_eq!(summary.root_run, *pending.stable_id());
    assert_eq!(summary.runs, 1);
    assert_eq!(summary.max_depth, 0);
    assert_eq!(summary.operations, 1);
    assert_eq!(summary.transcript_entries, 1);
    assert!(summary.encoded_bytes > 0);
    assert_eq!(
        serde_json::to_value(&summary).unwrap()["root_run"],
        pending.stable_id().as_str()
    );

    let mut version_four = snapshot.clone();
    version_four.version = 4;
    let migrated = migrate_active_run_snapshot(version_four).unwrap();
    assert_eq!(migrated.version, 6);
    assert_eq!(migrated.runs, snapshot.runs);

    let mut limited = runtime();
    limited.restore(domain.clone()).unwrap();
    let entities_before = limited.world().iter_entities().count();
    assert_eq!(
        limited
            .restore_active_run_with_limits(
                snapshot.clone(),
                ActiveRunSnapshotLimits {
                    max_runs: 0,
                    ..ActiveRunSnapshotLimits::default()
                },
            )
            .unwrap_err(),
        ActiveRunSnapshotError::LimitExceeded {
            field: "runs",
            actual: 1,
            limit: 0,
        }
    );
    assert_eq!(limited.world().iter_entities().count(), entities_before);

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let restored_runs = restored.restore_active_run(snapshot).unwrap();
    let restored_entity = restored_runs.0.get(pending.stable_id()).copied().unwrap();
    let restored_run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: restored_entity,
    };
    restored.handle().resume(restored_run).unwrap();
    restored.run_until_stalled().unwrap();
    let restored_request = restored.effects().try_recv().unwrap().unwrap();
    assert!(restored_request.generation > original.generation);
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: restored_request.operation,
            generation: restored_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "resumed".to_owned(),
                usage: Usage {
                    input_tokens: 2,
                    output_tokens: 1,
                },
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    assert_eq!(
        restored.observe_run(restored_run).unwrap(),
        Some(RunState::Completed(RunOutput {
            text: "resumed".to_owned(),
            usage: Usage {
                input_tokens: 2,
                output_tokens: 1,
            },
        }))
    );
    assert_eq!(
        restored.run_transcript(restored_run),
        Some(vec![
            TranscriptEntry::User("checkpoint me".to_owned()),
            TranscriptEntry::Assistant("resumed".to_owned()),
        ])
    );
}

#[test]
fn active_run_snapshot_rejects_unpersistable_extension_effect_phases() {
    #[derive(Clone)]
    struct SnapshotExtensionInput;

    struct SnapshotExtension;

    impl EcsEffect for SnapshotExtension {
        type Input = SnapshotExtensionInput;
        type Output = ();
        type Error = String;

        const KIND: &'static str = "tests.snapshot-extension.v1";
    }

    let mut source = runtime();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let pending = source
        .handle()
        .prompt(agent, "checkpoint extension")
        .unwrap();
    source.run_until_stalled().unwrap();
    let _model_request = source.effects().try_recv().unwrap().unwrap();
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();

    let operation = spawn_extension_effect::<SnapshotExtension>(
        source.world_mut(),
        run.entity(),
        id("snapshot-extension"),
        SnapshotExtensionInput,
    )
    .unwrap();
    assert!(matches!(
        source.snapshot_active_run(run),
        Err(ActiveRunSnapshotError::UnsafeExtensionEffect {
            phase: "prepared",
            ..
        })
    ));

    let (generation, _) =
        dispatch_extension_effect::<SnapshotExtension>(source.world_mut(), operation).unwrap();
    assert!(matches!(
        source.snapshot_active_run(run),
        Err(ActiveRunSnapshotError::UnsafeExtensionEffect {
            phase: "in-flight",
            ..
        })
    ));

    settle_extension_effect::<SnapshotExtension>(source.world_mut(), operation, generation, Ok(()))
        .unwrap();
    assert!(matches!(
        source.snapshot_active_run(run),
        Err(ActiveRunSnapshotError::UnsafeExtensionEffect {
            phase: "extension-settled",
            ..
        })
    ));
}

#[test]
fn resumed_run_matches_uninterrupted_output_transcript_usage_and_policy_decisions() {
    let mut definition = runtime();
    let model = definition
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
    let agent = definition
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    definition
        .spawn_policy(
            id("request-patch"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 9,
                rule: PolicyRule::PatchRequest(
                    RequestPatch::new()
                        .instructions("checkpoint policy")
                        .temperature(0.25),
                ),
            },
            agent,
        )
        .unwrap();
    let domain = definition.snapshot().unwrap();
    let response = ModelEffectOutput {
        assistant_message: None,
        text: "identical answer".to_owned(),
        usage: Usage {
            input_tokens: 11,
            output_tokens: 7,
        },
        tool_calls: Vec::new(),
    };

    let mut uninterrupted = runtime();
    let entities = uninterrupted.restore(domain.clone()).unwrap();
    let uninterrupted_agent = AgentHandle {
        runtime_id: uninterrupted.handle.runtime_id,
        entity: entities.0[&id("agent")],
    };
    let uninterrupted_pending = uninterrupted
        .handle()
        .prompt(uninterrupted_agent, "compare checkpoint")
        .unwrap();
    uninterrupted.run_until_stalled().unwrap();
    let uninterrupted_request = uninterrupted.effects().try_recv().unwrap().unwrap();
    let uninterrupted_input = uninterrupted_request.model_input().unwrap().clone();
    uninterrupted
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: uninterrupted_request.operation,
            generation: uninterrupted_request.generation,
            result: Ok(EffectOutput::Model(response.clone())),
        })
        .unwrap();
    uninterrupted.run_until_stalled().unwrap();
    let uninterrupted_run = uninterrupted.resolve_run(&uninterrupted_pending).unwrap();
    let uninterrupted_state = uninterrupted
        .world()
        .get::<RunState>(uninterrupted_run.entity())
        .cloned();
    let uninterrupted_transcript = uninterrupted.run_transcript(uninterrupted_run).unwrap();
    let uninterrupted_usage = uninterrupted
        .world()
        .get::<RunRecord>(uninterrupted_run.entity())
        .unwrap()
        .usage;
    let uninterrupted_turns = uninterrupted.run_committed_turns(uninterrupted_run);
    let uninterrupted_policies = uninterrupted
        .world()
        .get::<AcceptedPolicies>(uninterrupted_request.operation)
        .unwrap()
        .0
        .iter()
        .map(|policy| (policy.id.clone(), policy.revision))
        .collect::<Vec<_>>();

    let mut checkpoint_source = runtime();
    let entities = checkpoint_source.restore(domain.clone()).unwrap();
    let checkpoint_agent = AgentHandle {
        runtime_id: checkpoint_source.handle.runtime_id,
        entity: entities.0[&id("agent")],
    };
    let checkpoint_pending = checkpoint_source
        .handle()
        .prompt(checkpoint_agent, "compare checkpoint")
        .unwrap();
    checkpoint_source.run_until_stalled().unwrap();
    let original_request = checkpoint_source.effects().try_recv().unwrap().unwrap();
    assert_eq!(original_request.model_input(), Some(&uninterrupted_input));
    let checkpoint_run = checkpoint_source.resolve_run(&checkpoint_pending).unwrap();
    checkpoint_source
        .handle()
        .pause_with_mode(checkpoint_run, PauseMode::CancelAndSuspend)
        .unwrap();
    checkpoint_source.run_until_stalled().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(
            &checkpoint_source
                .snapshot_active_run(checkpoint_run)
                .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();

    let mut resumed = runtime();
    resumed.restore(domain).unwrap();
    let restored_runs = resumed.restore_active_run(snapshot).unwrap();
    let resumed_run = RunHandle {
        runtime_id: resumed.handle.runtime_id,
        entity: restored_runs.0[checkpoint_pending.stable_id()],
    };
    resumed.handle().resume(resumed_run).unwrap();
    resumed.run_until_stalled().unwrap();
    let resumed_request = resumed.effects().try_recv().unwrap().unwrap();
    assert!(resumed_request.generation > original_request.generation);
    assert_eq!(resumed_request.model_input(), Some(&uninterrupted_input));
    let resumed_policies = resumed
        .world()
        .get::<AcceptedPolicies>(resumed_request.operation)
        .unwrap()
        .0
        .iter()
        .map(|policy| (policy.id.clone(), policy.revision))
        .collect::<Vec<_>>();
    resumed
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: resumed_request.operation,
            generation: resumed_request.generation,
            result: Ok(EffectOutput::Model(response)),
        })
        .unwrap();
    resumed.run_until_stalled().unwrap();

    assert_eq!(
        resumed.world().get::<RunState>(resumed_run.entity()),
        uninterrupted_state.as_ref()
    );
    assert_eq!(
        resumed.run_transcript(resumed_run).unwrap(),
        uninterrupted_transcript
    );
    assert_eq!(
        resumed
            .world()
            .get::<RunRecord>(resumed_run.entity())
            .unwrap()
            .usage,
        uninterrupted_usage
    );
    assert_eq!(
        resumed.run_committed_turns(resumed_run),
        uninterrupted_turns
    );
    assert_eq!(resumed_policies, uninterrupted_policies);
}

#[test]
fn active_run_snapshot_extension_requires_exact_rebinding_before_mutation() {
    let mut source = runtime();
    register_active_run_snapshot_codec(
        source.world_mut(),
        std::sync::Arc::new(SnapshotNoteCodec { revision: 1 }),
    )
    .unwrap();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let domain = source.snapshot().unwrap();
    let pending = source.handle().prompt(agent, "extension state").unwrap();
    source.run_until_stalled().unwrap();
    let request = source.effects().try_recv().unwrap().unwrap();
    source
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
    source.run_until_stalled().unwrap();
    let run = source.resolve_run(&pending).unwrap();
    source
        .world_mut()
        .entity_mut(run.entity())
        .insert(SnapshotNote("retained".to_owned()));
    let snapshot = source.snapshot_active_run(run).unwrap();
    assert_eq!(snapshot.extensions.len(), 1);
    assert_eq!(snapshot.extensions[0].binding_id, "tests.snapshot-note");

    let mut missing = runtime();
    missing.restore(domain.clone()).unwrap();
    let entities_before = missing.world().iter_entities().count();
    assert_eq!(
        missing.restore_active_run(snapshot.clone()).unwrap_err(),
        ActiveRunSnapshotError::MissingExtensionBinding("tests.snapshot-note".to_owned())
    );
    assert_eq!(missing.world().iter_entities().count(), entities_before);

    let mut wrong_revision = runtime();
    wrong_revision.restore(domain.clone()).unwrap();
    register_active_run_snapshot_codec(
        wrong_revision.world_mut(),
        std::sync::Arc::new(SnapshotNoteCodec { revision: 2 }),
    )
    .unwrap();
    assert_eq!(
        wrong_revision
            .restore_active_run(snapshot.clone())
            .unwrap_err(),
        ActiveRunSnapshotError::ExtensionRevisionMismatch {
            id: "tests.snapshot-note".to_owned(),
            expected: 1,
            found: 2,
        }
    );

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    register_active_run_snapshot_codec(
        restored.world_mut(),
        std::sync::Arc::new(SnapshotNoteCodec { revision: 1 }),
    )
    .unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = runs.0.get(pending.stable_id()).copied().unwrap();
    assert_eq!(
        restored.world().get::<SnapshotNote>(run),
        Some(&SnapshotNote("retained".to_owned()))
    );
}

#[test]
fn active_run_snapshot_restores_request_policy_cursor_and_pending_approval() {
    let mut source = runtime();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    source
        .spawn_policy(
            id("approve-request"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 4,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::Request,
                    prompt: "approve request".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let domain = source.snapshot().unwrap();
    let pending = source.handle().prompt(agent, "policy checkpoint").unwrap();
    source.run_until_stalled().unwrap();
    let approval = source.effects().try_recv().unwrap().unwrap();
    assert!(approval.policy_approval_input().is_some());
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
    )
    .unwrap();

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    restored.handle().resume(run).unwrap();
    restored.run_until_stalled().unwrap();
    let approval = restored.effects().try_recv().unwrap().unwrap();
    assert_eq!(approval.policy_approval_input().unwrap().revision, 4);
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: Some("restored approval".to_owned()),
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    let model = restored.effects().try_recv().unwrap().unwrap();
    assert!(model.model_input().is_some());
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "approved after restore".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    assert!(matches!(
        restored.observe_run(run).unwrap(),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "approved after restore"
    ));
}

#[test]
fn active_run_snapshot_remaps_rebound_multi_point_typed_policy() {
    let mut source = runtime();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let policy = source
        .spawn_typed_policy(
            id("typed-policy"),
            tenant("a"),
            PolicyMeta {
                order: 3,
                revision: 8,
            },
            PolicyCapabilities::new([PolicyPoint::Request, PolicyPoint::CompletionResponse]),
            MultiPointPolicy {
                instructions: "before checkpoint".to_owned(),
                completion: "after checkpoint".to_owned(),
            },
            agent,
        )
        .unwrap();
    source
        .register_request_policy_responder(
            policy,
            PolicyResponderId::new("typed-policy:request").unwrap(),
            apply_multi_point_request,
        )
        .unwrap();
    source
        .register_completion_response_policy_responder(
            policy,
            PolicyResponderId::new("typed-policy:completion").unwrap(),
            apply_multi_point_completion,
        )
        .unwrap();
    let domain = source.snapshot().unwrap();

    let pending = source
        .handle()
        .prompt(agent, "checkpoint typed policy")
        .unwrap();
    source.run_until_stalled().unwrap();
    let original_model = source.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        original_model.model_input().unwrap().instructions,
        "before checkpoint"
    );
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    assert_eq!(
        source.effects().try_recv_cancellation().unwrap(),
        Some(EffectCancellation {
            operation: original_model.operation,
            generation: original_model.generation,
        })
    );
    let snapshot = source.snapshot_active_run(run).unwrap();

    let mut restored = runtime();
    let restored_domain = restored.restore(domain).unwrap();
    let restored_agent = AgentHandle {
        runtime_id: restored.handle.runtime_id,
        entity: restored_domain.0[&id("agent")],
    };
    let restored_policy = restored
        .spawn_typed_policy(
            id("typed-policy"),
            tenant("a"),
            PolicyMeta {
                order: 3,
                revision: 8,
            },
            PolicyCapabilities::new([PolicyPoint::Request, PolicyPoint::CompletionResponse]),
            MultiPointPolicy {
                instructions: "before checkpoint".to_owned(),
                completion: "after checkpoint".to_owned(),
            },
            restored_agent,
        )
        .unwrap();
    restored
        .register_request_policy_responder(
            restored_policy,
            PolicyResponderId::new("typed-policy:request").unwrap(),
            apply_multi_point_request,
        )
        .unwrap();
    restored
        .register_completion_response_policy_responder(
            restored_policy,
            PolicyResponderId::new("typed-policy:completion").unwrap(),
            apply_multi_point_completion,
        )
        .unwrap();

    let restored_runs = restored.restore_active_run(snapshot).unwrap();
    let restored_run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: restored_runs.0[pending.stable_id()],
    };
    restored.handle().resume(restored_run).unwrap();
    restored.run_until_stalled().unwrap();
    let resumed_model = restored.effects().try_recv().unwrap().unwrap();
    assert!(resumed_model.generation > original_model.generation);
    assert_eq!(
        restored.accepted_policies(resumed_model.operation).unwrap(),
        vec![AcceptedPolicyDebug {
            id: id("typed-policy"),
            revision: 8,
            order: 3,
            point: PolicyPoint::Request,
        }]
    );
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: resumed_model.operation,
            generation: resumed_model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "provider text".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    assert!(matches!(
        restored.observe_run(restored_run).unwrap(),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "after checkpoint"
    ));
}

#[test]
fn active_run_snapshot_restores_waiting_tool_effect() {
    let (mut source, agent) = runtime_with_tool();
    let domain = source.snapshot().unwrap();
    let (pending, original_tool) = advance_to_lookup_tool(&mut source, agent);
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    assert_eq!(
        source.effects().try_recv_cancellation().unwrap(),
        Some(EffectCancellation {
            operation: original_tool.operation,
            generation: original_tool.generation,
        })
    );
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
    )
    .unwrap();

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    restored.handle().resume(run).unwrap();
    restored.run_until_stalled().unwrap();
    let tool = restored.effects().try_recv().unwrap().unwrap();
    assert!(tool.generation > original_tool.generation);
    let input = tool.tool_input().unwrap();
    assert_eq!(input.decision.name, "lookup");
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: input.call_id.clone(),
                provider_result_id: input.provider_result_id.clone(),
                provider_call_id: input.provider_call_id.clone(),
                name: input.decision.name.clone(),
                raw: serde_json::json!({"answer": 42}).into(),
                presentation: "42".into(),
                failure: None,
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    let model = restored.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        model.model_input().unwrap().tool_results[0].presentation,
        "42"
    );
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "restored after tool".to_owned(),
                usage: Usage {
                    input_tokens: 6,
                    output_tokens: 3,
                },
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();

    assert_eq!(
        restored.observe_run(run).unwrap(),
        Some(RunState::Completed(RunOutput {
            text: "restored after tool".to_owned(),
            usage: Usage {
                input_tokens: 6,
                output_tokens: 3,
            },
        }))
    );
    assert!(restored.run_transcript(run).unwrap().iter().any(|entry| {
        matches!(
            entry,
            TranscriptEntry::ToolResult { content, .. } if content == "42"
        )
    }));
}

#[test]
fn active_run_snapshot_restores_run_scoped_policy_definition_and_decision() {
    let (mut source, agent) = runtime_with_tool();
    let domain = source.snapshot().unwrap();
    let pending = source
        .handle()
        .prompt(agent, "run policy checkpoint")
        .unwrap();
    source.run_until_stalled().unwrap();
    let model = source.effects().try_recv().unwrap().unwrap();
    let run = source.resolve_run(&pending).unwrap();
    source
        .spawn_run_policy(
            id("run-rewrite"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 6,
                rule: PolicyRule::RewriteToolArguments {
                    tool: Some("lookup".to_owned()),
                    arguments: serde_json::json!({"restored": true}),
                },
            },
            run,
        )
        .unwrap();
    source
        .spawn_run_policy(
            id("run-rewrite-later"),
            tenant("a"),
            Policy {
                order: 1,
                revision: 7,
                rule: PolicyRule::RewriteToolArguments {
                    tool: Some("lookup".to_owned()),
                    arguments: serde_json::json!({"restored": true}),
                },
            },
            run,
        )
        .unwrap();
    source
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "call".to_owned(),
                    provider_result_id: "call".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    arguments: serde_json::json!({"original": true}),
                }],
            })),
        })
        .unwrap();
    source.run_until_stalled().unwrap();
    let original_tool = source.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        original_tool.tool_input().unwrap().arguments,
        serde_json::json!({"restored": true})
    );
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(snapshot.version, 6);
    assert_eq!(snapshot.run_policies.len(), 2);
    assert_eq!(snapshot.run_policies[0].id, id("run-rewrite"));
    assert_eq!(snapshot.run_policies[0].run_id, *pending.stable_id());
    let accepted = snapshot
        .operations
        .iter()
        .find_map(|operation| operation.tool_call_policies.as_deref())
        .and_then(|policies| policies.first())
        .unwrap();
    assert_eq!(accepted.order, 0);
    assert_eq!(accepted.point, PolicyPoint::ToolCall);

    let mut incompatible = snapshot.clone();
    incompatible
        .operations
        .iter_mut()
        .find_map(|operation| operation.tool_call_policies.as_mut())
        .and_then(|policies| policies.first_mut())
        .unwrap()
        .order = 1;
    let mut rejected = runtime();
    rejected.restore(domain.clone()).unwrap();
    assert!(matches!(
        rejected.restore_active_run(incompatible),
        Err(ActiveRunSnapshotError::InvalidSnapshot(_))
    ));

    let mut wrong_point = snapshot.clone();
    wrong_point
        .operations
        .iter_mut()
        .find_map(|operation| operation.tool_call_policies.as_mut())
        .and_then(|policies| policies.first_mut())
        .unwrap()
        .point = PolicyPoint::Request;
    let mut rejected = runtime();
    rejected.restore(domain.clone()).unwrap();
    assert!(matches!(
        rejected.restore_active_run(wrong_point),
        Err(ActiveRunSnapshotError::InvalidSnapshot(_))
    ));

    let mut reordered = snapshot.clone();
    reordered
        .operations
        .iter_mut()
        .find_map(|operation| operation.tool_call_policies.as_mut())
        .unwrap()
        .reverse();
    let mut rejected = runtime();
    rejected.restore(domain.clone()).unwrap();
    assert!(matches!(
        rejected.restore_active_run(reordered),
        Err(ActiveRunSnapshotError::InvalidSnapshot(_))
    ));

    let mut exhausted_generation = snapshot.clone();
    exhausted_generation.operations[0].generation = u64::MAX;
    let mut rejected = runtime();
    rejected.restore(domain.clone()).unwrap();
    assert!(matches!(
        rejected.restore_active_run(exhausted_generation),
        Err(ActiveRunSnapshotError::InvalidSnapshot(_))
    ));

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    let restored_policy = restored
        .world()
        .iter_entities()
        .find(|entity| {
            entity
                .get::<StableId>()
                .is_some_and(|stable_id| stable_id == &id("run-rewrite"))
        })
        .map(|entity| entity.id())
        .unwrap();
    let restored_later_policy = restored
        .world()
        .iter_entities()
        .find(|entity| {
            entity
                .get::<StableId>()
                .is_some_and(|stable_id| stable_id == &id("run-rewrite-later"))
        })
        .map(|entity| entity.id())
        .unwrap();
    assert_eq!(
        restored
            .world()
            .get::<PolicyForRun>(restored_policy)
            .map(Relationship::get),
        Some(run.entity())
    );
    assert_eq!(
        restored
            .world()
            .get::<Policy>(restored_policy)
            .unwrap()
            .revision,
        6
    );
    restored.handle().resume(run).unwrap();
    restored.run_until_stalled().unwrap();
    let tool = restored.effects().try_recv().unwrap().unwrap();
    assert!(tool.generation > original_tool.generation);
    assert_eq!(
        tool.tool_input().unwrap().arguments,
        serde_json::json!({"restored": true})
    );
    assert_eq!(
        restored
            .world()
            .get::<AcceptedToolCallPolicies>(tool.operation),
        Some(&AcceptedToolCallPolicies(vec![
            AcceptedPolicy {
                id: id("run-rewrite"),
                entity: restored_policy,
                revision: 6,
                order: 0,
                point: PolicyPoint::ToolCall,
            },
            AcceptedPolicy {
                id: id("run-rewrite-later"),
                entity: restored_later_policy,
                revision: 7,
                order: 1,
                point: PolicyPoint::ToolCall,
            },
        ]))
    );
}

#[test]
fn active_run_snapshot_restores_settled_terminal_before_publication() {
    let mut source = runtime();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let domain = source.snapshot().unwrap();
    let pending = source.handle().prompt(agent, "freeze terminal").unwrap();
    source.run_until_stalled().unwrap();
    let request = source.effects().try_recv().unwrap().unwrap();
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::FreezeAfterIngress)
        .unwrap();
    source.run_until_stalled().unwrap();
    let completion_sender = source.effects().completion_sender();
    completion_sender
        .try_send_provider_diagnostics(ProviderDiagnosticsIngress::serialized(
            request.operation,
            request.generation,
            serde_json::json!({"provider_request_id": "response-7"}),
        ))
        .unwrap();
    completion_sender
        .try_send(EffectCompletion {
            operation: request.operation,
            generation: request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "frozen answer".to_owned(),
                usage: Usage {
                    input_tokens: 4,
                    output_tokens: 2,
                },
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    source.run_until_stalled().unwrap();
    assert!(matches!(
        source.world().get::<OperationState>(request.operation),
        Some(OperationState {
            phase: OperationPhase::Settled(OperationOutcome::Success(EffectOutput::Model(_))),
            ..
        })
    ));
    assert!(matches!(
        source.world().get::<RunState>(run.entity()),
        Some(RunState::WaitingModel { .. })
    ));
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        snapshot
            .operations
            .iter()
            .find_map(|operation| { operation.provider_diagnostics.as_ref() }),
        Some(&serde_json::json!({"provider_request_id": "response-7"}))
    );

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    restored.handle().resume(run).unwrap();
    restored.run_until_stalled().unwrap();

    let mut diagnostics = restored.world_mut().query::<&ProviderResponseDiagnostics>();
    assert_eq!(
        diagnostics.single(restored.world()).unwrap().0,
        serde_json::json!({"provider_request_id": "response-7"})
    );

    assert_eq!(
        restored.observe_run(run).unwrap(),
        Some(RunState::Completed(RunOutput {
            text: "frozen answer".to_owned(),
            usage: Usage {
                input_tokens: 4,
                output_tokens: 2,
            },
        }))
    );
    assert_eq!(
        restored.run_transcript(run),
        Some(vec![
            TranscriptEntry::User("freeze terminal".to_owned()),
            TranscriptEntry::Assistant("frozen answer".to_owned()),
        ])
    );
    assert_eq!(restored.effects().try_recv().unwrap(), None);
}

#[test]
fn active_run_snapshot_restores_tool_batch_and_result_policy() {
    let (mut source, agent) = runtime_with_tool();
    source
        .spawn_policy(
            id("approve-result"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 2,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::ToolResult,
                    prompt: "approve tool result".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let domain = source.snapshot().unwrap();
    let (pending, tool) = advance_to_lookup_tool(&mut source, agent);
    let input = tool.tool_input().unwrap();
    source
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool.operation,
            generation: tool.generation,
            result: Ok(EffectOutput::Tool(ToolEffectOutput {
                call_id: input.call_id.clone(),
                provider_result_id: input.provider_result_id.clone(),
                provider_call_id: input.provider_call_id.clone(),
                name: input.decision.name.clone(),
                raw: serde_json::json!({"answer": 42}).into(),
                presentation: "42".into(),
                failure: None,
            })),
        })
        .unwrap();
    source.run_until_stalled().unwrap();
    let approval = source.effects().try_recv().unwrap().unwrap();
    assert!(approval.policy_approval_input().is_some());
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
    )
    .unwrap();

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    restored.handle().resume(run).unwrap();
    restored.run_until_stalled().unwrap();
    let approval = restored.effects().try_recv().unwrap().unwrap();
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: None,
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    let model = restored.effects().try_recv().unwrap().unwrap();
    assert_eq!(
        model
            .model_input()
            .unwrap()
            .tool_results
            .first()
            .unwrap()
            .presentation,
        "42"
    );
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "tool batch resumed".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    assert!(matches!(
        restored.observe_run(run).unwrap(),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "tool batch resumed"
    ));
}

#[test]
fn active_run_snapshot_restores_pending_invalid_call_resolution() {
    let (mut source, agent) = runtime_with_tool();
    source
        .spawn_policy(
            id("approve-invalid"),
            tenant("a"),
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::RequireApproval {
                    point: PolicyPoint::InvalidToolCall,
                    prompt: "approve invalid repair".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    source
        .spawn_policy(
            id("repair-invalid"),
            tenant("a"),
            Policy {
                order: 1,
                revision: 3,
                rule: PolicyRule::RepairInvalidTool {
                    from: Some("lookpu".to_owned()),
                    to: "lookup".to_owned(),
                },
            },
            agent,
        )
        .unwrap();
    let domain = source.snapshot().unwrap();
    let pending = source
        .handle()
        .prompt(agent, "repair after restore")
        .unwrap();
    source.run_until_stalled().unwrap();
    let model = source.effects().try_recv().unwrap().unwrap();
    source
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: String::new(),
                usage: Usage::default(),
                tool_calls: vec![ModelToolCall {
                    id: "invalid-call".to_owned(),
                    provider_result_id: "invalid-result".to_owned(),
                    provider_call_id: Some("provider-call".to_owned()),
                    name: "lookpu".to_owned(),
                    arguments: serde_json::json!({"query": "persist me"}),
                }],
            })),
        })
        .unwrap();
    source.run_until_stalled().unwrap();
    let approval = source.effects().try_recv().unwrap().unwrap();
    assert!(approval.policy_approval_input().is_some());
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
    )
    .unwrap();

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    restored.handle().resume(run).unwrap();
    restored.run_until_stalled().unwrap();
    let approval = restored.effects().try_recv().unwrap().unwrap();
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: approval.operation,
            generation: approval.generation,
            result: Ok(EffectOutput::PolicyApproval(PolicyApprovalEffectOutput {
                approved: true,
                reason: None,
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    let tool = restored.effects().try_recv().unwrap().unwrap();
    let input = tool.tool_input().unwrap();
    assert_eq!(input.decision.name, "lookup");
    assert_eq!(input.call_id, "invalid-call");
    assert_eq!(input.provider_result_id, "invalid-result");
    assert_eq!(input.provider_call_id.as_deref(), Some("provider-call"));
    assert_eq!(input.arguments, serde_json::json!({"query": "persist me"}));
}

#[test]
fn active_run_snapshot_restores_memory_load_and_terminal_persistence() {
    let mut source = runtime();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let store = source
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
    source
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
    let domain = source.snapshot().unwrap();
    let pending = source
        .handle()
        .prompt_in_conversation(agent, id("conversation"), "new question")
        .unwrap();
    source.run_until_stalled().unwrap();
    let load = source.effects().try_recv().unwrap().unwrap();
    assert!(matches!(
        load.store_input().unwrap().operation,
        StoreOperation::LoadConversation { .. }
    ));
    let run = source.resolve_run(&pending).unwrap();
    source
        .handle()
        .pause_with_mode(run, PauseMode::CancelAndSuspend)
        .unwrap();
    source.run_until_stalled().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(run).unwrap()).unwrap(),
    )
    .unwrap();

    let mut middle = runtime();
    middle.restore(domain).unwrap();
    let runs = middle.restore_active_run(snapshot).unwrap();
    let middle_run = RunHandle {
        runtime_id: middle.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    middle.handle().resume(middle_run).unwrap();
    middle.run_until_stalled().unwrap();
    let load = middle.effects().try_recv().unwrap().unwrap();
    middle
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
    middle.run_until_stalled().unwrap();
    let model = middle.effects().try_recv().unwrap().unwrap();
    middle
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: model.operation,
            generation: model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "new answer".to_owned(),
                usage: Usage {
                    input_tokens: 5,
                    output_tokens: 2,
                },
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    middle.run_until_stalled().unwrap();
    let persist = middle.effects().try_recv().unwrap().unwrap();
    assert!(matches!(
        persist.store_input().unwrap().operation,
        StoreOperation::PersistConversation { .. }
    ));
    middle
        .handle()
        .pause_with_mode(middle_run, PauseMode::CancelAndSuspend)
        .unwrap();
    middle.run_until_stalled().unwrap();
    let domain = middle.snapshot().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&middle.snapshot_active_run(middle_run).unwrap()).unwrap(),
    )
    .unwrap();

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let run = RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(pending.stable_id()).copied().unwrap(),
    };
    restored.handle().resume(run).unwrap();
    restored.run_until_stalled().unwrap();
    let persist = restored.effects().try_recv().unwrap().unwrap();
    let store_input = persist.store_input().unwrap();
    assert!(matches!(
        store_input.operation,
        StoreOperation::PersistConversation { .. }
    ));
    let StoreOperation::PersistConversation { entries, .. } = &store_input.operation else {
        return;
    };
    assert!(entries.contains(&TranscriptEntry::Assistant("new answer".to_owned())));
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: persist.operation,
            generation: persist.generation,
            result: Ok(EffectOutput::Store(StoreEffectOutput::Persisted)),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    assert_eq!(
        restored.observe_run(run).unwrap(),
        Some(RunState::Completed(RunOutput {
            text: "new answer".to_owned(),
            usage: Usage {
                input_tokens: 5,
                output_tokens: 2,
            },
        }))
    );
}

#[test]
fn active_run_snapshot_restores_child_topology_and_result_order() {
    let mut source = runtime();
    let model = source
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
    let agent = source
        .spawn_agent(id("agent"), tenant("a"), Agent::default(), model)
        .unwrap();
    let domain = source.snapshot().unwrap();
    let parent_pending = source.handle().prompt(agent, "parent").unwrap();
    source.run_until_stalled().unwrap();
    let _ = source.effects().try_recv().unwrap().unwrap();
    let parent = source.resolve_run(&parent_pending).unwrap();
    let first_pending = source
        .handle()
        .spawn_child_run(parent, agent, "first child")
        .unwrap();
    let second_pending = source
        .handle()
        .spawn_child_run(parent, agent, "second child")
        .unwrap();
    source.run_until_stalled().unwrap();
    let _ = source.effects().try_recv().unwrap().unwrap();
    let _ = source.effects().try_recv().unwrap().unwrap();
    let first = source.resolve_run(&first_pending).unwrap();
    let second = source.resolve_run(&second_pending).unwrap();
    for run in [parent, first, second] {
        source
            .handle()
            .pause_with_mode(run, PauseMode::CancelAndSuspend)
            .unwrap();
    }
    source.run_until_stalled().unwrap();
    let snapshot: ActiveRunSnapshot = serde_json::from_str(
        &serde_json::to_string(&source.snapshot_active_run(parent).unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(snapshot.runs.len(), 3);

    let mut restored = runtime();
    restored.restore(domain).unwrap();
    let runs = restored.restore_active_run(snapshot).unwrap();
    let handle = |id: &StableId| RunHandle {
        runtime_id: restored.handle.runtime_id,
        entity: runs.0.get(id).copied().unwrap(),
    };
    let parent = handle(parent_pending.stable_id());
    let first = handle(first_pending.stable_id());
    let second = handle(second_pending.stable_id());
    assert_eq!(
        restored.world().get::<ParentRun>(first.entity()),
        Some(&ParentRun(parent.entity()))
    );
    assert_eq!(
        restored.world().get::<ParentRun>(second.entity()),
        Some(&ParentRun(parent.entity()))
    );
    assert_eq!(
        restored.world().get::<WaitingForChildren>(parent.entity()),
        Some(&WaitingForChildren)
    );
    for run in [parent, first, second] {
        restored.handle().resume(run).unwrap();
    }
    restored.run_until_stalled().unwrap();
    let requests = (0..2)
        .map(|_| restored.effects().try_recv().unwrap().unwrap())
        .collect::<Vec<_>>();
    let operation_for = |run: RunHandle| match restored.world().get::<RunState>(run.entity()) {
        Some(RunState::WaitingModel { operation }) => Some(*operation),
        _ => None,
    };
    let first_operation = operation_for(first).unwrap();
    let second_operation = operation_for(second).unwrap();
    let find_request = |operation| {
        requests
            .iter()
            .find(|request| request.operation == operation)
            .cloned()
            .unwrap()
    };
    for (request, text) in [
        (find_request(second_operation), "second output"),
        (find_request(first_operation), "first output"),
    ] {
        restored
            .effects()
            .completion_sender()
            .try_send(EffectCompletion {
                operation: request.operation,
                generation: request.generation,
                result: Ok(EffectOutput::Model(ModelEffectOutput {
                    assistant_message: None,
                    text: text.to_owned(),
                    usage: Usage::default(),
                    tool_calls: Vec::new(),
                })),
            })
            .unwrap();
        restored.run_until_stalled().unwrap();
    }
    let child_results = restored
        .world()
        .get::<RunRecord>(parent.entity())
        .unwrap()
        .transcript
        .iter()
        .filter_map(|entry| match entry {
            TranscriptEntry::ChildResult { result, .. } => result.as_ref().ok().cloned(),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(child_results, vec!["first output", "second output"]);
    assert!(
        restored
            .world()
            .get::<WaitingForChildren>(parent.entity())
            .is_none()
    );
    let parent_request = restored.effects().try_recv().unwrap().unwrap();
    restored
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: parent_request.operation,
            generation: parent_request.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "parent output".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })
        .unwrap();
    restored.run_until_stalled().unwrap();
    assert!(matches!(
        restored.observe_run(parent).unwrap(),
        Some(RunState::Completed(RunOutput { text, .. })) if text == "parent output"
    ));
}
