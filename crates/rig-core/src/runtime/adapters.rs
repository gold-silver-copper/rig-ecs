//! Typed authoring adapters for executing owned ECS effects without world access.

use crate::{
    OneOrMany,
    completion::{
        AssistantContent, Chat, CompletionError, CompletionModel, CompletionRequest, Document,
        Message, Prompt, PromptError, StructuredOutputError, ToolDefinition, TypedPrompt,
        Usage as CompletionUsage,
    },
    message::{
        ToolCall, ToolChoice, ToolFunction, ToolResult as MessageToolResult, ToolResultContent,
        UserContent,
    },
    runtime::{
        Agent, AgentHandle, CanonicalError, DiscoveryEffectInput, DiscoveryEffectOutput,
        DriveError, EffectCompletion, EffectDelta, EffectDeltaSender, EffectInput, EffectIoError,
        EffectOutput, EffectRequest, InstallError, ModelCapability, ModelDecision,
        ModelEffectInput, ModelEffectOutput, ModelToolCall, ModelToolChoice, OutputRequirement,
        RetrievalRequirement, RetrievedDocument, RunOutput, RunState, Runtime, RuntimeConfig,
        SpawnError, StableId, StoreCapability, StoreEffectInput, StoreEffectOutput, StoreGrant,
        StoreOperation, StreamItem, StreamReceiveError, StreamTerminal, SubmitError, TenantId,
        ToolCapability, ToolEffectInput, ToolEffectOutput, ToolGrant, TranscriptEntry, Usage,
    },
    streaming::{StreamedAssistantContent, ToolCallDeltaContent},
    tool::IntoToolOutput,
    vector_store::{VectorStoreIndexDyn, request::VectorSearchRequest},
};
use futures::future::BoxFuture;
use futures::{FutureExt, Stream, StreamExt};
use serde::de::DeserializeOwned;
use std::{
    collections::{HashMap, HashSet},
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::{Arc, Mutex},
};
use thiserror::Error;

/// Executes ECS model-effect input through any typed [`CompletionModel`].
///
/// This adapter owns only a typed transport/model handle. It receives an immutable
/// effect snapshot and never receives a [`bevy_ecs::world::World`] or ECS borrow.
#[derive(Clone, Debug)]
pub struct CompletionModelAdapter<M> {
    model: M,
    binding: Option<ModelDecision>,
}

/// Standalone typed facade that drives the authoritative ECS schedule.
///
/// This convenience owner is intentionally a thin effect executor around
/// [`Runtime`]. Run phase, transcript, decisions, usage, and outcomes remain
/// ECS state; the facade contains no parallel agent state machine.
#[derive(Clone)]
pub struct LocalModelAgent<M> {
    runtime: Arc<Mutex<Runtime>>,
    driver: Arc<tokio::sync::Mutex<()>>,
    agent: AgentHandle,
    model: CompletionModelAdapter<M>,
    tools: Arc<HashMap<(StableId, u64), Arc<dyn LocalToolExecutor>>>,
    stores: Arc<HashMap<(StableId, u64), Arc<dyn LocalStoreExecutor>>>,
}

/// Incremental observation from a schedule-driven local agent.
#[derive(Clone, Debug, PartialEq)]
pub enum LocalStreamEvent {
    /// Ordered provider text delta committed through ECS ingress.
    Delta(String),
    /// Complete provider tool call observed during the model effect.
    ToolCall {
        /// Canonical tool call.
        tool_call: ToolCall,
        /// Stream-local correlation identity.
        internal_call_id: String,
    },
    /// Tool call after its logical ECS batch committed successfully.
    ToolCommitted {
        /// Canonical tool call.
        tool_call: ToolCall,
        /// Stream-local correlation identity.
        internal_call_id: String,
    },
    /// Partial provider tool call data.
    ToolCallDelta {
        /// Provider correlation identity.
        id: String,
        /// Stream-local correlation identity.
        internal_call_id: String,
        /// Partial call content.
        content: ToolCallDeltaContent,
    },
    /// Complete provider reasoning block.
    Reasoning(crate::message::Reasoning),
    /// Partial provider reasoning text.
    ReasoningDelta {
        /// Provider reasoning block identity.
        id: Option<String>,
        /// Partial reasoning text.
        reasoning: String,
    },
    /// Provider-native stream item without a canonical typed representation.
    Unknown(serde_json::Value),
    /// Tool result after its logical batch committed.
    ToolResult {
        /// Model-visible result.
        tool_result: MessageToolResult,
        /// Stream-local correlation identity.
        internal_call_id: String,
    },
    /// The same terminal output observed by blocking execution.
    Finished {
        /// Terminal output.
        output: RunOutput,
        /// Complete committed transcript.
        transcript: Vec<TranscriptEntry>,
        /// Ordered usage for every committed model operation.
        completion_calls: Vec<Usage>,
    },
}

/// Owned prompt command submitted when awaited.
pub struct AgentPromptRequest<M>
where
    M: CompletionModel,
{
    agent: LocalModelAgent<M>,
    prompt: Message,
    history: Vec<Message>,
    max_model_calls: Option<u32>,
    conversation: Result<Option<StableId>, crate::runtime::IdentityError>,
}

/// Awaitable prompt command that returns terminal usage and canonical messages.
pub struct ExtendedAgentPromptRequest<M>
where
    M: CompletionModel,
{
    request: AgentPromptRequest<M>,
}

struct LocalRunResult {
    output: RunOutput,
    transcript: Vec<TranscriptEntry>,
    completion_calls: Vec<Usage>,
}

fn catch_executor_panic<'a, T, F>(future: F) -> BoxFuture<'a, Result<T, CanonicalError>>
where
    T: Send + 'a,
    F: Future<Output = Result<T, CanonicalError>> + Send + 'a,
{
    Box::pin(async move {
        AssertUnwindSafe(future)
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| {
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_owned());
                Err(CanonicalError::ExecutorPanicked(message))
            })
    })
}

impl<M> AgentPromptRequest<M>
where
    M: CompletionModel,
{
    /// Requests terminal details in addition to the response text.
    pub fn extended_details(self) -> ExtendedAgentPromptRequest<M> {
        ExtendedAgentPromptRequest { request: self }
    }

    /// Routes the run through an ECS conversation-memory store relationship.
    pub fn conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = StableId::new(conversation).map(Some);
        self
    }

    /// Copies canonical history into this run at command ingress.
    pub fn history<I>(mut self, history: I) -> Self
    where
        I: IntoIterator,
        I::Item: std::borrow::Borrow<Message>,
    {
        self.history = history
            .into_iter()
            .map(|message| std::borrow::Borrow::borrow(&message).clone())
            .collect();
        self
    }

    /// Overrides the run's model-call budget.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_model_calls = Some(u32::try_from(max_turns).unwrap_or(u32::MAX));
        self
    }
}

impl<M> std::future::IntoFuture for ExtendedAgentPromptRequest<M>
where
    M: CompletionModel + 'static,
{
    type Output = Result<crate::agent::PromptResponse, PromptError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let request = self.request;
            let conversation = request
                .conversation
                .map_err(|error| local_prompt_error(LocalAgentError::Identity(error)))?;
            let details = request
                .agent
                .run_prompt_options_detailed(
                    request.prompt,
                    request.history,
                    None,
                    conversation,
                    request.max_model_calls,
                )
                .await
                .map_err(local_prompt_error)?;
            let messages = response_messages(&details.transcript)
                .map_err(|error| local_prompt_error(LocalAgentError::Canonical(error)))?;
            let output = details.output;
            let usage = CompletionUsage {
                input_tokens: output.usage.input_tokens,
                output_tokens: output.usage.output_tokens,
                total_tokens: output.usage.input_tokens + output.usage.output_tokens,
                ..CompletionUsage::default()
            };
            let mut response = crate::agent::PromptResponse::new(output.text, usage);
            response.messages = Some(messages);
            response.completion_calls = details
                .completion_calls
                .into_iter()
                .enumerate()
                .map(|(index, usage)| {
                    crate::agent::CompletionCall::new(
                        index,
                        CompletionUsage {
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            total_tokens: usage.input_tokens + usage.output_tokens,
                            ..CompletionUsage::default()
                        },
                    )
                })
                .collect();
            Ok(response)
        })
    }
}

impl<M> std::future::IntoFuture for AgentPromptRequest<M>
where
    M: CompletionModel + 'static,
{
    type Output = Result<String, PromptError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let conversation = self
                .conversation
                .map_err(|error| local_prompt_error(LocalAgentError::Identity(error)))?;
            self.agent
                .run_prompt_options(
                    self.prompt,
                    self.history,
                    None,
                    conversation,
                    self.max_model_calls,
                )
                .await
                .map(|output| output.text)
                .map_err(local_prompt_error)
        })
    }
}

trait LocalToolExecutor: Send + Sync {
    fn execute(
        &self,
        input: ToolEffectInput,
    ) -> BoxFuture<'_, Result<ToolEffectOutput, CanonicalError>>;
}

trait LocalStoreExecutor: Send + Sync {
    fn execute(
        &self,
        input: StoreEffectInput,
    ) -> BoxFuture<'_, Result<StoreEffectOutput, CanonicalError>>;
}

struct TypedLocalStore<S> {
    adapter: StoreAdapter<S>,
}

impl<S> LocalStoreExecutor for TypedLocalStore<S>
where
    S: EcsStore + 'static,
{
    fn execute(
        &self,
        input: StoreEffectInput,
    ) -> BoxFuture<'_, Result<StoreEffectOutput, CanonicalError>> {
        Box::pin(self.adapter.execute(input))
    }
}

struct TypedLocalTool<T> {
    adapter: ToolAdapter<T>,
}

struct AuthoredTool<T>(T);

struct DynamicEcsTool(crate::tool::DynamicTool);

impl EcsTool for DynamicEcsTool {
    type Args = serde_json::Value;
    type Output = crate::tool::ToolOutput;
    type Error = crate::tool::ToolExecutionError;

    fn call(
        &self,
        arguments: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        self.0.execute(arguments)
    }
}

impl<T> EcsTool for AuthoredTool<T>
where
    T: crate::tool::Tool,
    T::Output: Send,
{
    type Args = T::Args;
    type Output = T::Output;
    type Error = T::Error;

    fn call(
        &self,
        arguments: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        self.0.call(arguments)
    }

    fn map_error(&self, error: Self::Error) -> crate::tool::ToolExecutionError {
        self.0.map_error(error)
    }
}

impl<T> LocalToolExecutor for TypedLocalTool<T>
where
    T: EcsTool + 'static,
{
    fn execute(
        &self,
        input: ToolEffectInput,
    ) -> BoxFuture<'_, Result<ToolEffectOutput, CanonicalError>> {
        Box::pin(self.adapter.execute(input))
    }
}

struct LocalToolRegistration {
    id: StableId,
    capability: ToolCapability,
    executor: Arc<dyn LocalToolExecutor>,
}

struct LocalStoreRegistration {
    id: StableId,
    capability: StoreCapability,
    grant: StoreGrant,
    executor: Arc<dyn LocalStoreExecutor>,
}

struct VectorIndexStore {
    index: Arc<dyn VectorStoreIndexDyn>,
}

struct ConversationStore<B> {
    backend: B,
}

impl<B> EcsStore for ConversationStore<B>
where
    B: crate::memory::ConversationMemory,
{
    type Error = crate::memory::MemoryError;

    async fn execute(&self, operation: StoreOperation) -> Result<StoreEffectOutput, Self::Error> {
        match operation {
            StoreOperation::LoadConversation { conversation } => {
                let messages = self.backend.load(conversation.as_str()).await?;
                let entries = messages
                    .into_iter()
                    .map(|message| {
                        serde_json::to_value(message)
                            .map(TranscriptEntry::Message)
                            .map_err(|error| {
                                crate::memory::MemoryError::Internal(error.to_string())
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(StoreEffectOutput::LoadedConversation(entries))
            }
            StoreOperation::PersistConversation {
                conversation,
                entries,
            } => {
                let messages = transcript_messages(&entries)
                    .map_err(|error| crate::memory::MemoryError::Internal(error.to_string()))?;
                self.backend.clear(conversation.as_str()).await?;
                self.backend.append(conversation.as_str(), messages).await?;
                Ok(StoreEffectOutput::Persisted)
            }
            StoreOperation::Retrieve { .. } => Err(crate::memory::MemoryError::Internal(
                "conversation store received retrieval operation".to_owned(),
            )),
        }
    }
}

#[derive(Debug, Error)]
enum VectorIndexStoreError {
    #[error("vector index only supports retrieval operations")]
    UnsupportedOperation,
    #[error(transparent)]
    Search(#[from] crate::vector_store::VectorStoreError),
}

impl EcsStore for VectorIndexStore {
    type Error = VectorIndexStoreError;

    async fn execute(&self, operation: StoreOperation) -> Result<StoreEffectOutput, Self::Error> {
        let StoreOperation::Retrieve { query, limit } = operation else {
            return Err(VectorIndexStoreError::UnsupportedOperation);
        };
        let request = VectorSearchRequest::builder()
            .query(query)
            .samples(u64::try_from(limit).unwrap_or(u64::MAX))
            .build();
        let documents = self
            .index
            .top_n(request)
            .await?
            .into_iter()
            .map(|(_, id, document)| RetrievedDocument {
                id,
                text: serde_json::to_string_pretty(&document)
                    .unwrap_or_else(|_| document.to_string()),
                metadata: Default::default(),
            })
            .collect();
        Ok(StoreEffectOutput::Retrieved(documents))
    }
}

/// Construction helper that spawns a standalone ECS agent composition.
pub struct LocalModelAgentBuilder<M> {
    model: M,
    provider: String,
    model_name: String,
    name: Option<String>,
    description: Option<String>,
    instructions: String,
    temperature_bits: Option<u64>,
    max_tokens: Option<u64>,
    tool_choice: Option<ModelToolChoice>,
    additional_params: Option<serde_json::Value>,
    documents: Vec<RetrievedDocument>,
    max_model_calls: u32,
    output_schema: Option<serde_json::Value>,
    structured_output_mode: StructuredOutputMode,
    terminal_tool: Option<String>,
    retrieval_limit: Option<usize>,
    tools: Vec<LocalToolRegistration>,
    stores: Vec<LocalStoreRegistration>,
}

/// Provider-facing strategy used after an output schema is attached.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StructuredOutputMode {
    /// Use a terminal tool when ordinary tools are present, otherwise use the
    /// provider's native schema facility.
    #[default]
    Auto,
    /// Send the schema through the provider's native structured-output field.
    Native,
    /// Advertise a synthetic `final_result` tool whose arguments end the run.
    Tool,
    /// Put the schema in the instructions and validate the final model text.
    Prompted,
}

/// Ergonomic construction facade that spawns an ECS-native standalone agent.
///
/// This builder owns no runner or registry. [`Self::build`] creates one
/// authoritative world and installs the same schedule used by embedded hosts.
pub struct AgentBuilder<M> {
    inner: LocalModelAgentBuilder<M>,
}

/// Standalone ECS agent facade.
pub type AgentFacade<M> = LocalModelAgent<M>;

impl<M> AgentBuilder<M>
where
    M: CompletionModel,
{
    /// Begins construction around a configured typed completion model.
    pub fn new(model: M) -> Self {
        Self::with_identity(model, "configured", "")
    }

    /// Begins construction with provider and model identity used in ECS state.
    pub fn with_identity(
        model: M,
        provider: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            inner: LocalModelAgentBuilder::new(model, provider, model_name),
        }
    }

    /// Sets system instructions.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.inner = self.inner.preamble(preamble);
        self
    }

    /// Sets the agent entity's human-readable name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.name(name);
        self
    }

    /// Sets the agent entity's human-readable description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.inner = self.inner.description(description);
        self
    }

    /// Appends system instructions.
    pub fn append_preamble(mut self, preamble: impl AsRef<str>) -> Self {
        self.inner = self.inner.append_preamble(preamble);
        self
    }

    /// Adds static context.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.inner = self.inner.context(context);
        self
    }

    /// Adds an ECS vector-search capability used before model preparation.
    pub fn dynamic_context(
        mut self,
        limit: usize,
        index: impl VectorStoreIndexDyn + 'static,
    ) -> Self {
        self.inner = self.inner.dynamic_context(limit, index);
        self
    }

    /// Adds an ECS conversation-memory store capability.
    pub fn memory<B>(mut self, backend: B) -> Self
    where
        B: crate::memory::ConversationMemory + 'static,
    {
        self.inner = self.inner.memory(backend);
        self
    }

    /// Sets the sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.inner = self.inner.temperature(temperature);
        self
    }

    /// Sets the provider output-token limit.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.inner = self.inner.max_tokens(max_tokens);
        self
    }

    /// Sets provider-specific request parameters.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.inner = self.inner.additional_params(params);
        self
    }

    /// Sets provider-independent tool selection.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        let choice = match choice {
            ToolChoice::Auto => ModelToolChoice::Auto,
            ToolChoice::None => ModelToolChoice::None,
            ToolChoice::Required => ModelToolChoice::Required,
            ToolChoice::Specific { function_names } => ModelToolChoice::Specific(function_names),
        };
        self.inner = self.inner.tool_choice(choice);
        self
    }

    /// Sets the maximum number of model operations for each run.
    pub fn default_max_turns(mut self, max_turns: usize) -> Self {
        self.inner = self
            .inner
            .max_model_calls(u32::try_from(max_turns).unwrap_or(u32::MAX));
        self
    }

    /// Adds one typed tool as an entity and grants it to the agent.
    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: crate::tool::Tool + 'static,
        T::Output: Send,
    {
        self.inner = self.inner.authored_tool(tool);
        self
    }

    /// Adds one runtime-authored context-free tool capability.
    pub fn dynamic_tool(mut self, tool: crate::tool::DynamicTool) -> Self {
        self.inner = self.inner.dynamic_tool(tool);
        self
    }

    /// Adds runtime-authored context-free tool capabilities in iterator order.
    pub fn dynamic_tools(
        mut self,
        tools: impl IntoIterator<Item = crate::tool::DynamicTool>,
    ) -> Self {
        for tool in tools {
            self.inner = self.inner.dynamic_tool(tool);
        }
        self
    }

    /// Requires terminal text to conform to the schema generated for `T`.
    pub fn output_schema<T>(mut self) -> Self
    where
        T: schemars::JsonSchema,
    {
        self.inner = self
            .inner
            .output_schema(serde_json::Value::from(schemars::schema_for!(T)));
        self
    }

    /// Requires terminal text to conform to a raw JSON Schema.
    pub fn output_schema_raw(mut self, schema: schemars::Schema) -> Self {
        self.inner = self.inner.output_schema(serde_json::Value::from(schema));
        self
    }

    /// Selects how a configured output schema is presented to the model.
    pub fn structured_output_mode(mut self, mode: StructuredOutputMode) -> Self {
        self.inner = self.inner.structured_output_mode(mode);
        self
    }

    pub(crate) fn terminal_tool(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.terminal_tool(name);
        self
    }

    /// Attempts to spawn the complete composition into a new runtime world.
    pub fn try_build(self) -> Result<AgentFacade<M>, LocalAgentError> {
        self.inner.build()
    }

    /// Spawns the complete composition into a new runtime world.
    ///
    /// All identities used by this facade are generated and unique. Failure
    /// therefore indicates an internal installation invariant violation.
    #[allow(clippy::panic)]
    pub fn build(self) -> AgentFacade<M> {
        self.try_build().unwrap_or_else(|error| {
            panic!("failed to install generated ECS agent composition: {error}")
        })
    }
}

impl<M> LocalModelAgentBuilder<M>
where
    M: CompletionModel,
{
    /// Begins construction around a configured typed model handle.
    pub fn new(model: M, provider: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            model,
            provider: provider.into(),
            model_name: model_name.into(),
            name: None,
            description: None,
            instructions: String::new(),
            temperature_bits: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            documents: Vec::new(),
            max_model_calls: 16,
            output_schema: None,
            structured_output_mode: StructuredOutputMode::Auto,
            terminal_tool: None,
            retrieval_limit: None,
            tools: Vec::new(),
            stores: Vec::new(),
        }
    }

    /// Sets instructions stored on the agent entity.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Sets the agent's queryable human-readable name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the agent's queryable human-readable description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the system instructions stored on the agent entity.
    pub fn preamble(self, instructions: impl Into<String>) -> Self {
        self.instructions(instructions)
    }

    /// Appends text to the system instructions stored on the agent entity.
    pub fn append_preamble(mut self, instructions: impl AsRef<str>) -> Self {
        if !self.instructions.is_empty() {
            self.instructions.push('\n');
        }
        self.instructions.push_str(instructions.as_ref());
        self
    }

    /// Sets the sampling temperature snapshotted into each model operation.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature_bits = Some(temperature.to_bits());
        self
    }

    /// Sets the provider output-token limit.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets provider-independent tool-selection behavior.
    pub fn tool_choice(mut self, tool_choice: ModelToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets provider-specific parameters copied into immutable operations.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Adds one static context document.
    pub fn context(mut self, document: impl Into<String>) -> Self {
        let order = self.documents.len();
        self.documents.push(RetrievedDocument {
            id: format!("static_doc_{order}"),
            text: document.into(),
            metadata: Default::default(),
        });
        self
    }

    /// Sets the maximum number of model calls accepted for one run.
    pub fn max_model_calls(mut self, max_model_calls: u32) -> Self {
        self.max_model_calls = max_model_calls;
        self
    }

    /// Sets a raw JSON Schema requirement for terminal model text.
    pub fn output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Selects how a configured output schema is presented to the model.
    pub fn structured_output_mode(mut self, mode: StructuredOutputMode) -> Self {
        self.structured_output_mode = mode;
        self
    }

    /// Accepts the named tool's arguments as terminal structured output.
    pub fn terminal_tool(mut self, name: impl Into<String>) -> Self {
        self.terminal_tool = Some(name.into());
        self
    }

    fn append_structured_output_instructions(
        &mut self,
        schema: &serde_json::Value,
        terminal_tool: bool,
    ) {
        if !self.instructions.is_empty() {
            self.instructions.push_str("\n\n");
        }
        if terminal_tool {
            self.instructions.push_str(
                "When you have gathered enough information to answer, call the `final_result` tool exactly once with your final answer. Its arguments are the structured result and must satisfy the required schema. Do not return the final answer as plain text.",
            );
        } else {
            self.instructions.push_str(
                "Respond with ONLY a single JSON object that conforms to this JSON Schema. Do not include any prose, explanation, or markdown code fences.\n",
            );
            self.instructions.push_str(&schema.to_string());
        }
    }

    /// Adds a vector-search store entity used during run preparation.
    pub fn dynamic_context(
        mut self,
        limit: usize,
        index: impl VectorStoreIndexDyn + 'static,
    ) -> Self {
        let order = self.stores.len();
        self.retrieval_limit = Some(limit);
        self = self.store(
            StableId::generated(format!("local-vector-store-{order}")),
            StoreCapability {
                kind: "vector-search".to_owned(),
                revision: 1,
                retired: false,
            },
            StoreGrant {
                order: u32::try_from(order).unwrap_or(u32::MAX),
                enabled: true,
            },
            VectorIndexStore {
                index: Arc::new(index),
            },
        );
        self
    }

    /// Adds a conversation-memory store entity and grant.
    pub fn memory<B>(self, backend: B) -> Self
    where
        B: crate::memory::ConversationMemory + 'static,
    {
        let order = self.stores.len();
        self.store(
            StableId::generated(format!("local-conversation-store-{order}")),
            StoreCapability {
                kind: "conversation-memory".to_owned(),
                revision: 1,
                retired: false,
            },
            StoreGrant {
                order: u32::try_from(order).unwrap_or(u32::MAX),
                enabled: true,
            },
            ConversationStore { backend },
        )
    }

    /// Adds one typed implementation and its authoritative ECS capability.
    pub fn tool<T>(mut self, id: StableId, capability: ToolCapability, tool: T) -> Self
    where
        T: EcsTool + 'static,
    {
        let revision = capability.revision;
        self.tools.push(LocalToolRegistration {
            id: id.clone(),
            capability,
            executor: Arc::new(TypedLocalTool {
                adapter: ToolAdapter::new(id, revision, tool),
            }),
        });
        self
    }

    /// Adds a typed Rig tool as an ECS capability entity.
    pub fn authored_tool<T>(self, tool: T) -> Self
    where
        T: crate::tool::Tool + 'static,
        T::Output: Send,
    {
        let order = self.tools.len();
        self.tool(
            StableId::generated(format!("local-tool-{order}")),
            ToolCapability {
                name: T::NAME.to_owned(),
                description: tool.description(),
                parameters: tool.parameters(),
                order: u32::try_from(order).unwrap_or(u32::MAX),
                revision: 1,
                retired: false,
            },
            AuthoredTool(tool),
        )
    }

    /// Adds a runtime-authored tool as an ordinary ECS capability entity.
    pub fn dynamic_tool(self, tool: crate::tool::DynamicTool) -> Self {
        let order = self.tools.len();
        let definition = tool.definition();
        self.tool(
            StableId::generated(format!("local-tool-{order}")),
            ToolCapability {
                name: definition.name,
                description: definition.description,
                parameters: definition.parameters,
                order: u32::try_from(order).unwrap_or(u32::MAX),
                revision: 1,
                retired: false,
            },
            DynamicEcsTool(tool),
        )
    }

    /// Adds one typed store capability and grants it to the agent.
    pub fn store<S>(
        mut self,
        id: StableId,
        capability: StoreCapability,
        grant: StoreGrant,
        store: S,
    ) -> Self
    where
        S: EcsStore + 'static,
    {
        let revision = capability.revision;
        self.stores.push(LocalStoreRegistration {
            id: id.clone(),
            capability,
            grant,
            executor: Arc::new(TypedLocalStore {
                adapter: StoreAdapter::new(id, revision, store),
            }),
        });
        self
    }

    /// Spawns all domain entities and relationships into one runtime world.
    pub fn build(mut self) -> Result<LocalModelAgent<M>, LocalAgentError> {
        if let Some(schema) = self.output_schema.clone() {
            let mode = match self.structured_output_mode {
                StructuredOutputMode::Auto if self.terminal_tool.is_some() => None,
                StructuredOutputMode::Auto if self.tools.is_empty() => {
                    Some(StructuredOutputMode::Native)
                }
                StructuredOutputMode::Auto => Some(StructuredOutputMode::Tool),
                mode => Some(mode),
            };
            match mode {
                Some(StructuredOutputMode::Tool) => {
                    self.terminal_tool = Some("final_result".to_owned());
                    self.append_structured_output_instructions(&schema, true);
                }
                Some(StructuredOutputMode::Prompted) => {
                    self.terminal_tool = Some(String::new());
                    self.append_structured_output_instructions(&schema, false);
                }
                Some(StructuredOutputMode::Native | StructuredOutputMode::Auto) | None => {}
            }
        }
        let mut runtime = Runtime::new(RuntimeConfig::default())?;
        let tenant = TenantId::new("local")?;
        let model_id = StableId::new("local-model")?;
        let model_capability = ModelCapability {
            provider: self.provider,
            model: self.model_name,
            revision: 1,
            retired: false,
        };
        let model_entity =
            runtime.spawn_model(model_id.clone(), tenant.clone(), model_capability.clone())?;
        let model_binding = ModelDecision {
            model_entity,
            model_id,
            revision: model_capability.revision,
            provider: model_capability.provider,
            model: model_capability.model,
            tenant: tenant.clone(),
        };
        let agent = runtime.spawn_agent(
            StableId::new("local-agent")?,
            tenant.clone(),
            Agent {
                name: self.name,
                description: self.description,
                instructions: self.instructions,
                temperature_bits: self.temperature_bits,
                max_tokens: self.max_tokens,
                tool_choice: self.tool_choice,
                additional_params: self.additional_params,
                documents: self.documents,
                max_model_calls: self.max_model_calls,
                terminal_tool: self.terminal_tool,
            },
            model_entity,
        )?;
        if let Some(schema) = self.output_schema {
            runtime.set_output_requirement(agent, OutputRequirement { schema })?;
        }
        if let Some(limit) = self.retrieval_limit {
            runtime.set_retrieval_requirement(agent, RetrievalRequirement { limit })?;
        }
        let mut names = HashSet::new();
        let mut tools = self
            .tools
            .into_iter()
            .rev()
            .filter(|registration| names.insert(registration.capability.name.clone()))
            .collect::<Vec<_>>();
        tools.reverse();
        let mut executors = HashMap::new();
        for (order, registration) in tools.into_iter().enumerate() {
            let revision = registration.capability.revision;
            let tool = runtime.spawn_tool(
                registration.id.clone(),
                tenant.clone(),
                registration.capability,
            )?;
            runtime.grant_tool(
                StableId::new(format!("local-tool-grant-{order}"))?,
                tenant.clone(),
                ToolGrant {
                    order: u32::try_from(order).unwrap_or(u32::MAX),
                    enabled: true,
                },
                agent,
                tool,
            )?;
            executors.insert((registration.id, revision), registration.executor);
        }
        let mut store_executors = HashMap::new();
        for (order, registration) in self.stores.into_iter().enumerate() {
            let revision = registration.capability.revision;
            let store = runtime.spawn_store(
                registration.id.clone(),
                tenant.clone(),
                registration.capability,
            )?;
            runtime.grant_store(
                StableId::new(format!("local-store-grant-{order}"))?,
                tenant.clone(),
                registration.grant,
                agent,
                store,
            )?;
            store_executors.insert((registration.id, revision), registration.executor);
        }
        Ok(LocalModelAgent {
            runtime: Arc::new(Mutex::new(runtime)),
            driver: Arc::new(tokio::sync::Mutex::new(())),
            agent,
            model: CompletionModelAdapter::bound(self.model, model_binding),
            tools: Arc::new(executors),
            stores: Arc::new(store_executors),
        })
    }
}

impl<M> LocalModelAgent<M>
where
    M: CompletionModel,
{
    async fn execute_tool_batch(
        &self,
        requests: Vec<(bevy_ecs::entity::Entity, u64, ToolEffectInput)>,
        concurrency: usize,
    ) -> Vec<EffectCompletion> {
        futures::stream::iter(requests.into_iter().map(|(operation, generation, input)| {
            let key = (input.decision.tool_id.clone(), input.decision.revision);
            let executor = self.tools.get(&key).cloned();
            async move {
                let result = match executor {
                    Some(executor) => {
                        catch_executor_panic(async move {
                            executor.execute(input).await.map(EffectOutput::Tool)
                        })
                        .await
                    }
                    None => Err(CanonicalError::UnknownTool(input.decision.name)),
                };
                EffectCompletion {
                    operation,
                    generation,
                    result,
                }
            }
        }))
        .buffered(concurrency)
        .collect()
        .await
    }

    /// Creates an awaitable ECS prompt command with optional per-run settings.
    pub fn prompt(&self, prompt: impl Into<Message>) -> AgentPromptRequest<M> {
        AgentPromptRequest {
            agent: self.clone(),
            prompt: prompt.into(),
            history: Vec::new(),
            max_model_calls: None,
            conversation: Ok(None),
        }
    }

    /// Returns the world-scoped agent entity handle.
    pub fn handle(&self) -> AgentHandle {
        self.agent
    }

    /// Creates a one-model standalone runtime and spawns its model and agent
    /// entities into the authoritative world.
    pub fn new(
        model: M,
        provider: impl Into<String>,
        model_name: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Result<Self, LocalAgentError> {
        LocalModelAgentBuilder::new(model, provider, model_name)
            .instructions(instructions)
            .build()
    }

    /// Executes a read-only closure at an ECS safe point.
    pub fn with_runtime<R>(
        &self,
        access: impl FnOnce(&Runtime) -> R,
    ) -> Result<R, LocalAgentError> {
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| LocalAgentError::RuntimePoisoned)?;
        Ok(access(&runtime))
    }

    /// Executes a mutable closure at an ECS safe point.
    pub fn with_runtime_mut<R>(
        &self,
        access: impl FnOnce(&mut Runtime) -> R,
    ) -> Result<R, LocalAgentError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| LocalAgentError::RuntimePoisoned)?;
        Ok(access(&mut runtime))
    }

    /// Submits and resolves one prompt through [`crate::runtime::RigSchedule`].
    pub async fn run_prompt(
        &self,
        prompt: impl Into<Message>,
    ) -> Result<RunOutput, LocalAgentError> {
        self.run_prompt_configured(prompt.into(), Vec::new(), None)
            .await
    }

    pub(crate) async fn run_prompt_configured(
        &self,
        prompt: Message,
        history: Vec<Message>,
        output_schema: Option<serde_json::Value>,
    ) -> Result<RunOutput, LocalAgentError> {
        self.run_prompt_options(prompt, history, output_schema, None, None)
            .await
    }

    pub(crate) async fn run_prompt_options(
        &self,
        prompt: Message,
        history: Vec<Message>,
        output_schema: Option<serde_json::Value>,
        conversation: Option<StableId>,
        max_model_calls: Option<u32>,
    ) -> Result<RunOutput, LocalAgentError> {
        self.run_prompt_options_detailed(
            prompt,
            history,
            output_schema,
            conversation,
            max_model_calls,
        )
        .await
        .map(|result| result.output)
    }

    async fn run_prompt_options_detailed(
        &self,
        prompt: Message,
        history: Vec<Message>,
        output_schema: Option<serde_json::Value>,
        conversation: Option<StableId>,
        max_model_calls: Option<u32>,
    ) -> Result<LocalRunResult, LocalAgentError> {
        // One facade owns one effect outbox. Serialize its convenience drivers
        // so a blocking caller cannot consume a streaming caller's effect (or
        // vice versa). Hosted users that need concurrent scheduling drive the
        // shared `Runtime`/effect boundary directly.
        let _driver = self.driver.lock().await;
        let pending = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| LocalAgentError::RuntimePoisoned)?;
            runtime.handle().prompt_configured(
                self.agent,
                prompt,
                history,
                output_schema,
                max_model_calls,
                conversation,
            )?
        };
        loop {
            let (requests, completion_sender) = {
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| LocalAgentError::RuntimePoisoned)?;
                runtime.run_until_stalled()?;
                let mut requests = Vec::new();
                while let Some(request) = runtime.effects().try_recv()? {
                    requests.push(request);
                }
                (requests, runtime.effects().completion_sender())
            };
            let mut tool_requests = Vec::new();
            for request in requests {
                let operation = request.operation;
                let generation = request.generation;
                let result = match request.input {
                    EffectInput::Model(input) => catch_executor_panic(self.model.execute(input))
                        .await
                        .map(EffectOutput::Model),
                    EffectInput::Tool(input) => {
                        tool_requests.push((operation, generation, input));
                        continue;
                    }
                    EffectInput::Store(input) => {
                        let key = (input.decision.store_id.clone(), input.decision.revision);
                        match self.stores.get(&key) {
                            Some(executor) => {
                                catch_executor_panic(async move {
                                    executor.execute(input).await.map(EffectOutput::Store)
                                })
                                .await
                            }
                            None => Err(CanonicalError::StaleEntity(format!(
                                "store revision {}@{}",
                                key.0.as_str(),
                                key.1
                            ))),
                        }
                    }
                    EffectInput::Discovery(_) => Err(CanonicalError::EffectKindMismatch),
                };
                self.submit_completion(
                    &completion_sender,
                    EffectCompletion {
                        operation,
                        generation,
                        result,
                    },
                )?;
            }
            for completion in self.execute_tool_batch(tool_requests, usize::MAX).await {
                self.submit_completion(&completion_sender, completion)?;
            }
            let (state, transcript, completion_calls) = {
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| LocalAgentError::RuntimePoisoned)?;
                runtime.run_until_stalled()?;
                let Some(run) = runtime.resolve_run(&pending) else {
                    continue;
                };
                let transcript = runtime.run_transcript(run).unwrap_or_default();
                let completion_calls = runtime
                    .run_committed_turns(run)
                    .into_iter()
                    .map(|turn| turn.output.usage)
                    .collect();
                (runtime.observe_run(run)?, transcript, completion_calls)
            };
            match state {
                Some(RunState::Completed(output)) => {
                    return Ok(LocalRunResult {
                        output,
                        transcript,
                        completion_calls,
                    });
                }
                Some(RunState::Failed(error)) => return Err(error.into()),
                Some(RunState::Cancelled) => return Err(LocalAgentError::Cancelled),
                Some(
                    RunState::Queued
                    | RunState::WaitingModel { .. }
                    | RunState::WaitingTools { .. }
                    | RunState::WaitingStore { .. },
                )
                | None => {}
            }
        }
    }

    fn begin_stream(
        &self,
        prompt: Message,
        history: Vec<Message>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
    ) -> Result<(crate::runtime::PendingRunHandle, crate::runtime::RunStream), LocalAgentError>
    {
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| LocalAgentError::RuntimePoisoned)?;
        Ok(runtime.handle().prompt_stream_configured(
            self.agent,
            prompt,
            history,
            None,
            max_model_calls,
            conversation,
        )?)
    }

    fn drive_and_take_effects(
        &self,
    ) -> Result<
        (
            Vec<EffectRequest>,
            crate::runtime::EffectCompletionSender,
            EffectDeltaSender,
        ),
        LocalAgentError,
    > {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| LocalAgentError::RuntimePoisoned)?;
        runtime.run_until_stalled()?;
        let mut requests = Vec::new();
        while let Some(request) = runtime.effects().try_recv()? {
            requests.push(request);
        }
        Ok((
            requests,
            runtime.effects().completion_sender(),
            runtime.effects().delta_sender(),
        ))
    }

    fn drive_once(&self) -> Result<(), LocalAgentError> {
        self.runtime
            .lock()
            .map_err(|_| LocalAgentError::RuntimePoisoned)?
            .run_until_stalled()?;
        Ok(())
    }

    fn submit_completion(
        &self,
        sender: &crate::runtime::EffectCompletionSender,
        completion: EffectCompletion,
    ) -> Result<(), LocalAgentError> {
        loop {
            match sender.try_send(completion.clone()) {
                Ok(()) => return Ok(()),
                Err(EffectIoError::Backpressure) => self.drive_once()?,
                Err(error) => return Err(error.into()),
            }
        }
    }

    fn submit_delta(
        &self,
        sender: &EffectDeltaSender,
        delta: EffectDelta,
    ) -> Result<(), LocalAgentError> {
        loop {
            match sender.try_send(delta.clone()) {
                Ok(()) => return Ok(()),
                Err(EffectIoError::Backpressure) => self.drive_once()?,
                Err(error) => return Err(error.into()),
            }
        }
    }

    fn observed_state(
        &self,
        pending: &crate::runtime::PendingRunHandle,
    ) -> Result<Option<RunState>, LocalAgentError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| LocalAgentError::RuntimePoisoned)?;
        Ok(runtime
            .resolve_run(pending)
            .map(|run| runtime.observe_run(run))
            .transpose()?
            .flatten())
    }

    fn details_for_pending(
        &self,
        pending: &crate::runtime::PendingRunHandle,
    ) -> Result<(Vec<TranscriptEntry>, Vec<Usage>), LocalAgentError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| LocalAgentError::RuntimePoisoned)?;
        let Some(run) = runtime.resolve_run(pending) else {
            return Ok((Vec::new(), Vec::new()));
        };
        let transcript = runtime.run_transcript(run).unwrap_or_default();
        let completion_calls = runtime
            .run_committed_turns(run)
            .into_iter()
            .map(|turn| turn.output.usage)
            .collect();
        Ok((transcript, completion_calls))
    }

    /// Streams one run while all authoritative progression remains in
    /// [`crate::runtime::RigSchedule`].
    pub fn stream_run(
        &self,
        prompt: impl Into<Message>,
    ) -> Pin<Box<dyn Stream<Item = Result<LocalStreamEvent, LocalAgentError>> + Send>>
    where
        M: 'static,
    {
        self.stream_run_with_history(prompt.into(), Vec::new(), None, None, usize::MAX)
    }

    /// Streams one run with pre-existing canonical chat history.
    pub fn stream_run_with_history(
        &self,
        prompt: Message,
        history: Vec<Message>,
        max_model_calls: Option<u32>,
        conversation: Option<StableId>,
        tool_concurrency: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<LocalStreamEvent, LocalAgentError>> + Send>>
    where
        M: 'static,
    {
        let agent = self.clone();
        Box::pin(async_stream::try_stream! {
            let _driver = agent.driver.lock().await;
            let (pending, stream) = agent.begin_stream(prompt, history, max_model_calls, conversation)?;
            let stream = stream;
            let mut internal_call_ids = HashMap::<String, String>::new();
            let mut streamed_tool_calls = HashMap::<String, ToolCall>::new();
            'drive: loop {
                let (requests, completion_sender, delta_sender) =
                    agent.drive_and_take_effects()?;

                let mut tool_requests = Vec::new();
                for request in requests {
                    let operation = request.operation;
                    let generation = request.generation;
                    let result = match request.input {
                        EffectInput::Model(input) => 'model: {
                            let completion_request = match completion_request(&input) {
                                Ok(request) => request,
                                Err(error) => break 'model Err(error),
                            };
                            let mut provider_stream = match catch_executor_panic(async {
                                agent.model.validate_decision(&input)?;
                                agent.model.model.stream(completion_request).await.map_err(|error| {
                                    CanonicalError::Provider {
                                        message: error.to_string(),
                                        retryable: false,
                                    }
                                })
                            }).await {
                                Ok(stream) => stream,
                                Err(error) => break 'model Err(error),
                            };
                            let mut observed_tool_calls = Vec::new();
                            let mut sequence = 0;
                            let mut stream_error = None;
                            loop {
                                let next = match catch_executor_panic(async {
                                    Ok(provider_stream.next().await)
                                }).await {
                                    Ok(next) => next,
                                    Err(error) => {
                                        stream_error = Some(error);
                                        break;
                                    }
                                };
                                let Some(item) = next else {
                                    break;
                                };
                                let item = match item {
                                    Ok(item) => item,
                                    Err(error) => {
                                        stream_error = Some(CanonicalError::Provider {
                                        message: error.to_string(),
                                        retryable: false,
                                        });
                                        break;
                                    }
                                };
                                match item {
                                    StreamedAssistantContent::Text(text) => {
                                        if let Err(error) = agent.submit_delta(&delta_sender, EffectDelta {
                                            operation,
                                            generation,
                                            sequence,
                                            text: text.text,
                                        }) {
                                            stream_error = Some(CanonicalError::Provider {
                                                message: format!("stream delta ingress failed: {error}"),
                                                retryable: true,
                                            });
                                            break;
                                        }
                                        sequence = sequence.saturating_add(1);
                                        agent.drive_once()?;
                                        while let Some(item) = stream.try_recv()? {
                                            match item {
                                                StreamItem::Delta { text, .. } => {
                                                    yield LocalStreamEvent::Delta(text);
                                                }
                                                StreamItem::Finished(StreamTerminal::Completed(output)) => {
                                                    let (transcript, completion_calls) =
                                                        agent.details_for_pending(&pending)?;
                                                    yield LocalStreamEvent::Finished {
                                                        output,
                                                        transcript,
                                                        completion_calls,
                                                    };
                                                    break 'drive;
                                                }
                                                StreamItem::Finished(StreamTerminal::Failed(error)) => {
                                                    Err(LocalAgentError::Canonical(error))?;
                                                }
                                                StreamItem::Finished(StreamTerminal::Cancelled) => {
                                                    Err(LocalAgentError::Cancelled)?;
                                                }
                                            }
                                        }
                                    }
                                    StreamedAssistantContent::ToolCall { tool_call, internal_call_id } => {
                                        observed_tool_calls.push((tool_call, internal_call_id));
                                    }
                                    StreamedAssistantContent::ToolCallDelta { id, internal_call_id, content } => {
                                        yield LocalStreamEvent::ToolCallDelta { id, internal_call_id, content };
                                    }
                                    StreamedAssistantContent::Reasoning(reasoning) => {
                                        yield LocalStreamEvent::Reasoning(reasoning);
                                    }
                                    StreamedAssistantContent::ReasoningDelta { id, reasoning } => {
                                        yield LocalStreamEvent::ReasoningDelta { id, reasoning };
                                    }
                                    StreamedAssistantContent::Unknown(value) => {
                                        yield LocalStreamEvent::Unknown(value);
                                    }
                                    StreamedAssistantContent::Final(_) => {}
                                }
                            }
                            if let Some(error) = stream_error {
                                break 'model Err(error);
                            }
                            let output = match normalize_model_output(
                                provider_stream.choice.iter(),
                                provider_stream.usage(),
                                provider_stream.message_id.as_ref(),
                            ) {
                                Ok(output) => output,
                                Err(error) => break 'model Err(error),
                            };
                            let mut used_internal_ids = HashSet::new();
                            for (index, normalized) in output.tool_calls.iter().enumerate() {
                                let (tool_call, mut internal_call_id) = observed_tool_calls
                                    .get(index)
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        let mut tool_call = ToolCall::new(
                                            normalized.provider_result_id.clone(),
                                            ToolFunction::new(
                                                normalized.name.clone(),
                                                normalized.arguments.clone(),
                                            ),
                                        );
                                        tool_call.call_id = normalized.provider_call_id.clone();
                                        (
                                            tool_call,
                                            normalized.id.clone(),
                                        )
                                    });
                                if used_internal_ids.contains(&internal_call_id) {
                                    let base = internal_call_id.clone();
                                    let mut suffix = index;
                                    while used_internal_ids.contains(&internal_call_id) {
                                        internal_call_id = format!("{base}-{suffix}");
                                        suffix = suffix.saturating_add(1);
                                    }
                                }
                                used_internal_ids.insert(internal_call_id.clone());
                                internal_call_ids.insert(
                                    normalized.id.clone(),
                                    internal_call_id.clone(),
                                );
                                streamed_tool_calls
                                    .insert(normalized.id.clone(), tool_call.clone());
                                yield LocalStreamEvent::ToolCall {
                                    tool_call,
                                    internal_call_id,
                                };
                            }
                            Ok(EffectOutput::Model(output))
                        }
                        EffectInput::Tool(input) => {
                            tool_requests.push((operation, generation, input));
                            continue;
                        }
                        EffectInput::Store(input) => {
                            let key = (input.decision.store_id.clone(), input.decision.revision);
                            match agent.stores.get(&key) {
                                Some(executor) => catch_executor_panic(async move {
                                    executor.execute(input).await.map(EffectOutput::Store)
                                }).await,
                                None => Err(CanonicalError::StaleEntity(format!(
                                    "store revision {}@{}",
                                    key.0.as_str(), key.1
                                ))),
                            }
                        }
                        EffectInput::Discovery(_) => {
                            Err(CanonicalError::EffectKindMismatch)
                        }
                    };
                    agent.submit_completion(&completion_sender, EffectCompletion {
                        operation,
                        generation,
                        result,
                    })?;
                }
                let mut committed_tool_results = Vec::new();
                for completion in agent
                    .execute_tool_batch(tool_requests, tool_concurrency)
                    .await
                {
                    if let Ok(EffectOutput::Tool(output)) = &completion.result {
                        let internal_call_id = internal_call_ids
                            .get(&output.call_id)
                            .cloned()
                            .unwrap_or_else(|| output.call_id.clone());
                        if let Some(tool_call) = streamed_tool_calls.get(&output.call_id).cloned() {
                            committed_tool_results.push(LocalStreamEvent::ToolCommitted {
                                tool_call,
                                internal_call_id: internal_call_id.clone(),
                            });
                        }
                            committed_tool_results.push(LocalStreamEvent::ToolResult {
                                tool_result: MessageToolResult {
                                id: output.provider_result_id.clone(),
                                call_id: output.provider_call_id.clone(),
                                content: tool_result_content(
                                    &output.raw,
                                    &output.presentation,
                                ),
                            },
                            internal_call_id,
                        });
                    }
                    agent.submit_completion(&completion_sender, completion)?;
                }

                agent.drive_once()?;
                for result in committed_tool_results {
                    yield result;
                }
                while let Some(item) = stream.try_recv()? {
                    match item {
                        StreamItem::Delta { text, .. } => yield LocalStreamEvent::Delta(text),
                        StreamItem::Finished(StreamTerminal::Completed(output)) => {
                            let (transcript, completion_calls) =
                                agent.details_for_pending(&pending)?;
                            yield LocalStreamEvent::Finished {
                                output,
                                transcript,
                                completion_calls,
                            };
                            break 'drive;
                        }
                        StreamItem::Finished(StreamTerminal::Failed(error)) => {
                            Err(LocalAgentError::Canonical(error))?;
                        }
                        StreamItem::Finished(StreamTerminal::Cancelled) => {
                            Err(LocalAgentError::Cancelled)?;
                        }
                    }
                }

                let terminal = agent.observed_state(&pending)?;
                match terminal {
                    Some(RunState::Failed(error)) => Err(LocalAgentError::Canonical(error))?,
                    Some(RunState::Cancelled) => Err(LocalAgentError::Cancelled)?,
                    Some(RunState::Completed(output)) => {
                        let (transcript, completion_calls) =
                            agent.details_for_pending(&pending)?;
                        yield LocalStreamEvent::Finished {
                            output,
                            transcript,
                            completion_calls,
                        };
                        break;
                    }
                    _ => {}
                }
            }
        })
    }
}

impl<M> Prompt for LocalModelAgent<M>
where
    M: CompletionModel + 'static,
{
    fn prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> impl std::future::IntoFuture<Output = Result<String, PromptError>, IntoFuture: Send> {
        AgentPromptRequest {
            agent: self.clone(),
            prompt: prompt.into(),
            history: Vec::new(),
            max_model_calls: None,
            conversation: Ok(None),
        }
    }
}

impl<M> Chat for LocalModelAgent<M>
where
    M: CompletionModel + 'static,
{
    fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: &mut Vec<Message>,
    ) -> impl Future<Output = Result<String, PromptError>> + Send {
        let prompt = prompt.into();
        async move {
            let details = self
                .run_prompt_options_detailed(prompt, chat_history.clone(), None, None, None)
                .await
                .map_err(local_prompt_error)?;
            *chat_history = response_messages(&details.transcript)
                .map_err(|error| local_prompt_error(LocalAgentError::Canonical(error)))?;
            Ok(details.output.text)
        }
    }
}

impl<M> TypedPrompt for LocalModelAgent<M>
where
    M: CompletionModel + 'static,
{
    type TypedRequest<T>
        = BoxFuture<'static, Result<T, StructuredOutputError>>
    where
        T: schemars::JsonSchema + DeserializeOwned + Send + 'static;

    fn prompt_typed<T>(&self, prompt: impl Into<Message> + Send) -> Self::TypedRequest<T>
    where
        T: schemars::JsonSchema + DeserializeOwned + Send + 'static,
    {
        let agent = self.clone();
        let prompt = prompt.into();
        Box::pin(async move {
            let schema = serde_json::to_value(schemars::schema_for!(T))?;
            let output = agent
                .run_prompt_configured(prompt, Vec::new(), Some(schema))
                .await
                .map_err(local_prompt_error)
                .map_err(Box::new)?;
            if output.text.is_empty() {
                return Err(StructuredOutputError::EmptyResponse);
            }
            Ok(serde_json::from_str(&output.text)?)
        })
    }
}

fn local_prompt_error(error: LocalAgentError) -> PromptError {
    let message = error.to_string();
    match error {
        LocalAgentError::Canonical(CanonicalError::Provider { .. }) => {
            CompletionError::ProviderError(message).into()
        }
        LocalAgentError::Canonical(CanonicalError::UnknownTool(tool_name)) => {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools: Vec::new(),
                allowed_tools: Vec::new(),
                chat_history: Box::default(),
            }
        }
        LocalAgentError::Canonical(CanonicalError::ModelCallBudget { limit }) => {
            PromptError::MaxTurnsError {
                max_turns: limit as usize,
                chat_history: Box::default(),
                prompt: Box::new(Message::user(String::new())),
            }
        }
        LocalAgentError::Cancelled => PromptError::PromptCancelled {
            chat_history: Vec::new(),
            reason: message,
        },
        _ => CompletionError::ProviderError(message).into(),
    }
}

/// Failure from the standalone schedule-driven facade.
#[derive(Debug, Error)]
pub enum LocalAgentError {
    /// Runtime installation failed.
    #[error(transparent)]
    Install(#[from] InstallError),
    /// Domain construction failed.
    #[error(transparent)]
    Spawn(#[from] SpawnError),
    /// A facade command could not enter the bounded runtime queue.
    #[error(transparent)]
    Submit(#[from] SubmitError),
    /// Immediate schedule progression exceeded its guard.
    #[error(transparent)]
    Drive(#[from] DriveError),
    /// The external effect ingress or outbox failed.
    #[error(transparent)]
    Effect(#[from] EffectIoError),
    /// Incremental subscriber ingress was disconnected.
    #[error(transparent)]
    Stream(#[from] StreamReceiveError),
    /// Stable local identities could not be created.
    #[error(transparent)]
    Identity(#[from] crate::runtime::IdentityError),
    /// The canonical model operation failed.
    #[error(transparent)]
    Canonical(#[from] CanonicalError),
    /// The run was cancelled before completion.
    #[error("run was cancelled")]
    Cancelled,
    /// Another thread panicked while it had exclusive access to the local world.
    #[error("local ECS runtime lock was poisoned")]
    RuntimePoisoned,
}

/// Context-free typed tool authoring boundary for ECS execution.
///
/// Arguments and output are owned values. This trait cannot access or borrow
/// the runtime world and does not expose a general-purpose context type map.
pub trait EcsTool: Send + Sync {
    /// Typed JSON arguments.
    type Args: DeserializeOwned + Send;
    /// Canonical model-visible output.
    type Output: IntoToolOutput + Send;
    /// Concrete author-facing error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Executes one owned invocation.
    fn call(
        &self,
        arguments: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;

    /// Normalizes concrete failures without exposing diagnostics to the model.
    fn map_error(&self, error: Self::Error) -> crate::tool::ToolExecutionError {
        crate::tool::ToolExecutionError::from_error(error)
    }
}

/// Executes one exact tool capability revision through an [`EcsTool`].
#[derive(Clone, Debug)]
pub struct ToolAdapter<T> {
    id: StableId,
    revision: u64,
    tool: T,
}

impl<T> ToolAdapter<T>
where
    T: EcsTool,
{
    /// Binds a typed implementation to a stable capability revision.
    pub fn new(id: StableId, revision: u64, tool: T) -> Self {
        Self { id, revision, tool }
    }

    /// Executes only an input snapshotted for this exact implementation.
    pub async fn execute(
        &self,
        input: ToolEffectInput,
    ) -> Result<ToolEffectOutput, CanonicalError> {
        if input.decision.tool_id != self.id || input.decision.revision != self.revision {
            return Err(CanonicalError::StaleEntity(format!(
                "tool revision {}@{}",
                input.decision.tool_id.as_str(),
                input.decision.revision
            )));
        }
        let call_id = input.call_id;
        let provider_result_id = input.provider_result_id;
        let provider_call_id = input.provider_call_id;
        let name = input.decision.name;
        let encoded_arguments =
            serde_json::to_string(&input.arguments).map_err(|error| CanonicalError::Tool {
                message: format!("failed to serialize tool arguments: {error}"),
                retryable: false,
            })?;
        let arguments = match serde_json::from_str(&encoded_arguments) {
            Ok(arguments) => arguments,
            Err(error) => {
                return tool_failure_output(
                    call_id,
                    provider_result_id,
                    provider_call_id,
                    name,
                    crate::tool::ToolExecutionError::invalid_args(format!(
                        "failed to parse tool arguments: {error}"
                    )),
                );
            }
        };
        let output = match self.tool.call(arguments).await {
            Ok(output) => output,
            Err(error) => {
                return tool_failure_output(
                    call_id,
                    provider_result_id,
                    provider_call_id,
                    name,
                    self.tool.map_error(error),
                );
            }
        };
        let output = match output.into_tool_output() {
            Ok(output) => output,
            Err(error) => {
                return tool_failure_output(
                    call_id,
                    provider_result_id,
                    provider_call_id,
                    name,
                    error,
                );
            }
        };
        let presentation = output.render();
        let raw =
            serde_json::to_value(output.as_content()).map_err(|error| CanonicalError::Tool {
                message: format!("failed to serialize output: {error}"),
                retryable: false,
            })?;
        Ok(ToolEffectOutput {
            call_id,
            provider_result_id,
            provider_call_id,
            name,
            raw,
            presentation,
            failure: None,
        })
    }
}

fn tool_failure_output(
    call_id: String,
    provider_result_id: String,
    provider_call_id: Option<String>,
    name: String,
    error: crate::tool::ToolExecutionError,
) -> Result<ToolEffectOutput, CanonicalError> {
    let presentation = error.model_output().render();
    let raw = serde_json::to_value(error.model_output().as_content()).map_err(|source| {
        CanonicalError::Tool {
            message: format!("failed to serialize model-visible tool failure: {source}"),
            retryable: false,
        }
    })?;
    Ok(ToolEffectOutput {
        call_id,
        provider_result_id,
        provider_call_id,
        name,
        raw,
        presentation,
        failure: Some(crate::runtime::ToolEffectFailure {
            message: error.message().to_owned(),
            retryable: error.retryable(),
            kind: error.kind(),
            refusal: error.is_refusal(),
        }),
    })
}

/// Context-free typed store authoring boundary for ECS execution.
pub trait EcsStore: Send + Sync {
    /// Concrete author-facing error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Executes an owned provider-independent store operation.
    fn execute(
        &self,
        operation: StoreOperation,
    ) -> impl Future<Output = Result<StoreEffectOutput, Self::Error>> + Send;
}

/// Executes one exact addressable store capability revision.
#[derive(Clone, Debug)]
pub struct StoreAdapter<S> {
    id: StableId,
    revision: u64,
    store: S,
}

impl<S> StoreAdapter<S>
where
    S: EcsStore,
{
    /// Binds a typed implementation to a stable store revision.
    pub fn new(id: StableId, revision: u64, store: S) -> Self {
        Self {
            id,
            revision,
            store,
        }
    }

    /// Executes only an input snapshotted for this exact store revision.
    pub async fn execute(
        &self,
        input: StoreEffectInput,
    ) -> Result<StoreEffectOutput, CanonicalError> {
        if input.decision.store_id != self.id || input.decision.revision != self.revision {
            return Err(CanonicalError::StaleEntity(format!(
                "store revision {}@{}",
                input.decision.store_id.as_str(),
                input.decision.revision
            )));
        }
        self.store
            .execute(input.operation)
            .await
            .map_err(|error| CanonicalError::Store {
                message: error.to_string(),
                retryable: false,
            })
    }
}

/// Context-free typed discovery authoring boundary for ECS execution.
pub trait EcsDiscovery: Send + Sync {
    /// Concrete author-facing error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Refreshes one owned source-generation snapshot.
    fn refresh(
        &self,
        input: DiscoveryEffectInput,
    ) -> impl Future<Output = Result<DiscoveryEffectOutput, Self::Error>> + Send;
}

/// Binds a typed discovery implementation to one stable source identity.
#[derive(Clone, Debug)]
pub struct DiscoveryAdapter<D> {
    source_id: StableId,
    discovery: D,
}

impl<D> DiscoveryAdapter<D>
where
    D: EcsDiscovery,
{
    /// Creates a typed discovery adapter.
    pub fn new(source_id: StableId, discovery: D) -> Self {
        Self {
            source_id,
            discovery,
        }
    }

    /// Executes a refresh only for the bound stable source.
    pub async fn execute(
        &self,
        input: DiscoveryEffectInput,
    ) -> Result<DiscoveryEffectOutput, CanonicalError> {
        if input.source_id != self.source_id {
            return Err(CanonicalError::StaleEntity(format!(
                "discovery source {}",
                input.source_id.as_str()
            )));
        }
        self.discovery
            .refresh(input)
            .await
            .map_err(|error| CanonicalError::Discovery {
                message: error.to_string(),
                retryable: false,
            })
    }
}

impl<M> CompletionModelAdapter<M>
where
    M: CompletionModel,
{
    /// Wraps a configured typed model for use by an external effect executor.
    pub fn new(model: M) -> Self {
        Self {
            model,
            binding: None,
        }
    }

    fn bound(model: M, decision: ModelDecision) -> Self {
        Self {
            model,
            binding: Some(decision),
        }
    }

    fn validate_decision(&self, input: &ModelEffectInput) -> Result<(), CanonicalError> {
        if self
            .binding
            .as_ref()
            .is_some_and(|binding| binding != &input.decision)
        {
            return Err(CanonicalError::StaleEntity(format!(
                "model revision {}@{}",
                input.decision.model_id.as_str(),
                input.decision.revision
            )));
        }
        Ok(())
    }

    /// Executes one owned model effect and normalizes its response.
    pub async fn execute(
        &self,
        input: ModelEffectInput,
    ) -> Result<ModelEffectOutput, CanonicalError> {
        self.validate_decision(&input)?;
        let request = completion_request(&input)?;
        let response =
            self.model
                .completion(request)
                .await
                .map_err(|error| CanonicalError::Provider {
                    message: error.to_string(),
                    retryable: false,
                })?;

        normalize_model_output(
            response.choice.iter(),
            response.usage,
            response.message_id.as_ref(),
        )
    }

    /// Executes one streaming model effect through the same correlated ECS
    /// ingress used by every other external completion.
    ///
    /// Text chunks are submitted with a monotonic operation-local sequence.
    /// The returned final value is normalized from the provider stream's
    /// accumulated assistant message, so streaming and non-streaming execution
    /// commit the same canonical output shape.
    pub async fn execute_stream(
        &self,
        request: EffectRequest,
        deltas: &EffectDeltaSender,
    ) -> Result<ModelEffectOutput, CanonicalError> {
        let EffectInput::Model(input) = request.input else {
            return Err(CanonicalError::EffectKindMismatch);
        };
        self.validate_decision(&input)?;
        let completion_request = completion_request(&input)?;
        let mut stream = self
            .model
            .stream(completion_request)
            .await
            .map_err(|error| CanonicalError::Provider {
                message: error.to_string(),
                retryable: false,
            })?;
        let mut sequence = 0;
        while let Some(item) = stream.next().await {
            match item.map_err(|error| CanonicalError::Provider {
                message: error.to_string(),
                retryable: false,
            })? {
                StreamedAssistantContent::Text(text) => {
                    deltas
                        .try_send(EffectDelta {
                            operation: request.operation,
                            generation: request.generation,
                            sequence,
                            text: text.text,
                        })
                        .map_err(|error| CanonicalError::Provider {
                            message: format!("stream delta ingress failed: {error}"),
                            retryable: true,
                        })?;
                    sequence = sequence.saturating_add(1);
                }
                StreamedAssistantContent::ToolCall { .. }
                | StreamedAssistantContent::ToolCallDelta { .. }
                | StreamedAssistantContent::Reasoning(_)
                | StreamedAssistantContent::ReasoningDelta { .. }
                | StreamedAssistantContent::Final(_)
                | StreamedAssistantContent::Unknown(_) => {}
            }
        }
        normalize_model_output(
            stream.choice.iter(),
            stream.usage(),
            stream.message_id.as_ref(),
        )
    }
}

fn normalize_model_output<'a>(
    content: impl IntoIterator<Item = &'a AssistantContent>,
    usage: CompletionUsage,
    message_id: Option<&String>,
) -> Result<ModelEffectOutput, CanonicalError> {
    let content = content.into_iter().collect::<Vec<_>>();
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut used_call_ids = content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => call.call_id.clone(),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut assistant_content = Vec::new();
    for content in content {
        if let AssistantContent::Reasoning(reasoning) = content
            && let Some(existing) = assistant_content.iter_mut().rev().find(|item| {
                matches!(item, AssistantContent::Reasoning(candidate)
                    if candidate.id == reasoning.id
                        && candidate.display_text() == reasoning.display_text())
            })
        {
            *existing = content.clone();
        } else {
            assistant_content.push(content.clone());
        }
        match content {
            AssistantContent::Text(value) => text.push_str(&value.text),
            AssistantContent::ToolCall(call) => {
                let id = if let Some(call_id) = call.call_id.clone() {
                    call_id
                } else {
                    let base = call.id.clone();
                    let mut candidate = base.clone();
                    let mut suffix = tool_calls.len();
                    while used_call_ids.contains(&candidate) {
                        candidate = format!("{base}-{suffix}");
                        suffix = suffix.saturating_add(1);
                    }
                    candidate
                };
                used_call_ids.insert(id.clone());
                tool_calls.push(ModelToolCall {
                    id,
                    provider_result_id: if call.call_id.is_some() {
                        call.function.name.clone()
                    } else {
                        call.id.clone()
                    },
                    provider_call_id: call.call_id.clone(),
                    name: call.function.name.clone(),
                    arguments: if call.function.arguments.is_null() {
                        serde_json::json!({})
                    } else {
                        call.function.arguments.clone()
                    },
                });
            }
            AssistantContent::Reasoning(_) => {}
            AssistantContent::Image(_) => {
                return Err(CanonicalError::UnsupportedModelContent(
                    "assistant image".to_owned(),
                ));
            }
        }
    }
    let has_identified_reasoning = assistant_content.iter().any(|content| {
        matches!(content, AssistantContent::Reasoning(reasoning) if reasoning.id.is_some())
    });
    if has_identified_reasoning {
        assistant_content.retain(|content| {
            !matches!(content, AssistantContent::Reasoning(reasoning) if reasoning.id.is_none())
        });
    }
    let assistant_message = OneOrMany::many(assistant_content)
        .ok()
        .map(|content| Message::Assistant {
            id: message_id.cloned(),
            content,
        })
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| CanonicalError::Provider {
            message: format!("failed to preserve canonical assistant message: {error}"),
            retryable: false,
        })?;
    let input_tokens = usage.input_tokens;
    let output_tokens = if input_tokens == 0 && usage.output_tokens == 0 {
        usage.total_tokens
    } else {
        usage.output_tokens
    };
    Ok(ModelEffectOutput {
        assistant_message,
        text,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
        tool_calls,
    })
}

fn completion_request(input: &ModelEffectInput) -> Result<CompletionRequest, CanonicalError> {
    let mut messages = transcript_messages(&input.history)?;
    if messages.is_empty() {
        messages.push(
            serde_json::from_value(input.prompt.clone()).map_err(|error| {
                CanonicalError::Provider {
                    message: format!("invalid canonical prompt: {error}"),
                    retryable: false,
                }
            })?,
        );
    }
    if !input.instructions.is_empty() {
        messages.insert(0, Message::system(input.instructions.clone()));
    }
    let chat_history = OneOrMany::many(messages).map_err(|error| CanonicalError::Provider {
        message: error.to_string(),
        retryable: false,
    })?;
    let mut tools = input
        .tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        })
        .collect::<Vec<_>>();
    if let Some(terminal_tool) = input.terminal_tool.as_ref().filter(|name| !name.is_empty())
        && !tools.iter().any(|tool| &tool.name == terminal_tool)
    {
        tools.push(ToolDefinition {
            name: terminal_tool.clone(),
            description: "Call this tool exactly once with your final answer when you are done. Its arguments are the structured result and must satisfy the output schema.".to_owned(),
            parameters: input
                .output_schema
                .clone()
                .unwrap_or_else(|| serde_json::json!({ "type": "object" })),
        });
    }
    Ok(CompletionRequest {
        model: (!input.decision.model.is_empty()).then(|| input.decision.model.clone()),
        preamble: None,
        chat_history,
        documents: input
            .documents
            .iter()
            .map(|document| Document {
                id: document.id.clone(),
                text: document.text.clone(),
                additional_props: document.metadata.clone().into_iter().collect(),
            })
            .collect(),
        tools,
        temperature: input.temperature_bits.map(f64::from_bits),
        max_tokens: input.max_tokens,
        tool_choice: input.tool_choice.clone().map(|choice| match choice {
            ModelToolChoice::Auto => ToolChoice::Auto,
            ModelToolChoice::None => ToolChoice::None,
            ModelToolChoice::Required => ToolChoice::Required,
            ModelToolChoice::Specific(function_names) => ToolChoice::Specific { function_names },
        }),
        additional_params: input.additional_params.clone(),
        output_schema: input
            .terminal_tool
            .is_none()
            .then(|| input.output_schema.clone())
            .flatten()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| {
                CanonicalError::InvalidStructuredOutput(format!(
                    "invalid configured schema: {error}"
                ))
            })?,
    })
}

pub(crate) fn transcript_messages(
    history: &[TranscriptEntry],
) -> Result<Vec<Message>, CanonicalError> {
    let mut messages = Vec::new();
    let mut index = 0;
    while index < history.len() {
        let Some(entry) = history.get(index) else {
            break;
        };
        match entry {
            TranscriptEntry::User(text) => {
                messages.push(Message::user(text.clone()));
                index += 1;
            }
            TranscriptEntry::UserMessage(value) => {
                messages.push(serde_json::from_value(value.clone()).map_err(|error| {
                    CanonicalError::Provider {
                        message: format!("invalid canonical user message: {error}"),
                        retryable: false,
                    }
                })?);
                index += 1;
            }
            TranscriptEntry::Message(value) => {
                messages.push(serde_json::from_value(value.clone()).map_err(|error| {
                    CanonicalError::Provider {
                        message: format!("invalid canonical history message: {error}"),
                        retryable: false,
                    }
                })?);
                index += 1;
            }
            TranscriptEntry::AssistantMessage(value) => {
                messages.push(serde_json::from_value(value.clone()).map_err(|error| {
                    CanonicalError::Provider {
                        message: format!("invalid canonical assistant message: {error}"),
                        retryable: false,
                    }
                })?);
                index += 1;
            }
            TranscriptEntry::Assistant(text) => {
                let mut content = Vec::new();
                if !text.is_empty() {
                    content.push(AssistantContent::text(text.clone()));
                }
                index += 1;
                while let Some(TranscriptEntry::AssistantToolCall {
                    call_id,
                    name,
                    arguments,
                }) = history.get(index)
                {
                    content.push(AssistantContent::ToolCall(ToolCall::new(
                        call_id.clone(),
                        ToolFunction::new(name.clone(), arguments.clone()),
                    )));
                    index += 1;
                }
                if content.is_empty() {
                    content.push(AssistantContent::text(String::new()));
                }
                messages.push(Message::Assistant {
                    id: None,
                    content: OneOrMany::many(content).map_err(|error| {
                        CanonicalError::Provider {
                            message: error.to_string(),
                            retryable: false,
                        }
                    })?,
                });
            }
            TranscriptEntry::AssistantToolCall { .. } => {
                return Err(CanonicalError::Provider {
                    message: "tool call transcript entry lacked an assistant turn".to_owned(),
                    retryable: false,
                });
            }
            TranscriptEntry::ToolResult { .. } => {
                let mut content = Vec::new();
                while let Some(TranscriptEntry::ToolResult {
                    provider_result_id,
                    provider_call_id,
                    raw,
                    content: result,
                    ..
                }) = history.get(index)
                {
                    content.push(UserContent::ToolResult(MessageToolResult {
                        id: provider_result_id.clone(),
                        call_id: provider_call_id.clone(),
                        content: tool_result_content(raw, result),
                    }));
                    index += 1;
                }
                messages.push(Message::User {
                    content: OneOrMany::many(content).map_err(|error| {
                        CanonicalError::Provider {
                            message: error.to_string(),
                            retryable: false,
                        }
                    })?,
                });
            }
        }
    }
    Ok(messages)
}

fn response_messages(history: &[TranscriptEntry]) -> Result<Vec<Message>, CanonicalError> {
    let mut messages = transcript_messages(history)?;
    messages.retain(|message| {
        !matches!(message, Message::Assistant { content, .. }
            if content.iter().all(|item| matches!(item,
                AssistantContent::Text(text) if text.text.is_empty())))
    });
    Ok(messages)
}

fn tool_result_content(
    raw: &serde_json::Value,
    presentation: &str,
) -> OneOrMany<ToolResultContent> {
    serde_json::from_value(raw.clone())
        .unwrap_or_else(|_| OneOrMany::one(ToolResultContent::text(presentation.to_owned())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bevy_ecs::entity::Entity,
        runtime::{
            EffectIngress, ModelDecision, RuntimeWaker, StableId, StoreDecision, TenantId,
            ToolDecision,
        },
        test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
    };
    use serde::Deserialize;
    use std::convert::Infallible;
    use std::sync::{Arc, mpsc::sync_channel};

    #[derive(Clone, Debug)]
    struct AddTool;

    #[derive(Clone, Debug)]
    struct BarrierTool(Arc<tokio::sync::Barrier>);

    #[derive(Clone, Debug)]
    struct FailingTool;

    #[derive(Clone, Debug)]
    struct PanickingTool;

    #[derive(Debug, thiserror::Error)]
    #[error("operator secret")]
    struct SecretToolError;

    #[derive(Deserialize)]
    struct AddArgs {
        left: i64,
        right: i64,
    }

    impl EcsTool for AddTool {
        type Args = AddArgs;
        type Output = i64;
        type Error = Infallible;

        fn call(
            &self,
            arguments: Self::Args,
        ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
            std::future::ready(Ok(arguments.left + arguments.right))
        }
    }

    impl EcsTool for BarrierTool {
        type Args = AddArgs;
        type Output = i64;
        type Error = Infallible;

        async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
            self.0.wait().await;
            Ok(arguments.left + arguments.right)
        }
    }

    impl EcsTool for FailingTool {
        type Args = AddArgs;
        type Output = i64;
        type Error = SecretToolError;

        fn call(
            &self,
            _arguments: Self::Args,
        ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
            std::future::ready(Err(SecretToolError))
        }
    }

    impl EcsTool for PanickingTool {
        type Args = serde_json::Value;
        type Output = serde_json::Value;
        type Error = Infallible;

        async fn call(&self, _arguments: Self::Args) -> Result<Self::Output, Self::Error> {
            panic!("tool task exploded")
        }
    }

    #[derive(Clone, Debug)]
    struct MemoryStore;

    impl EcsStore for MemoryStore {
        type Error = Infallible;

        fn execute(
            &self,
            operation: StoreOperation,
        ) -> impl Future<Output = Result<StoreEffectOutput, Self::Error>> + Send {
            std::future::ready(Ok(match operation {
                StoreOperation::LoadConversation { .. } => {
                    StoreEffectOutput::LoadedConversation(Vec::new())
                }
                StoreOperation::PersistConversation { .. } => StoreEffectOutput::Persisted,
                StoreOperation::Retrieve { .. } => StoreEffectOutput::Retrieved(Vec::new()),
            }))
        }
    }

    fn input() -> ModelEffectInput {
        ModelEffectInput {
            decision: ModelDecision {
                model_entity: Entity::PLACEHOLDER,
                model_id: StableId::new("model").unwrap(),
                revision: 3,
                provider: "mock".to_owned(),
                model: "mock-3".to_owned(),
                tenant: TenantId::new("tenant").unwrap(),
            },
            instructions: "system".to_owned(),
            prompt: serde_json::to_value(Message::user("question")).unwrap(),
            history: vec![
                TranscriptEntry::User("question".to_owned()),
                TranscriptEntry::Assistant(String::new()),
                TranscriptEntry::AssistantToolCall {
                    call_id: "call-1".to_owned(),
                    name: "lookup".to_owned(),
                    arguments: serde_json::json!({"id": 7}),
                },
                TranscriptEntry::ToolResult {
                    call_id: "call-1".to_owned(),
                    provider_result_id: "call-1".to_owned(),
                    provider_call_id: None,
                    name: "lookup".to_owned(),
                    raw: serde_json::Value::Null,
                    content: "result".to_owned(),
                },
            ],
            tools: vec![ToolDecision {
                tool_entity: Entity::PLACEHOLDER,
                tool_id: StableId::new("tool").unwrap(),
                revision: 4,
                name: "lookup".to_owned(),
                description: "look up an id".to_owned(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"]
                }),
                order: 0,
            }],
            tool_results: Vec::new(),
            output_schema: None,
            terminal_tool: None,
            documents: Vec::new(),
            temperature_bits: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
        }
    }

    #[test]
    fn request_preserves_schema_and_correlated_tool_history() {
        let request = completion_request(&input()).unwrap();
        assert_eq!(request.model.as_deref(), Some("mock-3"));
        assert_eq!(request.tools[0].parameters["required"][0], "id");
        let history = request.chat_history.iter().collect::<Vec<_>>();
        assert!(matches!(history[0], Message::System { content } if content == "system"));
        assert!(matches!(
            history[2],
            Message::Assistant { content, .. }
                if matches!(content.first_ref(), AssistantContent::ToolCall(call)
                    if call.id == "call-1" && call.function.name == "lookup")
        ));
    }

    #[tokio::test]
    async fn adapter_normalizes_provider_tool_calls() {
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "wire-id",
            "lookup",
            serde_json::json!({"id": 9}),
        )
        .with_call_id("call-id")]);
        let adapter = CompletionModelAdapter::new(model);
        let output = adapter.execute(input()).await.unwrap();
        assert_eq!(
            output.tool_calls,
            vec![ModelToolCall {
                id: "call-id".to_owned(),
                provider_result_id: "lookup".to_owned(),
                provider_call_id: Some("call-id".to_owned()),
                name: "lookup".to_owned(),
                arguments: serde_json::json!({"id": 9}),
            }]
        );
    }

    #[tokio::test]
    async fn adapter_uniquifies_only_missing_provider_tool_call_ids() {
        let model = MockCompletionModel::new([MockTurn::from_contents([
            AssistantContent::ToolCall(ToolCall::new(
                "lookup".to_owned(),
                ToolFunction::new("lookup".to_owned(), serde_json::json!({"id": 1})),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                "lookup".to_owned(),
                ToolFunction::new("lookup".to_owned(), serde_json::json!({"id": 2})),
            )),
        ])
        .unwrap()]);
        let output = CompletionModelAdapter::new(model)
            .execute(input())
            .await
            .unwrap();

        assert_eq!(output.tool_calls[0].id, "lookup");
        assert_eq!(output.tool_calls[1].id, "lookup-1");
        let message: Message = serde_json::from_value(output.assistant_message.unwrap()).unwrap();
        let Message::Assistant { content, .. } = message else {
            panic!("expected assistant message");
        };
        let provider_ids = content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call.id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(provider_ids, ["lookup", "lookup"]);
    }

    #[tokio::test]
    async fn adapter_reserves_later_explicit_ids_before_synthesizing() {
        let mut explicit = ToolCall::new(
            "wire".to_owned(),
            ToolFunction::new("lookup".to_owned(), serde_json::json!({"id": 2})),
        );
        explicit.call_id = Some("lookup".to_owned());
        let model = MockCompletionModel::new([MockTurn::from_contents([
            AssistantContent::ToolCall(ToolCall::new(
                "lookup".to_owned(),
                ToolFunction::new("lookup".to_owned(), serde_json::json!({"id": 1})),
            )),
            AssistantContent::ToolCall(explicit),
        ])
        .unwrap()]);

        let output = CompletionModelAdapter::new(model)
            .execute(input())
            .await
            .unwrap();

        assert_eq!(output.tool_calls[0].id, "lookup-0");
        assert_eq!(output.tool_calls[1].id, "lookup");
    }

    #[tokio::test]
    async fn streaming_adapter_emits_correlated_deltas_and_final_output() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::tool_call("wire-id", "lookup", serde_json::json!({"id": 4}))
                .with_call_id("call-id"),
        ]]);
        let adapter = CompletionModelAdapter::new(model);
        let (ingress, receiver) = sync_channel(4);
        let deltas = EffectDeltaSender {
            ingress,
            waker: RuntimeWaker(Arc::new(|| {})),
        };
        let request = EffectRequest {
            operation: Entity::PLACEHOLDER,
            generation: 7,
            input: EffectInput::Model(input()),
        };
        let output = adapter.execute_stream(request, &deltas).await.unwrap();
        assert_eq!(output.text, "hello");
        assert_eq!(output.tool_calls[0].id, "call-id");
        for (sequence, text) in [(0, "hel"), (1, "lo")] {
            let EffectIngress::Delta(delta) = receiver.recv().unwrap() else {
                panic!("expected a delta");
            };
            assert_eq!(delta.operation, Entity::PLACEHOLDER);
            assert_eq!(delta.generation, 7);
            assert_eq!(delta.sequence, sequence);
            assert_eq!(delta.text, text);
        }
    }

    #[tokio::test]
    async fn typed_tool_adapter_enforces_snapshotted_revision() {
        let adapter = ToolAdapter::new(StableId::new("add").unwrap(), 2, AddTool);
        let mut decision = input().tools.remove(0);
        decision.tool_id = StableId::new("add").unwrap();
        decision.revision = 2;
        decision.name = "add".to_owned();
        let output = adapter
            .execute(ToolEffectInput {
                decision: decision.clone(),
                call_id: "call".to_owned(),
                provider_result_id: "call".to_owned(),
                provider_call_id: None,
                arguments: serde_json::json!({"left": 2, "right": 3}),
                index: 0,
            })
            .await
            .unwrap();
        assert_eq!(
            output.raw,
            serde_json::json!([{"type": "json", "value": 5}])
        );
        assert_eq!(output.presentation, "5");

        decision.revision = 3;
        assert!(matches!(
            adapter
                .execute(ToolEffectInput {
                    decision,
                    call_id: "stale".to_owned(),
                    provider_result_id: "stale".to_owned(),
                    provider_call_id: None,
                    arguments: serde_json::json!({"left": 1, "right": 1}),
                    index: 0,
                })
                .await,
            Err(CanonicalError::StaleEntity(_))
        ));
    }

    #[tokio::test]
    async fn tool_failures_separate_diagnostics_from_model_presentation() {
        let adapter = ToolAdapter::new(StableId::new("add").unwrap(), 2, FailingTool);
        let mut decision = input().tools.remove(0);
        decision.tool_id = StableId::new("add").unwrap();
        decision.revision = 2;
        decision.name = "add".to_owned();

        let output = adapter
            .execute(ToolEffectInput {
                decision,
                call_id: "call".to_owned(),
                provider_result_id: "call".to_owned(),
                provider_call_id: None,
                arguments: serde_json::json!({"left": 2, "right": 3}),
                index: 0,
            })
            .await
            .unwrap();

        assert_eq!(output.presentation, "the tool failed");
        assert!(!output.raw.to_string().contains("operator secret"));
        let failure = output.failure.unwrap();
        assert_eq!(failure.message, "operator secret");
        assert_eq!(failure.kind, crate::tool::ToolErrorKind::Other);
        assert_eq!(failure.retryable, None);
    }

    #[tokio::test]
    async fn typed_store_adapter_enforces_snapshotted_revision() {
        let adapter = StoreAdapter::new(StableId::new("memory").unwrap(), 4, MemoryStore);
        let output = adapter
            .execute(StoreEffectInput {
                decision: StoreDecision {
                    store_entity: Entity::PLACEHOLDER,
                    store_id: StableId::new("memory").unwrap(),
                    revision: 4,
                    kind: "conversation-memory".to_owned(),
                    tenant: TenantId::new("tenant").unwrap(),
                },
                operation: StoreOperation::LoadConversation {
                    conversation: StableId::new("conversation").unwrap(),
                },
            })
            .await
            .unwrap();
        assert_eq!(output, StoreEffectOutput::LoadedConversation(Vec::new()));
    }

    #[tokio::test]
    async fn local_facade_drives_the_ecs_schedule() {
        let agent = LocalModelAgent::new(
            MockCompletionModel::text("schedule result"),
            "mock",
            "mock-1",
            "be concise",
        )
        .unwrap();
        let output = agent.run_prompt("hello").await.unwrap();
        assert_eq!(output.text, "schedule result");
        assert_eq!(
            agent
                .with_runtime(|runtime| runtime.metrics().unobserved_terminal_runs)
                .unwrap(),
            1
        );
        agent
            .with_runtime_mut(Runtime::run_until_stalled)
            .unwrap()
            .unwrap();
        assert_eq!(
            agent
                .with_runtime(|runtime| runtime.metrics().unobserved_terminal_runs)
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn streaming_provider_error_settles_the_authoritative_run() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("partial"),
            MockStreamEvent::error("stream failed"),
        ]]);
        let agent = LocalModelAgent::new(model, "mock", "mock-1", "be concise").unwrap();

        let events = agent.stream_run("hello").collect::<Vec<_>>().await;

        assert!(
            events.iter().any(|event| matches!(
                event,
                Err(LocalAgentError::Canonical(CanonicalError::Provider { message, .. }))
                    if message.contains("stream failed")
            )),
            "unexpected stream events: {events:?}"
        );
        let states = agent
            .with_runtime(|runtime| {
                runtime
                    .world()
                    .iter_entities()
                    .filter_map(|entity| entity.get::<RunState>().cloned())
                    .collect::<Vec<_>>()
            })
            .unwrap();
        assert_eq!(
            states,
            vec![RunState::Failed(CanonicalError::Provider {
                message: "ProviderError: stream failed".to_owned(),
                retryable: false,
            })]
        );
    }

    #[tokio::test]
    async fn local_facade_executes_tools_through_ecs_operations() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call(
                "wire-call",
                "add",
                serde_json::json!({"left": 2, "right": 3}),
            ),
            MockTurn::text("five"),
        ]);
        let agent = LocalModelAgentBuilder::new(model, "mock", "mock-1")
            .tool(
                StableId::new("add").unwrap(),
                crate::runtime::ToolCapability {
                    name: "add".to_owned(),
                    description: "adds integers".to_owned(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "left": {"type": "integer"},
                            "right": {"type": "integer"}
                        },
                        "required": ["left", "right"]
                    }),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
                AddTool,
            )
            .build()
            .unwrap();
        let output = agent.run_prompt("add").await.unwrap();
        assert_eq!(output.text, "five");
    }

    #[tokio::test]
    async fn streaming_repeated_missing_call_ids_keep_distinct_correlation() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "add",
                    "add",
                    serde_json::json!({"left": 1, "right": 2}),
                ),
                MockStreamEvent::tool_call(
                    "add",
                    "add",
                    serde_json::json!({"left": 3, "right": 4}),
                ),
            ],
            vec![MockStreamEvent::text("done")],
        ]);
        let agent = LocalModelAgentBuilder::new(model, "mock", "mock-1")
            .tool(
                StableId::new("add").unwrap(),
                ToolCapability {
                    name: "add".to_owned(),
                    description: "adds integers".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
                AddTool,
            )
            .build()
            .unwrap();

        let events = agent
            .stream_run_with_history(
                Message::user("add twice"),
                Vec::new(),
                None,
                None,
                usize::MAX,
            )
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let calls = events
            .iter()
            .filter_map(|event| match event {
                LocalStreamEvent::ToolCall {
                    tool_call,
                    internal_call_id,
                } => Some((
                    tool_call
                        .call_id
                        .as_deref()
                        .unwrap_or(tool_call.id.as_str()),
                    internal_call_id.as_str(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        let committed = events
            .iter()
            .filter_map(|event| match event {
                LocalStreamEvent::ToolCommitted {
                    tool_call,
                    internal_call_id,
                } => Some((
                    tool_call
                        .call_id
                        .as_deref()
                        .unwrap_or(tool_call.id.as_str()),
                    internal_call_id.as_str(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        let result_ids = events
            .iter()
            .filter_map(|event| match event {
                LocalStreamEvent::ToolResult { tool_result, .. } => Some(tool_result.id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            calls
                .iter()
                .map(|(call_id, _)| *call_id)
                .collect::<Vec<_>>(),
            ["add", "add"]
        );
        assert_ne!(calls[0].1, calls[1].1);
        assert_eq!(committed, calls);
        assert_eq!(result_ids, ["add", "add"]);
    }

    #[tokio::test]
    async fn extended_details_report_each_committed_model_call() {
        let first_usage = CompletionUsage {
            input_tokens: 3,
            output_tokens: 1,
            total_tokens: 4,
            ..CompletionUsage::default()
        };
        let second_usage = CompletionUsage {
            input_tokens: 5,
            output_tokens: 2,
            total_tokens: 7,
            ..CompletionUsage::default()
        };
        let model = MockCompletionModel::new([
            MockTurn::tool_call(
                "wire-call",
                "add",
                serde_json::json!({"left": 2, "right": 3}),
            )
            .with_usage(first_usage),
            MockTurn::text("five").with_usage(second_usage),
        ]);
        let agent = LocalModelAgentBuilder::new(model, "mock", "mock-1")
            .tool(
                StableId::new("add").unwrap(),
                ToolCapability {
                    name: "add".to_owned(),
                    description: "adds integers".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
                AddTool,
            )
            .build()
            .unwrap();

        let response = agent.prompt("add").extended_details().await.unwrap();

        assert_eq!(response.requests(), 2);
        assert_eq!(response.completion_calls()[0].usage, first_usage);
        assert_eq!(response.completion_calls()[1].usage, second_usage);
    }

    #[tokio::test]
    async fn local_model_adapter_rejects_mutated_capability_identity() {
        let agent = LocalModelAgent::new(
            MockCompletionModel::text("must not execute"),
            "mock",
            "mock-1",
            "",
        )
        .unwrap();
        let model = agent
            .with_runtime(|runtime| {
                runtime
                    .world()
                    .get::<crate::runtime::UsesModel>(agent.handle().entity())
                    .unwrap()
                    .0
            })
            .unwrap();
        agent
            .with_runtime_mut(|runtime| {
                runtime
                    .world_mut()
                    .get_mut::<ModelCapability>(model)
                    .unwrap()
                    .revision = 2;
            })
            .unwrap();

        assert!(matches!(
            agent.run_prompt("hello").await,
            Err(LocalAgentError::Canonical(CanonicalError::StaleEntity(message)))
                if message.contains("local-model@2")
        ));
    }

    #[tokio::test]
    async fn local_executor_panics_settle_as_correlated_failures() {
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "wire-call",
            "panic",
            serde_json::json!({}),
        )]);
        let agent = LocalModelAgentBuilder::new(model, "mock", "mock-1")
            .tool(
                StableId::new("panic").unwrap(),
                ToolCapability {
                    name: "panic".to_owned(),
                    description: "panics".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: 0,
                    revision: 1,
                    retired: false,
                },
                PanickingTool,
            )
            .build()
            .unwrap();

        assert!(matches!(
            agent.run_prompt("panic").await,
            Err(LocalAgentError::Canonical(CanonicalError::ExecutorPanicked(message)))
                if message == "tool task exploded"
        ));
    }

    #[tokio::test]
    async fn local_facade_executes_one_logical_tool_batch_concurrently() {
        let model = MockCompletionModel::new([
            MockTurn::from_contents([
                AssistantContent::ToolCall(ToolCall::new(
                    "first-call".to_owned(),
                    ToolFunction::new(
                        "first".to_owned(),
                        serde_json::json!({"left": 1, "right": 2}),
                    ),
                )),
                AssistantContent::ToolCall(ToolCall::new(
                    "second-call".to_owned(),
                    ToolFunction::new(
                        "second".to_owned(),
                        serde_json::json!({"left": 3, "right": 4}),
                    ),
                )),
            ])
            .unwrap(),
            MockTurn::text("done"),
        ]);
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let mut builder = LocalModelAgentBuilder::new(model, "mock", "mock-1");
        for (order, name) in ["first", "second"].into_iter().enumerate() {
            builder = builder.tool(
                StableId::new(name).unwrap(),
                ToolCapability {
                    name: name.to_owned(),
                    description: "waits for the other batch member".to_owned(),
                    parameters: serde_json::json!({"type": "object"}),
                    order: u32::try_from(order).unwrap(),
                    revision: 1,
                    retired: false,
                },
                BarrierTool(barrier.clone()),
            );
        }
        let agent = builder.build().unwrap();

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            agent.run_prompt("run both"),
        )
        .await
        .expect("a sequential dispatcher would wait forever at the barrier")
        .unwrap();

        assert_eq!(output.text, "done");
    }
}
