//! ECS-native agent construction and observation facades.
//!
//! The types in this module are handles around an authoritative
//! [`World`](crate::bevy_ecs::world::World). They do not contain a runner,
//! registry, callback stack, or duplicated orchestration state.

use std::{future::IntoFuture, pin::Pin};

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    OneOrMany,
    completion::{
        AssistantContent, CompletionError, CompletionModel, GetTokenUsage, Message, PromptError,
        Usage,
    },
    runtime::adapters::{LocalAgentError, LocalStreamEvent, local_prompt_error},
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat, StreamingPrompt},
};

pub use crate::message::{Text, ToolCall};
pub use crate::runtime::adapters::{
    AgentBuilder, AgentFacade as Agent, AgentPromptRequest, ExtendedAgentPromptRequest,
    StructuredOutputMode,
};

/// Details for one model operation committed by a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionCall {
    /// Zero-based model-operation index.
    pub call_index: usize,
    /// Provider-reported usage for this operation.
    pub usage: Usage,
}

impl CompletionCall {
    /// Creates one operation record.
    pub fn new(call_index: usize, usage: Usage) -> Self {
        Self { call_index, usage }
    }
}

/// Terminal response observed from an ECS run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromptResponse {
    /// Concatenated assistant text.
    pub output: String,
    /// Usage accumulated by the run.
    pub usage: Usage,
    /// Successfully committed model operations.
    pub completion_calls: Vec<CompletionCall>,
    /// Optional canonical history when requested by a chat facade.
    pub messages: Option<Vec<Message>>,
    /// Canonical final assistant content.
    pub content: OneOrMany<AssistantContent>,
}

impl PromptResponse {
    /// Creates a terminal text response.
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        let output = output.into();
        Self {
            content: OneOrMany::one(AssistantContent::text(output.clone())),
            output,
            usage,
            completion_calls: Vec::new(),
            messages: None,
        }
    }

    /// Creates an empty response.
    pub fn empty() -> Self {
        Self::new(String::new(), Usage::default())
    }

    /// Attaches canonical messages to an already constructed response.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Returns concatenated final text.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Returns aggregate run usage.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Returns canonical history when the caller requested it.
    pub fn messages(&self) -> Option<&[Message]> {
        self.messages.as_deref()
    }

    /// Returns final structured assistant content.
    pub fn content(&self) -> &OneOrMany<AssistantContent> {
        &self.content
    }

    /// Returns committed model-operation details.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Returns the number of committed model operations.
    pub fn requests(&self) -> usize {
        self.completion_calls.len()
    }
}

impl std::fmt::Display for PromptResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output.fmt(formatter)
    }
}

/// Ordered observation emitted by a streaming ECS run.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MultiTurnStreamItem<R> {
    /// Incremental assistant content.
    StreamAssistantItem(StreamedAssistantContent<R>),
    /// A tool call whose logical ECS batch committed successfully.
    ToolExecutionCommitted {
        /// Exact call accepted by the immutable tool decision.
        tool_call: ToolCall,
        /// Runtime-local call correlation identity.
        internal_call_id: String,
    },
    /// A canonical tool result committed to the run transcript.
    StreamUserItem(StreamedUserContent),
    /// Details for one committed model operation.
    CompletionCall(CompletionCall),
    /// Terminal output from the same run state used by blocking execution.
    FinalResponse(PromptResponse),
}

impl<R> MultiTurnStreamItem<R> {
    /// Creates a terminal response item from canonical content.
    pub fn final_response(content: OneOrMany<AssistantContent>, usage: Usage) -> Self {
        let output = content
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                AssistantContent::ToolCall(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_) => None,
            })
            .collect::<String>();
        let mut response = PromptResponse::new(output, usage);
        response.content = content;
        Self::FinalResponse(response)
    }
}

/// Failure observed while streaming one ECS run.
#[derive(Debug, Error)]
pub enum StreamingError {
    /// Low-level provider failure.
    #[error(transparent)]
    Completion(#[from] CompletionError),
    /// Canonical run failure.
    #[error("agent stream failed: {0}")]
    Agent(#[from] LocalAgentError),
    /// High-level prompt failure.
    #[error(transparent)]
    Prompt(#[from] Box<PromptError>),
}

/// Boxed observation stream for a single ECS run.
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>> + Send>>;

/// Construction value for starting a streaming run.
pub struct StreamingPromptRequest<M>
where
    M: CompletionModel,
{
    agent: Agent<M>,
    prompt: Message,
    history: Vec<Message>,
    max_model_calls: Option<u32>,
    tool_concurrency: usize,
    conversation: Result<Option<crate::runtime::StableId>, crate::runtime::IdentityError>,
}

impl<M> StreamingPromptRequest<M>
where
    M: CompletionModel,
{
    /// Creates a request using the agent's authoritative world.
    pub fn new(agent: &Agent<M>, prompt: impl Into<Message>) -> Self {
        Self {
            agent: agent.clone(),
            prompt: prompt.into(),
            history: Vec::new(),
            max_model_calls: None,
            tool_concurrency: usize::MAX,
            conversation: Ok(None),
        }
    }

    /// Supplies caller-owned history copied into the run at ingress.
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

    /// Sets the host-side upper bound for concurrent tool effects.
    ///
    /// Tool results are still committed in canonical logical-batch order.
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        assert!(
            concurrency > 0,
            "tool concurrency must be greater than zero"
        );
        self.tool_concurrency = concurrency;
        self
    }

    /// Routes the stream through an ECS conversation-memory relationship.
    pub fn conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = crate::runtime::StableId::new(conversation).map(Some);
        self
    }

    async fn send(self) -> StreamingResult<M::StreamingResponse>
    where
        M: 'static,
    {
        let conversation = match self.conversation {
            Ok(conversation) => conversation,
            Err(error) => {
                return Box::pin(futures::stream::once(async move {
                    Err(StreamingError::Prompt(Box::new(local_prompt_error(
                        LocalAgentError::Identity(error),
                    ))))
                }));
            }
        };
        let source = self.agent.stream_run_with_history(
            self.prompt,
            self.history,
            self.max_model_calls,
            conversation,
            self.tool_concurrency,
        );
        Box::pin(source.map(|event| {
            event
                .map_err(|error| StreamingError::Prompt(Box::new(local_prompt_error(error))))
                .map(|event| match event {
                    LocalStreamEvent::Delta(text) => MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::Text(Text::from(text)),
                    ),
                    LocalStreamEvent::ToolCall {
                        tool_call,
                        internal_call_id,
                    } => MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::ToolCall {
                            tool_call,
                            internal_call_id,
                        },
                    ),
                    LocalStreamEvent::ToolCommitted {
                        tool_call,
                        internal_call_id,
                    } => MultiTurnStreamItem::ToolExecutionCommitted {
                        tool_call,
                        internal_call_id,
                    },
                    LocalStreamEvent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    } => MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::ToolCallDelta {
                            id,
                            internal_call_id,
                            content,
                        },
                    ),
                    LocalStreamEvent::Reasoning(reasoning) => {
                        MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::Reasoning(reasoning),
                        )
                    }
                    LocalStreamEvent::ReasoningDelta { id, reasoning } => {
                        MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::ReasoningDelta { id, reasoning },
                        )
                    }
                    LocalStreamEvent::Unknown(value) => MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::Unknown(value),
                    ),
                    LocalStreamEvent::ToolResult {
                        tool_result,
                        internal_call_id,
                    } => MultiTurnStreamItem::StreamUserItem(StreamedUserContent::tool_result(
                        tool_result,
                        internal_call_id,
                    )),
                    LocalStreamEvent::Finished {
                        output,
                        transcript,
                        completion_calls,
                    } => {
                        let mut response = PromptResponse::new(
                            output.text,
                            Usage {
                                input_tokens: output.usage.input_tokens,
                                output_tokens: output.usage.output_tokens,
                                total_tokens: output.usage.input_tokens
                                    + output.usage.output_tokens,
                                ..Usage::default()
                            },
                        );
                        response.completion_calls = completion_calls
                            .into_iter()
                            .enumerate()
                            .map(|(index, usage)| {
                                CompletionCall::new(
                                    index,
                                    Usage {
                                        input_tokens: usage.input_tokens,
                                        output_tokens: usage.output_tokens,
                                        total_tokens: usage.input_tokens + usage.output_tokens,
                                        ..Usage::default()
                                    },
                                )
                            })
                            .collect();
                        response.messages =
                            crate::runtime::adapters::transcript_messages(&transcript).ok();
                        MultiTurnStreamItem::FinalResponse(response)
                    }
                })
        }))
    }
}

impl<M> IntoFuture for StreamingPromptRequest<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: Send + GetTokenUsage,
{
    type Output = StreamingResult<M::StreamingResponse>;
    type IntoFuture = futures::future::BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<M> StreamingPrompt<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: Send + GetTokenUsage,
{
    fn stream_prompt(&self, prompt: impl Into<Message> + Send) -> StreamingPromptRequest<M> {
        StreamingPromptRequest::new(self, prompt)
    }
}

impl<M> StreamingChat<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: Send + GetTokenUsage,
{
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: I,
    ) -> StreamingPromptRequest<M>
    where
        I: IntoIterator<Item = T> + Send,
        T: Into<Message>,
    {
        StreamingPromptRequest::new(self, prompt).history(chat_history.into_iter().map(Into::into))
    }
}

/// Prints assistant text and returns the terminal response.
pub async fn stream_to_stdout<R>(
    stream: &mut StreamingResult<R>,
) -> Result<PromptResponse, std::io::Error>
where
    R: Clone,
{
    let mut final_response = PromptResponse::empty();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
                print!("{}", text.text);
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => final_response = response,
            Ok(
                MultiTurnStreamItem::StreamAssistantItem(_)
                | MultiTurnStreamItem::ToolExecutionCommitted { .. }
                | MultiTurnStreamItem::StreamUserItem(_)
                | MultiTurnStreamItem::CompletionCall(_),
            ) => {}
            Err(error) => return Err(std::io::Error::other(error)),
        }
    }
    Ok(final_response)
}
