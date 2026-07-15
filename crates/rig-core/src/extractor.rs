//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::openai};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the OpenAI client
//! let openai = openai::Client::new("your-open-ai-api-key")?;
//!
//! // Define the structure of the data you want to extract
//! #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
//! struct Person {
//!    name: Option<String>,
//!    age: Option<u8>,
//!    profession: Option<String>,
//! }
//!
//! // Create the extractor
//! let extractor = openai.extractor::<Person>(openai::GPT_4O)
//!     .build();
//!
//! // Extract structured data from text
//! let person = extractor.extract("John Doe is a 30 year old doctor.").await?;
//! # Ok(())
//! # }
//! ```

use std::marker::PhantomData;

use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::{
    agent::{Agent, AgentBuilder},
    completion::{CompletionModel, Usage},
    message::{Message, ToolChoice},
    runtime::adapters::LocalAgentError,
    tool::Tool,
    vector_store::VectorStoreIndexDyn,
};

struct SubmitTool<T>(PhantomData<fn() -> T>);

impl<T> Tool for SubmitTool<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
{
    const NAME: &'static str = "submit";
    type Error = serde_json::Error;
    type Args = T;
    type Output = serde_json::Value;

    fn description(&self) -> String {
        "Submit the structured data you extracted from the provided text.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(T)).unwrap_or(serde_json::Value::Bool(false))
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        serde_json::to_value(arguments)
    }
}

/// Response from an extraction operation containing the extracted data and usage information.
#[derive(Debug, Clone)]
pub struct ExtractionResponse<T> {
    /// The extracted structured data
    pub data: T,
    /// Accumulated token usage across all attempts (including retries)
    pub usage: Usage,
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("No data extracted")]
    NoData,

    #[error("Failed to deserialize the extracted data: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("AgentError: {0}")]
    AgentError(#[from] LocalAgentError),
}

/// Extractor for structured data from text
pub struct Extractor<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
{
    agent: Agent<M>,
    _t: PhantomData<T>,
    retries: u64,
}

impl<M, T> Extractor<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
{
    /// Attempts to extract data from the given text with a number of retries.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    pub async fn extract(&self, text: impl Into<Message> + Send) -> Result<T, ExtractionError> {
        let (data, _usage) = self.retry_extract(text.into(), vec![]).await?;
        Ok(data)
    }

    /// Attempts to extract data from the given text with a number of retries.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    pub async fn extract_with_chat_history(
        &self,
        text: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<T, ExtractionError> {
        let (data, _usage) = self.retry_extract(text.into(), chat_history).await?;
        Ok(data)
    }

    /// Attempts to extract data from the given text with a number of retries,
    /// returning both the extracted data and accumulated token usage.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    ///
    /// Usage accumulates across all retry attempts, including attempts that received
    /// a billed response but failed extraction (e.g. the model never called `submit`).
    /// Attempts whose completion call itself returned an error (e.g. network failures
    /// or unparseable provider responses) contribute no usage, and when every attempt
    /// fails the returned error carries no usage information at all.
    pub async fn extract_with_usage(
        &self,
        text: impl Into<Message> + Send,
    ) -> Result<ExtractionResponse<T>, ExtractionError> {
        let (data, usage) = self.retry_extract(text.into(), vec![]).await?;
        Ok(ExtractionResponse { data, usage })
    }

    /// Attempts to extract data from the given text with a number of retries,
    /// providing chat history context, and returning both the extracted data
    /// and accumulated token usage.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    ///
    /// Usage accumulates across all retry attempts, including attempts that received
    /// a billed response but failed extraction (e.g. the model never called `submit`).
    /// Attempts whose completion call itself returned an error (e.g. network failures
    /// or unparseable provider responses) contribute no usage, and when every attempt
    /// fails the returned error carries no usage information at all.
    pub async fn extract_with_chat_history_with_usage(
        &self,
        text: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<ExtractionResponse<T>, ExtractionError> {
        let (data, usage) = self.retry_extract(text.into(), chat_history).await?;
        Ok(ExtractionResponse { data, usage })
    }

    /// Runs the extraction with the retry semantics shared by all public
    /// `extract*` methods, returning the extracted data and the token usage
    /// accumulated across all attempts, including failed ones. The accumulated
    /// usage is only observable on success: when every attempt fails, the
    /// returned error cannot carry it.
    async fn retry_extract(
        &self,
        text: Message,
        chat_history: Vec<Message>,
    ) -> Result<(T, Usage), ExtractionError> {
        let mut last_error = None;
        let mut usage = Usage::new();

        for i in 0..=self.retries {
            tracing::debug!(
                "Attempting to extract JSON. Retries left: {retries}",
                retries = self.retries - i
            );
            let (result, attempt_usage) = self.extract_json_with_usage(&text, &chat_history).await;
            usage += attempt_usage;
            match result {
                Ok(data) => return Ok((data, usage)),
                Err(e) => {
                    let suffix = if i < self.retries { " Retrying..." } else { "" };
                    tracing::warn!("Attempt {i} to extract JSON failed: {e:?}.{suffix}");
                    last_error = Some(e);
                }
            }
        }

        // If the loop finishes without a successful extraction, return the last error encountered.
        Err(last_error.unwrap_or(ExtractionError::NoData))
    }

    /// Performs a single extraction attempt, returning its outcome alongside
    /// the token usage it consumed. Usage is reported even when the attempt
    /// fails after a billed completion (e.g. the model never called `submit`);
    /// it is zero whenever the completion call itself returns an error, since
    /// `CompletionError` carries no usage — even if the provider billed the
    /// request (e.g. an unparseable response body).
    async fn extract_json_with_usage(
        &self,
        text: &Message,
        messages: &[Message],
    ) -> (Result<T, ExtractionError>, Usage) {
        let schema = serde_json::to_value(schema_for!(T));
        let schema = match schema {
            Ok(schema) => schema,
            Err(error) => return (Err(error.into()), Usage::new()),
        };
        let response = match self
            .agent
            .run_prompt_configured(text.clone(), messages.to_vec(), Some(schema))
            .await
        {
            Ok(response) => response,
            Err(e) => return (Err(e.into()), Usage::new()),
        };
        let usage = Usage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            total_tokens: response.usage.input_tokens + response.usage.output_tokens,
            ..Usage::new()
        };
        if response.text.is_empty() {
            return (Err(ExtractionError::NoData), usage);
        }

        (
            serde_json::from_str(&response.text).map_err(ExtractionError::from),
            usage,
        )
    }
}

/// Builder for the Extractor
pub struct ExtractorBuilder<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
{
    agent_builder: AgentBuilder<M>,
    _t: PhantomData<T>,
    retries: Option<u64>,
}

impl<M, T> ExtractorBuilder<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync + 'static,
{
    pub fn new(model: M) -> Self {
        Self::with_agent_builder(AgentBuilder::new(model))
    }

    pub(crate) fn with_identity(
        model: M,
        provider: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        Self::with_agent_builder(AgentBuilder::with_identity(model, provider, model_name))
    }

    fn with_agent_builder(agent_builder: AgentBuilder<M>) -> Self {
        Self {
            agent_builder: agent_builder
                .preamble("\
                    You are an AI assistant whose purpose is to extract structured data from the provided text.\n\
                    You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n\
                    Use the `submit` function to submit the structured data.\n\
                    Be sure to fill out every field and ALWAYS CALL THE `submit` function, even with default values!!!.
                ")
                .tool(SubmitTool::<T>(PhantomData))
                .terminal_tool("submit")
                .tool_choice(ToolChoice::Required)
                .output_schema::<T>(),
            retries: None,
            _t: PhantomData,
        }
    }

    /// Add additional preamble to the extractor
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.agent_builder = self.agent_builder.append_preamble(format!(
            "\n=============== ADDITIONAL INSTRUCTIONS ===============\n{preamble}"
        ));
        self
    }

    /// Add a context document to the extractor
    pub fn context(mut self, doc: &str) -> Self {
        self.agent_builder = self.agent_builder.context(doc);
        self
    }

    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.agent_builder = self.agent_builder.additional_params(params);
        self
    }

    /// Set the maximum number of tokens for the completion
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.agent_builder = self.agent_builder.max_tokens(max_tokens);
        self
    }

    /// Set the maximum number of retries for the extractor.
    pub fn retries(mut self, retries: u64) -> Self {
        self.retries = Some(retries);
        self
    }

    /// Set the `tool_choice` option for the inner Agent.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.agent_builder = self.agent_builder.tool_choice(choice);
        self
    }

    /// Build the Extractor
    pub fn build(self) -> Extractor<M, T> {
        Extractor {
            agent: self.agent_builder.build(),
            _t: PhantomData,
            retries: self.retries.unwrap_or(0),
        }
    }

    /// Add dynamic context (RAG) to the extractor.
    ///
    /// On each prompt, `sample` documents will be retrieved from the index based on the RAG text
    /// and inserted in the request.
    pub fn dynamic_context(
        mut self,
        sample: usize,
        dynamic_context: impl VectorStoreIndexDyn + 'static,
    ) -> Self {
        self.agent_builder = self.agent_builder.dynamic_context(sample, dynamic_context);
        self
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::test_utils::{MockCompletionModel, MockTurn};

    #[derive(Debug, PartialEq, Deserialize, Serialize, JsonSchema)]
    struct Person {
        name: String,
    }

    fn usage(total_tokens: u64) -> Usage {
        Usage {
            total_tokens,
            ..Usage::new()
        }
    }

    fn extractor(
        model: MockCompletionModel,
        retries: u64,
    ) -> Extractor<MockCompletionModel, Person> {
        ExtractorBuilder::new(model).retries(retries).build()
    }

    fn submit_turn(name: &str) -> MockTurn {
        MockTurn::tool_call("submit-call", "submit", json!({ "name": name }))
    }

    #[tokio::test]
    async fn usage_accumulates_across_failed_attempts() {
        let model = MockCompletionModel::new([
            MockTurn::text("no submit call").with_usage(usage(10)),
            submit_turn("John").with_usage(usage(5)),
        ]);

        let response = extractor(model, 1)
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(
            response.data,
            Person {
                name: "John".to_string()
            }
        );
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn transport_errors_contribute_no_usage() {
        let model = MockCompletionModel::new([
            MockTurn::error("boom"),
            submit_turn("John").with_usage(usage(5)),
        ]);

        let response = extractor(model, 1)
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(response.usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn single_successful_attempt_reports_its_own_usage() {
        let model = MockCompletionModel::new([submit_turn("John").with_usage(usage(7))]);

        let response = extractor(model, 0)
            .extract_with_usage("John")
            .await
            .expect("extraction should succeed");

        assert_eq!(response.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn exhausted_retries_return_last_error() {
        let model =
            MockCompletionModel::new([MockTurn::text("no submit call").with_usage(usage(10))]);

        let err = extractor(model, 0)
            .extract("John")
            .await
            .expect_err("extraction should fail");

        assert!(matches!(err, ExtractionError::NoData));
    }

    #[tokio::test]
    async fn exhausted_retries_return_error_from_final_attempt() {
        let model = MockCompletionModel::new([MockTurn::error("first"), MockTurn::error("second")]);

        let err = extractor(model, 1)
            .extract("John")
            .await
            .expect_err("extraction should fail");

        assert!(matches!(
            err,
            ExtractionError::AgentError(LocalAgentError::Canonical(
                crate::runtime::CanonicalError::Provider { message, .. }
            )) if message == "ProviderError: second"
        ));
    }
}
