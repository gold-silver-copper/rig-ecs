//! Context-free typed authoring for executable ECS tool capabilities.
//!
//! Tool identity, grants, revisions, policy, call state, and lifecycle live in
//! [`crate::runtime`] components. This module contains only the typed external
//! effect boundary and canonical output/error values; it is not a registry.

pub mod builtin;
mod output;
mod result;

pub use output::{IntoToolOutput, ToolOutput};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};

use std::{future::Future, sync::Arc};

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::completion::ToolDefinition;

/// A typed tool authoring boundary.
///
/// Calls receive only owned, deserialized arguments. Per-call identity,
/// authorization, policy, and result metadata remain ECS state and immutable
/// effect input; tools cannot access a world or a general-purpose type map.
pub trait Tool: Sized + Send + Sync {
    /// Provider-facing capability name.
    const NAME: &'static str;
    /// Owned JSON arguments.
    type Args: for<'de> Deserialize<'de> + Send + Sync;
    /// Canonical model-visible output.
    type Output: IntoToolOutput + Send;
    /// Concrete author-facing failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Provider-facing description.
    fn description(&self) -> String;

    /// JSON Schema for arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Normalizes concrete failures at the external effect boundary.
    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::from_error(error)
    }

    /// Executes one owned invocation without ECS access.
    fn call(
        &self,
        arguments: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

/// A tool that can be embedded and reconstructed by an authoring integration.
///
/// Runtime discovery still reconciles the resulting capability as an entity;
/// this trait does not provide or own a tool registry.
pub trait ToolEmbedding: Tool {
    /// Failure returned while reconstructing the typed implementation.
    type InitError: std::error::Error + Send + Sync + 'static;
    /// Serializable reconstruction data.
    type Context: for<'de> Deserialize<'de> + Serialize;
    /// Runtime initialization state supplied by the authoring integration.
    type State: Send;

    /// Documents used by a discovery implementation.
    fn embedding_docs(&self) -> Vec<String>;
    /// Serializable reconstruction data.
    fn context(&self) -> Self::Context;
    /// Reconstructs the typed implementation.
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}

type DynamicCallback = dyn Fn(serde_json::Value) -> BoxFuture<'static, Result<ToolOutput, ToolExecutionError>>
    + Send
    + Sync;

/// Runtime-authored context-free tool implementation.
#[derive(Clone)]
pub struct DynamicTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
    callback: Arc<DynamicCallback>,
}

impl std::fmt::Debug for DynamicTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl DynamicTool {
    /// Creates a runtime-authored tool from an owned async callback.
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self
    where
        F: Fn(serde_json::Value) -> BoxFuture<'static, Result<ToolOutput, ToolExecutionError>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            callback: Arc::new(callback),
        }
    }

    /// Provider-facing name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Provider-facing definition.
    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    pub(crate) async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, ToolExecutionError> {
        (self.callback)(arguments).await
    }
}

/// Generates provider-facing metadata for a typed tool.
pub fn tool_definition<T>(tool: &T) -> ToolDefinition
where
    T: Tool,
{
    ToolDefinition {
        name: T::NAME.to_owned(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;
