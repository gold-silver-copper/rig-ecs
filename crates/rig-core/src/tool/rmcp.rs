//! MCP discovery and execution adapters for the ECS effect boundary.
//!
//! MCP connections do not own a parallel tool registry. A connected endpoint
//! implements [`crate::runtime::adapters::EcsDiscovery`], producing complete
//! generation snapshots for normal ECS reconciliation, and executes immutable
//! [`crate::runtime::ToolEffectInput`] values outside the world.

use std::{
    borrow::Cow,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use rmcp::model::{
    CallToolRequest, CallToolResult, ClientRequest, ContentBlock, ListToolsRequest,
    PaginatedRequestParams, ResourceContents, ServerResult,
};
use rmcp::service::PeerRequestOptions;
use rmcp::{ServiceExt, handler::client::ClientHandler};

use crate::{
    OneOrMany,
    completion::ToolDefinition,
    message::{ImageMediaType, MimeType, ToolResultContent},
    runtime::{
        CanonicalError, DiscoveredTool, DiscoveryEffectInput, DiscoveryEffectOutput,
        ToolEffectFailure, ToolEffectInput, ToolEffectOutput, adapters::EcsDiscovery,
    },
    tool::{ToolErrorKind, ToolExecutionError, ToolOutput},
};

/// Default deadline for an MCP tool invocation.
pub const DEFAULT_MCP_TOOL_TIMEOUT: Duration = Duration::from_secs(300);
/// Default deadline for fetching a complete MCP tool snapshot.
pub const DEFAULT_MCP_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

impl From<&rmcp::model::Tool> for ToolDefinition {
    fn from(tool: &rmcp::model::Tool) -> Self {
        Self {
            name: tool.name.to_string(),
            description: tool
                .description
                .clone()
                .unwrap_or(Cow::Borrowed(""))
                .to_string(),
            parameters: tool.schema_as_json_value(),
        }
    }
}

impl From<rmcp::model::Tool> for ToolDefinition {
    fn from(tool: rmcp::model::Tool) -> Self {
        Self::from(&tool)
    }
}

/// MCP connection and protocol failure.
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    /// Failed to establish the protocol service.
    #[error("MCP connection error: {0}")]
    Connection(String),
    /// The peer returned an unexpected or failed response.
    #[error("MCP service error: {0}")]
    Service(#[from] rmcp::ServiceError),
    /// A complete tool snapshot exceeded its deadline.
    #[error("MCP tool discovery timed out after {0:?}")]
    DiscoveryTimeout(Duration),
}

/// Notification-aware client handler.
///
/// Notifications only record that a newer refresh is wanted. The host submits
/// the corresponding ECS refresh command, preserving schedule-boundary
/// reconciliation and stale-generation rejection.
#[derive(Clone)]
pub struct McpClientHandler {
    info: rmcp::model::ClientInfo,
    refresh_signal: Arc<AtomicU64>,
    tool_timeout: Option<Duration>,
    refresh_timeout: Duration,
}

impl McpClientHandler {
    /// Creates an MCP client handler without a tool registry.
    pub fn new(info: rmcp::model::ClientInfo) -> Self {
        Self {
            info,
            refresh_signal: Arc::new(AtomicU64::new(0)),
            tool_timeout: Some(DEFAULT_MCP_TOOL_TIMEOUT),
            refresh_timeout: DEFAULT_MCP_REFRESH_TIMEOUT,
        }
    }

    /// Sets or disables the per-call timeout.
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.tool_timeout = timeout.into();
        self
    }

    /// Sets the complete-snapshot discovery deadline.
    pub fn with_refresh_timeout(mut self, timeout: Duration) -> Self {
        self.refresh_timeout = timeout;
        self
    }

    /// Connects and returns both the running protocol service and its ECS
    /// effect adapter.
    pub async fn connect<T, E, A>(self, transport: T) -> Result<McpConnection, McpClientError>
    where
        T: rmcp::transport::IntoTransport<rmcp::service::RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let tool_timeout = self.tool_timeout;
        let refresh_timeout = self.refresh_timeout;
        let refresh_signal = Arc::clone(&self.refresh_signal);
        let service = ServiceExt::serve(self, transport)
            .await
            .map_err(|error| McpClientError::Connection(error.to_string()))?;
        let endpoint = McpEndpoint {
            peer: service.peer().clone(),
            refresh_signal,
            tool_timeout,
            refresh_timeout,
        };
        Ok(McpConnection { service, endpoint })
    }
}

impl ClientHandler for McpClientHandler {
    fn get_info(&self) -> rmcp::model::ClientInfo {
        self.info.clone()
    }

    async fn on_tool_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<rmcp::service::RoleClient>,
    ) {
        self.refresh_signal.fetch_add(1, Ordering::Release);
    }
}

/// A live MCP service together with its context-free ECS effect adapter.
pub struct McpConnection {
    service: rmcp::service::RunningService<rmcp::service::RoleClient, McpClientHandler>,
    endpoint: McpEndpoint,
}

impl McpConnection {
    /// Returns the discovery and execution adapter.
    pub fn endpoint(&self) -> &McpEndpoint {
        &self.endpoint
    }

    /// Returns an independently owned adapter for an effect host.
    pub fn clone_endpoint(&self) -> McpEndpoint {
        self.endpoint.clone()
    }

    /// Returns the running RMCP service for lifecycle management.
    pub fn service(
        &self,
    ) -> &rmcp::service::RunningService<rmcp::service::RoleClient, McpClientHandler> {
        &self.service
    }
}

/// Cloneable MCP discovery and tool-execution endpoint.
#[derive(Clone)]
pub struct McpEndpoint {
    peer: rmcp::service::ServerSink,
    refresh_signal: Arc<AtomicU64>,
    tool_timeout: Option<Duration>,
    refresh_timeout: Duration,
}

impl McpEndpoint {
    /// Returns the latest list-changed notification counter.
    ///
    /// Hosts compare this value with their last submitted value and enqueue one
    /// ECS refresh command when it advances; repeated notifications coalesce.
    pub fn refresh_signal(&self) -> u64 {
        self.refresh_signal.load(Ordering::Acquire)
    }

    /// Executes one exact immutable tool decision outside the ECS world.
    pub async fn execute(
        &self,
        input: ToolEffectInput,
    ) -> Result<ToolEffectOutput, CanonicalError> {
        let ToolEffectInput {
            decision,
            call_id,
            provider_result_id,
            provider_call_id,
            arguments,
            ..
        } = input;
        let name = decision.name;
        let arguments = match arguments {
            serde_json::Value::Null => None,
            serde_json::Value::Object(object) => Some(object),
            value => {
                return Err(CanonicalError::Tool {
                    message: format!(
                        "MCP tool `{}` requires object or null arguments, received {}",
                        name,
                        json_kind(&value)
                    ),
                    retryable: false,
                });
            }
        };
        let mut params = rmcp::model::CallToolRequestParams::new(name.clone());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        let result = call_tool(&self.peer, params, self.tool_timeout)
            .await
            .map_err(|error| CanonicalError::Tool {
                message: error.to_string(),
                retryable: matches!(error, rmcp::ServiceError::Timeout { .. }),
            })?;
        mcp_effect_output(call_id, provider_result_id, provider_call_id, name, &result)
    }

    async fn fetch_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpClientError> {
        let deadline = tokio::time::Instant::now() + self.refresh_timeout;
        let mut cursor = None;
        let mut tools = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(McpClientError::DiscoveryTimeout(self.refresh_timeout));
            }
            let mut params = PaginatedRequestParams::default();
            params.cursor = cursor;
            let response = send_request(
                &self.peer,
                ClientRequest::ListToolsRequest(ListToolsRequest::with_param(params)),
                Some((deadline, self.refresh_timeout)),
            )
            .await
            .map_err(|error| match error {
                rmcp::ServiceError::Timeout { .. } => {
                    McpClientError::DiscoveryTimeout(self.refresh_timeout)
                }
                error => McpClientError::Service(error),
            })?;
            let ServerResult::ListToolsResult(page) = response else {
                return Err(McpClientError::Service(
                    rmcp::ServiceError::UnexpectedResponse,
                ));
            };
            tools.extend(page.tools);
            cursor = page.next_cursor;
            if cursor.is_none() {
                return Ok(tools);
            }
        }
    }
}

fn mcp_effect_output(
    call_id: String,
    provider_result_id: String,
    provider_call_id: Option<String>,
    name: String,
    result: &CallToolResult,
) -> Result<ToolEffectOutput, CanonicalError> {
    let output = mcp_result_output(result).map_err(|error| CanonicalError::Tool {
        message: error.to_string(),
        retryable: error.retryable().unwrap_or(false),
    })?;
    let presentation = output.render();
    let raw = serde_json::to_value(output.as_content()).map_err(|error| CanonicalError::Tool {
        message: format!("failed to serialize MCP output: {error}"),
        retryable: false,
    })?;
    let failure = result
        .is_error
        .is_some_and(|is_error| is_error)
        .then(|| ToolEffectFailure {
            message: presentation.clone(),
            retryable: Some(false),
            kind: ToolErrorKind::Provider,
            refusal: false,
        });
    Ok(ToolEffectOutput {
        call_id,
        provider_result_id,
        provider_call_id,
        name,
        raw: raw.into(),
        presentation: presentation.into(),
        failure,
    })
}

impl EcsDiscovery for McpEndpoint {
    type Error = McpClientError;

    async fn refresh(
        &self,
        _input: DiscoveryEffectInput,
    ) -> Result<DiscoveryEffectOutput, Self::Error> {
        let tools = self.fetch_tools().await?;
        Ok(DiscoveryEffectOutput {
            tools: tools
                .into_iter()
                .enumerate()
                .map(|(order, tool)| {
                    let definition = ToolDefinition::from(&tool);
                    let revision = definition_revision(&definition);
                    DiscoveredTool {
                        key: definition.name.clone(),
                        name: definition.name,
                        description: definition.description,
                        parameters: definition.parameters,
                        order: u32::try_from(order).unwrap_or(u32::MAX),
                        revision,
                    }
                })
                .collect(),
        })
    }
}

fn definition_revision(definition: &ToolDefinition) -> u64 {
    let mut hasher = DefaultHasher::new();
    definition.name.hash(&mut hasher);
    definition.description.hash(&mut hasher);
    definition.parameters.to_string().hash(&mut hasher);
    hasher.finish()
}

fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

async fn call_tool(
    peer: &rmcp::service::ServerSink,
    params: rmcp::model::CallToolRequestParams,
    timeout: Option<Duration>,
) -> Result<CallToolResult, rmcp::ServiceError> {
    let deadline = timeout.map(|duration| (tokio::time::Instant::now() + duration, duration));
    let response = send_request(
        peer,
        ClientRequest::CallToolRequest(CallToolRequest::new(params)),
        deadline,
    )
    .await?;
    match response {
        ServerResult::CallToolResult(result) => Ok(result),
        _ => Err(rmcp::ServiceError::UnexpectedResponse),
    }
}

async fn send_request(
    peer: &rmcp::service::ServerSink,
    request: ClientRequest,
    deadline: Option<(tokio::time::Instant, Duration)>,
) -> Result<ServerResult, rmcp::ServiceError> {
    let handle = match deadline {
        Some((deadline, timeout)) => {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            crate::time::timeout(
                remaining,
                peer.send_cancellable_request(request, PeerRequestOptions::no_options()),
            )
            .await
            .map_err(|_| rmcp::ServiceError::Timeout { timeout })??
        }
        None => {
            peer.send_cancellable_request(request, PeerRequestOptions::no_options())
                .await?
        }
    };
    let response = match deadline {
        Some((deadline, timeout)) => {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let mut handle = handle;
            crate::time::timeout(remaining, &mut handle.rx)
                .await
                .map_err(|_| rmcp::ServiceError::Timeout { timeout })?
                .map_err(|_| rmcp::ServiceError::TransportClosed)??
        }
        None => handle.await_response().await?,
    };
    Ok(response)
}

fn mcp_content(content: &ContentBlock) -> Result<ToolResultContent, ToolExecutionError> {
    match content {
        ContentBlock::Text(text) => Ok(ToolResultContent::text(text.text.clone())),
        ContentBlock::Image(image) => ImageMediaType::from_mime_type(&image.mime_type)
            .map(|media_type| {
                ToolResultContent::image_base64(image.data.clone(), Some(media_type), None)
            })
            .map(Ok)
            .unwrap_or_else(|| preserve_block(content)),
        ContentBlock::Resource(resource) => match &resource.resource {
            ResourceContents::BlobResourceContents {
                mime_type, blob, ..
            } => mime_type
                .as_deref()
                .and_then(ImageMediaType::from_mime_type)
                .map(|media_type| {
                    ToolResultContent::image_base64(blob.clone(), Some(media_type), None)
                })
                .map(Ok)
                .unwrap_or_else(|| preserve_block(content)),
            _ => preserve_block(content),
        },
        _ => preserve_block(content),
    }
}

fn preserve_block(content: &ContentBlock) -> Result<ToolResultContent, ToolExecutionError> {
    serde_json::to_value(content)
        .map(ToolResultContent::json)
        .map_err(|error| ToolExecutionError::provider(error.to_string()).with_source(error))
}

fn mcp_result_output(result: &CallToolResult) -> Result<ToolOutput, ToolExecutionError> {
    let mut content = result
        .content
        .iter()
        .map(mcp_content)
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(structured) = result.structured_content.clone() {
        content.insert(0, ToolResultContent::json(structured));
    }
    if content.is_empty() {
        return Ok(ToolOutput::text(""));
    }
    let content =
        OneOrMany::many(content).map_err(|error| ToolExecutionError::other(error.to_string()))?;
    Ok(ToolOutput::content(content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_tool_errors_remain_model_visible_effect_results() {
        let result = CallToolResult::error(vec![ContentBlock::text("choose another input")]);

        let output = mcp_effect_output(
            "call-1".to_owned(),
            "call-1".to_owned(),
            None,
            "lookup".to_owned(),
            &result,
        )
        .expect("application-level MCP errors are successful boundary effects");

        assert_eq!(output.call_id, "call-1");
        assert!(output.presentation.contains("choose another input"));
        let failure = output.failure.expect("failure metadata must be retained");
        assert_eq!(failure.kind, ToolErrorKind::Provider);
        assert_eq!(failure.retryable, Some(false));
    }
}
