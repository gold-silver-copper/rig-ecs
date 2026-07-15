//! Discover and execute a real RMCP tool through the ECS effect boundary.
//!
//! The MCP endpoint owns protocol I/O, not a parallel registry. A complete
//! tool-list generation is reconciled into capability entities; the accepted
//! revision is then executed from an immutable `ToolEffectInput`. MCP
//! `tools/list_changed` notifications increment a coalescing refresh counter
//! that hosts translate into `RefreshDiscovery` commands.

use anyhow::{Context, Result};
use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use rig::bevy_ecs::prelude::Entity;
use rig::runtime::{
    Agent, DiscoveryKey, EffectCompletion, EffectOutput, ModelCapability, ModelEffectOutput,
    ModelToolCall, Runtime, RuntimeConfig, StableId, TenantId, ToolCapability, ToolGrant, Usage,
    adapters::DiscoveryAdapter,
};
use rig::tool::rmcp::McpClientHandler;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct SumRequest {
    a: i32,
    b: i32,
}

#[derive(Clone)]
struct Calculator {
    tool_router: ToolRouter<Self>,
}

impl Calculator {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl Calculator {
    #[tool(description = "Calculate the sum of two integers")]
    fn sum(
        &self,
        Parameters(SumRequest { a, b }): Parameters<SumRequest>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![ContentBlock::text(
            (a + b).to_string(),
        )]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Calculator {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("rig-ecs-rmcp-example", "1.0.0"))
            .with_instructions("Use sum for integer addition")
    }
}

fn id(value: &str) -> Result<StableId> {
    Ok(StableId::new(value)?)
}

fn tenant() -> Result<TenantId> {
    Ok(TenantId::new("rmcp-example")?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let service = TowerToHyperService::new(StreamableHttpService::new(
        || Ok(Calculator::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let service = service.clone();
            tokio::spawn(async move {
                let _ = Builder::new(TokioExecutor::default())
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });

    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("rig-core", env!("CARGO_PKG_VERSION")),
    );
    let transport =
        rmcp::transport::StreamableHttpClientTransport::from_uri(format!("http://{address}"));
    let connection = McpClientHandler::new(client_info)
        .connect(transport)
        .await?;
    let endpoint = connection.clone_endpoint();
    println!(
        "connected to MCP server: {:#?}",
        connection.service().peer_info()
    );

    let mut runtime = Runtime::new(RuntimeConfig::default())?;
    let source_id = id("calculator-mcp")?;
    let source = runtime.spawn_discovery_source(source_id.clone(), tenant()?, "mcp")?;
    runtime.handle().refresh_discovery(source)?;
    runtime.run_until_stalled()?;
    let refresh = runtime
        .effects()
        .try_recv()?
        .context("expected an MCP discovery effect")?;
    let refresh_input = refresh
        .discovery_input()
        .cloned()
        .context("expected typed discovery input")?;
    let discovery = DiscoveryAdapter::new(source_id, endpoint.clone());
    let discovered = discovery.execute(refresh_input).await?;
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: refresh.operation,
            generation: refresh.generation,
            result: Ok(EffectOutput::Discovery(discovered)),
        })?;
    runtime.run_until_stalled()?;

    let sum_tool = {
        let world = runtime.world_mut();
        let mut tools = world.query::<(Entity, &DiscoveryKey, &ToolCapability)>();
        tools
            .iter(world)
            .find_map(|(entity, _, tool)| (tool.name == "sum").then_some(entity))
            .context("MCP discovery did not reconcile the sum tool")?
    };
    let model = runtime.spawn_model(
        id("model")?,
        tenant()?,
        ModelCapability {
            provider: "example".to_owned(),
            model: "deterministic".to_owned(),
            revision: 1,
            retired: false,
        },
    )?;
    let agent = runtime.spawn_agent(
        id("agent")?,
        tenant()?,
        Agent {
            instructions: "Use the discovered MCP calculator".to_owned(),
            ..Agent::default()
        },
        model,
    )?;
    runtime.grant_tool(
        id("sum-grant")?,
        tenant()?,
        ToolGrant {
            order: 0,
            enabled: true,
        },
        agent,
        sum_tool,
    )?;

    runtime.handle().prompt(agent, "What is 2 + 5?")?;
    runtime.run_until_stalled()?;
    let model_request = runtime
        .effects()
        .try_recv()?
        .context("expected a model effect")?;
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
                    id: "sum-call".to_owned(),
                    provider_result_id: "sum-call".to_owned(),
                    provider_call_id: None,
                    name: "sum".to_owned(),
                    arguments: serde_json::json!({"a": 2, "b": 5}),
                }],
            })),
        })?;
    runtime.run_until_stalled()?;
    let tool_request = runtime
        .effects()
        .try_recv()?
        .context("expected the reconciled MCP tool effect")?;
    let tool_input = tool_request
        .tool_input()
        .cloned()
        .context("expected typed MCP tool input")?;
    let tool_output = endpoint.execute(tool_input).await?;
    println!("MCP sum result: {}", tool_output.presentation);
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: tool_request.operation,
            generation: tool_request.generation,
            result: Ok(EffectOutput::Tool(tool_output)),
        })?;
    runtime.run_until_stalled()?;
    let final_model = runtime
        .effects()
        .try_recv()?
        .context("expected a post-tool model effect")?;
    runtime
        .effects()
        .completion_sender()
        .try_send(EffectCompletion {
            operation: final_model.operation,
            generation: final_model.generation,
            result: Ok(EffectOutput::Model(ModelEffectOutput {
                assistant_message: None,
                text: "2 + 5 = 7".to_owned(),
                usage: Usage::default(),
                tool_calls: Vec::new(),
            })),
        })?;
    runtime.run_until_stalled()?;

    println!(
        "list_changed refresh generation: {} (submit one refresh when this advances)",
        endpoint.refresh_signal()
    );
    drop(connection);
    server.abort();
    Ok(())
}
