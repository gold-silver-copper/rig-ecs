//! Smoke coverage for issue #1604 against a local llama.cpp OpenAI-compatible server.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::CompletionClient;
use rig::completion::TypedPrompt;
use rig::tool::Tool;

use super::support;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct WeatherResponse {
    city: String,
    weather: String,
}

#[derive(Debug, Deserialize)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone)]
struct WeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl WeatherTool {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

fn pretty_json<T>(value: &T) -> String
where
    T: Serialize + ?Sized,
{
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|err| format!("<failed to serialize debug payload as JSON: {err}>"))
}

impl Tool for WeatherTool {
    const NAME: &'static str = "weather";
    type Error = std::io::Error;
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a city".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        })
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl std::future::Future<Output = Result<Self::Output, Self::Error>> + Send {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let result = format!("The weather in {} is all fire and brimstone", args.city);

        println!("\n=== weather tool implementation ===");
        println!(
            "{}",
            pretty_json(&serde_json::json!({
                "args": { "city": args.city },
                "returned": result,
            }))
        );

        std::future::ready(Ok(result))
    }
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn prompt_typed_with_tool_call_verbatim_roundtrip() -> Result<()> {
    let model = support::model_name();
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = support::completions_client();

    let agent = client
        .agent(model)
        .tool(WeatherTool::new(call_count.clone()))
        .preamble(
            "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. After calling the tool, return a JSON response with the city name and the weather description. DO NOT modify the description from the tool result.",
        )
        .build();

    let result = agent
        .prompt_typed::<WeatherResponse>("Hello, whats the weather in London?")
        .await;

    println!("prompt_typed result: {result:#?}");

    let response = result?;
    println!("agent response: {response:#?}");

    anyhow::ensure!(
        call_count.load(Ordering::SeqCst) >= 1,
        "expected the weather tool to be executed at least once"
    );
    crate::support::assert_weather_tool_roundtrip_response(
        &response.city,
        &response.weather,
        "London",
    );

    Ok(())
}
