use anyhow::Result;
use rig::prelude::*;
use rig::providers::openai;
use rig::{
    embeddings::{EmbeddingsBuilder, ToolSchema},
    providers::openai::Client,
    tool::{Tool, ToolEmbedding},
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct InitError;

#[derive(Deserialize, Serialize)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The first number to add"
                },
                "y": {
                    "type": "number",
                    "description": "The second number to add"
                }
            }
        })
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}
impl ToolEmbedding for Add {
    type InitError = InitError;
    type Context = ();
    type State = ();
    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Add)
    }
    fn embedding_docs(&self) -> Vec<String> {
        vec!["Add x and y together".into()]
    }
    fn context(&self) -> Self::Context {}
}
#[derive(Deserialize, Serialize)]
struct Subtract;
impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;
    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The number to subtract from"
                },
                "y": {
                    "type": "number",
                    "description": "The number to subtract"
                }
            }
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}

impl ToolEmbedding for Subtract {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Subtract)
    }

    fn context(&self) -> Self::Context {}

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Subtract y from x (i.e.: x - y)".into()]
    }
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // required to enable CloudWatch error logging by the runtime
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        .init();

    // Create OpenAI client
    let openai_client = Client::from_env()?;
    let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
    let add = Add;
    let subtract = Subtract;
    let schemas = vec![
        ToolSchema::try_from(&add)?,
        ToolSchema::try_from(&subtract)?,
    ];
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(schemas)?
        .build()
        .await?;
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone());
    let index = vector_store.index(embedding_model);

    // Candidate tools remain executable ECS capability entities, while a
    // scheduled store operation selects one immutable per-turn tool snapshot.
    let calculator_rag = openai_client
        .agent(openai::GPT_4)
        .preamble("You are a calculator here to help the user perform arithmetic operations.")
        .retrieved_tool(add)
        .retrieved_tool(subtract)
        .retrieved_tools(1, index)
        .default_max_turns(2)
        .build();

    // Prompt the agent and print the response
    let response = calculator_rag.prompt("Calculate 3 - 7").await?;

    println!("{response}");

    Ok(())
}
