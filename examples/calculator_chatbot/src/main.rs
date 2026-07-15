use anyhow::Result;
use rig::integrations::cli_chatbot::ChatBotBuilder;
use rig::prelude::*;
use rig::providers::openai;
use rig::{
    providers::openai::Client,
    tool::{Tool, ToolEmbedding},
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
#[error("Init error")]
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
            },
            "required": [ "x", "y" ]
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
            },
            "required": [ "x", "y" ]
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

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Subtract y from x (i.e.: x - y)".into()]
    }

    fn context(&self) -> Self::Context {}
}

struct Multiply;

impl Tool for Multiply {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Compute the product of x and y (i.e.: x * y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The first factor in the product"
                },
                "y": {
                    "type": "number",
                    "description": "The second factor in the product"
                }
            },
            "required": [ "x", "y" ]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x * args.y;
        Ok(result)
    }
}

impl ToolEmbedding for Multiply {
    type InitError = InitError;
    type Context = ();
    type State = ();
    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Multiply)
    }
    fn embedding_docs(&self) -> Vec<String> {
        vec!["Compute the product of x and y (i.e.: x * y)".into()]
    }
    fn context(&self) -> Self::Context {}
}

struct Divide;

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;
    fn description(&self) -> String {
        "Compute the Quotient of x and y (i.e.: x / y). Useful for ratios.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The Dividend of the division. The number being divided"
                },
                "y": {
                    "type": "number",
                    "description": "The Divisor of the division. The number by which the dividend is being divided"
                }
            },
            "required": [ "x", "y" ]
        })
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x / args.y;
        Ok(result)
    }
}

impl ToolEmbedding for Divide {
    type InitError = InitError;
    type Context = ();
    type State = ();

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Divide)
    }

    fn embedding_docs(&self) -> Vec<String> {
        vec!["Compute the Quotient of x and y (i.e.: x / y). Useful for ratios.".into()]
    }

    fn context(&self) -> Self::Context {}
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env()?;

    // Install executable capabilities as ECS entities.
    let calculator_rag = openai_client
        .agent(openai::GPT_4)
        .preamble(
            "You are an assistant here to help the user select which tool is most appropriate to perform arithmetic operations.
            Follow these instructions closely.
            1. Consider the user's request carefully and identify the core elements of the request.
            2. Select which tool among those made available to you is appropriate given the context.
            3. This is very important: never perform the operation yourself and never give me the direct result.
            Always respond with the name of the tool that should be used and the appropriate inputs
            in the following format:
            Tool: <tool name>
            Inputs: <list of inputs>
            "
        )
        .tool(Add)
        .tool(Subtract)
        .tool(Multiply)
        .tool(Divide)
        .build();

    // Create a CLI chatbot from the agent
    let chatbot = ChatBotBuilder::new()
        .agent(calculator_rag)
        .max_turns(2)
        .build();

    chatbot.run().await?;

    Ok(())
}
