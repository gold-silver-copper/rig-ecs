//! Shared deterministic fixtures retained for the ECS hook-stress migrations.
//!
//! The former callback and scratchpad helpers now live as ECS policy installers,
//! targeted observers, and typed components beside the individual cassette
//! suites. This module keeps their common prompts and third arithmetic tool in
//! a form that compiles without the removed hook traits.
#![allow(dead_code)]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub(crate) const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided \
     tools for every arithmetic operation instead of computing results yourself. Perform the steps \
     in order, using the result of each step as an input to the next. Once you have the final tool \
     result, reply with the final numeric answer in plain text.";

pub(crate) const INDEPENDENT_TOOLS_PREAMBLE: &str = "You are a calculator assistant. You MUST use \
     the provided tools for every arithmetic operation instead of computing results yourself. Once \
     you have the tool results you need, reply with the requested numbers in plain text.";

#[derive(Debug, thiserror::Error)]
#[error("math error")]
pub(crate) struct MathError;

#[derive(Deserialize, Serialize)]
pub(crate) struct OperationArgs {
    pub(crate) x: i64,
    pub(crate) y: i64,
}

#[derive(Clone, Default)]
pub(crate) struct CallCounter(Arc<AtomicUsize>);

impl CallCounter {
    pub(crate) fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    fn bump(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Third counting tool shared by the policy/observer stress workflows.
#[derive(Clone, Default)]
pub(crate) struct CountingMultiply {
    pub(crate) counter: CallCounter,
}

impl Tool for CountingMultiply {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Multiply x and y together".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first operand" },
                "y": { "type": "number", "description": "The second operand" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.counter.bump();
        Ok(args.x * args.y)
    }
}
