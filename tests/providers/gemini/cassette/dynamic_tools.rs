//! Dynamic (RAG) tools sampled through an ECS store operation per prompt and
//! merged with static tool entities.
//!
//! Each cassette records the Gemini embedding calls (toolset embedding at
//! build time, query embedding at prompt time) alongside the completion
//! turns.

use std::sync::{Arc, Mutex};

use rig::bevy_ecs::prelude::In;
use rig::client::{CompletionClient, EmbeddingsClient};
use rig::completion::{Chat, Message};
use rig::embeddings::{EmbeddingsBuilder, ToolSchema};
use rig::providers::gemini;
use rig::runtime::{
    PolicyPoint, PolicyResponderId, PolicyRule, RequestPolicyDecision, RequestPolicyInvocation,
};
use rig::vector_store::in_memory_store::InMemoryVectorStore;

use super::super::agent_run_support::{history_has_assistant_tool_call, tool_result_texts};
use super::super::support::with_gemini_cassette;
use super::super::tools_support::{
    CountingAdd, EmbedAdd, EmbedMultiply, EmbedSubtract, FORCE_TOOLS_PREAMBLE,
};
use crate::support::{assert_mentions_expected_number, install_policy};

/// Build an in-memory index over the toolset's embeddable schemas.
async fn build_tool_index(
    client: &gemini::Client,
    schemas: Vec<ToolSchema>,
) -> rig::vector_store::in_memory_store::InMemoryVectorIndex<
    gemini::embedding::EmbeddingModel,
    rig::embeddings::ToolSchema,
> {
    let embedding_model = client.embedding_model(gemini::embedding::EMBEDDING_001);
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(schemas)
        .expect("documents should be added")
        .build()
        .await
        .expect("tool schema embeddings should succeed");

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |tool| tool.name.clone());
    vector_store.index(embedding_model)
}

#[tokio::test]
async fn dynamic_tool_retrieved_and_merged_with_static() {
    let add = CountingAdd::default();
    let subtract = EmbedSubtract::default();
    let subtract_counter = subtract.counter.clone();

    with_gemini_cassette(
        "dynamic_tools/dynamic_tool_retrieved_and_merged_with_static",
        |client| async move {
            let multiply = EmbedMultiply::default();
            let schemas = vec![
                ToolSchema::try_from(&subtract).expect("subtract schema should build"),
                ToolSchema::try_from(&multiply).expect("multiply schema should build"),
            ];
            let index = build_tool_index(&client, schemas).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(add)
                .retrieved_tool(subtract)
                .retrieved_tool(multiply)
                .retrieved_tools(1, index)
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat("Subtract 8 from 50 to get their difference.", &mut history)
                .await
                .expect("dynamic tool prompt should succeed");

            assert_mentions_expected_number(&response, 42);
            assert!(
                history_has_assistant_tool_call(&history, "subtract"),
                "the retrieved dynamic tool should be called: {history:?}"
            );
            assert_eq!(
                subtract_counter.count(),
                1,
                "the retrieved dynamic tool should execute exactly once"
            );
        },
    )
    .await;
}
#[tokio::test]
async fn dynamic_only_agent_retrieves_tool_per_prompt() {
    let add = EmbedAdd::default();
    let add_counter = add.counter.clone();

    with_gemini_cassette(
        "dynamic_tools/dynamic_only_agent_retrieves_tool_per_prompt",
        |client| async move {
            let subtract = EmbedSubtract::default();
            let schemas = vec![
                ToolSchema::try_from(&add).expect("add schema should build"),
                ToolSchema::try_from(&subtract).expect("subtract schema should build"),
            ];
            let index = build_tool_index(&client, schemas).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .retrieved_tool(add)
                .retrieved_tool(subtract)
                .retrieved_tools(1, index)
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            let response = agent
                .chat("Add 19 and 23 together to get their sum.", &mut history)
                .await
                .expect("dynamic-only tool prompt should succeed");

            assert_mentions_expected_number(&response, 42);
            let texts: Vec<String> = history.iter().flat_map(tool_result_texts).collect();
            assert_eq!(texts, vec!["42".to_string()]);
            assert_eq!(
                add_counter.count(),
                1,
                "the retrieved dynamic tool should execute exactly once"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn sample_caps_retrieved_definitions() {
    with_gemini_cassette(
        "dynamic_tools/sample_caps_retrieved_definitions",
        |client| async move {
            let add = EmbedAdd::default();
            let subtract = EmbedSubtract::default();
            let multiply = EmbedMultiply::default();
            let schemas = vec![
                ToolSchema::try_from(&add).expect("add schema should build"),
                ToolSchema::try_from(&subtract).expect("subtract schema should build"),
                ToolSchema::try_from(&multiply).expect("multiply schema should build"),
            ];
            let index = build_tool_index(&client, schemas).await;

            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .retrieved_tool(add)
                .retrieved_tool(subtract)
                .retrieved_tool(multiply)
                .retrieved_tools(2, index)
                .build();

            let policy = install_policy(
                &agent,
                "capture-retrieved-tools",
                0,
                1,
                PolicyRule::Custom(PolicyPoint::Request),
            )
            .expect("capture policy should install");
            let captured = Arc::new(Mutex::new(Vec::new()));
            let captured_for_policy = Arc::clone(&captured);
            agent
                .with_runtime_mut(move |runtime| {
                    runtime
                        .register_request_policy_responder(
                            policy,
                            PolicyResponderId::new("capture-retrieved-tools").unwrap(),
                            move |In(event): In<RequestPolicyInvocation>| {
                                *captured_for_policy.lock().expect("captured tools") = event
                                    .request
                                    .tools
                                    .iter()
                                    .map(|tool| tool.name.clone())
                                    .collect();
                                Some(RequestPolicyDecision::Stop(
                                    "captured retrieved definitions".to_owned(),
                                ))
                            },
                        )
                        .expect("capture responder should register");
                })
                .expect("capture observer should install");

            agent
                .prompt("Multiply two numbers together to get their product.")
                .await
                .expect_err("capture policy should stop before model dispatch");
            let defs = captured.lock().expect("captured tools").clone();

            assert_eq!(
                defs.len(),
                2,
                "the sample size should cap how many dynamic definitions are returned: {:?}",
                defs
            );
            assert!(
                defs.iter().any(|name| name == "multiply"),
                "the best-matching tool should be retrieved: {:?}",
                defs
            );
        },
    )
    .await;
}
