use anyhow::Result;
use rig::prelude::*;
use rig::providers::anthropic::{self, Client};
use rig::{
    Embed, embeddings::EmbeddingsBuilder, tool::builtin::ThinkTool,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use serde::{Deserialize, Serialize};
use std::env;

// Define a knowledge base entry for our vector store
#[derive(Embed, Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct KnowledgeEntry {
    id: String,
    title: String,
    #[embed]
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Set up logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Create Anthropic client
    let anthropic_api_key = env::var("ANTHROPIC_API_KEY")?;
    let anthropic_client = Client::builder().api_key(&anthropic_api_key).build()?;

    // Create the embedding model for our vector store
    // We'll use OpenAI's embedding model for this example
    let openai_client = rig::providers::openai::Client::from_env()?;
    let embedding_model =
        openai_client.embedding_model(rig::providers::openai::TEXT_EMBEDDING_ADA_002);

    // Create a knowledge base with sample entries
    let knowledge_entries = vec![
        KnowledgeEntry {
            id: "kb1".to_string(),
            title: "Climate Change Effects".to_string(),
            content: "Climate change is causing rising sea levels, increased frequency of extreme weather events, \
                     and disruptions to ecosystems worldwide. The IPCC has projected that global temperatures \
                     could rise by 1.5°C to 4.5°C by 2100, depending on emission scenarios.".to_string(),
        },
        KnowledgeEntry {
            id: "kb2".to_string(),
            title: "Renewable Energy Technologies".to_string(),
            content: "Solar photovoltaic technology converts sunlight directly into electricity using semiconductor materials. \
                     Wind turbines convert kinetic energy from wind into mechanical power, which generators then convert to electricity. \
                     Hydroelectric power generates electricity by using flowing water to turn turbines connected to generators.".to_string(),
        },
        KnowledgeEntry {
            id: "kb3".to_string(),
            title: "Sustainable Agriculture Practices".to_string(),
            content: "Crop rotation improves soil health by alternating different crops in the same area across seasons. \
                     Agroforestry integrates trees with crop or livestock systems, enhancing biodiversity and resilience. \
                     Precision agriculture uses technology to optimize field-level management, reducing resource use while maximizing yields.".to_string(),
        },
        KnowledgeEntry {
            id: "kb4".to_string(),
            title: "Carbon Capture Methods".to_string(),
            content: "Direct air capture (DAC) extracts CO2 directly from the atmosphere using chemical processes. \
                     Bioenergy with carbon capture and storage (BECCS) combines biomass energy with geological CO2 storage. \
                     Enhanced weathering accelerates natural geological processes that remove CO2 from the atmosphere.".to_string(),
        },
    ];

    // Create embeddings for our knowledge base
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(knowledge_entries)?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |entry| entry.id.clone());

    // Create vector store index
    let vector_index = vector_store.index(embedding_model);

    // Create the ECS agent and install its executable capabilities as entities.
    let orchestrator_agent = anthropic_client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(
            "You are an environmental sustainability advisor that helps users understand complex environmental issues
            and find practical solutions. You have access to several specialized tools:

            1. A knowledge base with information on climate change, renewable energy, sustainable agriculture, and carbon capture.
            2. A research agent that can provide detailed information on environmental science topics.
            3. A data analysis agent that can interpret environmental data and statistics.
            4. A recommendation agent that can suggest practical sustainability solutions.
            5. A think tool that allows you to reason through complex problems step by step.

            Your workflow:
            1. Use the knowledge base to retrieve relevant background information
            2. Use the research agent to gather detailed information on specific topics
            3. Use the data analysis agent to interpret any data or statistics
            4. Use the think tool to reason through the problem and plan your approach
            5. Use the recommendation agent to generate practical solutions

            Combine these tools effectively to provide comprehensive, accurate, and actionable advice on
            environmental sustainability issues."
        )
        .tool(ThinkTool)
        .tool(vector_index)
        .name("orchestrator_agent")
        .build();

    println!("=== Complex Agentic Loop with Claude ===");
    println!("This example demonstrates a complex agentic loop using Claude with:");
    println!("- Multiple specialized agents used as tools");
    println!("- Vector store for knowledge retrieval");
    println!("- Think tool for complex reasoning");
    println!();

    // Example query that will exercise the complex agentic loop
    let query = "I'm a small business owner looking to reduce my company's carbon footprint. \
                We have 25 employees in a 5000 sq ft office space and a small fleet of 5 delivery vehicles. \
                What are the most cost-effective sustainability measures we could implement in the next 6-12 months? Try to stay concise.";

    println!("Query: {}", query);
    println!("\nProcessing...\n");

    let response = orchestrator_agent.prompt(query).await?;

    // Print the final response
    println!("\nFinal Response:\n{response}");

    Ok(())
}
