//! Demonstrates ECS-native request policy and configuration.
//!
//! Policy is data on an entity interpreted at `RigSet::InvokeRequestPolicy`; request
//! transformation is ordinary agent component configuration snapshotted by
//! preparation. No callback stack sits outside the schedule.

use anyhow::Result;
use rig::{
    client::{CompletionClient, ProviderClient},
    providers::openai,
    runtime::{Policy, PolicyRule, StableId, TenantId},
};

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .context("House style: keep jokes short and family-friendly.")
        .temperature(0.2)
        .build();

    agent.with_runtime_mut(|runtime| {
        runtime.spawn_policy(
            StableId::new("deny-private-material")?,
            TenantId::new("local")?,
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::DenyPromptContains("private".to_owned()),
            },
            agent.handle(),
        )?;
        Ok::<_, anyhow::Error>(())
    })??;

    let response = agent.prompt("Entertain me!").await?;
    println!("\nFinal response:\n{response}");
    Ok(())
}
