//! Fail-closed human approval before submitting an ECS run.
//!
//! Runtime policy is represented by ECS entities. This small interactive
//! example obtains approval before the command enters the runtime, then an
//! installed policy independently rejects unapproved prompt wording.

use std::io::Write;

use anyhow::Result;
use rig::{
    client::{CompletionClient, ProviderClient},
    providers::openai,
    runtime::{Policy, PolicyRule, StableId, TenantId},
};

fn approved() -> bool {
    print!("Submit the operations request? [y/N] ");
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).is_ok()
        && matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

#[tokio::main]
async fn main() -> Result<()> {
    if !approved() {
        println!("Request was not submitted.");
        return Ok(());
    }

    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("You are an operations assistant. Confirm the approved request succinctly.")
        .build();

    agent.with_runtime_mut(|runtime| {
        runtime.spawn_policy(
            StableId::new("reject-unapproved-marker")?,
            TenantId::new("local")?,
            Policy {
                order: 0,
                revision: 1,
                rule: PolicyRule::DenyPromptContains("UNAPPROVED".to_owned()),
            },
            agent.handle(),
        )?;
        Ok::<_, anyhow::Error>(())
    })??;

    let response = agent
        .prompt("APPROVED: email Alice a reminder about the 3pm budget review")
        .await?;
    println!("{response}");
    Ok(())
}
