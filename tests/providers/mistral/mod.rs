mod agent;
mod agent_tool_sessions;
#[cfg(feature = "derive")]
mod embeddings;
mod extractor;
mod extractor_usage;
mod models;
mod multi_extract;
mod streaming;
mod streaming_tools;
mod support;
mod transcription;

pub(super) const DEFAULT_MODEL: &str = "mistral-small-latest";
pub(super) const TOOL_MODEL: &str = DEFAULT_MODEL;
