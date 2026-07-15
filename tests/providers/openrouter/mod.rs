mod support;

mod cassette {
    mod agent;
    mod agent_tool_sessions;
    mod document_file_data;
    mod document_ordering;
    mod extractor;
    mod extractor_usage;
    mod models;
    mod multi_extract;
    mod multimodal;
    mod openai_responses_compat;
    mod provider_selection;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod streaming;
    mod streaming_tools;
    mod transcription;
}

#[cfg(feature = "audio")]
mod audio_generation;
mod document_file_data;
mod file_id;

pub(super) const DEFAULT_MODEL: &str = "openai/gpt-4o-mini";
pub(super) const TOOL_MODEL: &str = "openai/gpt-4o";
