mod agent_run_support;
mod support;

mod cassette {
    mod agent;
    mod chat_history;
    mod document_ordering;
    mod embeddings;
    mod extractor;
    mod generate_behaviors;
    mod generate_sessions;
    mod generate_tool_args;
    mod generate_tool_modes;
    #[cfg(feature = "image")]
    mod image_generation;
    mod interactions_api;
    mod models;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod streaming;
    mod streaming_multimodal_tool_results;
    mod streaming_tools;
    mod structured_output;
    mod tool_choice;
    mod tool_definitions;
    mod transcription;
}

mod live {}
