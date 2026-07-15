mod agent_run_support;
mod support;
mod tools_support;

mod cassette {
    mod agent;
    mod agent_tools_e2e;
    mod chat_history;
    mod document_ordering;
    mod embeddings;
    mod extractor;
    mod generate_behaviors;
    mod generate_sessions;
    mod generate_tool_args;
    mod generate_tool_modes;
    mod hook_stress_tools;
    #[cfg(feature = "image")]
    mod image_generation;
    mod interactions_api;
    mod models;
    mod multi_turn_streaming;
    mod reasoning_roundtrip;
    mod reasoning_tool_roundtrip;
    mod streaming;
    mod streaming_multimodal_tool_results;
    mod streaming_tools;
    mod structured_output;
    mod tool_choice;
    mod tool_definitions;
    mod tool_hooks;
    mod transcription;
}

mod live {}
