//! ECS facade coverage replacing the former hand-driven `AgentRun` stepping tests.

use rig::client::CompletionClient;
use rig::completion::PromptError;
use rig::message::{Message, ToolChoice};
use rig::providers::gemini;

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, Subtract, assistant_tool_call_names,
    history_has_assistant_tool_call, is_tool_result_user_message,
};
use super::super::support::with_gemini_cassette;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_mentions_expected_number, assert_nonempty_response,
};

#[tokio::test]
async fn hand_driven_single_turn_completes() {
    with_gemini_cassette(
        "agent_run_stepping/hand_driven_single_turn_completes",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(BASIC_PREAMBLE)
                .build();
            let response = agent
                .prompt(BASIC_PROMPT)
                .extended_details()
                .await
                .expect("single-turn run should complete");

            assert_nonempty_response(&response.output);
            assert_eq!(response.requests(), 1);
            assert!(response.usage.total_tokens > 0);
            let messages = response.messages.expect("canonical messages");
            assert_eq!(messages.len(), 2);
            assert!(matches!(messages.first(), Some(Message::User { .. })));
            assert!(matches!(messages.last(), Some(Message::Assistant { .. })));
        },
    )
    .await;
}

#[tokio::test]
async fn hand_driven_multi_turn_tool_run_completes() {
    with_gemini_cassette(
        "agent_run_stepping/hand_driven_multi_turn_tool_run_completes",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let response = agent
                .prompt(
                    "Use the tools to compute (7 + 4) - 2: first compute 7 + 4 with the add tool, then subtract 2 from that result with the subtract tool, then state the final result.",
                )
                .max_turns(5)
                .extended_details()
                .await
                .expect("multi-turn run should complete");

            assert_mentions_expected_number(&response.output, 9);
            assert!(response.requests() >= 2);
            let messages = response.messages.expect("canonical messages");
            assert!(history_has_assistant_tool_call(&messages, "add"));
            assert!(history_has_assistant_tool_call(&messages, "subtract"));
            assert!(messages.iter().any(is_tool_result_user_message));
        },
    )
    .await;
}

#[tokio::test]
async fn hand_driven_parallel_tool_calls_arrive_in_one_step() {
    with_gemini_cassette(
        "agent_run_stepping/hand_driven_parallel_tool_calls_arrive_in_one_step",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let response = agent
                .prompt(
                    "Compute 3 + 5 and 10 - 4. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.",
                )
                .max_turns(3)
                .extended_details()
                .await
                .expect("parallel run should complete");

            let messages = response.messages.expect("canonical messages");
            let mut names = messages
                .iter()
                .map(assistant_tool_call_names)
                .find(|names| names.len() == 2)
                .expect("one turn should contain both calls");
            names.sort();
            assert_eq!(names, vec!["add".to_owned(), "subtract".to_owned()]);
            assert_mentions_expected_number(&response.output, 8);
            assert_mentions_expected_number(&response.output, 6);
        },
    )
    .await;
}

#[tokio::test]
async fn max_turns_error_carries_pending_tool_results_message() {
    with_gemini_cassette(
        "agent_run_stepping/max_turns_error_carries_pending_tool_results_message",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            let error = agent
                .prompt("What is 21 + 21? Use the add tool.")
                .max_turns(2)
                .await
                .expect_err("the required-tool loop should exhaust its model budget");

            let PromptError::MaxTurnsError {
                max_turns,
                chat_history,
                prompt,
            } = error
            else {
                panic!("expected MaxTurnsError, got {error:?}");
            };
            assert_eq!(max_turns, 2);
            assert!(chat_history.iter().any(is_tool_result_user_message));
            assert!(is_tool_result_user_message(&prompt));
        },
    )
    .await;
}
