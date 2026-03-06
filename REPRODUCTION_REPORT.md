# Reproduction Report — Issue #4315
**Python: [Bug]: Azure open ai agent session error**

*Generated: 2026-03-06 03:33:44 UTC*

## Result

| Field | Value |
|-------|-------|
| Status | `reproduced` |
| Failure observed | `yes` |
| Evidence strength | `high` |
| Confidence | `high` |
| Failing test | `python/packages/core/tests/azure/test_azure_chat_client_conversation_id_bug.py` |
| Gate passed | Yes |

## Files Examined

python/packages/core/agent_framework/azure/_chat_client.py, python/packages/core/agent_framework/openai/_chat_client.py, python/packages/core/agent_framework/openai/_responses_client.py, python/packages/core/agent_framework/_agents.py, python/packages/core/tests/azure/conftest.py, python/packages/core/tests/azure/test_azure_chat_client.py

## Tests Run

packages/core/tests/azure/test_azure_chat_client_conversation_id_bug.py::test_conversation_id_not_passed_to_completions_create, packages/core/tests/azure/test_azure_chat_client_conversation_id_bug.py::test_conversation_id_not_passed_to_streaming_completions_create, packages/core/tests/azure/test_azure_chat_client_conversation_id_bug.py::test_agent_run_with_session_service_session_id

## Finding

The bug is confirmed. In `RawOpenAIChatClient._prepare_options()` (line 325 of `_chat_client.py`), `conversation_id` is not excluded from the options dict before it's passed to `AsyncCompletions.create()`. The `OpenAIResponsesClient` correctly excludes it at line 902, but the chat completions client does not. When `agent.run()` is called with a session that has `service_session_id`, the agent sets `conversation_id` in options (line 1067 of `_agents.py`), and it leaks through to the OpenAI API call.

## Full Analysis

Now let me look at the key files - the Azure chat client and how `conversation_id` flows through.I can see the bug. In `_prepare_options` at line 325, `conversation_id` is NOT excluded from the options dict that gets passed to `AsyncCompletions.create()`. Let me verify this is handled differently in the responses client:Confirmed. The `OpenAIResponsesClient` explicitly excludes `conversation_id` (line 902) and handles it separately, but `RawOpenAIChatClient._prepare_options` does NOT exclude it, so it gets passed straight through to `AsyncCompletions.create()`.

Let me now look at existing tests and write a reproduction test.Now I have everything I need. Let me write a test that reproduces the bug.All 3 tests fail, confirming the bug. Let me fix the streaming test's minor issue (wrong method name) so all failures are purely about the bug:I see — streaming uses `get_response(stream=True)`. Let me fix the streaming test:All 3 tests fail cleanly, confirming the bug. The `conversation_id` leaks through to `AsyncCompletions.create()` in all paths.
