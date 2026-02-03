# Copyright (c) Microsoft. All rights reserved.

from agent_framework import ChatMessage, Content

from agent_framework_ag_ui._message_adapters import _deduplicate_messages, _sanitize_tool_history


def test_sanitize_tool_history_injects_confirm_changes_result() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            contents=[
                Content.from_function_call(
                    name="confirm_changes",
                    call_id="call_confirm_123",
                    arguments='{"changes": "test"}',
                )
            ],
        ),
        ChatMessage(
            role="user",
            contents=[Content.from_text(text='{"accepted": true}')],
        ),
    ]

    sanitized = _sanitize_tool_history(messages)

    tool_messages = [
        msg for msg in sanitized if (msg.role.value if hasattr(msg.role, "value") else str(msg.role)) == "tool"
    ]
    assert len(tool_messages) == 1
    assert str(tool_messages[0].contents[0].call_id) == "call_confirm_123"
    assert tool_messages[0].contents[0].result == "Confirmed"


def test_deduplicate_messages_prefers_non_empty_tool_results() -> None:
    messages = [
        ChatMessage(
            role="tool",
            contents=[Content.from_function_result(call_id="call1", result="")],
        ),
        ChatMessage(
            role="tool",
            contents=[Content.from_function_result(call_id="call1", result="result data")],
        ),
    ]

    deduped = _deduplicate_messages(messages)
    assert len(deduped) == 1
    assert deduped[0].contents[0].result == "result data"


def test_convert_approval_results_to_tool_messages() -> None:
    """Test that function_result content in user messages gets converted to tool messages.

    This is a regression test for the MCP tool double-call bug where approved tool
    results ended up in user messages instead of tool messages, causing OpenAI to
    reject the request with 'tool_call_ids did not have response messages'.
    """
    from agent_framework_ag_ui._run import _convert_approval_results_to_tool_messages

    # Simulate what happens after _resolve_approval_responses:
    # A user message contains function_result content (the executed tool result)
    messages = [
        ChatMessage(
            role="assistant",
            contents=[
                Content.from_function_call(call_id="call_123", name="my_mcp_tool", arguments="{}"),
            ],
        ),
        ChatMessage(
            role="user",
            contents=[
                Content.from_function_result(call_id="call_123", result="tool execution result"),
            ],
        ),
    ]

    _convert_approval_results_to_tool_messages(messages)

    # After conversion, the function result should be in a tool message, not user message
    assert len(messages) == 2

    # First message unchanged
    assert messages[0].role.value == "assistant"

    # Second message should now be role="tool"
    assert messages[1].role.value == "tool"
    assert messages[1].contents[0].type == "function_result"
    assert messages[1].contents[0].call_id == "call_123"


def test_convert_approval_results_preserves_other_user_content() -> None:
    """Test that user messages with mixed content are handled correctly.

    If a user message has both function_result content and other content (like text),
    the function_result content should be extracted to a tool message while the
    remaining content stays in the user message.
    """
    from agent_framework_ag_ui._run import _convert_approval_results_to_tool_messages

    messages = [
        ChatMessage(
            role="assistant",
            contents=[
                Content.from_function_call(call_id="call_123", name="my_tool", arguments="{}"),
            ],
        ),
        ChatMessage(
            role="user",
            contents=[
                Content.from_text(text="User also said something"),
                Content.from_function_result(call_id="call_123", result="tool result"),
            ],
        ),
    ]

    _convert_approval_results_to_tool_messages(messages)

    # Should have 3 messages now: assistant, user (with text), tool (with result)
    assert len(messages) == 3

    # First message unchanged
    assert messages[0].role.value == "assistant"

    # Second message should be user with just text
    assert messages[1].role.value == "user"
    assert len(messages[1].contents) == 1
    assert messages[1].contents[0].type == "text"

    # Third message should be tool with result
    assert messages[2].role.value == "tool"
    assert messages[2].contents[0].type == "function_result"


def test_sanitize_tool_history_multiple_requests_with_confirm_changes() -> None:
    """Test that confirm_changes is properly handled across multiple requests.

    This is a regression test for the MCP tool double-call bug. When a second
    MCP tool call happens after the first one completed, the message history
    should have proper synthetic results for all confirm_changes calls.
    """
    # Simulate history from first request (already completed)
    # Note: confirm_changes synthetic result might be missing from frontend's snapshot
    messages = [
        # First request - user asks something
        ChatMessage(
            role="user",
            contents=[Content.from_text(text="What time is it?")],
        ),
        # First request - assistant calls MCP tool + confirm_changes
        ChatMessage(
            role="assistant",
            contents=[
                Content.from_function_call(call_id="call_1", name="get_datetime", arguments="{}"),
                Content.from_function_call(call_id="call_c1", name="confirm_changes", arguments="{}"),
            ],
        ),
        # First request - tool result for the actual MCP tool
        ChatMessage(
            role="tool",
            contents=[Content.from_function_result(call_id="call_1", result="2024-01-01 12:00:00")],
        ),
        # Note: NO tool result for call_c1 (confirm_changes) - this is the bug scenario!
        # The synthetic was injected in request 1 but not persisted in snapshot
        # Second request - user asks something else
        ChatMessage(
            role="user",
            contents=[Content.from_text(text="What's the date?")],
        ),
    ]

    sanitized = _sanitize_tool_history(messages)

    # After sanitization, call_c1 should have a synthetic result
    tool_messages = [
        msg for msg in sanitized if (msg.role.value if hasattr(msg.role, "value") else str(msg.role)) == "tool"
    ]

    # Should have 2 tool messages: one for call_1 (real) and one for call_c1 (synthetic)
    assert len(tool_messages) == 2

    tool_call_ids = {str(msg.contents[0].call_id) for msg in tool_messages}
    assert "call_1" in tool_call_ids
    assert "call_c1" in tool_call_ids
