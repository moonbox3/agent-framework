# Copyright (c) Microsoft. All rights reserved.

"""Tests for AG-UI orchestrators."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

from ag_ui.core import BaseEvent, RunFinishedEvent
from agent_framework import (
    AgentResponseUpdate,
    AgentThread,
    BaseChatClient,
    ChatAgent,
    ChatResponseUpdate,
    Content,
    FunctionInvocationConfiguration,
    ai_function,
)

from agent_framework_ag_ui._agent import AgentConfig
from agent_framework_ag_ui._orchestrators import DefaultOrchestrator, ExecutionContext


@ai_function
def server_tool() -> str:
    """Server-executable tool."""
    return "server"


def _create_mock_chat_agent(
    tools: list[Any] | None = None,
    response_format: Any = None,
    capture_tools: list[Any] | None = None,
    capture_messages: list[Any] | None = None,
) -> ChatAgent:
    """Create a ChatAgent with mocked chat client for testing.

    Args:
        tools: Tools to configure on the agent.
        response_format: Response format to configure.
        capture_tools: If provided, tools passed to run_stream will be appended here.
        capture_messages: If provided, messages passed to run_stream will be appended here.
    """
    mock_chat_client = MagicMock(spec=BaseChatClient)
    mock_chat_client.function_invocation_configuration = FunctionInvocationConfiguration()

    agent = ChatAgent(
        chat_client=mock_chat_client,
        tools=tools or [server_tool],
        response_format=response_format,
    )

    # Create a mock run_stream that captures parameters and yields a simple response
    async def mock_run_stream(
        messages: list[Any],
        *,
        #     thread: AgentThread,
        #     tools: list[Any] | None = None,
        #     **kwargs: Any,
        # ) -> AsyncGenerator[AgentRunResponseUpdate, None]:
        #     self.seen_tools = tools
        #     yield AgentRunResponseUpdate(
        #         contents=[TextContent(text="ok")],
        #         role="assistant",
        #         response_id=thread.metadata.get("ag_ui_run_id"),  # type: ignore[attr-defined] (metadata always created in orchestrator)
        #         raw_representation=ChatResponseUpdate(
        #             contents=[TextContent(text="ok")],
        #             conversation_id=thread.metadata.get("ag_ui_thread_id"),  # type: ignore[attr-defined] (metadata always created in orchestrator)
        #             response_id=thread.metadata.get("ag_ui_run_id"),  # type: ignore[attr-defined] (metadata always created in orchestrator)
        #         ),
        #     )
        thread: AgentThread,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[AgentResponseUpdate, None]:
        if capture_tools is not None and tools is not None:
            capture_tools.extend(tools)
        if capture_messages is not None:
            capture_messages.extend(messages)
        yield AgentResponseUpdate(
            contents=[Content.from_text(text="ok")],
            role="assistant",
            response_id=thread.metadata.get("ag_ui_run_id"),  # type: ignore[attr-defined] (metadata always created in orchestrator)
            raw_representation=ChatResponseUpdate(
                contents=[Content.from_text(text="ok")],
                conversation_id=thread.metadata.get("ag_ui_thread_id"),  # type: ignore[attr-defined] (metadata always created in orchestrator)
                response_id=thread.metadata.get("ag_ui_run_id"),  # type: ignore[attr-defined] (metadata always created in orchestrator)
            ),
        )

    # Patch the run_stream method
    agent.run_stream = mock_run_stream  # type: ignore[method-assign]

    return agent


async def test_default_orchestrator_merges_client_tools() -> None:
    """Client tool declarations are merged with server tools before running agent."""
    captured_tools: list[Any] = []
    agent = _create_mock_chat_agent(tools=[server_tool], capture_tools=captured_tools)
    orchestrator = DefaultOrchestrator()

    input_data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Client weather lookup.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
    }

    context = ExecutionContext(
        input_data=input_data,
        agent=agent,
        config=AgentConfig(),
    )

    events = []
    async for event in orchestrator.run(context):
        events.append(event)

    assert len(captured_tools) > 0
    tool_names = [getattr(tool, "name", "?") for tool in captured_tools]
    assert "server_tool" in tool_names
    assert "get_weather" in tool_names
    assert agent.chat_client.function_invocation_configuration.additional_tools


async def test_default_orchestrator_with_camel_case_ids() -> None:
    """Client tool is able to extract camelCase IDs."""
    agent = _create_mock_chat_agent()
    orchestrator = DefaultOrchestrator()

    input_data = {
        "runId": "test-camelcase-runid",
        "threadId": "test-camelcase-threadid",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        "tools": [],
    }

    context = ExecutionContext(
        input_data=input_data,
        agent=agent,
        config=AgentConfig(),
    )

    events = []
    async for event in orchestrator.run(context):
        events.append(event)

    # assert the last event has the expected run_id and thread_id
    assert isinstance(events[-1], RunFinishedEvent)
    last_event = events[-1]
    assert last_event.run_id == "test-camelcase-runid"
    assert last_event.thread_id == "test-camelcase-threadid"


async def test_default_orchestrator_with_snake_case_ids() -> None:
    """Client tool is able to extract snake_case IDs."""
    agent = _create_mock_chat_agent()
    orchestrator = DefaultOrchestrator()

    input_data = {
        "run_id": "test-snakecase-runid",
        "thread_id": "test-snakecase-threadid",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        "tools": [],
    }

    context = ExecutionContext(
        input_data=input_data,
        agent=agent,
        config=AgentConfig(),
    )

    events: list[BaseEvent] = []
    async for event in orchestrator.run(context):
        events.append(event)

    # assert the last event has the expected run_id and thread_id
    assert isinstance(events[-1], RunFinishedEvent)
    last_event = events[-1]
    assert last_event.run_id == "test-snakecase-runid"
    assert last_event.thread_id == "test-snakecase-threadid"


async def test_state_context_injected_when_tool_call_state_mismatch() -> None:
    """State context should be injected when current state differs from tool call args."""
    captured_messages: list[Any] = []
    agent = _create_mock_chat_agent(tools=[], capture_messages=captured_messages)
    orchestrator = DefaultOrchestrator()

    tool_recipe = {"title": "Salad", "special_preferences": []}
    current_recipe = {"title": "Salad", "special_preferences": ["Vegetarian"]}

    input_data = {
        "state": {"recipe": current_recipe},
        "messages": [
            {"role": "system", "content": "Instructions"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "update_recipe", "arguments": {"recipe": tool_recipe}},
                    }
                ],
            },
            {"role": "user", "content": "What are the dietary preferences?"},
        ],
    }

    context = ExecutionContext(
        input_data=input_data,
        agent=agent,
        config=AgentConfig(
            state_schema={"recipe": {"type": "object"}},
            predict_state_config={"recipe": {"tool": "update_recipe", "tool_argument": "recipe"}},
            require_confirmation=False,
        ),
    )

    async for _event in orchestrator.run(context):
        pass

    assert len(captured_messages) > 0
    state_messages = []
    for msg in captured_messages:
        role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role_value != "system":
            continue
        for content in msg.contents or []:
            if content.type == "text" and content.text.startswith("Current state of the application:"):
                state_messages.append(content.text)
    assert state_messages
    assert "Vegetarian" in state_messages[0]


async def test_state_context_not_injected_when_tool_call_matches_state() -> None:
    """State context should be skipped when tool call args match current state."""
    captured_messages: list[Any] = []
    agent = _create_mock_chat_agent(tools=[], capture_messages=captured_messages)
    orchestrator = DefaultOrchestrator()

    input_data = {
        "messages": [
            {"role": "system", "content": "Instructions"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "update_recipe", "arguments": {"recipe": {}}},
                    }
                ],
            },
            {"role": "user", "content": "What are the dietary preferences?"},
        ],
    }

    context = ExecutionContext(
        input_data=input_data,
        agent=agent,
        config=AgentConfig(
            state_schema={"recipe": {"type": "object"}},
            predict_state_config={"recipe": {"tool": "update_recipe", "tool_argument": "recipe"}},
            require_confirmation=False,
        ),
    )

    async for _event in orchestrator.run(context):
        pass

    assert len(captured_messages) > 0
    state_messages = []
    for msg in captured_messages:
        role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role_value != "system":
            continue
        for content in msg.contents or []:
            if content.type == "text" and content.text.startswith("Current state of the application:"):
                state_messages.append(content.text)
    assert not state_messages


def test_options_filtered_from_tool_kwargs() -> None:
    """Verify 'options' is filtered when creating tool_kwargs from run_kwargs.

    The AG-UI orchestrator adds 'options' (containing metadata/store for Azure AI)
    to run_kwargs, but this should NOT be passed to _try_execute_function_calls
    as external tools like MCP servers don't understand these kwargs.

    This test verifies the filtering logic inline rather than through the full
    orchestrator flow, matching the pattern in _resolve_approval_responses:
        tool_kwargs = {k: v for k, v in run_kwargs.items() if k != "options"}
    """
    # Simulate the run_kwargs that the orchestrator creates
    run_kwargs: dict[str, Any] = {
        "thread": MagicMock(),
        "tools": [server_tool],
        "options": {"metadata": {"thread_id": "test-123"}, "store": True},
    }

    # This is the exact filtering logic from _resolve_approval_responses
    tool_kwargs = {k: v for k, v in run_kwargs.items() if k != "options"}

    # Verify 'options' was filtered out
    assert "options" not in tool_kwargs, "'options' should be filtered out before tool execution"

    # Verify other kwargs are preserved
    assert "thread" in tool_kwargs, "'thread' should be preserved"
    assert "tools" in tool_kwargs, "'tools' should be preserved"

    # Verify the original run_kwargs still has options (it's needed for run_stream)
    assert "options" in run_kwargs, "Original run_kwargs should still have 'options'"


def test_orchestrator_filters_options_in_resolve_approval_responses() -> None:
    """Verify the orchestrator code filters 'options' before tool execution.

    This is a code inspection test that verifies the fix is present in the
    _resolve_approval_responses function within DefaultOrchestrator.run().
    """
    import inspect

    # Get the source code of the DefaultOrchestrator.run method
    source = inspect.getsource(DefaultOrchestrator.run)

    # Verify the filtering pattern is present
    assert 'k != "options"' in source, (
        "Expected 'options' filtering in DefaultOrchestrator.run(). "
        "The line 'tool_kwargs = {k: v for k, v in run_kwargs.items() if k != \"options\"}' "
        "should be present in _resolve_approval_responses."
    )

    # Verify tool_kwargs is passed to _try_execute_function_calls (not run_kwargs)
    assert "custom_args=tool_kwargs" in source, (
        "Expected _try_execute_function_calls to receive tool_kwargs (not run_kwargs). "
        "This ensures 'options' is filtered out before tool execution."
    )


def test_agui_internal_metadata_filtered_from_client_metadata() -> None:
    """Verify AG-UI internal metadata is filtered before passing to chat client.

    AG-UI internal fields like 'ag_ui_thread_id', 'ag_ui_run_id', and 'current_state'
    are used for orchestration tracking but should NOT be passed to chat clients
    (e.g., Anthropic API only accepts 'user_id' in metadata).
    """
    import inspect

    # Get the source code of the DefaultOrchestrator.run method
    source = inspect.getsource(DefaultOrchestrator.run)

    # Verify the AG-UI internal metadata keys are defined
    assert "AG_UI_INTERNAL_METADATA_KEYS" in source, (
        "Expected AG_UI_INTERNAL_METADATA_KEYS to be defined for filtering internal metadata."
    )

    # Verify the internal keys include the AG-UI specific fields
    assert '"ag_ui_thread_id"' in source, "Expected 'ag_ui_thread_id' to be filtered"
    assert '"ag_ui_run_id"' in source, "Expected 'ag_ui_run_id' to be filtered"
    assert '"current_state"' in source, "Expected 'current_state' to be filtered"

    # Verify client_metadata is used instead of safe_metadata for options
    assert "client_metadata = {k: v for k, v in safe_metadata.items()" in source, (
        "Expected client_metadata to be created by filtering safe_metadata."
    )
    assert '"options": {"metadata": client_metadata}' in source, (
        "Expected client_metadata (not safe_metadata) to be passed in options."
    )
