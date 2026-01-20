# Copyright (c) Microsoft. All rights reserved.

"""Simplified AG-UI orchestration - single linear flow."""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    MessagesSnapshotEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from agent_framework import (
    AgentProtocol,
    AgentThread,
    ChatMessage,
    FunctionApprovalRequestContent,
    FunctionCallContent,
    FunctionResultContent,
    TextContent,
    prepare_function_call_results,
)

from ._message_adapters import normalize_agui_input_messages
from ._orchestration._predictive_state import PredictiveStateHandler
from ._orchestration._tooling import collect_server_tools, merge_tools, register_additional_client_tools
from ._utils import convert_agui_tools_to_agent_framework, generate_event_id, get_conversation_id_from_update

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ._agent import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class FlowState:
    """Minimal explicit state for a single AG-UI run."""

    message_id: str | None = None  # Current text message being streamed
    tool_call_id: str | None = None  # Current tool call being streamed
    tool_call_name: str | None = None  # Name of current tool call
    waiting_for_approval: bool = False  # Stop after approval request
    current_state: dict[str, Any] = field(default_factory=dict)  # Shared state
    accumulated_text: str = ""  # For MessagesSnapshotEvent
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)  # For MessagesSnapshotEvent
    tool_calls_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


def _create_state_context_message(
    current_state: dict[str, Any],
    state_schema: dict[str, Any],
) -> ChatMessage | None:
    """Create a system message with current state context.

    This injects the current state into the conversation so the model
    knows what state exists and can make informed updates.

    Args:
        current_state: The current state to inject
        state_schema: The state schema (used to determine if injection is needed)

    Returns:
        ChatMessage with state context, or None if not needed
    """
    if not current_state or not state_schema:
        return None

    state_json = json.dumps(current_state, indent=2)
    return ChatMessage(
        role="system",
        contents=[
            TextContent(
                text=(
                    "Current state of the application:\n"
                    f"{state_json}\n\n"
                    "When modifying state, you MUST include ALL existing data plus your changes.\n"
                    "For example, if adding one new item to a list, include ALL existing items PLUS the new item.\n"
                    "Never replace existing data - always preserve and append or merge."
                )
            )
        ],
    )


def _inject_state_context(
    messages: list[ChatMessage],
    current_state: dict[str, Any],
    state_schema: dict[str, Any],
) -> list[ChatMessage]:
    """Inject state context message into messages if appropriate.

    The state context is injected before the last user message to give
    the model visibility into the current application state.

    Args:
        messages: The messages to potentially inject into
        current_state: The current state
        state_schema: The state schema

    Returns:
        Messages with state context injected if appropriate
    """
    state_msg = _create_state_context_message(current_state, state_schema)
    if not state_msg:
        return messages

    # Check if the last message is from a user (new user turn)
    if not messages:
        return messages

    from ._utils import get_role_value

    last_role = get_role_value(messages[-1])
    if last_role != "user":
        return messages

    # Always inject state context if state is provided
    # This ensures UI state changes are visible to the model

    # Insert state context before the last user message
    result = list(messages[:-1])
    result.append(state_msg)
    result.append(messages[-1])
    return result


def _emit_text(content: TextContent, flow: FlowState, skip_text: bool = False) -> list[BaseEvent]:
    """Emit TextMessage events for TextContent."""
    if not content.text:
        return []

    # Skip if we're in structured output mode or waiting for approval
    if skip_text or flow.waiting_for_approval:
        return []

    events: list[BaseEvent] = []
    if not flow.message_id:
        flow.message_id = generate_event_id()
        events.append(TextMessageStartEvent(message_id=flow.message_id, role="assistant"))

    events.append(TextMessageContentEvent(message_id=flow.message_id, delta=content.text))
    flow.accumulated_text += content.text
    return events


def _emit_tool_call(
    content: FunctionCallContent,
    flow: FlowState,
    predictive_handler: PredictiveStateHandler | None = None,
) -> list[BaseEvent]:
    """Emit ToolCall events for FunctionCallContent."""
    events: list[BaseEvent] = []

    tool_call_id = content.call_id or flow.tool_call_id or generate_event_id()

    # Emit start event when we have a new tool call
    if content.name and tool_call_id != flow.tool_call_id:
        flow.tool_call_id = tool_call_id
        flow.tool_call_name = content.name
        if predictive_handler:
            predictive_handler.reset_streaming()

        events.append(
            ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_call_name=content.name,
                parent_message_id=flow.message_id,
            )
        )

        # Track for MessagesSnapshotEvent
        tool_entry = {
            "id": tool_call_id,
            "type": "function",
            "function": {"name": content.name, "arguments": ""},
        }
        flow.pending_tool_calls.append(tool_entry)
        flow.tool_calls_by_id[tool_call_id] = tool_entry

    elif tool_call_id:
        flow.tool_call_id = tool_call_id

    # Emit args if present
    if content.arguments:
        delta = content.arguments if isinstance(content.arguments, str) else json.dumps(content.arguments)
        events.append(ToolCallArgsEvent(tool_call_id=tool_call_id, delta=delta))

        # Track args for MessagesSnapshotEvent
        if tool_call_id in flow.tool_calls_by_id:
            flow.tool_calls_by_id[tool_call_id]["function"]["arguments"] += delta

        # Emit predictive state deltas
        if predictive_handler and flow.tool_call_name:
            delta_events = predictive_handler.emit_streaming_deltas(flow.tool_call_name, delta)
            events.extend(delta_events)

    return events


def _emit_tool_result(
    content: FunctionResultContent,
    flow: FlowState,
    predictive_handler: PredictiveStateHandler | None = None,
) -> list[BaseEvent]:
    """Emit ToolCallResult events for FunctionResultContent."""
    events: list[BaseEvent] = []

    if content.call_id:
        events.append(ToolCallEndEvent(tool_call_id=content.call_id))

    result_content = prepare_function_call_results(content.result)
    message_id = generate_event_id()
    events.append(
        ToolCallResultEvent(
            message_id=message_id,
            tool_call_id=content.call_id,
            content=result_content,
            role="tool",
        )
    )

    # Track for MessagesSnapshotEvent
    flow.tool_results.append(
        {
            "id": message_id,
            "role": "tool",
            "toolCallId": content.call_id,
            "content": result_content,
        }
    )

    # Apply predictive state updates and emit snapshot
    if predictive_handler:
        predictive_handler.apply_pending_updates()
        if flow.current_state:
            events.append(StateSnapshotEvent(snapshot=flow.current_state))

    # Reset tool tracking
    flow.tool_call_id = None
    flow.tool_call_name = None

    return events


def _emit_approval_request(
    content: FunctionApprovalRequestContent,
    flow: FlowState,
    predictive_handler: PredictiveStateHandler | None = None,
    require_confirmation: bool = True,
) -> list[BaseEvent]:
    """Emit events for function approval request."""
    events: list[BaseEvent] = []

    # Extract state from function arguments if predictive
    if predictive_handler:
        parsed_args = content.function_call.parse_arguments()
        result = predictive_handler.extract_state_value(content.function_call.name, parsed_args)
        if result:
            state_key, state_value = result
            flow.current_state[state_key] = state_value
            events.append(StateSnapshotEvent(snapshot=flow.current_state))

    # End the original tool call
    if content.function_call.call_id:
        events.append(ToolCallEndEvent(tool_call_id=content.function_call.call_id))

    # Emit custom event for UI
    events.append(
        CustomEvent(
            name="function_approval_request",
            value={
                "id": content.id,
                "function_call": {
                    "call_id": content.function_call.call_id,
                    "name": content.function_call.name,
                    "arguments": content.function_call.parse_arguments(),
                },
            },
        )
    )

    # Emit confirm_changes tool call for UI compatibility
    if require_confirmation:
        confirm_id = generate_event_id()
        events.append(
            ToolCallStartEvent(
                tool_call_id=confirm_id,
                tool_call_name="confirm_changes",
                parent_message_id=flow.message_id,
            )
        )
        args = {
            "function_name": content.function_call.name,
            "function_call_id": content.function_call.call_id,
            "function_arguments": content.function_call.parse_arguments() or {},
            "steps": [{"description": f"Execute {content.function_call.name}", "status": "enabled"}],
        }
        events.append(ToolCallArgsEvent(tool_call_id=confirm_id, delta=json.dumps(args)))
        events.append(ToolCallEndEvent(tool_call_id=confirm_id))

    flow.waiting_for_approval = True
    return events


def _emit_content(
    content: Any,
    flow: FlowState,
    predictive_handler: PredictiveStateHandler | None = None,
    skip_text: bool = False,
    require_confirmation: bool = True,
) -> list[BaseEvent]:
    """Emit appropriate events for any content type."""
    if isinstance(content, TextContent):
        return _emit_text(content, flow, skip_text)
    elif isinstance(content, FunctionCallContent):
        return _emit_tool_call(content, flow, predictive_handler)
    elif isinstance(content, FunctionResultContent):
        return _emit_tool_result(content, flow, predictive_handler)
    elif isinstance(content, FunctionApprovalRequestContent):
        return _emit_approval_request(content, flow, predictive_handler, require_confirmation)
    return []


def _is_confirm_changes_response(messages: list[Any]) -> bool:
    """Check if the last message is a confirm_changes tool result (state confirmation flow).

    This returns True for confirm_changes flows where we emit a confirmation message
    and stop. The key indicator is the presence of a 'steps' key in the tool result
    (even if empty), combined with 'accepted' boolean.
    """
    if not messages:
        return False
    last = messages[-1]
    if not last.additional_properties.get("is_tool_result", False):
        return False

    # Parse the content to check if it has the confirm_changes structure
    for content in last.contents:
        if isinstance(content, TextContent):
            try:
                result = json.loads(content.text)
                # confirm_changes results have 'accepted' and 'steps' keys
                if "accepted" in result and "steps" in result:
                    return True
            except json.JSONDecodeError:
                pass
    return False


def _handle_step_based_approval(messages: list[Any]) -> list[BaseEvent]:
    """Handle step-based approval response and emit confirmation message."""
    events: list[BaseEvent] = []
    last = messages[-1]

    # Parse the approval content
    approval_text = ""
    for content in last.contents:
        if isinstance(content, TextContent):
            approval_text = content.text
            break

    try:
        result = json.loads(approval_text)
        accepted = result.get("accepted", False)
        steps = result.get("steps", [])

        if accepted:
            # Generate acceptance message with step descriptions
            enabled_steps = [s for s in steps if s.get("status") == "enabled"]
            if enabled_steps:
                message_parts = [f"Executing {len(enabled_steps)} approved steps:\n\n"]
                for i, step in enumerate(enabled_steps, 1):
                    message_parts.append(f"{i}. {step.get('description', 'Step')}\n")
                message_parts.append("\nAll steps completed successfully!")
                message = "".join(message_parts)
            else:
                message = "Changes confirmed and applied successfully!"
        else:
            # Rejection message
            message = "No problem! What would you like me to change about the plan?"
    except json.JSONDecodeError:
        message = "Acknowledged."

    message_id = generate_event_id()
    events.append(TextMessageStartEvent(message_id=message_id, role="assistant"))
    events.append(TextMessageContentEvent(message_id=message_id, delta=message))
    events.append(TextMessageEndEvent(message_id=message_id))

    return events


async def _resolve_approval_responses(
    messages: list[Any],
    tools: list[Any],
    agent: AgentProtocol,
    run_kwargs: dict[str, Any],
) -> None:
    """Execute approved function calls and replace approval content with results.

    This modifies the messages list in place, replacing FunctionApprovalResponseContent
    with FunctionResultContent containing the actual tool execution result.

    Args:
        messages: List of messages (will be modified in place)
        tools: List of available tools
        agent: The agent instance (to get chat_client and config)
        run_kwargs: Kwargs for tool execution
    """
    from agent_framework._middleware import extract_and_merge_function_middleware
    from agent_framework._tools import (
        FunctionInvocationConfiguration,
        _collect_approval_responses,
        _replace_approval_contents_with_results,
        _try_execute_function_calls,
    )

    fcc_todo = _collect_approval_responses(messages)
    if not fcc_todo:
        return

    approved_responses = [resp for resp in fcc_todo.values() if resp.approved]
    rejected_responses = [resp for resp in fcc_todo.values() if not resp.approved]
    approved_function_results: list[Any] = []

    # Execute approved tool calls
    if approved_responses and tools:
        chat_client = getattr(agent, "chat_client", None)
        config = getattr(chat_client, "function_invocation_configuration", None) or FunctionInvocationConfiguration()
        middleware_pipeline = extract_and_merge_function_middleware(chat_client, run_kwargs)
        # Filter out AG-UI-specific kwargs that should not be passed to tool execution
        tool_kwargs = {k: v for k, v in run_kwargs.items() if k != "options"}
        try:
            results, _ = await _try_execute_function_calls(
                custom_args=tool_kwargs,
                attempt_idx=0,
                function_calls=approved_responses,
                tools=tools,
                middleware_pipeline=middleware_pipeline,
                config=config,
            )
            approved_function_results = list(results)
        except Exception:
            logger.error("Failed to execute approved tool calls; injecting error results.")
            approved_function_results = []

    # Build normalized results for approved responses
    normalized_results: list[FunctionResultContent] = []
    for idx, approval in enumerate(approved_responses):
        if idx < len(approved_function_results) and isinstance(approved_function_results[idx], FunctionResultContent):
            normalized_results.append(approved_function_results[idx])
            continue
        call_id = approval.function_call.call_id or approval.id
        normalized_results.append(FunctionResultContent(call_id=call_id, result="Error: Tool call invocation failed."))

    # Build rejection results
    for rejection in rejected_responses:
        call_id = rejection.function_call.call_id or rejection.id
        normalized_results.append(
            FunctionResultContent(call_id=call_id, result="Error: Tool call invocation was rejected by user.")
        )

    _replace_approval_contents_with_results(messages, fcc_todo, normalized_results)  # type: ignore


def _build_messages_snapshot(
    flow: FlowState,
    snapshot_messages: list[dict[str, Any]],
) -> MessagesSnapshotEvent:
    """Build MessagesSnapshotEvent from current flow state."""
    all_messages = list(snapshot_messages)

    # Add assistant message with tool calls
    if flow.pending_tool_calls:
        tool_call_message = {
            "id": flow.message_id or generate_event_id(),
            "role": "assistant",
            "tool_calls": flow.pending_tool_calls.copy(),
        }
        if flow.accumulated_text:
            tool_call_message["content"] = flow.accumulated_text
        all_messages.append(tool_call_message)

    # Add tool results
    all_messages.extend(flow.tool_results)

    # Add text-only assistant message if no tool calls
    if flow.accumulated_text and not flow.pending_tool_calls:
        all_messages.append(
            {
                "id": flow.message_id or generate_event_id(),
                "role": "assistant",
                "content": flow.accumulated_text,
            }
        )

    return MessagesSnapshotEvent(messages=all_messages)  # type: ignore[arg-type]


async def run_agent_stream(
    input_data: dict[str, Any],
    agent: AgentProtocol,
    config: "AgentConfig",
) -> "AsyncGenerator[BaseEvent, None]":
    """Run agent and yield AG-UI events.

    This is the single entry point for all AG-UI agent runs. It follows a simple
    linear flow: RunStarted -> content events -> RunFinished.

    Args:
        input_data: AG-UI request data with messages, state, tools, etc.
        agent: The Agent Framework agent to run
        config: Agent configuration

    Yields:
        AG-UI events
    """
    # Parse IDs
    thread_id = input_data.get("thread_id") or input_data.get("threadId") or str(uuid.uuid4())
    run_id = input_data.get("run_id") or input_data.get("runId") or str(uuid.uuid4())

    # Initialize flow state with schema defaults
    flow = FlowState()
    if input_data.get("state"):
        flow.current_state = dict(input_data["state"])

    # Apply schema defaults for missing state keys
    if config.state_schema:
        for key, schema in config.state_schema.items():
            if key in flow.current_state:
                continue
            if isinstance(schema, dict) and schema.get("type") == "array":
                flow.current_state[key] = []
            else:
                flow.current_state[key] = {}

    # Initialize predictive state handler if configured
    predictive_handler: PredictiveStateHandler | None = None
    if config.predict_state_config:
        predictive_handler = PredictiveStateHandler(
            predict_state_config=config.predict_state_config,
            current_state=flow.current_state,
        )

    # Normalize messages
    raw_messages = input_data.get("messages", [])
    messages, snapshot_messages = normalize_agui_input_messages(raw_messages)

    # Check for structured output mode (skip text content)
    skip_text = False
    response_format = None
    from agent_framework import ChatAgent

    if isinstance(agent, ChatAgent):
        response_format = agent.default_options.get("response_format")
        skip_text = response_format is not None

    # Handle empty messages (emit RunStarted immediately since no agent response)
    if not messages:
        logger.warning("No messages provided in AG-UI input")
        yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
        yield RunFinishedEvent(run_id=run_id, thread_id=thread_id)
        return

    # Prepare tools
    client_tools = convert_agui_tools_to_agent_framework(input_data.get("tools"))
    server_tools = collect_server_tools(agent)
    register_additional_client_tools(agent, client_tools)
    tools = merge_tools(server_tools, client_tools)

    # Create thread (with service thread support)
    if config.use_service_thread:
        supplied_thread_id = input_data.get("thread_id") or input_data.get("threadId")
        thread = AgentThread(service_thread_id=supplied_thread_id)
    else:
        thread = AgentThread()

    # Inject metadata for AG-UI orchestration
    thread.metadata = {  # type: ignore[attr-defined]
        "ag_ui_thread_id": thread_id,
        "ag_ui_run_id": run_id,
    }
    if flow.current_state:
        thread.metadata["current_state"] = flow.current_state  # type: ignore[attr-defined]

    # Build run kwargs
    run_kwargs: dict[str, Any] = {"thread": thread}
    if tools:
        run_kwargs["tools"] = tools

    # Resolve approval responses (execute approved tools, replace approvals with results)
    # This must happen before running the agent so it sees the tool results
    tools_for_execution = tools if tools is not None else server_tools
    await _resolve_approval_responses(messages, tools_for_execution, agent, run_kwargs)

    # Handle confirm_changes response (state confirmation flow - emit confirmation and stop)
    if _is_confirm_changes_response(messages):
        yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
        for event in _handle_step_based_approval(messages):
            yield event
        yield RunFinishedEvent(run_id=run_id, thread_id=thread_id)
        return

    # Inject state context message so the model knows current application state
    # This is critical for shared state scenarios where the UI state needs to be visible
    if config.state_schema and flow.current_state:
        messages = _inject_state_context(messages, flow.current_state, config.state_schema)

    # Stream from agent - emit RunStarted after first update to get service IDs
    run_started_emitted = False
    all_updates: list[Any] = []  # Collect for structured output processing
    async for update in agent.run_stream(messages, **run_kwargs):
        # Collect updates for structured output processing
        if response_format is not None:
            all_updates.append(update)

        # Update IDs from service response on first update and emit RunStarted
        if not run_started_emitted:
            conv_id = get_conversation_id_from_update(update)
            if conv_id:
                thread_id = conv_id
            if update.response_id:
                run_id = update.response_id
            # NOW emit RunStarted with proper IDs
            yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
            # Emit PredictState custom event if configured
            if config.predict_state_config:
                predict_state_value = [
                    {
                        "state_key": state_key,
                        "tool": cfg["tool"],
                        "tool_argument": cfg["tool_argument"],
                    }
                    for state_key, cfg in config.predict_state_config.items()
                ]
                yield CustomEvent(name="PredictState", value=predict_state_value)
            # Emit initial state snapshot only if we have both state_schema and state
            if config.state_schema and flow.current_state:
                yield StateSnapshotEvent(snapshot=flow.current_state)
            run_started_emitted = True

        # Emit events for each content item
        for content in update.contents:
            for event in _emit_content(
                content,
                flow,
                predictive_handler,
                skip_text,
                config.require_confirmation,
            ):
                yield event

        # Stop if waiting for approval
        if flow.waiting_for_approval:
            break

    # If no updates at all, still emit RunStarted
    if not run_started_emitted:
        yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
        if config.predict_state_config:
            predict_state_value = [
                {
                    "state_key": state_key,
                    "tool": cfg["tool"],
                    "tool_argument": cfg["tool_argument"],
                }
                for state_key, cfg in config.predict_state_config.items()
            ]
            yield CustomEvent(name="PredictState", value=predict_state_value)
        if config.state_schema and flow.current_state:
            yield StateSnapshotEvent(snapshot=flow.current_state)

    # Process structured output if response_format is set
    if response_format is not None and all_updates:
        from agent_framework import AgentResponse
        from pydantic import BaseModel

        logger.info(f"Processing structured output, update count: {len(all_updates)}")
        final_response = AgentResponse.from_agent_run_response_updates(all_updates, output_format_type=response_format)

        if final_response.value and isinstance(final_response.value, BaseModel):
            response_dict = final_response.value.model_dump(mode="json", exclude_none=True)
            logger.info(f"Received structured output keys: {list(response_dict.keys())}")

            # Extract state updates - if no state_schema, all non-message fields are state
            state_keys = (
                set(config.state_schema.keys()) if config.state_schema else set(response_dict.keys()) - {"message"}
            )
            state_updates = {k: v for k, v in response_dict.items() if k in state_keys}

            if state_updates:
                flow.current_state.update(state_updates)
                yield StateSnapshotEvent(snapshot=flow.current_state)
                logger.info(f"Emitted StateSnapshotEvent with updates: {list(state_updates.keys())}")

            # Emit message field as text if present
            if "message" in response_dict and response_dict["message"]:
                message_id = generate_event_id()
                yield TextMessageStartEvent(message_id=message_id, role="assistant")
                yield TextMessageContentEvent(message_id=message_id, delta=response_dict["message"])
                yield TextMessageEndEvent(message_id=message_id)
                logger.info(f"Emitted conversational message with length={len(response_dict['message'])}")

    # Close any open message
    if flow.message_id:
        yield TextMessageEndEvent(message_id=flow.message_id)

    # Emit MessagesSnapshotEvent if we have tool calls or results
    if flow.pending_tool_calls or flow.tool_results or flow.accumulated_text:
        yield _build_messages_snapshot(flow, snapshot_messages)

    yield RunFinishedEvent(run_id=run_id, thread_id=thread_id)


__all__ = ["FlowState", "run_agent_stream"]
