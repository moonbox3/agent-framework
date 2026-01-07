# Copyright (c) Microsoft. All rights reserved.

"""Orchestrators for multi-turn agent flows."""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ag_ui.core import (
    BaseEvent,
    MessagesSnapshotEvent,
    RunErrorEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from agent_framework import (
    AgentProtocol,
    AgentThread,
    ChatAgent,
    FunctionApprovalResponseContent,
    FunctionCallContent,
    FunctionResultContent,
    TextContent,
)
from agent_framework._middleware import extract_and_merge_function_middleware
from agent_framework._tools import (
    FunctionInvocationConfiguration,
    _collect_approval_responses,
    _replace_approval_contents_with_results,
    _try_execute_function_calls,
)

from ._utils import convert_agui_tools_to_agent_framework, generate_event_id

if TYPE_CHECKING:
    from ._agent import AgentConfig
    from ._confirmation_strategies import ConfirmationStrategy


logger = logging.getLogger(__name__)


class ExecutionContext:
    """Shared context for orchestrators."""

    def __init__(
        self,
        input_data: dict[str, Any],
        agent: AgentProtocol,
        config: "AgentConfig",  # noqa: F821
        confirmation_strategy: "ConfirmationStrategy | None" = None,  # noqa: F821
    ):
        """Initialize execution context.

        Args:
            input_data: AG-UI run input containing messages, state, etc.
            agent: The Agent Framework agent to execute
            config: Agent configuration
            confirmation_strategy: Strategy for generating confirmation messages
        """
        self.input_data = input_data
        self.agent = agent
        self.config = config
        self.confirmation_strategy = confirmation_strategy

        # Lazy-loaded properties
        self._messages = None
        self._last_message = None
        self._run_id: str | None = None
        self._thread_id: str | None = None

    @property
    def messages(self):
        """Get converted Agent Framework messages (lazy loaded)."""
        if self._messages is None:
            from ._message_adapters import agui_messages_to_agent_framework

            raw = self.input_data.get("messages", [])
            self._messages = agui_messages_to_agent_framework(raw)
        return self._messages

    @property
    def last_message(self):
        """Get the last message in the conversation (lazy loaded)."""
        if self._last_message is None and self.messages:
            self._last_message = self.messages[-1]
        return self._last_message

    @property
    def run_id(self) -> str:
        """Get or generate run ID."""
        if self._run_id is None:
            self._run_id = self.input_data.get("run_id") or self.input_data.get("runId") or str(uuid.uuid4())
        # This should never be None after the if block above, but satisfy type checkers
        if self._run_id is None:  # pragma: no cover
            raise RuntimeError("Failed to initialize run_id")
        return self._run_id

    @property
    def thread_id(self) -> str:
        """Get or generate thread ID."""
        if self._thread_id is None:
            self._thread_id = self.input_data.get("thread_id") or self.input_data.get("threadId") or str(uuid.uuid4())
        # This should never be None after the if block above, but satisfy type checkers
        if self._thread_id is None:  # pragma: no cover
            raise RuntimeError("Failed to initialize thread_id")
        return self._thread_id


class Orchestrator(ABC):
    """Base orchestrator for agent execution flows."""

    @abstractmethod
    def can_handle(self, context: ExecutionContext) -> bool:
        """Determine if this orchestrator handles the current request.

        Args:
            context: Execution context with input data and agent

        Returns:
            True if this orchestrator should handle the request
        """
        ...

    @abstractmethod
    async def run(
        self,
        context: ExecutionContext,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Execute the orchestration and yield events.

        Args:
            context: Execution context

        Yields:
            AG-UI events
        """
        # This is never executed - just satisfies mypy's requirement for async generators
        if False:  # pragma: no cover
            yield
        raise NotImplementedError


class HumanInTheLoopOrchestrator(Orchestrator):
    """Handles tool approval responses from user."""

    def can_handle(self, context: ExecutionContext) -> bool:
        """Check if last message is a tool approval response.

        Args:
            context: Execution context

        Returns:
            True if last message is a tool result
        """
        msg = context.last_message
        if not msg:
            return False

        return bool(msg.additional_properties.get("is_tool_result", False))

    async def run(
        self,
        context: ExecutionContext,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Process approval response and generate confirmation events.

        This implementation is extracted from the legacy _agent.py lines 144-244.

        Args:
            context: Execution context

        Yields:
            AG-UI events (TextMessage, RunFinished)
        """
        from ._confirmation_strategies import DefaultConfirmationStrategy
        from ._events import AgentFrameworkEventBridge

        logger.info("=== TOOL RESULT DETECTED (HumanInTheLoopOrchestrator) ===")

        # Create event bridge for run events
        event_bridge = AgentFrameworkEventBridge(
            run_id=context.run_id,
            thread_id=context.thread_id,
        )

        # CRITICAL: Every AG-UI run must start with RunStartedEvent
        yield event_bridge.create_run_started_event()

        # Get confirmation strategy (use default if none provided)
        strategy = context.confirmation_strategy
        if strategy is None:
            strategy = DefaultConfirmationStrategy()

        # Parse the tool result content
        tool_content_text = ""
        last_message = context.last_message
        if last_message:
            for content in last_message.contents:
                if isinstance(content, TextContent):
                    tool_content_text = content.text
                    break

        try:
            tool_result = json.loads(tool_content_text)
            accepted = tool_result.get("accepted", False)
            steps = tool_result.get("steps", [])

            logger.info(f"  Accepted: {accepted}")
            logger.info(f"  Steps count: {len(steps)}")

            # Emit a text message confirming execution
            message_id = generate_event_id()

            yield TextMessageStartEvent(message_id=message_id, role="assistant")

            # Check if this is confirm_changes (no steps) or function approval (has steps)
            if not steps:
                # This is confirm_changes for predictive state updates
                if accepted:
                    confirmation_message = strategy.on_state_confirmed()
                else:
                    confirmation_message = strategy.on_state_rejected()
            elif accepted:
                # User approved - execute the enabled steps (function approval flow)
                confirmation_message = strategy.on_approval_accepted(steps)
            else:
                # User rejected
                confirmation_message = strategy.on_approval_rejected(steps)

            yield TextMessageContentEvent(
                message_id=message_id,
                delta=confirmation_message,
            )

            yield TextMessageEndEvent(message_id=message_id)

            # Emit run finished
            yield event_bridge.create_run_finished_event()

        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool result: {tool_content_text}")
            yield RunErrorEvent(message=f"Invalid tool result format: {tool_content_text[:100]}")
            yield event_bridge.create_run_finished_event()


class DefaultOrchestrator(Orchestrator):
    """Standard agent execution (no special handling)."""

    def can_handle(self, context: ExecutionContext) -> bool:
        """Always returns True as this is the fallback orchestrator.

        Args:
            context: Execution context

        Returns:
            Always True
        """
        return True

    async def run(
        self,
        context: ExecutionContext,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Standard agent run with event translation.

        This implements the default agent execution flow using the event bridge
        to translate Agent Framework events to AG-UI events.

        Args:
            context: Execution context

        Yields:
            AG-UI events
        """
        from ._events import AgentFrameworkEventBridge
        from ._message_adapters import agui_messages_to_snapshot_format
        from ._orchestration._message_hygiene import deduplicate_messages, sanitize_tool_history
        from ._orchestration._state_manager import StateManager
        from ._orchestration._tooling import (
            collect_server_tools,
            merge_tools,
            register_additional_client_tools,
        )

        logger.info(f"Starting default agent run for thread_id={context.thread_id}, run_id={context.run_id}")

        response_format = None
        if isinstance(context.agent, ChatAgent):
            response_format = context.agent.chat_options.response_format
        skip_text_content = response_format is not None

        state_manager = StateManager(
            state_schema=context.config.state_schema,
            predict_state_config=context.config.predict_state_config,
            require_confirmation=context.config.require_confirmation,
        )
        current_state = state_manager.initialize(context.input_data.get("state"))

        event_bridge = AgentFrameworkEventBridge(
            run_id=context.run_id,
            thread_id=context.thread_id,
            predict_state_config=context.config.predict_state_config,
            current_state=current_state,
            skip_text_content=skip_text_content,
            input_messages=context.input_data.get("messages", []),
            require_confirmation=context.config.require_confirmation,
        )

        yield event_bridge.create_run_started_event()

        predict_event = state_manager.predict_state_event()
        if predict_event:
            yield predict_event

        snapshot_event = state_manager.initial_snapshot_event(event_bridge)
        if snapshot_event:
            yield snapshot_event

        thread = AgentThread()
        thread.metadata = {  # type: ignore[attr-defined]
            "ag_ui_thread_id": context.thread_id,
            "ag_ui_run_id": context.run_id,
        }
        if current_state:
            thread.metadata["current_state"] = current_state  # type: ignore[attr-defined]

        raw_messages = context.messages or []
        if not raw_messages:
            logger.warning("No messages provided in AG-UI input")
            yield event_bridge.create_run_finished_event()
            return

        logger.info(f"Received {len(raw_messages)} raw messages from client")
        for i, msg in enumerate(raw_messages):
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            msg_id = getattr(msg, "message_id", None)
            logger.info(f"  Raw message {i}: role={role}, id={msg_id}")
            if hasattr(msg, "contents") and msg.contents:
                for j, content in enumerate(msg.contents):
                    content_type = type(content).__name__
                    if isinstance(content, TextContent):
                        logger.debug("    Content %s: %s - text_length=%s", j, content_type, len(content.text))
                    elif isinstance(content, FunctionCallContent):
                        arg_length = len(str(content.arguments)) if content.arguments else 0
                        logger.debug(
                            "    Content %s: %s - %s args_length=%s", j, content_type, content.name, arg_length
                        )
                    elif isinstance(content, FunctionResultContent):
                        result_preview = type(content.result).__name__ if content.result is not None else "None"
                        logger.debug(
                            "    Content %s: %s - call_id=%s, result_type=%s",
                            j,
                            content_type,
                            content.call_id,
                            result_preview,
                        )
                    else:
                        logger.debug(f"    Content {j}: {content_type}")

        sanitized_messages = sanitize_tool_history(raw_messages)
        provider_messages = deduplicate_messages(sanitized_messages)

        if not provider_messages:
            logger.info("No provider-eligible messages after filtering; finishing run without invoking agent.")
            yield event_bridge.create_run_finished_event()
            return

        logger.info(f"Processing {len(provider_messages)} provider messages after sanitization/deduplication")
        for i, msg in enumerate(provider_messages):
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            logger.info(f"  Message {i}: role={role}")
            if hasattr(msg, "contents") and msg.contents:
                for j, content in enumerate(msg.contents):
                    content_type = type(content).__name__
                    if isinstance(content, TextContent):
                        logger.info(f"    Content {j}: {content_type} - text_length={len(content.text)}")
                    elif isinstance(content, FunctionCallContent):
                        arg_length = len(str(content.arguments)) if content.arguments else 0
                        logger.info("    Content %s: %s - %s args_length=%s", j, content_type, content.name, arg_length)
                    elif isinstance(content, FunctionResultContent):
                        result_preview = type(content.result).__name__ if content.result is not None else "None"
                        logger.info(
                            "    Content %s: %s - call_id=%s, result_type=%s",
                            j,
                            content_type,
                            content.call_id,
                            result_preview,
                        )
                    else:
                        logger.info(f"    Content {j}: {content_type}")

        # Check for FunctionApprovalResponseContent and emit updated state snapshot
        # This ensures the UI shows the approved state (e.g., 2 steps) not the original (3 steps)
        for msg in provider_messages:
            role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            if role_value != "user":
                continue
            for content in msg.contents or []:
                if type(content) is FunctionApprovalResponseContent:
                    if content.function_call and content.approved:
                        parsed_args = content.function_call.parse_arguments()
                        state_args = None
                        if content.additional_properties:
                            state_args = content.additional_properties.get("ag_ui_state_args")
                        if not isinstance(state_args, dict):
                            state_args = parsed_args
                        if state_args and context.config.predict_state_config:
                            for state_key, config in context.config.predict_state_config.items():
                                if config["tool"] == content.function_call.name:
                                    tool_arg_name = config["tool_argument"]
                                    if tool_arg_name == "*":
                                        state_value = state_args
                                    elif isinstance(state_args, dict) and tool_arg_name in state_args:
                                        state_value = state_args[tool_arg_name]
                                    else:
                                        continue
                                    # Update current_state and emit snapshot
                                    current_state[state_key] = state_value
                                    event_bridge.current_state[state_key] = state_value
                                    logger.info(
                                        f"Emitting StateSnapshotEvent for approved state key '{state_key}' "
                                        f"with {len(state_value) if isinstance(state_value, list) else 'N/A'} items"
                                    )
                                    yield StateSnapshotEvent(snapshot=current_state)
                                    break

        messages_to_run: list[Any] = []
        is_new_user_turn = False
        if provider_messages:
            last_msg = provider_messages[-1]
            role_value = last_msg.role.value if hasattr(last_msg.role, "value") else str(last_msg.role)
            is_new_user_turn = role_value == "user"

        def _tool_calls_match_state() -> bool:
            if not state_manager.predict_state_config or not state_manager.current_state:
                return False

            def _parse_args(arguments: Any) -> dict[str, Any] | None:
                if isinstance(arguments, dict):
                    return arguments
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                    except json.JSONDecodeError:
                        return None
                    if isinstance(parsed, dict):
                        return parsed
                return None

            for state_key, config in state_manager.predict_state_config.items():
                tool_name = config["tool"]
                tool_arg_name = config["tool_argument"]
                tool_args: dict[str, Any] | None = None

                for msg in reversed(provider_messages):
                    role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    if role_value != "assistant":
                        continue
                    for content in msg.contents or []:
                        if isinstance(content, FunctionCallContent) and content.name == tool_name:
                            tool_args = _parse_args(content.arguments)
                            break
                    if tool_args is not None:
                        break

                if not tool_args:
                    return False

                if tool_arg_name == "*":
                    state_value = tool_args
                elif tool_arg_name in tool_args:
                    state_value = tool_args[tool_arg_name]
                else:
                    return False

                if state_manager.current_state.get(state_key) != state_value:
                    return False

            return True

        conversation_has_tool_calls = _tool_calls_match_state()

        state_context_msg = state_manager.state_context_message(
            is_new_user_turn=is_new_user_turn, conversation_has_tool_calls=conversation_has_tool_calls
        )

        def _is_state_context_message(message: Any) -> bool:
            role_value = message.role.value if hasattr(message.role, "value") else str(message.role)
            if role_value != "system":
                return False
            for content in message.contents or []:
                if isinstance(content, TextContent) and content.text.startswith("Current state of the application:"):
                    return True
            return False

        def _pending_tool_call_ids(messages: list[Any]) -> set[str]:
            pending_ids: set[str] = set()
            resolved_ids: set[str] = set()
            for msg in messages:
                for content in msg.contents or []:
                    if isinstance(content, FunctionCallContent) and content.call_id:
                        pending_ids.add(str(content.call_id))
                    elif isinstance(content, FunctionResultContent) and content.call_id:
                        resolved_ids.add(str(content.call_id))
            return pending_ids - resolved_ids

        if state_context_msg:
            messages_to_run = [msg for msg in provider_messages if not _is_state_context_message(msg)]
            if not _pending_tool_call_ids(messages_to_run):
                insert_index = len(messages_to_run) - 1 if is_new_user_turn else len(messages_to_run)
                if insert_index < 0:
                    insert_index = 0
                messages_to_run.insert(insert_index, state_context_msg)
        else:
            messages_to_run.extend(provider_messages)

        client_tools = convert_agui_tools_to_agent_framework(context.input_data.get("tools"))
        logger.info(f"[TOOLS] Client sent {len(client_tools) if client_tools else 0} tools")
        if client_tools:
            for tool in client_tools:
                tool_name = getattr(tool, "name", "unknown")
                declaration_only = getattr(tool, "declaration_only", None)
                logger.info(f"[TOOLS]   - Client tool: {tool_name}, declaration_only={declaration_only}")

        server_tools = collect_server_tools(context.agent)
        register_additional_client_tools(context.agent, client_tools)
        tools_param = merge_tools(server_tools, client_tools)

        all_updates: list[Any] = []
        update_count = 0
        # Prepare metadata for chat client (Azure requires string values)
        safe_metadata: dict[str, Any] = {}
        thread_metadata = getattr(thread, "metadata", None)
        if thread_metadata:
            for key, value in thread_metadata.items():
                value_str = value if isinstance(value, str) else json.dumps(value)
                if len(value_str) > 512:
                    value_str = value_str[:512]
                safe_metadata[key] = value_str

        run_kwargs: dict[str, Any] = {
            "thread": thread,
            "tools": tools_param,
            "metadata": safe_metadata,
        }
        if safe_metadata:
            run_kwargs["store"] = True

        async def _resolve_approval_responses(
            messages: list[Any],
            tools_for_execution: list[Any],
        ) -> None:
            fcc_todo = _collect_approval_responses(messages)
            if not fcc_todo:
                return

            approved_responses = [resp for resp in fcc_todo.values() if resp.approved]
            approved_function_results: list[Any] = []
            if approved_responses and tools_for_execution:
                chat_client = getattr(context.agent, "chat_client", None)
                config = (
                    getattr(chat_client, "function_invocation_configuration", None) or FunctionInvocationConfiguration()
                )
                middleware_pipeline = extract_and_merge_function_middleware(chat_client, run_kwargs)
                try:
                    results, _ = await _try_execute_function_calls(
                        custom_args=run_kwargs,
                        attempt_idx=0,
                        function_calls=approved_responses,
                        tools=tools_for_execution,
                        middleware_pipeline=middleware_pipeline,
                        config=config,
                    )
                    approved_function_results = list(results)
                except Exception:
                    logger.error("Failed to execute approved tool calls; injecting error results.")
                    approved_function_results = []

            normalized_results: list[FunctionResultContent] = []
            for idx, approval in enumerate(approved_responses):
                if idx < len(approved_function_results) and isinstance(
                    approved_function_results[idx], FunctionResultContent
                ):
                    normalized_results.append(approved_function_results[idx])
                    continue
                call_id = approval.function_call.call_id or approval.id
                normalized_results.append(
                    FunctionResultContent(call_id=call_id, result="Error: Tool call invocation failed.")
                )

            _replace_approval_contents_with_results(messages, fcc_todo, normalized_results)  # type: ignore

        await _resolve_approval_responses(messages_to_run, server_tools)

        async for update in context.agent.run_stream(messages_to_run, **run_kwargs):
            update_count += 1
            logger.info(f"[STREAM] Received update #{update_count} from agent")
            all_updates.append(update)
            if event_bridge.current_message_id is None and update.contents:
                has_tool_call = any(isinstance(content, FunctionCallContent) for content in update.contents)
                has_text = any(isinstance(content, TextContent) for content in update.contents)
                if has_tool_call and not has_text:
                    tool_message_id = generate_event_id()
                    event_bridge.current_message_id = tool_message_id
                    logger.info(
                        "[STREAM] Emitting TextMessageStartEvent for tool-only response message_id=%s",
                        tool_message_id,
                    )
                    yield TextMessageStartEvent(message_id=tool_message_id, role="assistant")
            events = await event_bridge.from_agent_run_update(update)
            logger.info(f"[STREAM] Update #{update_count} produced {len(events)} events")
            for event in events:
                logger.info(f"[STREAM] Yielding event: {type(event).__name__}")
                yield event

        logger.info(f"[STREAM] Agent stream completed. Total updates: {update_count}")

        if event_bridge.should_stop_after_confirm:
            logger.info("Stopping run after confirm_changes - waiting for user response")
            yield event_bridge.create_run_finished_event()
            return

        if event_bridge.pending_tool_calls:
            pending_without_end = [
                tc for tc in event_bridge.pending_tool_calls if tc.get("id") not in event_bridge.tool_calls_ended
            ]
            if pending_without_end:
                logger.info(
                    "Found %s pending tool calls without end event - emitting ToolCallEndEvent",
                    len(pending_without_end),
                )
                for tool_call in pending_without_end:
                    tool_call_id = tool_call.get("id")
                    if tool_call_id:
                        from ag_ui.core import ToolCallEndEvent

                        end_event = ToolCallEndEvent(tool_call_id=tool_call_id)
                        logger.info(f"Emitting ToolCallEndEvent for declaration-only tool call '{tool_call_id}'")
                        yield end_event

        if all_updates and response_format:
            from agent_framework import AgentRunResponse
            from pydantic import BaseModel

            logger.info(f"Processing structured output, update count: {len(all_updates)}")
            final_response = AgentRunResponse.from_agent_run_response_updates(
                all_updates, output_format_type=response_format
            )

            if final_response.value and isinstance(final_response.value, BaseModel):
                response_dict = final_response.value.model_dump(mode="json", exclude_none=True)
                logger.info(f"Received structured output keys: {list(response_dict.keys())}")

                state_updates = state_manager.extract_state_updates(response_dict)
                if state_updates:
                    state_manager.apply_state_updates(state_updates)
                    state_snapshot = event_bridge.create_state_snapshot_event(current_state)
                    yield state_snapshot
                    logger.info(f"Emitted StateSnapshotEvent with updates: {list(state_updates.keys())}")

                if "message" in response_dict and response_dict["message"]:
                    message_id = generate_event_id()
                    yield TextMessageStartEvent(message_id=message_id, role="assistant")
                    yield TextMessageContentEvent(message_id=message_id, delta=response_dict["message"])
                    yield TextMessageEndEvent(message_id=message_id)
                    logger.info(f"Emitted conversational message with length={len(response_dict['message'])}")

        logger.info(f"[FINALIZE] Checking for unclosed message. current_message_id={event_bridge.current_message_id}")
        if event_bridge.current_message_id:
            logger.info(f"[FINALIZE] Emitting TextMessageEndEvent for message_id={event_bridge.current_message_id}")
            yield event_bridge.create_message_end_event(event_bridge.current_message_id)

            has_text_content = bool(event_bridge.accumulated_text_content)
            assistant_text_message = {
                "id": event_bridge.current_message_id,
                "role": "assistant",
                "content": event_bridge.accumulated_text_content,
            }

            converted_input_messages = agui_messages_to_snapshot_format(event_bridge.input_messages)
            all_messages = converted_input_messages.copy()

            if event_bridge.pending_tool_calls:
                tool_call_message = {
                    "id": event_bridge.current_message_id if not has_text_content else generate_event_id(),
                    "role": "assistant",
                    "tool_calls": event_bridge.pending_tool_calls.copy(),
                }
                all_messages.append(tool_call_message)

            all_messages.extend(event_bridge.tool_results.copy())
            if has_text_content:
                all_messages.append(assistant_text_message)

            messages_snapshot = MessagesSnapshotEvent(
                messages=all_messages,  # type: ignore[arg-type]
            )
            logger.info(
                "[FINALIZE] Emitting MessagesSnapshotEvent with %s messages (text content length: %s)",
                len(all_messages),
                len(event_bridge.accumulated_text_content),
            )
            yield messages_snapshot
        else:
            logger.info("[FINALIZE] No current_message_id - skipping TextMessageEndEvent")
            if not event_bridge.messages_snapshot_emitted and (
                event_bridge.pending_tool_calls or event_bridge.tool_results
            ):
                converted_input_messages = agui_messages_to_snapshot_format(event_bridge.input_messages)
                all_messages = converted_input_messages.copy()

                if event_bridge.pending_tool_calls:
                    tool_call_message = {
                        "id": generate_event_id(),
                        "role": "assistant",
                        "tool_calls": event_bridge.pending_tool_calls.copy(),
                    }
                    all_messages.append(tool_call_message)

                all_messages.extend(event_bridge.tool_results.copy())

                messages_snapshot = MessagesSnapshotEvent(
                    messages=all_messages,  # type: ignore[arg-type]
                )
                logger.info(f"[FINALIZE] Emitting MessagesSnapshotEvent with {len(all_messages)} messages (tool-only)")
                yield messages_snapshot

        logger.info("[FINALIZE] Emitting RUN_FINISHED event")
        yield event_bridge.create_run_finished_event()
        logger.info(f"Completed agent run for thread_id={context.thread_id}, run_id={context.run_id}")


__all__ = [
    "Orchestrator",
    "ExecutionContext",
    "HumanInTheLoopOrchestrator",
    "DefaultOrchestrator",
]
