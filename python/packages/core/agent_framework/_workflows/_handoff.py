# Copyright (c) Microsoft. All rights reserved.

"""High-level builder for conversational handoff workflows.

The handoff pattern models a triage/dispatcher agent that optionally routes
control to specialist agents before handing the conversation back to the user.
The flow is intentionally cyclical:

    user input -> starting agent -> optional specialist -> request user input -> ...

Key properties:
- The entire conversation is maintained by default and reused on every hop
- Developers can opt into a rolling context window (last N messages)
- The starting agent determines whether to hand off by emitting metadata
- After a specialist responds, the workflow immediately requests new user input
"""

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_framework import AgentProtocol, AgentRunResponse, ChatMessage, Role

from ._agent_executor import AgentExecutor, AgentExecutorRequest, AgentExecutorResponse
from ._checkpoint import CheckpointStorage
from ._conversation_state import decode_chat_messages, encode_chat_messages
from ._executor import Executor, handler
from ._request_info_executor import RequestInfoExecutor, RequestInfoMessage, RequestResponse
from ._workflow import Workflow, WorkflowBuilder
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


_HANDOFF_HINT_KEYS = ("handoff_to", "handoff", "transfer_to", "agent_id", "agent")
_HANDOFF_TEXT_PATTERN = re.compile(r"handoff[_\s-]*to\s*:?\s*(?P<target>[\w-]+)", re.IGNORECASE)


@dataclass
class HandoffUserInputRequest(RequestInfoMessage):
    """Request message emitted when the workflow needs fresh user input."""

    conversation: list[ChatMessage] = field(default_factory=list)
    awaiting_agent_id: str | None = None
    prompt: str | None = None


@dataclass
class _ConversationWithUserInput:
    """Internal message carrying full conversation + new user messages from gateway to coordinator."""

    full_conversation: list[ChatMessage] = field(default_factory=list)


class _InputToConversation(Executor):
    """Normalises initial workflow input into a list[ChatMessage]."""

    @handler
    async def from_str(self, prompt: str, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        await ctx.send_message([ChatMessage(Role.USER, text=prompt)])

    @handler
    async def from_message(self, message: ChatMessage, ctx: WorkflowContext[list[ChatMessage]]) -> None:  # type: ignore[name-defined]
        await ctx.send_message([message])

    @handler
    async def from_messages(
        self,
        messages: list[ChatMessage],
        ctx: WorkflowContext[list[ChatMessage]],
    ) -> None:  # type: ignore[name-defined]
        await ctx.send_message(list(messages))


def _default_handoff_resolver(response: AgentExecutorResponse) -> str | None:
    """Extract a target specialist identifier from an agent response."""
    agent_response = response.agent_run_response

    # Structured value
    value = agent_response.value
    candidate = _extract_handoff_candidate(value)
    if candidate:
        return candidate

    # Additional properties on the response payload
    props = agent_response.additional_properties or {}
    candidate = _extract_from_mapping(props)
    if candidate:
        return candidate

    # Inspect most recent assistant message metadata
    for msg in reversed(agent_response.messages):
        props = getattr(msg, "additional_properties", {}) or {}
        candidate = _extract_from_mapping(props)
        if candidate:
            return candidate
        text = msg.text or ""
        match = _HANDOFF_TEXT_PATTERN.search(text)
        if match:
            parsed = match.group("target").strip()
            if parsed:
                return parsed

    return None


def _extract_handoff_candidate(candidate: Any) -> str | None:
    if candidate is None:
        return None
    if isinstance(candidate, str):
        return candidate.strip() or None
    if isinstance(candidate, Mapping):
        return _extract_from_mapping(candidate)
    attr = getattr(candidate, "handoff_to", None)
    if isinstance(attr, str) and attr.strip():
        return attr.strip()
    attr = getattr(candidate, "agent_id", None)
    if isinstance(attr, str) and attr.strip():
        return attr.strip()
    return None


def _extract_from_mapping(mapping: Mapping[str, Any]) -> str | None:
    for key in _HANDOFF_HINT_KEYS:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class _HandoffCoordinator(Executor):
    """Coordinates agent-to-agent transfers and user turn requests."""

    def __init__(
        self,
        *,
        starting_agent_id: str,
        specialist_ids: Mapping[str, str],
        input_gateway_id: str,
        context_window: int | None,
        resolver: Callable[[AgentExecutorResponse], str | AgentProtocol | Executor | None],
        termination_condition: Callable[[list[ChatMessage]], bool],
        id: str,
    ) -> None:
        super().__init__(id)
        self._starting_agent_id = starting_agent_id
        self._specialist_by_alias = dict(specialist_ids)
        self._specialist_ids = set(specialist_ids.values())
        self._input_gateway_id = input_gateway_id
        self._context_window = context_window
        self._resolver = resolver
        self._termination_condition = termination_condition
        self._full_conversation: list[ChatMessage] = []

    @handler
    async def handle_agent_response(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[AgentExecutorRequest | list[ChatMessage], list[ChatMessage]],
    ) -> None:
        # Hydrate coordinator state (and detect new run) using checkpointable executor state
        state = await ctx.get_state()
        if not state:
            self._full_conversation = []
        elif not self._full_conversation:
            restored = self._restore_conversation_from_state(state)
            if restored:
                self._full_conversation = restored

        source = ctx.get_source_executor_id()
        is_starting_agent = source == self._starting_agent_id

        # On first turn of a run, full_conversation is empty
        # On subsequent turns with context window, response.full_conversation may be trimmed
        # Solution: Track new messages only, build authoritative history incrementally
        if not self._full_conversation:
            # First response from starting agent - initialize with authoritative conversation snapshot
            self._full_conversation = self._conversation_from_response(response)
        else:
            # Subsequent responses - append only new messages from this agent
            new_messages = list(response.agent_run_response.messages)
            self._full_conversation.extend(new_messages)

        self._apply_response_metadata(self._full_conversation, response.agent_run_response)

        conversation = list(self._full_conversation)
        await self._persist_state(ctx)

        if is_starting_agent:
            target = self._resolve_specialist(response)
            if target is not None:
                trimmed = self._trim(conversation)
                request = AgentExecutorRequest(messages=trimmed, should_respond=True)
                await ctx.send_message(request, target_id=target)
                return

            # Check termination condition before requesting more user input
            if self._termination_condition(conversation):
                logger.info("Handoff workflow termination condition met. Ending conversation.")
                await ctx.yield_output(list(conversation))
                return

            await ctx.send_message(list(conversation), target_id=self._input_gateway_id)
            return

        if source not in self._specialist_ids:
            raise RuntimeError(f"HandoffCoordinator received response from unknown executor '{source}'.")

        # Check termination condition after specialist response
        if self._termination_condition(conversation):
            logger.info("Handoff workflow termination condition met. Ending conversation.")
            await ctx.yield_output(list(conversation))
            return

        await ctx.send_message(list(conversation), target_id=self._input_gateway_id)

    @handler
    async def handle_user_input(
        self,
        message: _ConversationWithUserInput,
        ctx: WorkflowContext[AgentExecutorRequest, list[ChatMessage]],
    ) -> None:
        """Receive full conversation with new user input from gateway, update history, trim for agent."""
        # Update authoritative full conversation
        self._full_conversation = list(message.full_conversation)
        await self._persist_state(ctx)

        # Check termination before sending to agent
        if self._termination_condition(self._full_conversation):
            logger.info("Handoff workflow termination condition met. Ending conversation.")
            await ctx.yield_output(list(self._full_conversation))
            return

        # Trim and send to starting agent
        trimmed = self._trim(self._full_conversation)
        request = AgentExecutorRequest(messages=trimmed, should_respond=True)
        await ctx.send_message(request, target_id=self._starting_agent_id)

    def _resolve_specialist(self, response: AgentExecutorResponse) -> str | None:
        try:
            resolved = self._resolver(response)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("handoff resolver raised %s", exc)
            return None

        if resolved is None:
            return None

        resolved_id: str | None
        if isinstance(resolved, Executor):
            resolved_id = resolved.id
        elif isinstance(resolved, AgentProtocol):
            name = getattr(resolved, "name", None)
            if name is None:
                raise ValueError("Resolver returned AgentProtocol without a name; cannot map to executor id.")
            resolved_id = self._specialist_by_alias.get(name) or name
        elif isinstance(resolved, str):
            resolved_id = self._specialist_by_alias.get(resolved)
            if resolved_id is None:
                lowered = resolved.lower()
                for alias, exec_id in self._specialist_by_alias.items():
                    if alias.lower() == lowered:
                        resolved_id = exec_id
                        break
            if resolved_id is None:
                resolved_id = resolved
        else:
            raise TypeError(
                f"Resolver must return Executor, AgentProtocol, str, or None. Got {type(resolved).__name__}."
            )

        if resolved_id not in self._specialist_ids:
            logger.warning("Resolver selected '%s' which is not a registered specialist.", resolved_id)
            return None
        return resolved_id

    def _conversation_from_response(self, response: AgentExecutorResponse) -> list[ChatMessage]:
        conversation = response.full_conversation
        if conversation is None:
            raise RuntimeError(
                "AgentExecutorResponse.full_conversation missing; AgentExecutor must populate it in handoff workflows."
            )
        return list(conversation)

    def _trim(self, conversation: list[ChatMessage]) -> list[ChatMessage]:
        if self._context_window is None:
            return list(conversation)
        return list(conversation[-self._context_window :])

    async def _persist_state(self, ctx: WorkflowContext[Any, Any]) -> None:
        """Store authoritative conversation snapshot without losing rich metadata."""
        state_payload = {"full_conversation": encode_chat_messages(self._full_conversation)}
        await ctx.set_state(state_payload)

    def _restore_conversation_from_state(self, state: Mapping[str, Any]) -> list[ChatMessage]:
        raw_conv = state.get("full_conversation")
        if not isinstance(raw_conv, list):
            return []
        return decode_chat_messages(raw_conv)

    def _apply_response_metadata(self, conversation: list[ChatMessage], agent_response: AgentRunResponse) -> None:
        if not agent_response.additional_properties:
            return

        # Find the most recent assistant message contributed by this response
        for message in reversed(conversation):
            if message.role == Role.ASSISTANT:
                metadata = agent_response.additional_properties or {}
                if not metadata:
                    return
                # Merge metadata without mutating shared dict from agent response
                merged = dict(message.additional_properties or {})
                for key, value in metadata.items():
                    merged.setdefault(key, value)
                message.additional_properties = merged
                break


class _UserInputGateway(Executor):
    """Bridges conversation context with RequestInfoExecutor and re-enters the loop."""

    def __init__(
        self,
        *,
        request_executor_id: str,
        starting_agent_id: str,
        prompt: str | None,
        id: str,
    ) -> None:
        super().__init__(id)
        self._request_executor_id = request_executor_id
        self._starting_agent_id = starting_agent_id
        self._prompt = prompt or "Provide your next input for the conversation."

    @handler
    async def request_input(
        self,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[HandoffUserInputRequest],
    ) -> None:
        if not conversation:
            raise ValueError("Handoff workflow requires non-empty conversation before requesting user input.")
        request = HandoffUserInputRequest(
            conversation=list(conversation),
            awaiting_agent_id=self._starting_agent_id,
            prompt=self._prompt,
        )
        request.source_executor_id = self.id
        await ctx.send_message(request, target_id=self._request_executor_id)

    @handler
    async def resume_from_user(
        self,
        response: RequestResponse[HandoffUserInputRequest, Any],
        ctx: WorkflowContext[_ConversationWithUserInput],
    ) -> None:
        # Reconstruct full conversation with new user input
        conversation = list(response.original_request.conversation)
        user_messages = _as_user_messages(response.data)
        conversation.extend(user_messages)

        # Send full conversation back to coordinator (not trimmed)
        # Coordinator will update its authoritative history and trim for agent
        message = _ConversationWithUserInput(full_conversation=conversation)
        # CRITICAL: Must specify target to avoid broadcasting to all connected executors
        # Gateway is connected to both request_info and coordinator, we want coordinator only
        await ctx.send_message(message, target_id="handoff-coordinator")


def _as_user_messages(payload: Any) -> list[ChatMessage]:
    if isinstance(payload, ChatMessage):
        if payload.role == Role.USER:
            return [payload]
        return [ChatMessage(Role.USER, text=payload.text)]
    if isinstance(payload, list) and all(isinstance(msg, ChatMessage) for msg in payload):
        return [msg if msg.role == Role.USER else ChatMessage(Role.USER, text=msg.text) for msg in payload]
    if isinstance(payload, Mapping):  # User supplied structured data
        text = payload.get("text") or payload.get("content")
        if isinstance(text, str) and text.strip():
            return [ChatMessage(Role.USER, text=text.strip())]
    return [ChatMessage(Role.USER, text=str(payload))]


def _default_termination_condition(conversation: list[ChatMessage]) -> bool:
    """Default termination: stop after 10 user messages to prevent infinite loops."""
    user_message_count = sum(1 for msg in conversation if msg.role == Role.USER)
    return user_message_count >= 10


class HandoffBuilder:
    r"""Fluent builder for conversational handoff workflows with triage and specialist agents.

    The handoff pattern models a customer support or multi-agent conversation where:
    - A **triage/dispatcher agent** receives user input and decides whether to handle it directly
      or hand off to a **specialist agent**.
    - After a specialist responds, control returns to the user for more input, creating a cyclical flow:
      user -> triage -> [optional specialist] -> user -> triage -> ...
    - The workflow automatically requests user input after each agent response, maintaining conversation continuity.
    - A **termination condition** determines when the workflow should stop requesting input and complete.

    Key Features:
    - **Automatic handoff detection**: The triage agent includes "HANDOFF_TO: <specialist_name>" in its response
      to trigger a handoff. Custom resolvers can parse different formats.
    - **Full conversation history**: By default, the entire conversation (including any
      `ChatMessage.additional_properties`) is preserved and passed to each agent. Use
      `.with_context_window(N)` to limit the history to the last N messages when you want a rolling window.
    - **Termination control**: By default, terminates after 10 user messages. Override with
      `.with_termination_condition(lambda conv: ...)` for custom logic (e.g., detect "goodbye").
    - **Checkpointing**: Optional persistence for resumable workflows.

    Usage:

    .. code-block:: python

        from agent_framework import HandoffBuilder
        from agent_framework.openai import OpenAIChatClient

        chat_client = OpenAIChatClient()

        # Create triage and specialist agents
        triage = chat_client.create_agent(
            instructions=(
                "You are a frontline support agent. Assess the user's issue and decide "
                "whether to hand off to 'refund_agent' or 'shipping_agent'. If handing off, "
                "include 'HANDOFF_TO: <agent_name>' in your response."
            ),
            name="triage_agent",
        )

        refund = chat_client.create_agent(
            instructions="You handle refund requests. Ask for order details and process refunds.",
            name="refund_agent",
        )

        shipping = chat_client.create_agent(
            instructions="You resolve shipping issues. Track packages and update delivery status.",
            name="shipping_agent",
        )

        # Build the handoff workflow with default termination (10 user messages)
        workflow = (
            HandoffBuilder(
                name="customer_support",
                participants=[triage, refund, shipping],
            )
            .starting_agent("triage_agent")
            .build()
        )

        # Run the workflow
        events = await workflow.run_stream("My package hasn't arrived yet")
        async for event in events:
            if isinstance(event, RequestInfoEvent):
                # Request user input
                user_response = input("You: ")
                await workflow.send_response(event.data.request_id, user_response)

    **Custom Termination Condition:**

    .. code-block:: python

        # Terminate when user says goodbye or after 5 exchanges
        workflow = (
            HandoffBuilder(participants=[triage, refund, shipping])
            .starting_agent("triage_agent")
            .with_termination_condition(
                lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 5
                or any("goodbye" in msg.text.lower() for msg in conv[-2:])
            )
            .build()
        )

    **Context Window (Rolling History):**

    .. code-block:: python

        # Only keep last 10 messages in conversation for each agent
        workflow = (
            HandoffBuilder(participants=[triage, refund, shipping])
            .starting_agent("triage_agent")
            .with_context_window(10)
            .build()
        )

    **Custom Handoff Resolver:**

    .. code-block:: python

        # Parse handoff from structured agent response
        def custom_resolver(response):
            # Check additional_properties for handoff metadata
            props = response.agent_run_response.additional_properties or {}
            return props.get("route_to")


        workflow = (
            HandoffBuilder(participants=[triage, refund, shipping])
            .starting_agent("triage_agent")
            .handoff_resolver(custom_resolver)
            .build()
        )

    **Checkpointing:**

    .. code-block:: python

        from agent_framework import InMemoryCheckpointStorage

        storage = InMemoryCheckpointStorage()
        workflow = (
            HandoffBuilder(participants=[triage, refund, shipping])
            .starting_agent("triage_agent")
            .with_checkpointing(storage)
            .build()
        )

    Args:
        name: Optional workflow name for identification and logging.
        participants: List of agents (AgentProtocol) or executors to participate in the handoff.
                     The first agent you specify as starting_agent becomes the triage agent.
        description: Optional human-readable description of the workflow.

    Raises:
        ValueError: If participants list is empty, contains duplicates, or starting_agent not specified.
        TypeError: If participants are not AgentProtocol or Executor instances.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        participants: Sequence[AgentProtocol | Executor] | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize a HandoffBuilder for creating conversational handoff workflows.

        The builder starts in an unconfigured state and requires you to call:
        1. `.participants([...])` - Register agents
        2. `.starting_agent(...)` - Designate which agent receives initial user input
        3. `.build()` - Construct the final Workflow

        Optional configuration methods allow you to customize handoff detection,
        context management, termination logic, and persistence.

        Args:
            name: Optional workflow identifier used in logging and debugging.
                 If not provided, a default name will be generated.
            participants: Optional list of agents (AgentProtocol) or executors that will
                         participate in the handoff workflow. You can also call
                         `.participants([...])` later. Each participant must have a
                         unique identifier (name for agents, id for executors).
            description: Optional human-readable description explaining the workflow's
                        purpose. Useful for documentation and observability.

        Note:
            Participants must have stable names/ids because the handoff resolver
            uses these identifiers to route control between agents. Agent names
            should match the handoff target strings (e.g., "HANDOFF_TO: billing"
            requires an agent named "billing").
        """
        self._name = name
        self._description = description
        self._executors: dict[str, Executor] = {}
        self._aliases: dict[str, str] = {}
        self._starting_agent_id: str | None = None
        self._context_window: int | None = None
        self._resolver: Callable[[AgentExecutorResponse], Any] = _default_handoff_resolver
        self._checkpoint_storage: CheckpointStorage | None = None
        self._request_prompt: str | None = None
        self._termination_condition: Callable[[list[ChatMessage]], bool] = _default_termination_condition

        if participants:
            self.participants(participants)

    def participants(self, participants: Sequence[AgentProtocol | Executor]) -> "HandoffBuilder":
        """Register the agents or executors that will participate in the handoff workflow.

        Each participant must have a unique identifier (name for agents, id for executors).
        The workflow will automatically create an alias map so agents can be referenced by
        their name, display_name, or executor id when routing.

        Args:
            participants: Sequence of AgentProtocol or Executor instances. Each must have
                         a unique identifier. For agents, the name attribute is used as the
                         primary identifier and must match handoff target strings.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If participants is empty or contains duplicates.
            TypeError: If participants are not AgentProtocol or Executor instances.

        Example:

        .. code-block:: python

            from agent_framework import HandoffBuilder
            from agent_framework.openai import OpenAIChatClient

            client = OpenAIChatClient()
            triage = client.create_agent(instructions="...", name="triage")
            refund = client.create_agent(instructions="...", name="refund_agent")
            billing = client.create_agent(instructions="...", name="billing_agent")

            builder = HandoffBuilder().participants([triage, refund, billing])
            # Now you can call .starting_agent() to designate the entry point

        Note:
            This method resets any previously configured starting_agent, so you must call
            `.starting_agent(...)` again after changing participants.
        """
        if not participants:
            raise ValueError("participants cannot be empty")

        wrapped: list[Executor] = []
        seen_ids: set[str] = set()
        alias_map: dict[str, str] = {}

        for p in participants:
            executor = self._wrap_participant(p)
            if executor.id in seen_ids:
                raise ValueError(f"Duplicate participant with id '{executor.id}' detected")
            seen_ids.add(executor.id)
            wrapped.append(executor)

            alias_map[executor.id] = executor.id
            if isinstance(p, AgentProtocol):
                name = getattr(p, "name", None)
                if name:
                    alias_map[name] = executor.id
            display = getattr(p, "display_name", None)
            if isinstance(display, str) and display:
                alias_map[display] = executor.id

        self._executors = {executor.id: executor for executor in wrapped}
        self._aliases = alias_map
        self._starting_agent_id = None
        return self

    def starting_agent(self, agent: str | AgentProtocol | Executor) -> "HandoffBuilder":
        """Designate which agent receives initial user input and acts as the triage/dispatcher.

        The starting agent is responsible for analyzing user requests and deciding whether to:
        1. Handle the request directly and respond to the user, OR
        2. Hand off to a specialist agent by including handoff metadata in the response

        After a specialist responds, the workflow automatically returns control to the user,
        creating a cyclical flow: user -> starting_agent -> [specialist] -> user -> ...

        Args:
            agent: The agent to use as the entry point. Can be:
                  - Agent name (str): e.g., "triage_agent"
                  - AgentProtocol instance: The actual agent object
                  - Executor instance: A custom executor wrapping an agent

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If participants(...) hasn't been called yet, or if the specified
                       agent is not in the participants list.

        Example:

        .. code-block:: python

            builder = (
                HandoffBuilder().participants([triage, refund, billing]).starting_agent("triage")  # Use agent name
            )

            # Or pass the agent object directly:
            builder = (
                HandoffBuilder().participants([triage, refund, billing]).starting_agent(triage)  # Use agent instance
            )

        Note:
            The starting agent determines routing by including "HANDOFF_TO: <agent_name>"
            in its response, or by setting structured metadata that the handoff resolver
            can parse. Use `.handoff_resolver(...)` to customize detection logic.
        """
        if not self._executors:
            raise ValueError("Call participants(...) before starting_agent(...)")
        resolved = self._resolve_to_id(agent)
        if resolved not in self._executors:
            raise ValueError(f"starting_agent '{resolved}' is not part of the participants list")
        self._starting_agent_id = resolved
        return self

    def with_context_window(self, message_count: int | None) -> "HandoffBuilder":
        """Limit conversation history to a rolling window of recent messages.

        By default, the handoff workflow passes the entire conversation history to each agent.
        This can lead to excessive token usage in long conversations. Use a context window to
        send only the most recent N messages to agents, reducing costs while maintaining focus
        on recent context.

        Args:
            message_count: Maximum number of recent messages to include when calling agents.
                          If None, uses the full conversation history (default behavior).
                          Must be positive if specified.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If message_count is not positive (when provided).

        Example:

        .. code-block:: python

            # Keep only last 10 messages for each agent call
            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .starting_agent("triage")
                .with_context_window(10)
                .build()
            )

            # After 15 messages in the conversation:
            # - Full conversation: 15 messages stored
            # - Agent sees: Only messages 6-15 (last 10)
            # - User sees: All 15 messages in output

        Use Cases:
            - Long support conversations where early context becomes irrelevant
            - Cost optimization by reducing tokens sent to LLM
            - Forcing agents to focus on recent exchanges rather than full history
            - Conversations with repetitive patterns where distant history adds noise

        Note:
            The context window applies to messages sent TO agents, not the conversation
            stored by the workflow. The full conversation is maintained internally and
            returned in the final output. This is purely for token efficiency.
        """
        if message_count is not None and message_count <= 0:
            raise ValueError("message_count must be positive when provided")
        self._context_window = message_count
        return self

    def handoff_resolver(
        self,
        resolver: Callable[[AgentExecutorResponse], str | AgentProtocol | Executor | None],
    ) -> "HandoffBuilder":
        r"""Customize how the workflow detects handoff requests from the starting agent.

        By default, the workflow looks for "HANDOFF_TO: <agent_name>" in the starting agent's
        response text. Use this method to implement custom detection logic that reads structured
        metadata, function call results, or parses different text patterns.

        The resolver is called after each starting agent response to determine if a handoff
        should occur and which specialist to route to.

        Args:
            resolver: Function that receives an AgentExecutorResponse and returns:
                     - str: Name or ID of the specialist agent to hand off to
                     - AgentProtocol: The specialist agent instance
                     - Executor: A custom executor to route to
                     - None: No handoff, starting agent continues handling the conversation

        Returns:
            Self for method chaining.

        Example (Structured Metadata):

        .. code-block:: python

            def custom_resolver(response: AgentExecutorResponse) -> str | None:
                # Read handoff from response metadata
                props = response.agent_run_response.additional_properties or {}
                return props.get("route_to")


            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .starting_agent("triage")
                .handoff_resolver(custom_resolver)
                .build()
            )

        Example (Function Call Result):

        .. code-block:: python

            def function_call_resolver(response: AgentExecutorResponse) -> str | None:
                # Check if agent used a function call to specify routing
                value = response.agent_run_response.value
                if isinstance(value, dict):
                    return value.get("handoff_to")
                return None


            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .starting_agent("triage")
                .handoff_resolver(function_call_resolver)
                .build()
            )

        Example (Custom Text Pattern):

        .. code-block:: python

            import re


            def regex_resolver(response: AgentExecutorResponse) -> str | None:
                # Look for "ROUTE: agent_name" instead of "HANDOFF_TO: agent_name"
                for msg in response.agent_run_response.messages:
                    match = re.search(r"ROUTE:\s*(\w+)", msg.text or "")
                    if match:
                        return match.group(1)
                return None


            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .starting_agent("triage")
                .handoff_resolver(regex_resolver)
                .build()
            )

        Note:
            If the resolver returns an agent name that doesn't match any specialist,
            a warning is logged and no handoff occurs. Make sure resolver returns
            match the names of agents in participants.
        """
        self._resolver = resolver
        return self

    def request_prompt(self, prompt: str | None) -> "HandoffBuilder":
        """Set a custom prompt message displayed when requesting user input.

        By default, the workflow uses a generic prompt: "Provide your next input for the
        conversation." Use this method to customize the message shown to users when the
        workflow needs their response.

        Args:
            prompt: Custom prompt text to display, or None to use the default prompt.

        Returns:
            Self for method chaining.

        Example:

        .. code-block:: python

            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .starting_agent("triage")
                .request_prompt("How can we help you today?")
                .build()
            )

            # For more context-aware prompts, you can access the prompt via
            # RequestInfoEvent.data.prompt in your event handling loop

        Note:
            The prompt is static and set once during workflow construction. If you need
            dynamic prompts based on conversation state, you'll need to handle that in
            your application's event processing logic.
        """
        self._request_prompt = prompt
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "HandoffBuilder":
        """Enable workflow state persistence for resumable conversations.

        Checkpointing allows the workflow to save its state at key points, enabling you to:
        - Resume conversations after application restarts
        - Implement long-running support tickets that span multiple sessions
        - Recover from failures without losing conversation context
        - Audit and replay conversation history

        Args:
            checkpoint_storage: Storage backend implementing CheckpointStorage interface.
                               Common implementations: InMemoryCheckpointStorage (testing),
                               database-backed storage (production).

        Returns:
            Self for method chaining.

        Example (In-Memory):

        .. code-block:: python

            from agent_framework import InMemoryCheckpointStorage

            storage = InMemoryCheckpointStorage()
            workflow = (
                HandoffBuilder(participants=[triage, refund, billing])
                .starting_agent("triage")
                .with_checkpointing(storage)
                .build()
            )

            # Run workflow with a session ID for resumption
            async for event in workflow.run_stream("Help me", session_id="user_123"):
                # Process events...
                pass

            # Later, resume the same conversation
            async for event in workflow.run_stream("I need a refund", session_id="user_123"):
                # Conversation continues from where it left off
                pass

        Use Cases:
            - Customer support systems with persistent ticket history
            - Multi-day conversations that need to survive server restarts
            - Compliance requirements for conversation auditing
            - A/B testing different agent configurations on same conversation

        Note:
            Checkpointing adds overhead for serialization and storage I/O. Use it when
            persistence is required, not for simple stateless request-response patterns.
        """
        self._checkpoint_storage = checkpoint_storage
        return self

    def with_termination_condition(self, condition: Callable[[list[ChatMessage]], bool]) -> "HandoffBuilder":
        """Set a custom termination condition for the handoff workflow.

        Args:
            condition: Function that receives the full conversation and returns True
                      if the workflow should terminate (not request further user input).

        Returns:
            Self for chaining.

        Example:

        .. code-block:: python

            builder.with_termination_condition(
                lambda conv: len(conv) > 20 or any("goodbye" in msg.text.lower() for msg in conv[-2:])
            )
        """
        self._termination_condition = condition
        return self

    def build(self) -> Workflow:
        """Construct the final Workflow instance from the configured builder.

        This method validates the configuration and assembles all internal components:
        - Input normalization executor
        - Starting agent executor
        - Handoff coordinator
        - Specialist agent executors
        - User input gateway
        - Request/response handling

        Returns:
            A fully configured Workflow ready to execute via `.run()` or `.run_stream()`.

        Raises:
            ValueError: If participants or starting_agent were not configured, or if
                       required configuration is invalid.

        Example (Minimal):

        .. code-block:: python

            workflow = HandoffBuilder(participants=[triage, refund, billing]).starting_agent("triage").build()

            # Run the workflow
            async for event in workflow.run_stream("I need help"):
                # Handle events...
                pass

        Example (Full Configuration):

        .. code-block:: python

            from agent_framework import InMemoryCheckpointStorage

            storage = InMemoryCheckpointStorage()
            workflow = (
                HandoffBuilder(
                    name="support_workflow",
                    participants=[triage, refund, billing],
                    description="Customer support with specialist routing",
                )
                .starting_agent("triage")
                .with_context_window(10)
                .with_termination_condition(lambda conv: len(conv) > 20)
                .handoff_resolver(custom_resolver)
                .request_prompt("How can we help?")
                .with_checkpointing(storage)
                .build()
            )

        Note:
            After calling build(), the builder instance should not be reused. Create a
            new builder if you need to construct another workflow with different configuration.
        """
        if not self._executors:
            raise ValueError("No participants provided. Call participants([...]) first.")
        if self._starting_agent_id is None:
            raise ValueError("starting_agent must be defined before build().")

        starting_executor = self._executors[self._starting_agent_id]
        specialists = {
            exec_id: executor for exec_id, executor in self._executors.items() if exec_id != self._starting_agent_id
        }

        if not specialists:
            logger.warning("Handoff workflow has no specialist agents; the starting agent will loop with the user.")

        input_node = _InputToConversation(id="input-conversation")
        request_info = RequestInfoExecutor(id=f"{starting_executor.id}_handoff_requests")
        user_gateway = _UserInputGateway(
            request_executor_id=request_info.id,
            starting_agent_id=starting_executor.id,
            prompt=self._request_prompt,
            id="handoff-user-input",
        )
        coordinator = _HandoffCoordinator(
            starting_agent_id=starting_executor.id,
            specialist_ids={alias: exec_id for alias, exec_id in self._aliases.items() if exec_id in specialists},
            input_gateway_id=user_gateway.id,
            context_window=self._context_window,
            resolver=self._resolver,
            termination_condition=self._termination_condition,
            id="handoff-coordinator",
        )

        builder = WorkflowBuilder(name=self._name, description=self._description)
        builder.set_start_executor(input_node)
        builder.add_edge(input_node, starting_executor)
        builder.add_edge(starting_executor, coordinator)

        for specialist in specialists.values():
            builder.add_edge(coordinator, specialist)
            builder.add_edge(specialist, coordinator)

        builder.add_edge(coordinator, user_gateway)
        builder.add_edge(user_gateway, request_info)
        builder.add_edge(request_info, user_gateway)
        builder.add_edge(user_gateway, coordinator)  # Route back to coordinator, not directly to agent
        builder.add_edge(coordinator, starting_executor)  # Coordinator sends trimmed request to agent

        if self._checkpoint_storage is not None:
            builder = builder.with_checkpointing(self._checkpoint_storage)

        return builder.build()

    def _wrap_participant(self, participant: AgentProtocol | Executor) -> Executor:
        if isinstance(participant, Executor):
            return participant
        if isinstance(participant, AgentProtocol):
            name = getattr(participant, "name", None)
            if not name:
                raise ValueError(
                    "Agents used in handoff workflows must have a stable name so they can be addressed during routing."
                )
            return AgentExecutor(participant, id=name)
        raise TypeError(f"Participants must be AgentProtocol or Executor instances. Got {type(participant).__name__}.")

    def _resolve_to_id(self, candidate: str | AgentProtocol | Executor) -> str:
        if isinstance(candidate, Executor):
            return candidate.id
        if isinstance(candidate, AgentProtocol):
            name: str | None = getattr(candidate, "name", None)
            if not name:
                raise ValueError("AgentProtocol without a name cannot be resolved to an executor id.")
            return self._aliases.get(name, name)
        if isinstance(candidate, str):
            if candidate in self._aliases:
                return self._aliases[candidate]
            return candidate
        raise TypeError(f"Invalid starting agent reference: {type(candidate).__name__}")
