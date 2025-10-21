# Copyright (c) Microsoft. All rights reserved.

"""Group chat orchestration primitives.

This module introduces a reusable orchestration surface for manager-directed
multi-agent conversations. The key components are:

- GroupChatRequestMessage / GroupChatResponseMessage: canonical envelopes used
  between the orchestrator and participants.
- GroupChatManagerFn: minimal asynchronous callable contract for pluggable coordination logic.
- GroupChatOrchestratorExecutor: runtime state machine that delegates to a
  manager to select the next participant or complete the task.
- GroupChatBuilder: high-level builder that wires managers and participants
  into a workflow graph. It mirrors the ergonomics of SequentialBuilder and
  ConcurrentBuilder while allowing Magentic to reuse the same infrastructure.

The default wiring uses AgentExecutor under the hood for agent participants so
existing observability and streaming semantics continue to apply.
"""

import itertools
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from .._agents import AgentProtocol
from .._clients import ChatClientProtocol
from .._types import ChatMessage, Role
from ._agent_executor import AgentExecutor, AgentExecutorRequest, AgentExecutorResponse
from ._checkpoint import CheckpointStorage
from ._executor import Executor, handler
from ._workflow import Workflow
from ._workflow_builder import WorkflowBuilder
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


# region Message primitives


@dataclass
class GroupChatRequestMessage:
    """Request envelope sent from the orchestrator to a participant."""

    agent_name: str
    conversation: list[ChatMessage] = field(default_factory=list)  # type: ignore
    instruction: str = ""
    task: ChatMessage | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GroupChatResponseMessage:
    """Response envelope emitted by participants back to the orchestrator."""

    agent_name: str
    message: ChatMessage
    target_agent: str | None = None
    broadcast: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class GroupChatTurn:
    """Represents a single turn in the manager-participant conversation."""

    speaker: str
    role: str
    message: ChatMessage


@dataclass
class GroupChatDirective:
    """Instruction emitted by a GroupChatManagerFn implementation."""

    agent_name: str | None = None
    instruction: str | None = None
    metadata: dict[str, Any] | None = None
    finish: bool = False
    final_message: ChatMessage | None = None


# endregion


# region Manager callable


GroupChatStateSnapshot = Mapping[str, Any]
GroupChatManagerFn = Callable[[GroupChatStateSnapshot], Awaitable[GroupChatDirective]]


@dataclass
class GroupChatParticipantSpec:
    """Metadata describing a single participant in the orchestration.

    Attributes:
        name: Unique identifier for the participant used by the manager for selection
        participant: AgentProtocol or Executor instance representing the participant
        description: Human-readable description provided to the manager for selection context
    """

    name: str
    participant: AgentProtocol | Executor
    description: str


GroupChatParticipantPipeline: TypeAlias = Sequence[Executor]


@dataclass
class GroupChatWiring:
    """Configuration passed to factories during workflow assembly.

    Attributes:
        manager: Manager instance responsible for orchestration decisions (None when custom factory handles it)
        manager_name: Display name for the manager in conversation history
        participants: Mapping of participant names to their specifications
        max_rounds: Optional limit on manager selection rounds to prevent infinite loops
        orchestrator: Orchestrator executor instance (populated during build)
    """

    manager: GroupChatManagerFn | None
    manager_name: str
    participants: Mapping[str, GroupChatParticipantSpec]
    max_rounds: int | None = None
    orchestrator: Executor | None = None


# endregion


# region Default participant factory


def _default_participant_factory(
    spec: GroupChatParticipantSpec,
    _: GroupChatWiring,
) -> GroupChatParticipantPipeline:
    """Default factory for constructing participant pipeline nodes in the workflow graph.

    Creates a single AgentExecutor node for AgentProtocol participants or a passthrough executor
    for custom participants. Translation between group-chat envelopes and the agent runtime is now
    handled inside the orchestrator, removing the need for dedicated ingress/egress adapters.

    Args:
        spec: Participant specification containing name, instance, and description
        _: GroupChatWiring configuration (unused by default implementation)

    Returns:
        Sequence of executors representing the participant pipeline in execution order

    Behavior:
        - AgentProtocol participants are wrapped in AgentExecutor with deterministic IDs
        - Executor participants are wired directly without additional adapters
    """
    participant = spec.participant
    if isinstance(participant, Executor):
        return (participant,)

    agent = participant
    agent_executor = AgentExecutor(agent, id=f"groupchat_agent:{spec.name}")
    return (agent_executor,)


# endregion


# region Default orchestrator


class GroupChatOrchestratorExecutor(Executor):
    """Default orchestrator executor that implements manager-directed group chat coordination.

    This is the central runtime state machine that drives multi-agent conversations. It
    maintains conversation state, delegates speaker selection to a manager, routes messages
    to participants, and collects responses in a loop until the manager signals completion.

    Core responsibilities:
    - Accept initial input as str, ChatMessage, or list[ChatMessage]
    - Maintain conversation history and turn tracking
    - Query manager for next action (select participant or finish)
    - Route requests to selected participants using AgentExecutorRequest or GroupChatRequestMessage
    - Collect participant responses and append to conversation
    - Enforce optional round limits to prevent infinite loops
    - Yield final completion message and transition to idle state

    State management:
    - _conversation: Growing list of all messages (user, manager, agents)
    - _history: Turn-by-turn record with speaker attribution and roles
    - _task_message: Original user task extracted from input
    - _pending_agent: Name of agent currently processing a request
    - _round_index: Count of manager selection rounds for limit enforcement

    Manager interaction:
    The orchestrator builds immutable state snapshots and passes them to the manager
    callable. The manager returns a GroupChatDirective indicating either:
    - Next participant to speak (with optional instruction)
    - Finish signal (with optional final message)

    Message flow topology:
        User input -> orchestrator -> manager -> orchestrator -> participant -> orchestrator
        (loops until manager returns finish directive)

    Why this design:
    - Separates orchestration logic (this class) from selection logic (manager)
    - Manager is stateless and testable in isolation
    - Orchestrator handles all state mutations and message routing
    - Broadcast routing to participants keeps graph structure simple

    Args:
        manager: Callable that selects the next participant or finishes based on state snapshot
        participants: Mapping of participant names to descriptions (for manager context)
        manager_name: Display name for manager in conversation history
        max_rounds: Optional limit on manager selection rounds (None = unlimited)
        executor_id: Optional custom ID for observability (auto-generated if not provided)
    """

    def __init__(
        self,
        manager: GroupChatManagerFn,
        *,
        participants: Mapping[str, str],
        manager_name: str,
        max_rounds: int | None = None,
        executor_id: str | None = None,
    ) -> None:
        super().__init__(executor_id or f"groupchat_orchestrator_{uuid4().hex[:8]}")
        self._manager = manager
        self._participants = dict(participants)
        self._manager_name = manager_name
        self._conversation: list[ChatMessage] = []
        self._history: list[GroupChatTurn] = []
        self._task_message: ChatMessage | None = None
        self._pending_agent: str | None = None
        self._round_index = 0
        self._max_rounds = max_rounds
        # Stashes the initial conversation list until _handle_task_message normalizes it into _conversation.
        self._pending_initial_conversation: list[ChatMessage] | None = None
        self._participant_entry_ids: dict[str, str] = {}
        self._agent_executor_ids: dict[str, str] = {}
        self._executor_id_to_participant: dict[str, str] = {}
        self._non_agent_participants: set[str] = set()

    @staticmethod
    def _role_value(message: ChatMessage) -> str:
        """Extract string role value from a ChatMessage, handling enum and string cases.

        Args:
            message: Chat message with role attribute (may be enum or string)

        Returns:
            String representation of the role (e.g., "user", "assistant", "system")

        Why this exists:
            Different ChatMessage implementations may use Role enum or plain strings.
            This normalizes access for consistent turn tracking.
        """
        role = getattr(message.role, "value", None) or str(message.role)
        return str(role)

    def _build_state(self) -> GroupChatStateSnapshot:
        """Build a snapshot of current orchestration state for the manager.

        Packages conversation history, participant metadata, and round tracking into
        an immutable mapping that the manager uses to make speaker selection decisions.

        Returns:
            Mapping containing all context needed for manager decision-making

        Raises:
            RuntimeError: If called before task message initialization (defensive check)

        When this is called:
            - After initial input is processed (first manager query)
            - After each participant response (subsequent manager queries)
        """
        if self._task_message is None:
            raise RuntimeError("GroupChatOrchestratorExecutor state not initialized with task message.")
        snapshot: dict[str, Any] = {
            "task": self._task_message,
            "participants": dict(self._participants),
            "conversation": tuple(self._conversation),
            "history": tuple(self._history),
            "pending_agent": self._pending_agent,
            "round_index": self._round_index,
        }
        return MappingProxyType(snapshot)

    def register_participant_entry(self, name: str, *, entry_id: str, is_agent: bool) -> None:
        """Record routing details for a participant's entry executor."""
        self._participant_entry_ids[name] = entry_id
        if is_agent:
            self._agent_executor_ids[name] = entry_id
            self._executor_id_to_participant[entry_id] = name
        else:
            self._non_agent_participants.add(name)

    async def _apply_directive(
        self,
        directive: GroupChatDirective,
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Execute a manager directive by either finishing the workflow or routing to a participant.

        This is the core routing logic that interprets manager decisions. It handles two cases:
        1. Finish directive: append final message, update state, yield output, become idle
        2. Agent selection: build request envelope, route to participant, increment round counter

        Args:
            directive: Manager's decision (finish or select next participant)
            ctx: Workflow context for sending messages and yielding output

        Behavior for finish directive:
            - Uses provided final_message or creates default completion message
            - Ensures author_name is set to manager for attribution
            - Appends to conversation and history for complete record
            - Yields message as workflow output
            - Orchestrator becomes idle (no further processing)

        Behavior for agent selection:
            - Validates agent_name exists in participants
            - Optionally appends manager instruction as USER message
            - Prepares full conversation context for the participant
            - Routes request directly to the participant entry executor
            - Increments round counter and enforces max_rounds if configured

        Round limit enforcement:
            If max_rounds is reached, recursively calls _apply_directive with a finish
            directive to gracefully terminate the conversation.

        Raises:
            ValueError: If directive lacks agent_name when finish=False, or if
                       agent_name doesn't match any participant
        """
        if directive.finish:
            final_message = directive.final_message
            if final_message is None:
                final_message = ChatMessage(
                    role=Role.ASSISTANT,
                    text="Completed without final summary.",
                    author_name=self._manager_name,
                )
            elif not final_message.author_name:
                message_dict = final_message.to_dict()
                message_dict["author_name"] = self._manager_name
                final_message = ChatMessage.from_dict(message_dict)

            self._conversation.append(final_message)
            self._history.append(GroupChatTurn(self._manager_name, "manager", final_message))
            self._pending_agent = None
            await ctx.yield_output(final_message)
            return

        agent_name = directive.agent_name
        if not agent_name:
            raise ValueError("Directive must include agent_name when finish is False.")
        if agent_name not in self._participants:
            raise ValueError(f"Manager selected unknown participant '{agent_name}'.")

        entry_id = self._participant_entry_ids.get(agent_name)
        if entry_id is None:
            raise ValueError(f"No registered entry executor for participant '{agent_name}'.")

        instruction = directive.instruction or ""
        conversation = list(self._conversation)
        if instruction:
            manager_message = ChatMessage(
                role=Role.USER,
                text=instruction,
                author_name=self._manager_name,
            )
            conversation.append(manager_message)
            self._conversation.append(manager_message)
            self._history.append(GroupChatTurn(self._manager_name, "manager", manager_message))

        self._pending_agent = agent_name
        self._round_index += 1

        if agent_name in self._agent_executor_ids:
            await ctx.send_message(
                AgentExecutorRequest(messages=conversation, should_respond=True),
                target_id=entry_id,
            )
        else:
            request = GroupChatRequestMessage(
                agent_name=agent_name,
                conversation=conversation,
                task=self._task_message,
                metadata=dict(directive.metadata or {}),
            )
            await ctx.send_message(request, target_id=entry_id)

        if self._max_rounds is not None and self._round_index >= self._max_rounds:
            logger.warning(
                "GroupChatOrchestratorExecutor reached max_rounds=%s; forcing completion.",
                self._max_rounds,
            )
            await self._apply_directive(
                GroupChatDirective(
                    finish=True,
                    final_message=ChatMessage(
                        role=Role.ASSISTANT,
                        text="Conversation halted after reaching manager round limit.",
                        author_name=self._manager_name,
                    ),
                ),
                ctx,
            )

    async def _ingest_participant_message(
        self,
        participant_name: str,
        message: ChatMessage,
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Common response ingestion logic shared by agent and custom participants."""
        if participant_name not in self._participants:
            logger.debug("Ignoring response from unknown participant '%s'.", participant_name)
            return

        if not message.author_name:
            message_dict = message.to_dict()
            message_dict["author_name"] = participant_name
            message = ChatMessage.from_dict(message_dict)

        self._conversation.append(message)
        self._history.append(GroupChatTurn(participant_name, "agent", message))
        self._pending_agent = None

        if self._max_rounds is not None and self._round_index >= self._max_rounds:
            logger.warning(
                "GroupChatOrchestratorExecutor reached max_rounds=%s after receiving agent response.",
                self._max_rounds,
            )
            await ctx.yield_output(
                ChatMessage(
                    role=Role.ASSISTANT,
                    text="Conversation halted after reaching manager round limit.",
                    author_name=self._manager_name,
                )
            )
            return

        directive = await self._manager(self._build_state())
        await self._apply_directive(directive, ctx)

    @staticmethod
    def _extract_agent_message(response: AgentExecutorResponse, participant_name: str) -> ChatMessage:
        """Select the final assistant message from an AgentExecutor response."""
        final_message: ChatMessage | None = None
        candidate_sequences: tuple[Sequence[ChatMessage] | None, ...] = (
            response.agent_run_response.messages,
            response.full_conversation,
        )
        for sequence in candidate_sequences:
            if not sequence:
                continue
            for candidate in reversed(sequence):
                if getattr(candidate, "role", None) == Role.ASSISTANT:
                    final_message = candidate
                    break
            if final_message is not None:
                break

        if final_message is None:
            final_message = ChatMessage(role=Role.ASSISTANT, text="", author_name=participant_name)
        elif not final_message.author_name:
            message_dict = final_message.to_dict()
            message_dict["author_name"] = participant_name
            final_message = ChatMessage.from_dict(message_dict)
        return final_message

    async def _handle_task_message(
        self,
        task_message: ChatMessage,
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Initialize orchestrator state and start the manager-directed conversation loop.

        This internal method is called by all public handlers (str, ChatMessage, list[ChatMessage])
        after normalizing their input. It initializes conversation state, queries the manager
        for the first action, and applies the resulting directive.

        Args:
            task_message: The primary user task message (extracted or provided directly)
            ctx: Workflow context for sending messages and yielding output

        Behavior:
            - Sets task_message for manager context
            - Initializes conversation from pending_initial_conversation if present
            - Otherwise starts fresh with just the task message
            - Builds turn history with speaker attribution
            - Resets pending_agent and round_index
            - Queries manager for first action
            - Applies directive to start the conversation loop

        State initialization:
            - _conversation: Full message list for context
            - _history: Turn-by-turn record with speaker names and roles
            - _pending_agent: None (no active request)
            - _round_index: 0 (first manager query)

        Why pending_initial_conversation exists:
            The handle_conversation handler supplies an explicit task (the first message in
            the list) but still forwards the entire conversation for context. The full list is
            stashed in _pending_initial_conversation to preserve all context when initializing state.
        """
        self._task_message = task_message
        if self._pending_initial_conversation:
            initial_conversation = list(self._pending_initial_conversation)
            self._pending_initial_conversation = None
            self._conversation = initial_conversation
            self._history = [
                GroupChatTurn(
                    msg.author_name or self._role_value(msg),
                    self._role_value(msg),
                    msg,
                )
                for msg in initial_conversation
            ]
        else:
            self._conversation = [task_message]
            self._history = [GroupChatTurn("user", "user", task_message)]
        self._pending_agent = None
        self._round_index = 0
        directive = await self._manager(self._build_state())
        await self._apply_directive(directive, ctx)

    @handler
    async def handle_str(
        self,
        task: str,
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handler for string input as workflow entry point.

        Wraps the string in a USER role ChatMessage and delegates to _handle_task_message.

        Args:
            task: Plain text task description from user
            ctx: Workflow context

        Usage:
            workflow.run("Write a blog post about AI agents")
        """
        await self._handle_task_message(ChatMessage(role=Role.USER, text=task), ctx)

    @handler
    async def handle_chat_message(
        self,
        task_message: ChatMessage,
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handler for ChatMessage input as workflow entry point.

        Directly delegates to _handle_task_message for state initialization.

        Args:
            task_message: Structured chat message from user (may include metadata, role, etc.)
            ctx: Workflow context

        Usage:
            workflow.run(ChatMessage(role=Role.USER, text="Analyze this data"))
        """
        await self._handle_task_message(task_message, ctx)

    @handler
    async def handle_conversation(
        self,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handler for conversation history as workflow entry point.

        Accepts a pre-existing conversation and uses the first message in the list as the task.
        Preserves the full conversation for state initialization.

        Args:
            conversation: List of chat messages (system, user, assistant)
            ctx: Workflow context

        Raises:
            ValueError: If conversation list is empty

        Behavior:
            - Validates conversation is non-empty
            - Clones conversation to avoid mutation
            - Extracts task message (most recent USER message)
            - Stashes full conversation in _pending_initial_conversation
            - Delegates to _handle_task_message for initialization

        Usage:
            existing_messages = [
                ChatMessage(role=Role.SYSTEM, text="You are an expert"),
                ChatMessage(role=Role.USER, text="Help me with this task")
            ]
            workflow.run(existing_messages)
        """
        if not conversation:
            raise ValueError("GroupChat workflow requires at least one chat message.")
        self._pending_initial_conversation = list(conversation)
        task_message = conversation[0]
        await self._handle_task_message(task_message, ctx)

    @handler
    async def handle_agent_response(
        self,
        response: GroupChatResponseMessage,
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handle responses from custom participant executors."""
        await self._ingest_participant_message(response.agent_name, response.message, ctx)

    @handler
    async def handle_agent_executor_response(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[AgentExecutorRequest | GroupChatRequestMessage, ChatMessage],
    ) -> None:
        """Handle direct AgentExecutor responses."""
        participant_name = self._executor_id_to_participant.get(response.executor_id)
        if participant_name is None:
            logger.debug(
                "Ignoring response from unregistered agent executor '%s'.",
                response.executor_id,
            )
            return
        message = self._extract_agent_message(response, participant_name)
        await self._ingest_participant_message(participant_name, message, ctx)


def _default_orchestrator_factory(wiring: GroupChatWiring) -> Executor:
    """Default factory for creating the GroupChatOrchestratorExecutor instance.

    This is the internal implementation used by GroupChatBuilder to instantiate the
    orchestrator. It extracts participant descriptions from the wiring configuration
    and passes them to the orchestrator for manager context.

    Args:
        wiring: Complete workflow configuration assembled by the builder

    Returns:
        Initialized GroupChatOrchestratorExecutor ready to coordinate the conversation

    Behavior:
        - Extracts participant names and descriptions for manager context
        - Forwards manager instance, manager name, and max_rounds settings
        - Allows orchestrator to auto-generate its executor ID

    Why descriptions are extracted:
        The manager needs participant descriptions (not full specs) to make informed
        selection decisions. The orchestrator doesn't need participant instances directly
        since routing is handled by the workflow graph.

    Raises:
        RuntimeError: If manager is None (should not happen when using default factory)
    """
    if wiring.manager is None:
        raise RuntimeError("Default orchestrator factory requires a manager to be set")

    return GroupChatOrchestratorExecutor(
        manager=wiring.manager,
        participants={name: spec.description for name, spec in wiring.participants.items()},
        manager_name=wiring.manager_name,
        max_rounds=wiring.max_rounds,
    )


# endregion


# region Builder


class GroupChatBuilder:
    r"""High-level builder for manager-directed group chat workflows with dynamic orchestration.

    - `set_manager(...)` configures the orchestration manager (required)
    - `participants({...})` accepts a mapping of named AgentProtocol or Executor instances
    - The workflow wires an orchestrator that delegates speaker selection to the manager
    - Agents are automatically wrapped as AgentExecutor for consistent observability
    - The manager receives conversation state and returns directives (next speaker or finish)
    - The final output is the manager's completion message when the task is finished

    Usage:

    .. code-block:: python

        from agent_framework import GroupChatBuilder, StandardGroupChatManager

        manager = StandardGroupChatManager(chat_client)
        workflow = (
            GroupChatBuilder().set_manager(manager).participants(writer=writer_agent, reviewer=reviewer_agent).build()
        )

        # Enable checkpoint persistence
        workflow = (
            GroupChatBuilder()
            .set_manager(manager)
            .participants({"analyst": analyst_agent, "coder": coder_agent})
            .with_checkpointing(storage)
            .build()
        )

        # Limit conversation rounds
        workflow = (
            GroupChatBuilder()
            .set_manager(manager)
            .participants(agent1=agent1, agent2=agent2)
            .with_max_rounds(10)
            .build()
        )
    """

    def __init__(
        self,
        *,
        _orchestrator_factory: Callable[[GroupChatWiring], Executor] | None = None,
        _participant_factory: Callable[[GroupChatParticipantSpec, GroupChatWiring], GroupChatParticipantPipeline]
        | None = None,
    ) -> None:
        """Initialize the GroupChatBuilder.

        Args:
            _orchestrator_factory: Internal extension point for custom orchestrator implementations.
                Used by Magentic. Not part of public API - subject to change.
            _participant_factory: Internal extension point for custom participant pipelines.
                Used by Magentic. Not part of public API - subject to change.
        """
        self._participants: dict[str, AgentProtocol | Executor] = {}
        self._participant_descriptions: dict[str, str] = {}
        self._manager: GroupChatManagerFn | None = None
        self._manager_name: str = "manager"
        self._checkpoint_storage: CheckpointStorage | None = None
        self._max_rounds: int | None = None
        self._request_handler: tuple[Executor, Callable[[Any], bool]] | None = None
        self._orchestrator_factory = _orchestrator_factory or _default_orchestrator_factory
        self._participant_factory = _participant_factory or _default_participant_factory

    def set_manager(self, manager: GroupChatManagerFn, *, display_name: str | None = None) -> "GroupChatBuilder":
        """Configure the orchestration manager callable that selects participants and completes tasks.

        The callable receives an immutable conversation snapshot and returns directives indicating which
        participant should speak next or whether the task is complete.

        Args:
            manager: Awaitable callable accepting the state snapshot and returning a GroupChatDirective
            display_name: Optional custom name for the manager in conversation history

        Returns:
            Self for fluent chaining
        """
        self._manager = manager
        resolved_name = display_name or getattr(manager, "name", None) or "manager"
        self._manager_name = resolved_name
        return self

    def participants(
        self,
        participants: Mapping[str, AgentProtocol | Executor] | None = None,
        /,
        **named_participants: AgentProtocol | Executor,
    ) -> "GroupChatBuilder":
        """Define the named participants for this group chat workflow.

        Accepts AgentProtocol instances (auto-wrapped as AgentExecutor) or Executor instances.
        Participant names must be unique and non-empty. The manager uses these names when
        selecting the next speaker.

        Args:
            participants: Optional mapping of participant names to agent/executor instances
            **named_participants: Keyword arguments mapping names to agent/executor instances

        Returns:
            Self for fluent chaining

        Raises:
            ValueError: If participants are empty, names are duplicated, or names are empty strings

        Usage:

        .. code-block:: python

            from agent_framework import GroupChatBuilder

            # Using keyword arguments
            workflow = (
                GroupChatBuilder()
                .set_manager(manager)
                .participants(writer=writer_agent, editor=editor_agent, reviewer=reviewer_agent)
                .build()
            )

            # Using dictionary
            participants_dict = {"analyst": analyst_agent, "coder": coder_agent}
            workflow = GroupChatBuilder().set_manager(manager).participants(participants_dict).build()

            # Combining both approaches
            workflow = (
                GroupChatBuilder()
                .set_manager(manager)
                .participants({"agent1": agent1}, agent2=agent2, agent3=agent3)
                .build()
            )
        """
        combined: dict[str, AgentProtocol | Executor] = {}

        def _add(name: str, participant: AgentProtocol | Executor) -> None:
            if not name:
                raise ValueError("participant names must be non-empty strings")
            if name in combined or name in self._participants:
                raise ValueError(f"Duplicate participant name '{name}' supplied.")
            combined[name] = participant

        if participants:
            for name, participant in participants.items():
                _add(name, participant)
        for name, participant in named_participants.items():
            _add(name, participant)

        if not combined:
            raise ValueError("participants cannot be empty")

        for name, participant in combined.items():
            self._participants[name] = participant
            description = ""
            if isinstance(participant, Executor):
                description = participant.id
            else:
                description = getattr(participant, "description", None) or participant.__class__.__name__
            self._participant_descriptions[name] = description
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "GroupChatBuilder":
        """Enable checkpointing for the built workflow using the provided storage.

        Checkpointing allows the workflow to persist state and resume from interruption
        points, enabling long-running conversations and failure recovery.

        Args:
            checkpoint_storage: Storage implementation for persisting workflow state

        Returns:
            Self for fluent chaining

        Usage:

        .. code-block:: python

            from agent_framework import GroupChatBuilder, MemoryCheckpointStorage

            storage = MemoryCheckpointStorage()
            workflow = (
                GroupChatBuilder()
                .set_manager(manager)
                .participants(agent1=agent1, agent2=agent2)
                .with_checkpointing(storage)
                .build()
            )
        """
        self._checkpoint_storage = checkpoint_storage
        return self

    def with_request_handler(
        self,
        executor: Executor,
        *,
        condition: Callable[[Any], bool],
    ) -> "GroupChatBuilder":
        """Register an executor that intercepts and handles special orchestrator requests.

        This advanced feature allows custom executors to process specific messages
        emitted by the orchestrator before they reach participants. Useful for
        implementing plan review, validation gates, or custom routing logic.

        Args:
            executor: Executor instance that handles intercepted requests
            condition: Callable that returns True for messages this executor should handle

        Returns:
            Self for fluent chaining

        Usage:

        .. code-block:: python

            from agent_framework import GroupChatBuilder, Executor


            def is_plan_review(msg: Any) -> bool:
                return isinstance(msg, dict) and msg.get("type") == "plan_review"


            review_executor = PlanReviewExecutor()
            workflow = (
                GroupChatBuilder()
                .set_manager(manager)
                .participants(agent1=agent1)
                .with_request_handler(review_executor, condition=is_plan_review)
                .build()
            )
        """
        self._request_handler = (executor, condition)
        return self

    def with_max_rounds(self, max_rounds: int | None) -> "GroupChatBuilder":
        """Set a maximum number of manager rounds to prevent infinite conversations.

        When the round limit is reached, the workflow automatically completes with
        a default completion message. Setting to None allows unlimited rounds.

        Args:
            max_rounds: Maximum number of manager selection rounds, or None for unlimited

        Returns:
            Self for fluent chaining

        Usage:

        .. code-block:: python

            from agent_framework import GroupChatBuilder

            # Limit to 15 rounds
            workflow = (
                GroupChatBuilder()
                .set_manager(manager)
                .participants(agent1=agent1, agent2=agent2)
                .with_max_rounds(15)
                .build()
            )

            # Unlimited rounds
            workflow = GroupChatBuilder().set_manager(manager).participants(agent1=agent1).with_max_rounds(None).build()
        """
        self._max_rounds = max_rounds
        return self

    def _build_participant_specs(self) -> dict[str, GroupChatParticipantSpec]:
        specs: dict[str, GroupChatParticipantSpec] = {}
        for name, participant in self._participants.items():
            specs[name] = GroupChatParticipantSpec(
                name=name,
                participant=participant,
                description=self._participant_descriptions[name],
            )
        return specs

    def build(self) -> Workflow:
        """Build and validate the group chat workflow.

        Assembles the orchestrator, participants, and their interconnections into
        a complete workflow graph. The orchestrator delegates speaker selection to
        the manager, routes requests to the appropriate participants, and collects
        their responses to continue or complete the conversation.

        Returns:
            Validated Workflow instance ready for execution

        Raises:
            ValueError: If manager or participants are not configured (when using default factory)

        Wiring pattern:
        - Orchestrator receives initial input (str, ChatMessage, or list[ChatMessage])
        - Orchestrator queries manager for next action (participant selection or finish)
        - If participant selected: request routed directly to participant entry node
        - Participant pipeline: AgentExecutor for agents or custom executor chains
        - Participant response flows back to orchestrator
        - Orchestrator updates state and queries manager again
        - When manager returns finish directive: orchestrator yields final message and becomes idle

        Usage:

        .. code-block:: python

            from agent_framework import GroupChatBuilder, StandardGroupChatManager

            manager = StandardGroupChatManager(chat_client)
            workflow = GroupChatBuilder().set_manager(manager).participants(agent1=agent1, agent2=agent2).build()

            # Execute the workflow
            async for message in workflow.run("Solve this problem collaboratively"):
                print(message.text)
        """
        # Manager is only required when using the default orchestrator factory
        # Custom factories (e.g., MagenticBuilder) provide their own orchestrator with embedded manager
        if self._manager is None and self._orchestrator_factory == _default_orchestrator_factory:
            raise ValueError("manager must be configured before build() when using default orchestrator")
        if not self._participants:
            raise ValueError("participants must be configured before build()")

        participant_specs = self._build_participant_specs()
        wiring = GroupChatWiring(
            manager=self._manager,
            manager_name=self._manager_name,
            participants=participant_specs,
            max_rounds=self._max_rounds,
        )

        orchestrator = self._orchestrator_factory(wiring)
        wiring.orchestrator = orchestrator

        workflow_builder = WorkflowBuilder().set_start_executor(orchestrator)

        for name, spec in participant_specs.items():
            pipeline = list(self._participant_factory(spec, wiring))
            if not pipeline:
                raise ValueError(
                    f"Participant factory returned an empty pipeline for '{name}'. "
                    "Provide at least one executor per participant."
                )
            entry_executor = pipeline[0]
            exit_executor = pipeline[-1]
            register_entry = getattr(orchestrator, "register_participant_entry", None)
            if callable(register_entry):
                register_entry(
                    name,
                    entry_id=entry_executor.id,
                    is_agent=not isinstance(spec.participant, Executor),
                )

            workflow_builder = workflow_builder.add_edge(orchestrator, entry_executor)
            for upstream, downstream in itertools.pairwise(pipeline):
                workflow_builder = workflow_builder.add_edge(upstream, downstream)
            if exit_executor is not orchestrator:
                workflow_builder = workflow_builder.add_edge(exit_executor, orchestrator)

        if self._request_handler is not None:
            handler_executor, condition = self._request_handler
            workflow_builder = workflow_builder.add_edge(orchestrator, handler_executor, condition=condition)
            workflow_builder = workflow_builder.add_edge(handler_executor, orchestrator)

        if self._checkpoint_storage is not None:
            workflow_builder = workflow_builder.with_checkpointing(self._checkpoint_storage)

        return workflow_builder.build()


# endregion


# region Default manager implementation


class _ManagerDirectiveModel(BaseModel):
    """Pydantic model for structured output from LLM manager decisions.

    Defines the JSON schema that StandardGroupChatManager expects from the LLM's
    response_format output. This ensures type-safe parsing and validation of manager
    directives.

    Attributes:
        next_agent: Name of participant to speak next (null when finishing)
        message: Optional instruction for the selected participant
        finish: Boolean indicating if the task is complete
        final_response: Final answer to the user (only when finish=True)

    Usage:
        The LLM is prompted to return this exact structure via structured output,
        which is then parsed and converted to GroupChatDirective for orchestrator routing.
    """

    next_agent: str | None = None
    message: str | None = None
    finish: bool = False
    final_response: str | None = None


DEFAULT_MANAGER_INSTRUCTIONS = """You are coordinating a team conversation to solve the user's task.
Select the next participant to respond or finish the task. When selecting an agent you MUST return
the JSON fields:
- next_agent: name of the participant who should act next (use null when finish is true)
- message: instruction for that participant (empty string if not needed)
- finish: boolean indicating if the task is complete
- final_response: when finish is true, provide the final answer to the user
"""


class StandardGroupChatManager:
    """LLM-backed manager that produces directives via structured output.

    This is the default manager implementation for group chat workflows. It uses an LLM
    to make speaker selection decisions based on conversation state, participant
    descriptions, and custom instructions.

    Coordination strategy:
    - Receives immutable state snapshot with full conversation history
    - Formats system prompt with instructions, task, and participant descriptions
    - Appends conversation context and structured output prompt
    - Calls LLM with response_format=_ManagerDirectiveModel for type safety
    - Parses LLM response and converts to GroupChatDirective

    Flexibility:
    - Custom instructions allow domain-specific coordination strategies
    - Participant descriptions guide the LLM's selection logic
    - Structured output ensures reliable parsing (no regex or brittle prompts)

    Example coordination patterns:
    - Round-robin: "Rotate between participants in order"
    - Task-based: "Select the participant best suited for the current sub-task"
    - Dependency-aware: "Only call analyst after researcher provides data"

    Args:
        chat_client: ChatClientProtocol implementation for LLM inference
        instructions: Custom system instructions (defaults to DEFAULT_MANAGER_INSTRUCTIONS)
        name: Display name for the manager in conversation history

    Raises:
        RuntimeError: If LLM response cannot be parsed into _ManagerDirectiveModel
                     If directive is missing next_agent when finish=False
                     If selected agent is not in participants
    """

    def __init__(
        self,
        chat_client: ChatClientProtocol,
        *,
        instructions: str | None = None,
        name: str | None = None,
    ) -> None:
        self._chat_client = chat_client
        self._instructions = instructions or DEFAULT_MANAGER_INSTRUCTIONS
        self._name = name or "GroupChatManager"

    @property
    def name(self) -> str:
        return self._name

    async def __call__(self, state: GroupChatStateSnapshot) -> GroupChatDirective:
        participants = state["participants"]
        task_message = state["task"]
        conversation = state["conversation"]

        participants_section = "\n".join(f"- {agent}: {description}" for agent, description in participants.items())

        system_message = ChatMessage(
            role=Role.SYSTEM,
            text=(f"{self._instructions}\n\nTask:\n{task_message.text}\n\nParticipants:\n{participants_section}"),
        )

        messages: list[ChatMessage] = [system_message, *conversation]
        messages.append(
            ChatMessage(
                role=Role.USER,
                text=(
                    "Return a JSON object with keys (next_agent, message, finish, final_response). "
                    "If you decide to finish, next_agent must be null."
                ),
            )
        )

        try:
            response = await self._chat_client.get_response(
                messages,
                response_format=_ManagerDirectiveModel,
            )
            directive_obj: _ManagerDirectiveModel
            if response.value is not None:
                directive_obj = _ManagerDirectiveModel.model_validate(response.value)
            elif response.messages:
                payload = response.messages[-1].text or "{}"
                directive_obj = _ManagerDirectiveModel.model_validate_json(payload)
            else:
                raise RuntimeError("LLM response did not contain structured output.")
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse manager directive: %s", exc)
            raise RuntimeError("Unable to parse manager directive from chat client response.") from exc

        if directive_obj.finish:
            final_text = directive_obj.final_response or ""
            return GroupChatDirective(
                finish=True,
                final_message=ChatMessage(
                    role=Role.ASSISTANT,
                    text=final_text,
                    author_name=self._name,
                ),
            )

        next_agent = directive_obj.next_agent
        if not next_agent:
            raise RuntimeError("Manager directive missing next_agent while finish is False.")
        if next_agent not in participants:
            raise RuntimeError(f"Manager selected unknown participant '{next_agent}'.")

        return GroupChatDirective(
            agent_name=next_agent,
            instruction=directive_obj.message or "",
        )


# endregion
