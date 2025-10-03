# Copyright (c) Microsoft. All rights reserved.

"""Sequential builder for agent/executor workflows with shared conversation context.

This module provides a high-level, agent-focused API to assemble a sequential
workflow where:
- Participants are a sequence of AgentProtocol instances or Executors
- A shared conversation context (list[ChatMessage]) is owned by ConversationSession
- Internal adapters keep the session synchronized and expose resumable handles

Wiring pattern:
    input -> ConversationEntryExecutor -> participant -> ConversationProjectionExecutor -> ... -> ConversationOutputExecutor

Why include the internal adapters?
- `ConversationEntryExecutor` normalizes initial input, seeds ConversationSession, and hides plumbing from visualization.
- `ConversationProjectionExecutor` reconciles AgentExecutorResponse or custom list outputs with the session and forwards a list[ChatMessage].
- `ConversationOutputExecutor` yields a `ConversationSnapshot` containing both transcript and ConversationHandle.

Adapters are tagged with `visibility=internal` so default visualization omits them while maintaining strong typing and observability hooks when diagnostics are enabled.
"""  # noqa: E501

import logging
from collections.abc import Sequence
from typing import Any

from agent_framework import AgentProtocol, ChatMessage, Role

from ._checkpoint import CheckpointStorage
from ._conversation import ConversationHandle, ConversationManager, ConversationSnapshot
from ._executor import (
    AgentExecutor,
    AgentExecutorResponse,
    Executor,
    handler,
)
from ._workflow import Workflow, WorkflowBuilder
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class ConversationEntryExecutor(Executor):
    """Normalizes workflow input and seeds the shared ConversationSession."""

    def __init__(self, *, id: str = "conversation-entry") -> None:
        super().__init__(id=id)
        self.metadata["visibility"] = "internal"

    async def _initialize(self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]) -> None:
        manager = await ConversationManager.ensure_on_shared_state(ctx.shared_state)
        await manager.ensure_session()
        await manager.replace_transcript(messages)
        await ctx.send_message(list(messages))

    @handler
    async def from_str(self, prompt: str, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        await self._initialize([ChatMessage(Role.USER, text=prompt)], ctx)

    @handler
    async def from_message(self, message: ChatMessage, ctx: WorkflowContext[list[ChatMessage]]) -> None:  # type: ignore[name-defined]
        await self._initialize([message], ctx)

    @handler
    async def from_messages(self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]) -> None:  # type: ignore[name-defined]
        await self._initialize(list(messages), ctx)

    @handler
    async def from_handle(self, handle: ConversationHandle, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        manager = await ConversationManager.ensure_on_shared_state(ctx.shared_state)
        session = await manager.ensure_session(handle=handle)
        await ctx.send_message(list(session.transcript))


class ConversationProjectionExecutor(Executor):
    """Reconciles downstream outputs with the active ConversationSession."""

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self.metadata["visibility"] = "internal"

    @handler
    async def from_agent_response(self, response: AgentExecutorResponse, ctx: WorkflowContext[list[ChatMessage]]) -> None:
        manager = await ConversationManager.ensure_on_shared_state(ctx.shared_state)
        await manager.ensure_session()
        if response.full_conversation is None:
            transcript = list(response.agent_run_response.messages)
        else:
            transcript = list(response.full_conversation)
        await manager.replace_transcript(transcript)
        await ctx.send_message(transcript)

    @handler
    async def from_messages(self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]) -> None:  # type: ignore[name-defined]
        manager = await ConversationManager.ensure_on_shared_state(ctx.shared_state)
        await manager.ensure_session()
        transcript = list(messages)
        await manager.replace_transcript(transcript)
        await ctx.send_message(transcript)


class ConversationOutputExecutor(Executor):
    """Emits a ConversationSnapshot containing transcript and resumable handle."""

    def __init__(self, *, id: str = "conversation-output") -> None:
        super().__init__(id=id)
        self.metadata["visibility"] = "internal"

    @handler
    async def end(self, conversation: list[ChatMessage], ctx: WorkflowContext[Any, ConversationSnapshot]) -> None:
        manager = await ConversationManager.ensure_on_shared_state(ctx.shared_state)
        await manager.ensure_session()
        transcript = list(conversation)
        await manager.replace_transcript(transcript)
        handle = manager.handle
        snapshot = ConversationSnapshot(messages=transcript, handle=ConversationHandle(handle.session_id, handle.revision))
        await ctx.yield_output(snapshot)


class SequentialBuilder:
    r"""High-level builder for sequential agent/executor workflows with shared context.

    - `participants([...])` accepts a list of AgentProtocol (recommended) or Executor
    - The workflow wires participants in order, passing a list[ChatMessage] down the chain
    - Agents append their assistant messages to the conversation
    - Custom executors can transform/summarize and return a list[ChatMessage]
    - The final output is the conversation produced by the last participant

    Usage:

    .. code-block:: python

        from agent_framework import SequentialBuilder

        workflow = SequentialBuilder().participants([agent1, agent2, summarizer_exec]).build()

        # Enable checkpoint persistence
        workflow = SequentialBuilder().participants([agent1, agent2]).with_checkpointing(storage).build()
    """

    def __init__(self) -> None:
        self._participants: list[AgentProtocol | Executor] = []
        self._checkpoint_storage: CheckpointStorage | None = None

    def participants(self, participants: Sequence[AgentProtocol | Executor]) -> "SequentialBuilder":
        """Define the ordered participants for this sequential workflow.

        Accepts AgentProtocol instances (auto-wrapped as AgentExecutor) or Executor instances.
        Raises if empty or duplicates are provided for clarity.
        """
        if not participants:
            raise ValueError("participants cannot be empty")

        # Defensive duplicate detection
        seen_agent_ids: set[int] = set()
        seen_executor_ids: set[str] = set()
        for p in participants:
            if isinstance(p, Executor):
                if p.id in seen_executor_ids:
                    raise ValueError(f"Duplicate executor participant detected: id '{p.id}'")
                seen_executor_ids.add(p.id)
            else:
                # Treat non-Executor as agent-like (AgentProtocol). Structural checks can be brittle at runtime.
                pid = id(p)
                if pid in seen_agent_ids:
                    raise ValueError("Duplicate agent participant detected (same agent instance provided twice)")
                seen_agent_ids.add(pid)

        self._participants = list(participants)
        return self

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> "SequentialBuilder":
        """Enable checkpointing for the built workflow using the provided storage."""
        self._checkpoint_storage = checkpoint_storage
        return self

    def build(self) -> Workflow:
        """Build and validate the sequential workflow.

        Wiring pattern:
        - ConversationEntryExecutor normalizes the initial input and initializes ConversationSession.
        - For each participant: execute participant, then apply ConversationProjectionExecutor to sync session state.
        - ConversationOutputExecutor emits the final ConversationSnapshot and the workflow becomes idle.
        """
        if not self._participants:
            raise ValueError("No participants provided. Call .participants([...]) first.")

        # Internal nodes
        entry = ConversationEntryExecutor(id="conversation-entry")
        output = ConversationOutputExecutor(id="conversation-output")

        builder = WorkflowBuilder()
        builder.set_start_executor(entry)

        # Start of the chain is the input normalizer
        prior: Executor | AgentProtocol = entry

        for p in self._participants:
            builder.add_edge(prior, p)
            label = p.id if isinstance(p, Executor) else getattr(p, "name", None) or p.__class__.__name__
            projection = ConversationProjectionExecutor(id=f"conversation-view:{label}")
            builder.add_edge(p, projection)
            prior = projection

        # Terminate with the final conversation
        builder.add_edge(prior, output)

        if self._checkpoint_storage is not None:
            builder = builder.with_checkpointing(self._checkpoint_storage)

        return builder.build()
