# Copyright (c) Microsoft. All rights reserved.

"""Base orchestrator class for group chat patterns.

This module provides BaseGroupChatOrchestrator, an abstract base class that
consolidates shared orchestration logic across GroupChat, Handoff, and Magentic patterns.
"""

import logging
from abc import ABC
from typing import Any

from .._types import ChatMessage
from ._executor import Executor
from ._orchestrator_helpers import ParticipantRegistry
from ._workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class BaseGroupChatOrchestrator(Executor, ABC):
    """Abstract base class for group chat orchestrators.

    Provides shared functionality for participant registration, routing,
    and round limit checking that is common across all group chat patterns.

    Subclasses must implement pattern-specific orchestration logic while
    inheriting the common participant management infrastructure.
    """

    def __init__(self, executor_id: str) -> None:
        """Initialize base orchestrator.

        Args:
            executor_id: Unique identifier for this orchestrator executor
        """
        super().__init__(executor_id)
        self._registry = ParticipantRegistry()

    def register_participant_entry(self, name: str, *, entry_id: str, is_agent: bool) -> None:
        """Record routing details for a participant's entry executor.

        This method provides a unified interface for registering participants
        across all orchestrator patterns, whether they are agents or custom executors.

        Args:
            name: Participant name (used for selection and tracking)
            entry_id: Executor ID for this participant's entry point
            is_agent: Whether this is an AgentExecutor (True) or custom Executor (False)
        """
        self._registry.register(name, entry_id=entry_id, is_agent=is_agent)

    async def _route_to_participant(
        self,
        participant_name: str,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[Any, Any],
        *,
        instruction: str | None = None,
        task: ChatMessage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Route a conversation to a participant.

        This method handles the dual envelope pattern:
        - AgentExecutors receive AgentExecutorRequest (messages only)
        - Custom executors receive GroupChatRequestMessage (full context)

        Args:
            participant_name: Name of the participant to route to
            conversation: Conversation history to send
            ctx: Workflow context for message routing
            instruction: Optional instruction from manager/orchestrator
            task: Optional task context
            metadata: Optional metadata dict

        Raises:
            ValueError: If participant is not registered
        """
        from ._agent_executor import AgentExecutorRequest
        from ._orchestrator_helpers import prepare_participant_request

        entry_id = self._registry.get_entry_id(participant_name)
        if entry_id is None:
            raise ValueError(f"No registered entry executor for participant '{participant_name}'.")

        if self._registry.is_agent(participant_name):
            # AgentExecutors receive simple message list
            await ctx.send_message(
                AgentExecutorRequest(messages=conversation, should_respond=True),
                target_id=entry_id,
            )
        else:
            # Custom executors receive full context envelope
            request = prepare_participant_request(
                participant_name=participant_name,
                conversation=conversation,
                instruction=instruction or "",
                task=task,
                metadata=metadata,
            )
            await ctx.send_message(request, target_id=entry_id)

    def _check_round_limit(
        self,
        current_round: int,
        max_rounds: int | None,
        *,
        pattern_name: str = "orchestrator",
    ) -> bool:
        """Check if round limit has been reached.

        Args:
            current_round: Current round index
            max_rounds: Maximum allowed rounds (None = no limit)
            pattern_name: Name for logging (e.g., "GroupChat", "Handoff")

        Returns:
            True if limit reached, False otherwise
        """
        if max_rounds is None:
            return False

        if current_round >= max_rounds:
            logger.warning(
                "%s reached max_rounds=%s; forcing completion.",
                pattern_name,
                max_rounds,
            )
            return True

        return False

    def snapshot_state(self) -> dict[str, Any]:
        """Capture current orchestrator state for checkpointing.

        Subclasses should override this to serialize pattern-specific state.
        Default implementation returns empty dict.
        """
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore orchestrator state from checkpoint.

        Subclasses should override this to deserialize pattern-specific state.
        Default implementation does nothing.
        """
        pass
