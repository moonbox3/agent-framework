# Copyright (c) Microsoft. All rights reserved.

"""Sequential builder for agent/executor workflows with shared conversation context.

Participants (SupportsAgentRun or Executor instances) run in order, sharing a
conversation along the chain. Agents append their assistant messages; custom executors
transform and return a refined `list[Message]`.

Wiring: input -> _InputToConversation -> participant1 -> ... -> participantN -> _EndWithConversation

The workflow's final `output` event is either the last agent's `AgentResponse` (when the
terminator is an agent) or the custom executor's `list[Message]`. With
`intermediate_outputs=True`, intermediate agents emit `data` events (via
`AgentExecutor.emit_intermediate_data`) so consumers can observe them separately from the
terminal answer.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from agent_framework import Message, SupportsAgentRun
from agent_framework._workflows._agent_executor import (
    AgentExecutor,
    AgentExecutorResponse,
)
from agent_framework._workflows._agent_utils import resolve_agent_id
from agent_framework._workflows._checkpoint import CheckpointStorage
from agent_framework._workflows._executor import (
    Executor,
    handler,
)
from agent_framework._workflows._message_utils import normalize_messages_input
from agent_framework._workflows._workflow import Workflow
from agent_framework._workflows._workflow_builder import WorkflowBuilder
from agent_framework._workflows._workflow_context import WorkflowContext

from ._orchestration_request_info import AgentApprovalExecutor

logger = logging.getLogger(__name__)


class _InputToConversation(Executor):
    """Normalizes initial input into a list[Message] conversation."""

    @handler
    async def from_str(self, prompt: str, ctx: WorkflowContext[list[Message]]) -> None:
        await ctx.send_message(normalize_messages_input(prompt))

    @handler
    async def from_message(self, message: Message, ctx: WorkflowContext[list[Message]]) -> None:
        await ctx.send_message(normalize_messages_input(message))

    @handler
    async def from_messages(self, messages: list[str | Message], ctx: WorkflowContext[list[Message]]) -> None:
        await ctx.send_message(normalize_messages_input(messages))


class _EndWithConversation(Executor):
    """Graph terminator for the sequential workflow.

    For custom-executor terminators, this emits the final `list[Message]` as an `output`
    event (the executor's own contract). For agent terminators it is a passive sink: the
    last `AgentExecutor` is itself registered as the workflow's output executor in
    `SequentialBuilder.build()`, so its `yield_output` calls — a single `AgentResponse`
    non-streaming, or per-chunk `AgentResponseUpdate` events streaming — become the
    workflow's outputs directly.

    Intermediate participants emit observation `data` events (via
    `AgentExecutor.emit_intermediate_data`) when `intermediate_outputs=True`; they never
    emit `output` events because output_executors is restricted to the terminator
    executor (the last agent or this node).
    """

    @handler
    async def end_with_messages(
        self,
        conversation: list[Message],
        ctx: WorkflowContext[Any, list[Message]],
    ) -> None:
        """Yield the final conversation when the last participant is a custom executor."""
        await ctx.yield_output(list(conversation))

    @handler
    async def end_with_agent_executor_response(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[Any],
    ) -> None:
        """Convert the agent-terminator response into a workflow output.

        When the last participant is a regular AgentExecutor (registered as the
        output executor), this node is NOT in output_executors so the yield is
        silently filtered — no duplicate output. When the last participant is an
        AgentApprovalExecutor (or similar wrapper), this node IS the output
        executor so the yield produces the workflow's terminal answer.
        """
        await ctx.yield_output(response.agent_response)


class SequentialBuilder:
    r"""High-level builder for sequential agent/executor workflows with shared context.

    - `participants=[...]` accepts a list of SupportsAgentRun (recommended) or Executor instances
    - Executors must define a handler that consumes list[Message] and sends out a list[Message]
    - The workflow wires participants in order, passing a list[Message] down the chain
    - Agents append their assistant messages to the conversation
    - Custom executors can transform/summarize and return a list[Message]
    - The final output is the conversation produced by the last participant

    Usage:

    .. code-block:: python

        from agent_framework_orchestrations import SequentialBuilder

        # With agent instances
        workflow = SequentialBuilder(participants=[agent1, agent2, summarizer_exec]).build()

        # Enable checkpoint persistence
        workflow = SequentialBuilder(participants=[agent1, agent2], checkpoint_storage=storage).build()

        # Enable request info for mid-workflow feedback (pauses before each agent)
        workflow = SequentialBuilder(participants=[agent1, agent2]).with_request_info().build()

        # Enable request info only for specific agents
        workflow = (
            SequentialBuilder(participants=[agent1, agent2, agent3])
            .with_request_info(agents=[agent2])  # Only pause before agent2
            .build()
        )
    """

    def __init__(
        self,
        *,
        participants: Sequence[SupportsAgentRun | Executor],
        checkpoint_storage: CheckpointStorage | None = None,
        chain_only_agent_responses: bool = False,
        intermediate_outputs: bool = False,
    ) -> None:
        """Initialize the SequentialBuilder.

        Args:
            participants: Sequence of agent or executor instances to run sequentially.
            checkpoint_storage: Optional checkpoint storage for enabling workflow state persistence.
            chain_only_agent_responses: If True, only agent responses are chained between agents.
                By default, the full conversation context is passed to the next agent. This also applies
                to Executor -> Agent transitions if the executor sends `AgentExecutorResponse`.
            intermediate_outputs: If True, enables intermediate outputs from agent participants.
        """
        self._participants: list[SupportsAgentRun | Executor] = []
        self._checkpoint_storage: CheckpointStorage | None = checkpoint_storage
        self._chain_only_agent_responses: bool = chain_only_agent_responses
        self._request_info_enabled: bool = False
        self._request_info_filter: set[str] | None = None
        self._intermediate_outputs: bool = intermediate_outputs

        self._set_participants(participants)

    def _set_participants(self, participants: Sequence[SupportsAgentRun | Executor]) -> None:
        """Set participants (internal)."""
        if self._participants:
            raise ValueError("participants already set.")

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
                # Treat non-Executor as agent-like (SupportsAgentRun). Structural checks can be brittle at runtime.
                pid = id(p)
                if pid in seen_agent_ids:
                    raise ValueError("Duplicate agent participant detected (same agent instance provided twice)")
                seen_agent_ids.add(pid)

        self._participants = list(participants)

    def with_request_info(
        self,
        *,
        agents: Sequence[str | SupportsAgentRun] | None = None,
    ) -> "SequentialBuilder":
        """Enable request info after agent participant responses.

        This enables human-in-the-loop (HIL) scenarios for the sequential orchestration.
        When enabled, the workflow pauses after each agent participant runs, emitting
        a request_info event (type='request_info') that allows the caller to review the conversation and optionally
        inject guidance for the agent participant to iterate. The caller provides input via
        the standard response_handler/request_info pattern.

        Simulated flow with HIL:
        Input -> [Agent Participant <-> Request Info] -> [Agent Participant <-> Request Info] -> ...

        Note: This is only available for agent participants. Executor participants can incorporate
        request info handling in their own implementation if desired.

        Args:
            agents: Optional list of agents names or agent factories to enable request info for.
                    If None, enables HIL for all agent participants.

        Returns:
            Self for fluent chaining
        """
        from ._orchestration_request_info import resolve_request_info_filter

        self._request_info_enabled = True
        self._request_info_filter = resolve_request_info_filter(list(agents) if agents else None)

        return self

    def _resolve_participants(self) -> list[Executor]:
        """Resolve participant instances into Executor objects.

        Wraps `SupportsAgentRun` participants as `AgentExecutor`. When `intermediate_outputs=True`,
        every wrapped agent except the final one is constructed with `emit_intermediate_data=True`
        so its responses surface as workflow `data` events without polluting the single `output`
        event reserved for the final answer.
        """
        if not self._participants:
            raise ValueError("No participants provided. Pass participants to the constructor.")

        participants: list[Executor | SupportsAgentRun] = self._participants

        context_mode: Literal["full", "last_agent", "custom"] | None = (
            "last_agent" if self._chain_only_agent_responses else None
        )

        last_idx = len(participants) - 1
        executors: list[Executor] = []
        for idx, p in enumerate(participants):
            if isinstance(p, Executor):
                executors.append(p)
            elif isinstance(p, SupportsAgentRun):
                emit_intermediate = self._intermediate_outputs and idx != last_idx
                if self._request_info_enabled and (
                    not self._request_info_filter or resolve_agent_id(p) in self._request_info_filter
                ):
                    # Handle request info enabled agents
                    executors.append(
                        AgentApprovalExecutor(
                            p,
                            context_mode=context_mode,
                            emit_intermediate_data=emit_intermediate,
                        )
                    )
                else:
                    executors.append(
                        AgentExecutor(
                            p,
                            context_mode=context_mode,
                            emit_intermediate_data=emit_intermediate,
                        )
                    )
            else:
                raise TypeError(f"Participants must be SupportsAgentRun or Executor instances. Got {type(p).__name__}.")

        return executors

    def build(self) -> Workflow:
        """Build and validate the sequential workflow.

        Wiring pattern:
        - _InputToConversation normalizes the initial input into list[Message]
        - For each participant in order:
            - Agent or AgentExecutor: receives the conversation/AgentExecutorResponse,
              produces an AgentExecutorResponse forwarded downstream
            - Custom Executor: receives list[Message] and forwards a list[Message]
        - The workflow's `output_executor` is selected based on the last participant:
            - Agent terminator: the last AgentExecutor itself (its yield_output is the answer)
            - Custom-executor terminator: `_EndWithConversation` (yields the final list[Message])
        """
        # Internal nodes
        input_conv = _InputToConversation(id="input-conversation")
        end = _EndWithConversation(id="end")

        # Resolve participants and participant factories to executors
        participants: list[Executor] = self._resolve_participants()

        last_executor = participants[-1]
        output_executors: list[Executor | SupportsAgentRun] = [
            last_executor if isinstance(last_executor, AgentExecutor) else end
        ]

        builder = WorkflowBuilder(
            start_executor=input_conv,
            checkpoint_storage=self._checkpoint_storage,
            output_executors=output_executors,
        )

        # Start of the chain is the input normalizer
        prior: Executor | SupportsAgentRun = input_conv
        for p in participants:
            builder.add_edge(prior, p)
            prior = p
        # Terminate with the final conversation
        builder.add_edge(prior, end)

        return builder.build()
