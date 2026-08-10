# Copyright (c) Microsoft. All rights reserved.

"""Server-owned lifecycle for AG-UI approval-gated tool calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from agent_framework import Content


class ApprovalStatus(str, Enum):
    """Lifecycle state of one server-owned approval occurrence."""

    PENDING = "pending"
    CLAIMED = "claimed"
    EXECUTING = "executing"
    SETTLED = "settled"


@dataclass(frozen=True)
class ApprovalOccurrenceIdentity:
    """Identity of one approval occurrence within a server-owned thread."""

    thread_id: str
    interrupt_id: str
    call_id: str


@dataclass(frozen=True)
class ResumeDecision:
    """Canonical client decision presented to the approval lifecycle."""

    interrupt_id: str
    accepted: bool
    arguments: str
    original_arguments: str | None = None


@dataclass
class ApprovalOccurrence:
    """Server-owned state for one approval-gated call occurrence."""

    identity: ApprovalOccurrenceIdentity
    name: str
    arguments: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    replayable_results: list[ReplayableToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class AuthorizedExecution:
    """Authority for a Pending Tool Transition Owner to execute one local call."""

    identity: ApprovalOccurrenceIdentity
    name: str
    arguments: str


@dataclass(frozen=True)
class ReplayableToolResult:
    """A settled tool result retained under its original call identity."""

    content: Content


@dataclass(frozen=True)
class ApprovalOutcome:
    """Terminal outcome retained for one approval occurrence."""

    identity: ApprovalOccurrenceIdentity
    replayable_results: tuple[ReplayableToolResult, ...]
    result_group: tuple[Content, ...]


class ApprovalLifecycle:
    """Own registration, authority transitions, and settlement for approvals."""

    def __init__(self) -> None:
        self._occurrences: dict[ApprovalOccurrenceIdentity, ApprovalOccurrence] = {}
        self._pending_by_interrupt: dict[tuple[str, str], ApprovalOccurrenceIdentity] = {}

    def register_local(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register one server-generated local approval occurrence."""
        identity = ApprovalOccurrenceIdentity(
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            call_id=call_id,
        )
        occurrence = ApprovalOccurrence(identity=identity, name=name, arguments=arguments)
        self._occurrences[identity] = occurrence
        self._pending_by_interrupt[(thread_id, interrupt_id)] = identity
        return occurrence

    def get(self, identity: ApprovalOccurrenceIdentity) -> ApprovalOccurrence:
        """Return server-owned state for a registered occurrence."""
        return self._occurrences[identity]

    def claim(self, *, thread_id: str, decision: ResumeDecision) -> AuthorizedExecution:
        """Validate and reserve one accepted decision before execution."""
        identity = self._pending_by_interrupt[(thread_id, decision.interrupt_id)]
        occurrence = self._occurrences[identity]
        if occurrence.status is not ApprovalStatus.PENDING:
            raise ValueError(f"Approval occurrence is not pending: {occurrence.status}.")
        if not decision.accepted:
            raise ValueError("A rejected decision cannot authorize execution.")
        if (decision.original_arguments or decision.arguments) != occurrence.arguments:
            raise ValueError("Approval decision arguments do not match the registered occurrence.")
        occurrence.arguments = decision.arguments
        occurrence.status = ApprovalStatus.CLAIMED
        return AuthorizedExecution(identity=identity, name=occurrence.name, arguments=occurrence.arguments)

    def begin_execution(self, intent: AuthorizedExecution) -> None:
        """Mark a claimed occurrence immediately before its owner may invoke a tool."""
        occurrence = self._occurrences[intent.identity]
        if occurrence.status is not ApprovalStatus.CLAIMED:
            raise ValueError(f"Approval occurrence is not claimed: {occurrence.status}.")
        occurrence.status = ApprovalStatus.EXECUTING

    def settle(self, intent: AuthorizedExecution, results: list[Content]) -> ApprovalOutcome:
        """Settle an executing occurrence with results under its original call identity."""
        occurrence = self._occurrences[intent.identity]
        if occurrence.status is not ApprovalStatus.EXECUTING:
            raise ValueError(f"Approval occurrence is not executing: {occurrence.status}.")
        replayable_results = [
            ReplayableToolResult(content=result)
            for result in results
            if result.type == "function_result" and result.call_id == occurrence.identity.call_id
        ]
        if len(replayable_results) != 1:
            raise ValueError("A settled local approval must produce exactly one result for its original call.")
        occurrence.replayable_results = replayable_results
        occurrence.status = ApprovalStatus.SETTLED
        self._pending_by_interrupt.pop((intent.identity.thread_id, intent.identity.interrupt_id), None)
        return ApprovalOutcome(
            identity=occurrence.identity,
            replayable_results=tuple(replayable_results),
            result_group=tuple(results),
        )

    def defer(self, intent: AuthorizedExecution, results: list[Content]) -> ApprovalOutcome:
        """Return an execution that yielded only follow-up requests to pending."""
        occurrence = self._occurrences[intent.identity]
        if occurrence.status is not ApprovalStatus.EXECUTING:
            raise ValueError(f"Approval occurrence is not executing: {occurrence.status}.")
        occurrence.status = ApprovalStatus.PENDING
        return ApprovalOutcome(identity=occurrence.identity, replayable_results=(), result_group=tuple(results))


class LocalPendingToolTransitionOwner:
    """Execute an authorized call through the process-local transition owner."""

    def __init__(self, executor: Callable[[], Awaitable[list[Content]]]) -> None:
        self._executor = executor

    async def execute(
        self,
        intent: AuthorizedExecution,
        *,
        lifecycle: ApprovalLifecycle,
    ) -> ApprovalOutcome:
        """Execute and settle one call after lifecycle authorization."""
        lifecycle.begin_execution(intent)
        results = await self._executor()
        if not any(result.type == "function_result" for result in results):
            return lifecycle.defer(intent, results)
        return lifecycle.settle(intent, results)
