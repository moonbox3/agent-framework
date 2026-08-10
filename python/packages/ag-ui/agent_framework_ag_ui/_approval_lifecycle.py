# Copyright (c) Microsoft. All rights reserved.

"""Server-owned lifecycle for AG-UI approval-gated tool calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from agent_framework import Content


class ApprovalStatus(str, Enum):
    """Lifecycle state of one server-owned approval occurrence."""

    PENDING = "pending"
    CLAIMED = "claimed"
    EXECUTING = "executing"
    SETTLED = "settled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ApprovalExecutionOwner(str, Enum):
    """Runtime owner authorized to continue an approved occurrence."""

    LOCAL = "local"
    HOSTED = "hosted"
    DEFERRED = "deferred"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ApprovalOccurrenceIdentity:
    """Identity of one occurrence within a scoped server-owned thread."""

    thread_id: str
    occurrence_id: str
    interrupt_id: str
    call_id: str


@dataclass(frozen=True)
class ResumeDecision:
    """Canonical client decision presented to the approval lifecycle."""

    interrupt_id: str
    accepted: bool
    arguments: str | None
    name: str | None = None
    original_arguments: str | None = None


@dataclass
class ApprovalOccurrence:
    """Server-owned state for one approval-gated call occurrence."""

    identity: ApprovalOccurrenceIdentity
    thread_ids: tuple[str, ...]
    name: str
    arguments: str
    owner: ApprovalExecutionOwner
    status: ApprovalStatus = ApprovalStatus.PENDING
    replayable_results: list[ReplayableToolResult] = field(default_factory=list)
    decision: ResumeDecision | None = None
    outcome: ApprovalOutcome | None = None


@dataclass(frozen=True)
class AuthorizedExecution:
    """Authority for one Pending Tool Transition Owner to continue a call."""

    identity: ApprovalOccurrenceIdentity
    name: str
    arguments: str
    owner: ApprovalExecutionOwner


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


@dataclass(frozen=True)
class ApprovalBatchDecision:
    """Validated batch result containing new authority and retained outcomes."""

    authorized_executions: tuple[AuthorizedExecution, ...]
    retained_outcomes: tuple[ApprovalOutcome, ...] = ()

    def __iter__(self) -> Iterator[AuthorizedExecution]:
        """Iterate newly authorized executions for compatibility with existing callers."""
        return iter(self.authorized_executions)


class ApprovalLifecycle:
    """Own registration, authority transitions, and settlement for approvals."""

    def __init__(self) -> None:
        self._occurrences: dict[ApprovalOccurrenceIdentity, ApprovalOccurrence] = {}
        self._pending_by_interrupt: dict[tuple[str, str], ApprovalOccurrenceIdentity] = {}
        self._terminal_by_interrupt: dict[tuple[str, str], ApprovalOccurrenceIdentity] = {}

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
        return self.register_local_aliases(
            thread_ids=[thread_id],
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )

    def register_local_aliases(
        self,
        *,
        thread_ids: list[str],
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register one occurrence under its trusted scoped-thread aliases."""
        return self._register_aliases(
            thread_ids=thread_ids,
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
            owner=ApprovalExecutionOwner.LOCAL,
        )

    def register_hosted(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register one server-generated hosted approval occurrence."""
        return self.register_hosted_aliases(
            thread_ids=[thread_id],
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )

    def register_hosted_aliases(
        self,
        *,
        thread_ids: list[str],
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register one hosted occurrence under its trusted scoped-thread aliases."""
        return self._register_aliases(
            thread_ids=thread_ids,
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
            owner=ApprovalExecutionOwner.HOSTED,
        )

    def register_unowned(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register an occurrence that has no executable transition owner."""
        return self.register_unowned_aliases(
            thread_ids=[thread_id],
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )

    def register_deferred(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register one occurrence owned by the in-run transition pipeline."""
        return self.register_deferred_aliases(
            thread_ids=[thread_id],
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )

    def register_deferred_aliases(
        self,
        *,
        thread_ids: list[str],
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register one deferred occurrence under its trusted scoped-thread aliases."""
        return self._register_aliases(
            thread_ids=thread_ids,
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
            owner=ApprovalExecutionOwner.DEFERRED,
        )

    def register_unowned_aliases(
        self,
        *,
        thread_ids: list[str],
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
    ) -> ApprovalOccurrence:
        """Register an unowned occurrence under its trusted scoped-thread aliases."""
        return self._register_aliases(
            thread_ids=thread_ids,
            interrupt_id=interrupt_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
            owner=ApprovalExecutionOwner.UNAVAILABLE,
        )

    def _register_aliases(
        self,
        *,
        thread_ids: list[str],
        interrupt_id: str,
        call_id: str,
        name: str,
        arguments: str,
        owner: ApprovalExecutionOwner,
    ) -> ApprovalOccurrence:
        unique_thread_ids = tuple(dict.fromkeys(thread_ids))
        if not unique_thread_ids:
            raise ValueError("An approval occurrence requires at least one scoped thread identity.")
        existing_identities = {
            identity
            for thread_id in unique_thread_ids
            if (identity := self._pending_by_interrupt.get((thread_id, interrupt_id))) is not None
        }
        if len(existing_identities) > 1:
            raise ValueError("Approval aliases resolve to different pending occurrences.")
        if existing_identities:
            occurrence = self._occurrences[next(iter(existing_identities))]
            if occurrence.status is not ApprovalStatus.PENDING:
                raise ValueError(f"Approval occurrence is not pending: {occurrence.status}.")
            if (
                occurrence.identity.call_id != call_id
                or occurrence.name != name
                or occurrence.arguments != arguments
                or occurrence.owner is not owner
            ):
                raise ValueError("Approval alias conflicts with an existing pending occurrence.")
            occurrence.thread_ids = tuple(dict.fromkeys((*occurrence.thread_ids, *unique_thread_ids)))
            for thread_id in occurrence.thread_ids:
                self._pending_by_interrupt[(thread_id, interrupt_id)] = occurrence.identity
            return occurrence

        identity = ApprovalOccurrenceIdentity(
            thread_id=unique_thread_ids[0],
            occurrence_id=str(uuid4()),
            interrupt_id=interrupt_id,
            call_id=call_id,
        )
        occurrence = ApprovalOccurrence(
            identity=identity,
            thread_ids=unique_thread_ids,
            name=name,
            arguments=arguments,
            owner=owner,
        )
        self._occurrences[identity] = occurrence
        for thread_id in unique_thread_ids:
            self._pending_by_interrupt[(thread_id, interrupt_id)] = identity
        return occurrence

    def get(self, identity: ApprovalOccurrenceIdentity) -> ApprovalOccurrence:
        """Return server-owned state for a registered occurrence."""
        return self._occurrences[identity]

    def decision_context(self, *, thread_id: str, interrupt_id: str) -> tuple[str, str]:
        """Return canonical server-owned call data needed to normalize a typed retry."""
        key = (thread_id, interrupt_id)
        identity = self._pending_by_interrupt.get(key) or self._terminal_by_interrupt[key]
        occurrence = self._occurrences[identity]
        return occurrence.name, occurrence.arguments

    def claim(self, *, thread_id: str, decision: ResumeDecision) -> AuthorizedExecution:
        """Validate and reserve one accepted decision before execution."""
        if not decision.accepted:
            raise ValueError("A rejected decision cannot authorize execution.")
        batch = self.claim_batch(thread_id=thread_id, decisions=[decision])
        if not batch.authorized_executions:
            raise ValueError("A duplicate settled decision cannot authorize execution again.")
        return batch.authorized_executions[0]

    def claim_batch(
        self,
        *,
        thread_id: str,
        decisions: list[ResumeDecision],
    ) -> ApprovalBatchDecision:
        """Validate a complete decision batch before reserving accepted occurrences."""
        resolved: list[tuple[ResumeDecision, ApprovalOccurrence, bool]] = []
        seen_interrupt_ids: set[str] = set()
        for decision in decisions:
            if decision.interrupt_id in seen_interrupt_ids:
                raise ValueError(f"Approval batch repeats interrupt: {decision.interrupt_id}.")
            seen_interrupt_ids.add(decision.interrupt_id)
            key = (thread_id, decision.interrupt_id)
            identity = self._pending_by_interrupt.get(key)
            is_terminal = identity is None
            if identity is None:
                identity = self._terminal_by_interrupt[key]
            occurrence = self._occurrences[identity]
            if is_terminal:
                if occurrence.status is ApprovalStatus.EXPIRED:
                    raise ValueError("Approval authority has expired.")
                retained_decision = occurrence.decision
                if (
                    retained_decision is None
                    or retained_decision.accepted != decision.accepted
                    or (decision.name is not None and retained_decision.name != decision.name)
                    or (decision.arguments is not None and retained_decision.arguments != decision.arguments)
                    or (
                        decision.original_arguments is not None
                        and retained_decision.original_arguments != decision.original_arguments
                    )
                ):
                    raise ValueError("Approval decision conflicts with the retained terminal decision.")
                if occurrence.outcome is None:
                    raise ValueError(f"Approval occurrence has no replayable terminal outcome: {occurrence.status}.")
                resolved.append((decision, occurrence, True))
                continue
            if occurrence.status is not ApprovalStatus.PENDING:
                raise ValueError(f"Approval occurrence is not pending: {occurrence.status}.")
            if decision.name is not None and decision.name != occurrence.name:
                raise ValueError("Approval decision tool name does not match the registered occurrence.")
            canonical_arguments = decision.arguments
            if canonical_arguments is None:
                raise ValueError("A pending approval decision must include canonical arguments.")
            if (decision.original_arguments or canonical_arguments) != occurrence.arguments:
                raise ValueError("Approval decision arguments do not match the registered occurrence.")
            resolved.append((decision, occurrence, False))
        intents: list[AuthorizedExecution] = []
        retained_outcomes: list[ApprovalOutcome] = []
        for decision, occurrence, is_terminal in resolved:
            if is_terminal:
                if occurrence.outcome is None:
                    raise RuntimeError("Validated terminal approval is missing its retained outcome.")
                retained_outcomes.append(occurrence.outcome)
                continue
            occurrence.decision = decision
            if not decision.accepted:
                result = Content.from_function_result(
                    call_id=occurrence.identity.call_id,
                    result="Error: Tool call invocation was rejected by user.",
                )
                occurrence.replayable_results = [ReplayableToolResult(content=result)]
                occurrence.outcome = ApprovalOutcome(
                    identity=occurrence.identity,
                    replayable_results=tuple(occurrence.replayable_results),
                    result_group=(result,),
                )
                occurrence.status = ApprovalStatus.REJECTED
                self._remove_pending_aliases(occurrence)
                continue
            if decision.arguments is None:
                raise RuntimeError("Validated pending approval is missing canonical arguments.")
            occurrence.arguments = decision.arguments
            if occurrence.owner is ApprovalExecutionOwner.UNAVAILABLE:
                continue
            occurrence.status = ApprovalStatus.CLAIMED
            intents.append(
                AuthorizedExecution(
                    identity=occurrence.identity,
                    name=occurrence.name,
                    arguments=occurrence.arguments,
                    owner=occurrence.owner,
                )
            )
        return ApprovalBatchDecision(
            authorized_executions=tuple(intents),
            retained_outcomes=tuple(retained_outcomes),
        )

    def cancel_batch(self, *, thread_id: str, interrupt_ids: list[str]) -> None:
        """Validate and cancel selected occurrences without changing their siblings."""
        occurrences: list[ApprovalOccurrence] = []
        seen_interrupt_ids: set[str] = set()
        for interrupt_id in interrupt_ids:
            if interrupt_id in seen_interrupt_ids:
                raise ValueError(f"Approval batch repeats interrupt: {interrupt_id}.")
            seen_interrupt_ids.add(interrupt_id)
            key = (thread_id, interrupt_id)
            identity = self._pending_by_interrupt.get(key)
            if identity is None:
                identity = self._terminal_by_interrupt[key]
                terminal = self._occurrences[identity]
                if terminal.status is ApprovalStatus.CANCELLED:
                    continue
                raise ValueError("Approval cancellation conflicts with the retained terminal decision.")
            occurrence = self._occurrences[identity]
            if occurrence.status is not ApprovalStatus.PENDING:
                raise ValueError(f"Approval occurrence is not pending: {occurrence.status}.")
            occurrences.append(occurrence)

        for occurrence in occurrences:
            occurrence.status = ApprovalStatus.CANCELLED
            self._remove_pending_aliases(occurrence)

    def expire_batch(self, *, thread_id: str, interrupt_ids: list[str]) -> None:
        """Expire pending authority without permitting later execution."""
        occurrences: list[ApprovalOccurrence] = []
        seen_interrupt_ids: set[str] = set()
        for interrupt_id in interrupt_ids:
            if interrupt_id in seen_interrupt_ids:
                raise ValueError(f"Approval batch repeats interrupt: {interrupt_id}.")
            seen_interrupt_ids.add(interrupt_id)
            identity = self._pending_by_interrupt[(thread_id, interrupt_id)]
            occurrence = self._occurrences[identity]
            if occurrence.status is not ApprovalStatus.PENDING:
                raise ValueError(f"Approval occurrence is not pending: {occurrence.status}.")
            occurrences.append(occurrence)

        for occurrence in occurrences:
            occurrence.status = ApprovalStatus.EXPIRED
            self._remove_pending_aliases(occurrence)

    def _remove_pending_aliases(self, occurrence: ApprovalOccurrence) -> None:
        for thread_id in occurrence.thread_ids:
            self._pending_by_interrupt.pop((thread_id, occurrence.identity.interrupt_id), None)
            self._terminal_by_interrupt[(thread_id, occurrence.identity.interrupt_id)] = occurrence.identity

    def begin_execution(self, intent: AuthorizedExecution, *, owner: ApprovalExecutionOwner) -> None:
        """Mark a claimed occurrence immediately before its owner may invoke a tool."""
        occurrence = self._occurrences[intent.identity]
        if intent.owner is not owner or occurrence.owner is not owner:
            raise ValueError(
                f"Approval occurrence belongs to the {occurrence.owner.value} transition owner, not {owner.value}."
            )
        if occurrence.status is not ApprovalStatus.CLAIMED:
            raise ValueError(f"Approval occurrence is not claimed: {occurrence.status}.")
        occurrence.status = ApprovalStatus.EXECUTING

    def settle(self, intent: AuthorizedExecution, results: list[Content]) -> ApprovalOutcome:
        """Settle an executing occurrence with results under its original call identity."""
        occurrence = self._occurrences[intent.identity]
        if intent.owner is not ApprovalExecutionOwner.LOCAL or occurrence.owner is not ApprovalExecutionOwner.LOCAL:
            raise ValueError("Only the local transition owner can settle a local execution result.")
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
        self._remove_pending_aliases(occurrence)
        outcome = ApprovalOutcome(
            identity=occurrence.identity,
            replayable_results=tuple(replayable_results),
            result_group=tuple(results),
        )
        occurrence.outcome = outcome
        return outcome

    def settle_forwarded(
        self,
        intent: AuthorizedExecution,
        results: list[Content],
        *,
        owner: ApprovalExecutionOwner,
    ) -> ApprovalOutcome:
        """Record a forwarded owner's outcome under its original occurrence."""
        occurrence = self._occurrences[intent.identity]
        if owner not in {ApprovalExecutionOwner.HOSTED, ApprovalExecutionOwner.DEFERRED}:
            raise ValueError("Only a forwarding transition owner can record a forwarded outcome.")
        if intent.owner is not owner or occurrence.owner is not owner:
            raise ValueError(
                f"Approval occurrence belongs to the {occurrence.owner.value} transition owner, not {owner.value}."
            )
        if occurrence.status is not ApprovalStatus.EXECUTING:
            raise ValueError(f"Approval occurrence is not executing: {occurrence.status}.")
        replayable_results = [
            ReplayableToolResult(content=result)
            for result in results
            if result.type == "function_result" and result.call_id == occurrence.identity.call_id
        ]
        forwarded_responses = [
            result
            for result in results
            if result.type == "function_approval_response"
            and result.approved
            and result.function_call is not None
            and result.function_call.call_id == occurrence.identity.call_id
        ]
        if len(replayable_results) + len(forwarded_responses) != 1:
            raise ValueError("A hosted approval must record exactly one outcome for its original call.")
        outcome = ApprovalOutcome(
            identity=occurrence.identity,
            replayable_results=tuple(replayable_results),
            result_group=tuple(results),
        )
        occurrence.replayable_results = replayable_results
        occurrence.status = ApprovalStatus.SETTLED
        occurrence.outcome = outcome
        self._remove_pending_aliases(occurrence)
        return outcome

    def defer(self, intent: AuthorizedExecution, results: list[Content]) -> ApprovalOutcome:
        """Return an execution that yielded only follow-up requests to pending."""
        occurrence = self._occurrences[intent.identity]
        if intent.owner is not ApprovalExecutionOwner.LOCAL or occurrence.owner is not ApprovalExecutionOwner.LOCAL:
            raise ValueError("Only the local transition owner can defer a local execution.")
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
        lifecycle.begin_execution(intent, owner=ApprovalExecutionOwner.LOCAL)
        results = await self._executor()
        if not any(result.type == "function_result" for result in results):
            return lifecycle.defer(intent, results)
        return lifecycle.settle(intent, results)


class ForwardedPendingToolTransitionOwner:
    """Forward an authorized decision to a non-transport transition owner."""

    def __init__(
        self,
        forwarder: Callable[[], Awaitable[list[Content]]],
        *,
        owner: ApprovalExecutionOwner,
    ) -> None:
        self._forwarder = forwarder
        self._owner = owner

    async def forward(
        self,
        intent: AuthorizedExecution,
        *,
        lifecycle: ApprovalLifecycle,
    ) -> list[Content]:
        """Forward one hosted approval after lifecycle authorization."""
        lifecycle.begin_execution(intent, owner=self._owner)
        return await self._forwarder()

    def record_outcome(
        self,
        intent: AuthorizedExecution,
        results: list[Content],
        *,
        lifecycle: ApprovalLifecycle,
    ) -> ApprovalOutcome:
        """Record the hosted owner's outcome against the authorized occurrence."""
        return lifecycle.settle_forwarded(intent, results, owner=self._owner)

    async def execute(
        self,
        intent: AuthorizedExecution,
        *,
        lifecycle: ApprovalLifecycle,
    ) -> ApprovalOutcome:
        """Forward and immediately record an available hosted outcome."""
        return self.record_outcome(intent, await self.forward(intent, lifecycle=lifecycle), lifecycle=lifecycle)


class HostedPendingToolTransitionOwner(ForwardedPendingToolTransitionOwner):
    """Forward an authorized decision through the hosted transition owner."""

    def __init__(self, forwarder: Callable[[], Awaitable[list[Content]]]) -> None:
        super().__init__(forwarder, owner=ApprovalExecutionOwner.HOSTED)


class DeferredPendingToolTransitionOwner(ForwardedPendingToolTransitionOwner):
    """Forward an authorized decision to the in-run transition pipeline."""

    def __init__(self, forwarder: Callable[[], Awaitable[list[Content]]]) -> None:
        super().__init__(forwarder, owner=ApprovalExecutionOwner.DEFERRED)
