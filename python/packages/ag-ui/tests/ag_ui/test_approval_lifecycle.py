# Copyright (c) Microsoft. All rights reserved.

"""Behavior tests for the approval batch continuity lifecycle seam."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from agent_framework import Content

from agent_framework_ag_ui._approval_lifecycle import (
    ApprovalCapacityError,
    ApprovalClaimConflictError,
    ApprovalExecutionOwner,
    ApprovalIndeterminateError,
    ApprovalLifecycle,
    ApprovalSettlementConflictError,
    ApprovalSnapshotStatus,
    ApprovalStatus,
    ClaimRecoveryPolicy,
    HostedPendingToolTransitionOwner,
    LocalPendingToolTransitionOwner,
    ResumeDecision,
)


async def test_local_approval_crosses_lifecycle_before_execution_and_settlement() -> None:
    """One accepted local occurrence is claimed, executed by its owner, and settled."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="get_weather",
        arguments='{"city":"Seattle"}',
    )
    observed_statuses = [occurrence.status]

    intent = lifecycle.claim(
        thread_id="thread-1",
        decision=ResumeDecision(
            interrupt_id="approval-1",
            accepted=True,
            arguments='{"city":"Seattle"}',
        ),
    )
    observed_statuses.append(lifecycle.get(occurrence.identity).status)

    async def execute_authorized_call() -> list[Content]:
        observed_statuses.append(lifecycle.get(occurrence.identity).status)
        return [Content.from_function_result(call_id="call-1", result="Sunny")]

    owner = LocalPendingToolTransitionOwner(execute_authorized_call)
    outcome = await owner.execute(intent, lifecycle=lifecycle)
    observed_statuses.append(lifecycle.get(occurrence.identity).status)

    assert observed_statuses == [
        ApprovalStatus.PENDING,
        ApprovalStatus.CLAIMED,
        ApprovalStatus.EXECUTING,
        ApprovalStatus.SETTLED,
    ]
    assert [result.content.call_id for result in outcome.replayable_results] == ["call-1"]
    assert [result.content.result for result in outcome.replayable_results] == ["Sunny"]


def test_active_occurrence_is_not_evicted_when_capacity_is_exhausted() -> None:
    """Storage pressure fails explicitly instead of discarding pending authority."""
    lifecycle = ApprovalLifecycle(max_entries=1)
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"secret":"first"}',
    )

    with pytest.raises(ApprovalCapacityError):
        lifecycle.register_local(
            thread_id="thread-2",
            interrupt_id="approval-2",
            call_id="call-2",
            name="write_record",
            arguments='{"secret":"second"}',
        )

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.PENDING


async def test_terminal_outcome_expires_only_after_configured_retention_window() -> None:
    """Duplicate execution protection lasts for the configured terminal retention window."""
    now = 100.0
    lifecycle = ApprovalLifecycle(max_entries=1, terminal_retention_seconds=30, clock=lambda: now)
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )
    decision = ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}")
    intent = lifecycle.claim(thread_id="thread-1", decision=decision)

    async def execute() -> list[Content]:
        return [Content.from_function_result(call_id="call-1", result="done")]

    outcome = await LocalPendingToolTransitionOwner(execute).execute(intent, lifecycle=lifecycle)
    now = 129.0
    assert lifecycle.claim_batch(thread_id="thread-1", decisions=[decision]).retained_outcomes == (outcome,)

    now = 131.0
    replacement = lifecycle.register_local(
        thread_id="thread-2",
        interrupt_id="approval-2",
        call_id="call-2",
        name="write_record",
        arguments="{}",
    )

    assert replacement.status is ApprovalStatus.PENDING
    with pytest.raises(KeyError):
        lifecycle.claim_batch(thread_id="thread-1", decisions=[decision])
    with pytest.raises(KeyError):
        lifecycle.get(occurrence.identity)


def test_indeterminate_occurrence_remains_protected_after_terminal_retention_window() -> None:
    """Uncertain execution is never aged out as a retryable terminal tombstone."""
    now = 100.0
    lifecycle = ApprovalLifecycle(max_entries=1, terminal_retention_seconds=30, clock=lambda: now)
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )
    intent = lifecycle.claim(
        thread_id="thread-1",
        decision=ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}"),
    )
    lifecycle.begin_execution(intent, owner=ApprovalExecutionOwner.LOCAL)
    lifecycle.recover_execution(intent, owner=ApprovalExecutionOwner.LOCAL)
    now = 1_000.0

    with pytest.raises(ApprovalCapacityError):
        lifecycle.register_local(
            thread_id="thread-2",
            interrupt_id="approval-2",
            call_id="call-2",
            name="write_record",
            arguments="{}",
        )

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.INDETERMINATE


async def test_transition_telemetry_covers_lifecycle_without_sensitive_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operators can distinguish lifecycle transitions without logging tool inputs."""
    caplog.set_level("INFO", logger="agent_framework_ag_ui._approval_lifecycle")
    lifecycle = ApprovalLifecycle()
    secret = "sensitive-value"
    settled = lifecycle.register_local(
        thread_id="thread-settled",
        interrupt_id="approval-settled",
        call_id="call-settled",
        name="write_secret",
        arguments=f'{{"value":"{secret}"}}',
    )
    decision = ResumeDecision(
        interrupt_id="approval-settled",
        accepted=True,
        arguments=f'{{"value":"{secret}"}}',
    )
    intent = lifecycle.claim(thread_id="thread-settled", decision=decision)

    async def execute() -> list[Content]:
        return [Content.from_function_result(call_id="call-settled", result="done")]

    await LocalPendingToolTransitionOwner(execute).execute(intent, lifecycle=lifecycle)
    lifecycle.claim_batch(thread_id="thread-settled", decisions=[decision])

    lifecycle.register_local(
        thread_id="thread-rejected",
        interrupt_id="approval-rejected",
        call_id="call-rejected",
        name="reject_secret",
        arguments="{}",
    )
    lifecycle.claim_batch(
        thread_id="thread-rejected",
        decisions=[ResumeDecision(interrupt_id="approval-rejected", accepted=False, arguments="{}")],
    )
    lifecycle.register_local(
        thread_id="thread-cancelled",
        interrupt_id="approval-cancelled",
        call_id="call-cancelled",
        name="cancel_secret",
        arguments="{}",
    )
    lifecycle.cancel_batch(thread_id="thread-cancelled", interrupt_ids=["approval-cancelled"])
    lifecycle.register_local(
        thread_id="thread-expired",
        interrupt_id="approval-expired",
        call_id="call-expired",
        name="expire_secret",
        arguments="{}",
    )
    lifecycle.expire_batch(thread_id="thread-expired", interrupt_ids=["approval-expired"])
    uncertain = lifecycle.register_local(
        thread_id="thread-uncertain",
        interrupt_id="approval-uncertain",
        call_id="call-uncertain",
        name="uncertain_secret",
        arguments="{}",
    )
    uncertain_intent = lifecycle.claim(
        thread_id="thread-uncertain",
        decision=ResumeDecision(interrupt_id="approval-uncertain", accepted=True, arguments="{}"),
    )
    lifecycle.begin_execution(uncertain_intent, owner=ApprovalExecutionOwner.LOCAL)
    lifecycle.recover_execution(uncertain_intent, owner=ApprovalExecutionOwner.LOCAL)
    with pytest.raises(KeyError):
        lifecycle.claim_batch(
            thread_id="thread-missing",
            decisions=[ResumeDecision(interrupt_id="approval-missing", accepted=True, arguments="{}")],
        )
    capacity_lifecycle = ApprovalLifecycle(max_entries=1)
    capacity_lifecycle.register_local(
        thread_id="thread-capacity-1",
        interrupt_id="approval-capacity-1",
        call_id="call-capacity-1",
        name="capacity_secret",
        arguments="{}",
    )
    with pytest.raises(ApprovalCapacityError):
        capacity_lifecycle.register_local(
            thread_id="thread-capacity-2",
            interrupt_id="approval-capacity-2",
            call_id="call-capacity-2",
            name="capacity_secret",
            arguments="{}",
        )

    events = {getattr(record, "approval_event", None) for record in caplog.records}
    assert {
        "registration",
        "claim",
        "execution_start",
        "settlement",
        "rejection",
        "cancellation",
        "duplicate",
        "expiration",
        "indeterminate_recovery",
        "authority_failure",
        "capacity_failure",
    } <= events
    assert any(
        getattr(record, "approval_occurrence_id", None) == settled.identity.occurrence_id for record in caplog.records
    )
    assert lifecycle.get(uncertain.identity).status is ApprovalStatus.INDETERMINATE
    assert secret not in caplog.text
    assert all(secret not in repr(record.__dict__) for record in caplog.records)


def test_claim_and_settlement_conflicts_have_typed_outcomes() -> None:
    """Adapters can distinguish transition conflicts without parsing error messages."""
    lifecycle = ApprovalLifecycle()
    lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )
    decision = ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}")
    intent = lifecycle.claim(thread_id="thread-1", decision=decision)

    with pytest.raises(ApprovalClaimConflictError):
        lifecycle.claim_batch(thread_id="thread-1", decisions=[decision])
    with pytest.raises(ApprovalSettlementConflictError):
        lifecycle.settle(intent, [Content.from_function_result(call_id="call-1", result="done")])


def test_same_thread_transitions_serialize_without_blocking_an_independent_thread() -> None:
    """One scoped thread is serialized while another can claim concurrently."""
    lifecycle = ApprovalLifecycle()
    first = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )
    lifecycle.register_local(
        thread_id="thread-2",
        interrupt_id="approval-2",
        call_id="call-2",
        name="write_record",
        arguments="{}",
    )
    first_decision = ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}")
    second_decision = ResumeDecision(interrupt_id="approval-2", accepted=True, arguments="{}")
    first_claim_entered = Event()
    release_first_claim = Event()
    second_same_thread_started = Event()
    original_emit = lifecycle._emit_event

    def blocking_emit(event: str, occurrence=None, *, failure_type: str | None = None) -> None:
        if event == "claim" and occurrence is not None and occurrence.identity == first.identity:
            first_claim_entered.set()
            assert release_first_claim.wait(timeout=2)
        original_emit(event, occurrence, failure_type=failure_type)

    lifecycle._emit_event = blocking_emit  # type: ignore[method-assign]

    def repeat_first_claim():
        second_same_thread_started.set()
        return lifecycle.claim_batch(thread_id="thread-1", decisions=[first_decision])

    with ThreadPoolExecutor(max_workers=3) as executor:
        first_claim = executor.submit(lifecycle.claim, thread_id="thread-1", decision=first_decision)
        assert first_claim_entered.wait(timeout=2)
        conflicting_claim = executor.submit(repeat_first_claim)
        assert second_same_thread_started.wait(timeout=2)
        independent_claim = executor.submit(lifecycle.claim, thread_id="thread-2", decision=second_decision)

        assert independent_claim.result(timeout=2).identity.call_id == "call-2"
        assert not conflicting_claim.done()
        release_first_claim.set()
        assert first_claim.result(timeout=2).identity == first.identity
        with pytest.raises(ApprovalClaimConflictError):
            conflicting_claim.result(timeout=2)


async def test_hosted_approval_is_forwarded_only_by_its_owner_and_settles_same_occurrence() -> None:
    """Hosted authority cannot execute locally and records forwarding against its occurrence."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_hosted(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="hosted_search",
        arguments='{"query":"azure"}',
    )
    intent = lifecycle.claim(
        thread_id="thread-1",
        decision=ResumeDecision(
            interrupt_id="approval-1",
            accepted=True,
            arguments='{"query":"azure"}',
        ),
    )
    local_invocations = 0

    async def execute_locally() -> list[Content]:
        nonlocal local_invocations
        local_invocations += 1
        return [Content.from_function_result(call_id="call-1", result="local")]

    with pytest.raises(ValueError, match="hosted"):
        await LocalPendingToolTransitionOwner(execute_locally).execute(intent, lifecycle=lifecycle)

    forwarded_response = Content.from_function_approval_response(
        approved=True,
        id="approval-1",
        function_call=Content.from_function_call(
            call_id="call-1",
            name="hosted_search",
            arguments={"query": "azure"},
            additional_properties={"server_label": "hosted"},
        ),
    )
    hosted_forwards = 0

    async def forward_to_hosted_owner() -> list[Content]:
        nonlocal hosted_forwards
        hosted_forwards += 1
        return [forwarded_response]

    owner = HostedPendingToolTransitionOwner(forward_to_hosted_owner)
    forwarded = await owner.forward(intent, lifecycle=lifecycle)
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.EXECUTING
    remote_result = Content.from_function_result(call_id="call-1", result="hosted")
    outcome = owner.record_outcome(intent, [remote_result], lifecycle=lifecycle)

    assert intent.owner is ApprovalExecutionOwner.HOSTED
    assert local_invocations == 0
    assert hosted_forwards == 1
    assert forwarded == [forwarded_response]
    assert outcome.identity == occurrence.identity
    assert outcome.result_group == (remote_result,)
    assert [result.content for result in outcome.replayable_results] == [remote_result]
    assert outcome.snapshot_reconciliation.status is ApprovalSnapshotStatus.SETTLED
    assert outcome.snapshot_reconciliation.retire_interrupt is True
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.SETTLED


async def test_execution_failure_becomes_indeterminate_and_keeps_unexecuted_sibling_claimed() -> None:
    """A possibly started side effect is not retried and does not erase a claimed sibling."""
    lifecycle = ApprovalLifecycle()
    first = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )
    second = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-2",
        call_id="call-2",
        name="write_record",
        arguments='{"value":"second"}',
    )
    first_intent, second_intent = lifecycle.claim_batch(
        thread_id="thread-1",
        decisions=[
            ResumeDecision(interrupt_id="approval-1", accepted=True, arguments='{"value":"first"}'),
            ResumeDecision(interrupt_id="approval-2", accepted=True, arguments='{"value":"second"}'),
        ],
    )

    invocation_count = 0

    async def fail_first() -> list[Content]:
        nonlocal invocation_count
        invocation_count += 1
        raise RuntimeError("side effect failed")

    with pytest.raises(RuntimeError, match="side effect failed"):
        await LocalPendingToolTransitionOwner(fail_first).execute(first_intent, lifecycle=lifecycle)

    assert lifecycle.get(first.identity).status is ApprovalStatus.INDETERMINATE
    assert lifecycle.get(second.identity).status is ApprovalStatus.CLAIMED
    with pytest.raises(ApprovalIndeterminateError) as error:
        lifecycle.claim_batch(
            thread_id="thread-1",
            decisions=[ResumeDecision(interrupt_id="approval-1", accepted=True, arguments='{"value":"first"}')],
        )
    assert str(error.value) == "Approval execution outcome is indeterminate; automatic retry is unsafe."
    assert "first" not in str(error.value)
    assert invocation_count == 1

    async def execute_second() -> list[Content]:
        return [Content.from_function_result(call_id="call-2", result="wrote second")]

    await LocalPendingToolTransitionOwner(execute_second).execute(second_intent, lifecycle=lifecycle)
    assert lifecycle.get(second.identity).status is ApprovalStatus.SETTLED


def test_claim_can_be_released_before_execution_only_with_explicit_safe_policy() -> None:
    """Reserved authority can be reclaimed when the owner proves execution never began."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )
    decision = ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}")
    intent = lifecycle.claim(thread_id="thread-1", decision=decision)

    lifecycle.release_claim(intent, policy=ClaimRecoveryPolicy.SAFE_TO_RETRY)
    retry = lifecycle.claim(thread_id="thread-1", decision=decision)

    assert retry.identity == occurrence.identity
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.CLAIMED


def test_recovering_execution_without_a_result_becomes_indeterminate() -> None:
    """Recovery preserves an uncertain occurrence instead of granting authority again."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )
    decision = ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}")
    intent = lifecycle.claim(thread_id="thread-1", decision=decision)
    lifecycle.begin_execution(intent, owner=ApprovalExecutionOwner.LOCAL)

    recovered = lifecycle.recover_execution(intent, owner=ApprovalExecutionOwner.LOCAL)

    assert recovered is None
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.INDETERMINATE
    with pytest.raises(ApprovalIndeterminateError):
        lifecycle.claim_batch(thread_id="thread-1", decisions=[decision])


async def test_explicit_idempotency_key_allows_retry_after_execution_interruption() -> None:
    """A predeclared idempotency key permits retrying a potentially started side effect."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
        idempotency_key="operation-1",
    )
    intent = lifecycle.claim(
        thread_id="thread-1",
        decision=ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}"),
    )
    invocation_count = 0

    async def execute_idempotently() -> list[Content]:
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count == 1:
            raise RuntimeError("connection lost")
        return [Content.from_function_result(call_id="call-1", result="recorded")]

    owner = LocalPendingToolTransitionOwner(execute_idempotently)
    with pytest.raises(RuntimeError, match="connection lost"):
        await owner.execute(intent, lifecycle=lifecycle)

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.CLAIMED
    outcome = await owner.execute(intent, lifecycle=lifecycle)
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.SETTLED
    assert outcome.replayable_results[0].content.result == "recorded"
    assert invocation_count == 2


async def test_hosted_idempotency_key_allows_retry_after_forwarding_interruption() -> None:
    """A hosted owner uses the same explicit recovery rule as the local owner."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_hosted(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="hosted_write",
        arguments="{}",
        idempotency_key="hosted-operation-1",
    )
    intent = lifecycle.claim(
        thread_id="thread-1",
        decision=ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}"),
    )
    forwarding_count = 0

    async def forward_idempotently() -> list[Content]:
        nonlocal forwarding_count
        forwarding_count += 1
        if forwarding_count == 1:
            raise RuntimeError("host disconnected")
        return [Content.from_function_result(call_id="call-1", result="recorded")]

    owner = HostedPendingToolTransitionOwner(forward_idempotently)
    with pytest.raises(RuntimeError, match="host disconnected"):
        await owner.execute(intent, lifecycle=lifecycle)

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.CLAIMED
    await owner.execute(intent, lifecycle=lifecycle)
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.SETTLED
    assert forwarding_count == 2


def test_batch_validation_is_atomic_before_claiming_any_occurrence() -> None:
    """One invalid decision leaves every occurrence pending and eligible for a corrected batch."""
    lifecycle = ApprovalLifecycle()
    first = lifecycle.register_local(
        thread_id="tenant-a\x1fthread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )
    second = lifecycle.register_local(
        thread_id="tenant-a\x1fthread-1",
        interrupt_id="approval-2",
        call_id="call-2",
        name="write_record",
        arguments='{"value":"second"}',
    )

    with pytest.raises(ValueError, match="arguments do not match"):
        lifecycle.claim_batch(
            thread_id="tenant-a\x1fthread-1",
            decisions=[
                ResumeDecision(interrupt_id="approval-1", accepted=True, arguments='{"value":"first"}'),
                ResumeDecision(interrupt_id="approval-2", accepted=True, arguments='{"value":"forged"}'),
            ],
        )

    assert lifecycle.get(first.identity).status is ApprovalStatus.PENDING
    assert lifecycle.get(second.identity).status is ApprovalStatus.PENDING


def test_accepted_declaration_without_execution_owner_remains_pending() -> None:
    """Approval alone does not grant local authority to a declaration-only call."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_unowned(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="client_action",
        arguments="{}",
    )

    batch = lifecycle.claim_batch(
        thread_id="thread-1",
        decisions=[ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}")],
    )

    assert batch.authorized_executions == ()
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.PENDING


def test_mixed_batch_accounts_for_rejection_under_original_call_identity() -> None:
    """A rejected occurrence remains represented while its accepted sibling is claimed."""
    lifecycle = ApprovalLifecycle()
    accepted = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )
    rejected = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-2",
        call_id="call-2",
        name="write_record",
        arguments='{"value":"second"}',
    )

    intents = lifecycle.claim_batch(
        thread_id="thread-1",
        decisions=[
            ResumeDecision(interrupt_id="approval-1", accepted=True, arguments='{"value":"first"}'),
            ResumeDecision(interrupt_id="approval-2", accepted=False, arguments='{"value":"second"}'),
        ],
    )

    assert [intent.identity for intent in intents] == [accepted.identity]
    assert lifecycle.get(rejected.identity).status is ApprovalStatus.REJECTED
    assert [result.content.call_id for result in lifecycle.get(rejected.identity).replayable_results] == ["call-2"]


def test_batch_claims_preserve_order_and_scope_reused_raw_call_ids() -> None:
    """Raw call ids reused in another scoped thread cannot correlate approval authority."""
    lifecycle = ApprovalLifecycle()
    tenant_a = lifecycle.register_local(
        thread_id="tenant-a\x1fthread-1",
        interrupt_id="approval-shared",
        call_id="call-shared",
        name="write_record",
        arguments='{"tenant":"a"}',
    )
    tenant_b = lifecycle.register_local(
        thread_id="tenant-b\x1fthread-1",
        interrupt_id="approval-shared",
        call_id="call-shared",
        name="write_record",
        arguments='{"tenant":"b"}',
    )

    intents = lifecycle.claim_batch(
        thread_id="tenant-a\x1fthread-1",
        decisions=[
            ResumeDecision(
                interrupt_id="approval-shared",
                accepted=True,
                arguments='{"tenant":"a"}',
            )
        ],
    )

    assert [intent.identity for intent in intents] == [tenant_a.identity]
    assert tenant_a.identity != tenant_b.identity
    assert lifecycle.get(tenant_a.identity).status is ApprovalStatus.CLAIMED
    assert lifecycle.get(tenant_b.identity).status is ApprovalStatus.PENDING


def test_batch_cancellation_preserves_each_original_occurrence() -> None:
    """Cancelling selected occurrences is terminal without consuming an unrelated sibling."""
    lifecycle = ApprovalLifecycle()
    cancelled = lifecycle.register_local(
        thread_id="tenant-a\x1fthread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )
    pending = lifecycle.register_local(
        thread_id="tenant-a\x1fthread-1",
        interrupt_id="approval-2",
        call_id="call-2",
        name="write_record",
        arguments='{"value":"second"}',
    )

    reconciliations = lifecycle.cancel_batch(
        thread_id="tenant-a\x1fthread-1",
        interrupt_ids=["approval-1"],
    )

    assert lifecycle.get(cancelled.identity).status is ApprovalStatus.CANCELLED
    assert lifecycle.get(pending.identity).status is ApprovalStatus.PENDING
    assert [(item.identity, item.status, item.retire_interrupt) for item in reconciliations] == [
        (cancelled.identity, ApprovalSnapshotStatus.CANCELLED, True)
    ]


def test_snapshot_reconciliation_reports_terminal_pending_and_missing_occurrences() -> None:
    """Snapshot projection receives lifecycle semantics without recreating authority."""
    lifecycle = ApprovalLifecycle()
    settled = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-settled",
        call_id="call-settled",
        name="write_record",
        arguments="{}",
    )
    pending = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-pending",
        call_id="call-pending",
        name="write_record",
        arguments="{}",
    )
    intent = lifecycle.claim(
        thread_id="thread-1",
        decision=ResumeDecision(interrupt_id="approval-settled", accepted=True, arguments="{}"),
    )
    lifecycle.begin_execution(intent, owner=ApprovalExecutionOwner.LOCAL)
    outcome = lifecycle.settle(
        intent,
        [Content.from_function_result(call_id="call-settled", result="done")],
    )

    reconciliations = lifecycle.reconcile_snapshot(
        thread_id="thread-1",
        interrupt_ids=["approval-settled", "approval-pending", "approval-missing"],
    )

    assert outcome.snapshot_reconciliation == reconciliations[0]
    assert [(item.identity, item.status, item.retire_interrupt) for item in reconciliations] == [
        (settled.identity, ApprovalSnapshotStatus.SETTLED, True),
        (pending.identity, ApprovalSnapshotStatus.PENDING, False),
        (None, ApprovalSnapshotStatus.MISSING, True),
    ]


def test_one_occurrence_can_be_claimed_through_a_trusted_thread_alias() -> None:
    """Provider conversation aliases address one occurrence rather than duplicating authority."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local_aliases(
        thread_ids=["tenant-a\x1fag-ui-thread", "tenant-a\x1fprovider-thread"],
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )

    intents = lifecycle.claim_batch(
        thread_id="tenant-a\x1fprovider-thread",
        decisions=[ResumeDecision(interrupt_id="approval-1", accepted=True, arguments='{"value":"first"}')],
    )

    assert [intent.identity for intent in intents] == [occurrence.identity]
    with pytest.raises(ValueError, match="not pending"):
        lifecycle.register_local_aliases(
            thread_ids=["tenant-a\x1fag-ui-thread", "tenant-a\x1fprovider-thread"],
            interrupt_id="approval-1",
            call_id="call-1",
            name="write_record",
            arguments='{"value":"first"}',
        )
    with pytest.raises(ValueError, match="not pending"):
        lifecycle.claim_batch(
            thread_id="tenant-a\x1fag-ui-thread",
            decisions=[ResumeDecision(interrupt_id="approval-1", accepted=True, arguments='{"value":"first"}')],
        )


async def test_settled_raw_call_id_can_be_reused_for_a_new_occurrence() -> None:
    """Sequential reuse creates a fresh logical occurrence instead of reviving settled authority."""
    lifecycle = ApprovalLifecycle()
    first = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-shared",
        call_id="call-shared",
        name="write_record",
        arguments='{"value":"first"}',
    )
    first_intent = lifecycle.claim(
        thread_id="thread-1",
        decision=ResumeDecision(
            interrupt_id="approval-shared",
            accepted=True,
            arguments='{"value":"first"}',
        ),
    )

    async def execute_first() -> list[Content]:
        return [Content.from_function_result(call_id="call-shared", result="wrote first")]

    await LocalPendingToolTransitionOwner(execute_first).execute(first_intent, lifecycle=lifecycle)
    second = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-shared",
        call_id="call-shared",
        name="write_record",
        arguments='{"value":"second"}',
    )

    assert first.identity != second.identity
    assert lifecycle.get(first.identity).status is ApprovalStatus.SETTLED
    assert lifecycle.get(second.identity).status is ApprovalStatus.PENDING


async def test_identical_accepted_retry_returns_retained_outcome_without_execution() -> None:
    """A settled accepted decision reprojects its result instead of granting authority again."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )
    decision = ResumeDecision(
        interrupt_id="approval-1",
        accepted=True,
        arguments='{"value":"first"}',
    )
    intent = lifecycle.claim(thread_id="thread-1", decision=decision)
    invocation_count = 0

    async def execute_once() -> list[Content]:
        nonlocal invocation_count
        invocation_count += 1
        return [Content.from_function_result(call_id="call-1", result="wrote first")]

    first_outcome = await LocalPendingToolTransitionOwner(execute_once).execute(intent, lifecycle=lifecycle)
    retry = lifecycle.claim_batch(thread_id="thread-1", decisions=[decision])

    assert retry.authorized_executions == ()
    assert retry.retained_outcomes == (first_outcome,)
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.SETTLED
    assert invocation_count == 1


def test_accepted_retry_after_rejection_fails_as_a_conflict() -> None:
    """A terminal rejection cannot be changed into execution authority by a retry."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )
    lifecycle.claim_batch(
        thread_id="thread-1",
        decisions=[ResumeDecision(interrupt_id="approval-1", accepted=False, arguments='{"value":"first"}')],
    )

    with pytest.raises(ValueError, match="conflicts with the retained terminal decision"):
        lifecycle.claim_batch(
            thread_id="thread-1",
            decisions=[ResumeDecision(interrupt_id="approval-1", accepted=True, arguments='{"value":"first"}')],
        )

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.REJECTED


def test_identical_rejection_retry_returns_retained_outcome() -> None:
    """A repeated rejection preserves and returns the original rejection result."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments='{"value":"first"}',
    )
    decision = ResumeDecision(interrupt_id="approval-1", accepted=False, arguments='{"value":"first"}')

    first = lifecycle.claim_batch(thread_id="thread-1", decisions=[decision])
    retry = lifecycle.claim_batch(thread_id="thread-1", decisions=[decision])

    assert first.authorized_executions == ()
    assert first.retained_outcomes == ()
    assert retry.authorized_executions == ()
    assert retry.retained_outcomes == (lifecycle.get(occurrence.identity).outcome,)
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.REJECTED


def test_changed_tool_name_fails_before_authority_is_claimed() -> None:
    """A typed decision for another tool cannot claim the registered occurrence."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="safe_action",
        arguments="{}",
    )

    with pytest.raises(ValueError, match="tool name does not match"):
        lifecycle.claim_batch(
            thread_id="thread-1",
            decisions=[
                ResumeDecision(
                    interrupt_id="approval-1",
                    accepted=True,
                    name="dangerous_action",
                    arguments="{}",
                )
            ],
        )

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.PENDING


def test_identical_cancel_retry_keeps_terminal_cancellation() -> None:
    """Retrying an explicit cancellation is idempotent and cannot restore authority."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )

    lifecycle.cancel_batch(thread_id="thread-1", interrupt_ids=["approval-1"])
    lifecycle.cancel_batch(thread_id="thread-1", interrupt_ids=["approval-1"])

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.CANCELLED


def test_expired_authority_cannot_be_claimed() -> None:
    """Expiration is terminal and an otherwise valid decision cannot revive it."""
    lifecycle = ApprovalLifecycle()
    occurrence = lifecycle.register_local(
        thread_id="thread-1",
        interrupt_id="approval-1",
        call_id="call-1",
        name="write_record",
        arguments="{}",
    )

    lifecycle.expire_batch(thread_id="thread-1", interrupt_ids=["approval-1"])

    with pytest.raises(ValueError, match="expired"):
        lifecycle.claim_batch(
            thread_id="thread-1",
            decisions=[ResumeDecision(interrupt_id="approval-1", accepted=True, arguments="{}")],
        )

    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.EXPIRED
