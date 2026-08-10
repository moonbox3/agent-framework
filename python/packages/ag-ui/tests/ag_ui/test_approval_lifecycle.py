# Copyright (c) Microsoft. All rights reserved.

"""Behavior tests for the approval batch continuity lifecycle seam."""

from __future__ import annotations

import pytest
from agent_framework import Content

from agent_framework_ag_ui._approval_lifecycle import (
    ApprovalExecutionOwner,
    ApprovalLifecycle,
    ApprovalStatus,
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
    assert lifecycle.get(occurrence.identity).status is ApprovalStatus.SETTLED


async def test_execution_failure_keeps_unexecuted_batch_sibling_claimed() -> None:
    """A failed occurrence does not erase a claimed sibling that can still execute."""
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

    async def fail_first() -> list[Content]:
        raise RuntimeError("side effect failed")

    with pytest.raises(RuntimeError, match="side effect failed"):
        await LocalPendingToolTransitionOwner(fail_first).execute(first_intent, lifecycle=lifecycle)

    assert lifecycle.get(first.identity).status is ApprovalStatus.EXECUTING
    assert lifecycle.get(second.identity).status is ApprovalStatus.CLAIMED

    async def execute_second() -> list[Content]:
        return [Content.from_function_result(call_id="call-2", result="wrote second")]

    await LocalPendingToolTransitionOwner(execute_second).execute(second_intent, lifecycle=lifecycle)
    assert lifecycle.get(second.identity).status is ApprovalStatus.SETTLED


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

    lifecycle.cancel_batch(
        thread_id="tenant-a\x1fthread-1",
        interrupt_ids=["approval-1"],
    )

    assert lifecycle.get(cancelled.identity).status is ApprovalStatus.CANCELLED
    assert lifecycle.get(pending.identity).status is ApprovalStatus.PENDING


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
