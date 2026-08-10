# Copyright (c) Microsoft. All rights reserved.

"""Behavior tests for the approval batch continuity lifecycle seam."""

from __future__ import annotations

from agent_framework import Content

from agent_framework_ag_ui._approval_lifecycle import (
    ApprovalLifecycle,
    ApprovalStatus,
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
