# Copyright (c) Microsoft. All rights reserved.

"""Tests for server-side AG-UI approval state storage."""

import pytest

from agent_framework_ag_ui._approval_lifecycle import ApprovalCapacityError
from agent_framework_ag_ui._approval_state import InMemoryAGUIApprovalStateStore, approval_state_thread_id


def test_approval_state_thread_id_allows_unscoped_thread() -> None:
    assert approval_state_thread_id(scope=None, thread_id="thread-1") == "thread-1"


def test_approval_state_thread_id_scopes_thread() -> None:
    scoped_thread_id = approval_state_thread_id(scope="tenant-a", thread_id="thread-1")

    assert scoped_thread_id != "thread-1"
    assert "tenant-a" in scoped_thread_id
    assert "thread-1" in scoped_thread_id


@pytest.mark.parametrize("scope", ["", object()])
def test_approval_state_thread_id_rejects_invalid_scope(scope: object) -> None:
    with pytest.raises(ValueError, match="scope must be a non-empty string"):
        approval_state_thread_id(scope=scope, thread_id="thread-1")


def test_approval_state_store_rejects_invalid_max_entries() -> None:
    with pytest.raises(ValueError, match="max_entries must be greater than 0"):
        InMemoryAGUIApprovalStateStore(max_entries=0)


def test_approval_state_store_does_not_evict_active_entries() -> None:
    store = InMemoryAGUIApprovalStateStore(max_entries=1)
    store.register_local(
        thread_ids=["thread-1"],
        name="write_record",
        arguments="{}",
        request_id="request-1",
        interrupt_id="approval-1",
    )

    with pytest.raises(ApprovalCapacityError):
        store.register_local(
            thread_ids=["thread-2"],
            name="write_record",
            arguments="{}",
            request_id="request-2",
            interrupt_id="approval-2",
        )

    assert {entry["interrupt_id"] for entry in store.pending_approvals.values()} == {"approval-1"}


def test_approval_state_store_does_not_evict_active_middleware_state() -> None:
    store = InMemoryAGUIApprovalStateStore(max_entries=1)
    store.set_tool_approval_state("thread-1", {"call_id": "call-1"})

    with pytest.raises(ApprovalCapacityError):
        store.set_tool_approval_state("thread-2", {"call_id": "call-2"})

    assert store.tool_approval_states == {"thread-1": {"call_id": "call-1"}}
