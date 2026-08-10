# Copyright (c) Microsoft. All rights reserved.

"""Server-side AG-UI approval state storage."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ._approval_lifecycle import ApprovalLifecycle

ApprovalScope = str
"""Application-defined scope for server-side AG-UI Approval State."""

DEFAULT_MAX_APPROVAL_STATES = 10_000
_APPROVAL_SCOPE_INPUT_KEY = "__ag_ui_approval_scope"
_APPROVAL_THREAD_SEPARATOR = "\x1f"


def approval_state_thread_id(*, scope: object | None, thread_id: str) -> str:
    """Return the storage thread key for Approval State.

    ``None`` is the only unscoped value. A provided scope must be a non-empty
    string so accidental empty or malformed scopes cannot collapse into the
    unscoped namespace.
    """
    if scope is None:
        return thread_id
    if not isinstance(scope, str) or not scope:
        raise ValueError("scope must be a non-empty string when provided.")
    return f"{scope}{_APPROVAL_THREAD_SEPARATOR}{thread_id}"


class InMemoryAGUIApprovalStateStore:
    """Bounded process-local server-side store for AG-UI Approval State.

    The default store keeps only pending approval entries. It does not store
    general ``AgentSession.state`` or AG-UI Thread Snapshots.
    """

    def __init__(self, *, max_entries: int = DEFAULT_MAX_APPROVAL_STATES) -> None:
        """Initialize the process-local Approval State store.

        Keyword Args:
            max_entries: Maximum pending approval entries to retain.

        Raises:
            ValueError: If ``max_entries`` is less than 1.
        """
        if max_entries < 1:
            raise ValueError("max_entries must be greater than 0.")
        self.max_entries = max_entries
        self.pending_approvals: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self.tool_approval_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lifecycle = ApprovalLifecycle()

    def register_local(
        self,
        *,
        thread_ids: list[str],
        name: str,
        arguments: str,
        request_id: str,
        interrupt_id: str,
        already_approved_requests: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register one local occurrence and its trusted aliases."""
        entry: dict[str, Any] = {
            "name": name,
            "arguments": arguments,
            "request_id": request_id,
            "interrupt_id": interrupt_id,
        }
        if already_approved_requests:
            entry["already_approved_requests"] = already_approved_requests

        unique_thread_ids = list(dict.fromkeys(thread_ids))
        self.lifecycle.register_local_aliases(
            thread_ids=unique_thread_ids,
            interrupt_id=interrupt_id,
            call_id=interrupt_id,
            name=name,
            arguments=arguments,
        )
        for thread_id in unique_thread_ids:
            aliases = {(thread_id, request_id), (thread_id, interrupt_id)}
            replaced_entries = {id(existing) for key, existing in self.pending_approvals.items() if key in aliases}
            for key, existing in list(self.pending_approvals.items()):
                if key in aliases or id(existing) in replaced_entries:
                    self.pending_approvals.pop(key, None)
            for key in aliases:
                self.pending_approvals[key] = entry
        self.evict_oldest()

    def evict_oldest(self) -> None:
        """Evict oldest pending approval entries until the store is within bounds."""
        while len(self.pending_approvals) > self.max_entries:
            self.pending_approvals.popitem(last=False)
        while len(self.tool_approval_states) > self.max_entries:
            self.tool_approval_states.popitem(last=False)
