# Copyright (c) Microsoft. All rights reserved.

"""Server-side AG-UI approval state storage."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ._approval_lifecycle import ApprovalCapacityError, ApprovalExecutionOwner, ApprovalLifecycle

ApprovalScope = str
"""Application-defined scope for server-side AG-UI Approval State."""

DEFAULT_MAX_APPROVAL_STATES = 10_000
DEFAULT_TERMINAL_RETENTION_SECONDS = 900
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

    State is local to one process and is not durable across restarts or replicas.
    Active and indeterminate occurrences are protected from eviction. Terminal
    outcomes guarantee duplicate-execution protection for the configured
    retention interval.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_APPROVAL_STATES,
        terminal_retention_seconds: float = DEFAULT_TERMINAL_RETENTION_SECONDS,
    ) -> None:
        """Initialize the process-local Approval State store.

        Keyword Args:
            max_entries: Maximum approval occurrences or middleware state entries to retain.
            terminal_retention_seconds: Process-local duplicate-execution protection window.

        Raises:
            ValueError: If ``max_entries`` is less than 1.
        """
        if max_entries < 1:
            raise ValueError("max_entries must be greater than 0.")
        self.max_entries = max_entries
        self.pending_approvals: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self.tool_approval_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lifecycle = ApprovalLifecycle(
            max_entries=max_entries,
            terminal_retention_seconds=terminal_retention_seconds,
        )

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
        self._register(
            thread_ids=thread_ids,
            name=name,
            arguments=arguments,
            request_id=request_id,
            interrupt_id=interrupt_id,
            already_approved_requests=already_approved_requests,
            server_label=None,
            owner=ApprovalExecutionOwner.LOCAL,
        )

    def register_hosted(
        self,
        *,
        thread_ids: list[str],
        name: str,
        arguments: str,
        request_id: str,
        interrupt_id: str,
        server_label: str | None,
        already_approved_requests: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register one hosted occurrence and its trusted aliases."""
        self._register(
            thread_ids=thread_ids,
            name=name,
            arguments=arguments,
            request_id=request_id,
            interrupt_id=interrupt_id,
            already_approved_requests=already_approved_requests,
            server_label=server_label,
            owner=ApprovalExecutionOwner.HOSTED,
        )

    def register_unowned(
        self,
        *,
        thread_ids: list[str],
        name: str,
        arguments: str,
        request_id: str,
        interrupt_id: str,
        already_approved_requests: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register one occurrence that has no executable transition owner."""
        self._register(
            thread_ids=thread_ids,
            name=name,
            arguments=arguments,
            request_id=request_id,
            interrupt_id=interrupt_id,
            already_approved_requests=already_approved_requests,
            server_label=None,
            owner=ApprovalExecutionOwner.UNAVAILABLE,
        )

    def register_deferred(
        self,
        *,
        thread_ids: list[str],
        name: str,
        arguments: str,
        request_id: str,
        interrupt_id: str,
        already_approved_requests: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register one occurrence owned by the in-run transition pipeline."""
        self._register(
            thread_ids=thread_ids,
            name=name,
            arguments=arguments,
            request_id=request_id,
            interrupt_id=interrupt_id,
            already_approved_requests=already_approved_requests,
            server_label=None,
            owner=ApprovalExecutionOwner.DEFERRED,
        )

    def _register(
        self,
        *,
        thread_ids: list[str],
        name: str,
        arguments: str,
        request_id: str,
        interrupt_id: str,
        already_approved_requests: list[dict[str, Any]] | None,
        server_label: str | None,
        owner: ApprovalExecutionOwner,
    ) -> None:
        entry: dict[str, Any] = {
            "name": name,
            "arguments": arguments,
            "request_id": request_id,
            "interrupt_id": interrupt_id,
        }
        if already_approved_requests:
            entry["already_approved_requests"] = already_approved_requests
        if server_label:
            entry["server_label"] = server_label
        entry["execution_owner"] = owner.value

        unique_thread_ids = list(dict.fromkeys(thread_ids))
        if owner is ApprovalExecutionOwner.HOSTED:
            register_aliases = self.lifecycle.register_hosted_aliases
        elif owner is ApprovalExecutionOwner.DEFERRED:
            register_aliases = self.lifecycle.register_deferred_aliases
        elif owner is ApprovalExecutionOwner.UNAVAILABLE:
            register_aliases = self.lifecycle.register_unowned_aliases
        else:
            register_aliases = self.lifecycle.register_local_aliases
        register_aliases(
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

    def set_tool_approval_state(self, thread_id: str, state: dict[str, Any]) -> None:
        """Store approval middleware state without evicting another active thread."""
        if thread_id not in self.tool_approval_states and len(self.tool_approval_states) >= self.max_entries:
            raise ApprovalCapacityError("Approval state capacity is exhausted by protected occurrences.")
        self.tool_approval_states[thread_id] = state
        self.tool_approval_states.move_to_end(thread_id)
