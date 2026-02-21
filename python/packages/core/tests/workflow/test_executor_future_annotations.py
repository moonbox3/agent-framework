# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

import pytest

from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


def test_handler_future_annotations_get_type_hints_failure_does_not_propagate() -> None:
    class MyTypeA:
        pass

    # Force `typing.get_type_hints()` to raise without introducing ruff-flagged undefined names.
    # We do this by deleting a referenced symbol from the handler function globals after definition.
    class MyExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="my_exec")

        @handler
        async def example(self, message: str, ctx: WorkflowContext[MyTypeA]) -> None:
            _ = (message, ctx)

    del MyExecutor.example.__globals__["MyTypeA"]

    # Invariant: annotation-evaluation failures should not leak (e.g., NameError);
    # handler validation should fail with user-facing ValueError instead.
    with pytest.raises(ValueError, match=r"must be annotated as WorkflowContext"):
        MyExecutor()


def test_handler_future_annotations_get_type_hints_failure_does_not_propagate(self) -> None:
    class MyTypeA:
        pass

    # Force `typing.get_type_hints()` to raise without introducing undefined names.
    # We do this by deleting a referenced symbol from the function globals after definition.
    class MyExecutor(Executor):
        def __init__(self) -> None:
            super().__init__(id="my_exec")

        @handler
        async def example(self, message: str, ctx: WorkflowContext[MyTypeA]) -> None:
            _ = (message, ctx)

    # Remove MyTypeA from handler globals to cause NameError during type-hint evaluation.
    # Invariant: those errors must not leak.
    del MyExecutor.example.__globals__["MyTypeA"]

    with pytest.raises(ValueError, match=r"must be annotated as WorkflowContext"):
        MyExecutor()
