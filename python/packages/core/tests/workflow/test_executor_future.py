# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, WorkflowMessage, handler


@dataclass
class FutureMsg:
    value: int


@dataclass
class FutureTypeA:
    value: str


@dataclass
class FutureTypeB:
    value: int


def test_executor_handler_future_annotations_workflow_context_generics_are_resolved() -> None:
    """Regression test: @handler should work when annotations are strings (PEP563)."""

    class MyExecutor(Executor):
        @handler
        async def example(self, message: FutureMsg, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
            pass

    ex = MyExecutor(id="my")

    # Ensure handler registration uses real types (not string annotations)
    assert FutureMsg in ex._handlers

    handler_func = ex._handlers[FutureMsg]
    assert hasattr(handler_func, "_handler_spec")
    spec = handler_func._handler_spec  # type: ignore[attr-defined]

    assert spec["message_type"] is FutureMsg
    assert spec["output_types"] == [FutureTypeA]
    assert spec["workflow_output_types"] == [FutureTypeB]

    # And can_handle still works
    assert ex.can_handle(WorkflowMessage(data=FutureMsg(1), source_id="mock"))
