# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureTypeA:
    value: str


@dataclass
class FutureTypeB:
    value: int


class TestExecutorFutureAnnotations:
    """Regression tests for class-based Executor handlers with postponed annotations."""

    def test_handler_future_annotations_workflow_context_type_args_are_resolved(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
                pass

        ex = MyExecutor(id="test")

        # Ensure handler was registered and the ctx annotation is not a string.
        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [FutureTypeA]
        assert spec["workflow_output_types"] == [FutureTypeB]
        assert not isinstance(spec["ctx_annotation"], str)
