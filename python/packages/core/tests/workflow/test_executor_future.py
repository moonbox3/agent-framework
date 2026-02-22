# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler


class TestExecutorFutureAnnotations:
    """Regression tests for @handler with from __future__ import annotations."""

    def test_handler_future_annotations_workflow_context_two_types(self) -> None:
        class TypeA:
            pass

        class TypeB:
            pass

        class MyExecutor(Executor):
            @handler(input=str, output=TypeA, workflow_output=TypeB)
            async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        ex = MyExecutor(id="test")
        assert str in ex._handlers

        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]

    def test_handler_future_annotations_workflow_context_one_type(self) -> None:
        class TypeA:
            pass

        class MyExecutor(Executor):
            @handler(input=str, output=TypeA)
            async def example(self, input: str, ctx: WorkflowContext[TypeA]) -> None:
                pass

        ex = MyExecutor(id="test")
        assert str in ex._handlers

        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == []
