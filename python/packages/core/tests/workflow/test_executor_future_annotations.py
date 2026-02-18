# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler


class TestExecutorFutureAnnotations:
    """Test suite for Executor handlers with postponed annotations."""

    def test_executor_handler_future_annotations(self) -> None:
        """Handlers should resolve WorkflowContext annotations from __future__ import annotations."""

        class FutureAnnotationExecutor(Executor):
            @handler
            async def handle(self, value: int, ctx: WorkflowContext[int]) -> None:
                await ctx.send_message(value)

        executor = FutureAnnotationExecutor(id="future")
        spec = executor._handler_specs[0]
        assert spec["message_type"] is int
        assert spec["output_types"] == [int]

    def test_executor_handler_forward_ref_ctx_annotation(self) -> None:
        """Handlers should resolve quoted WorkflowContext annotations."""

        class ForwardRefCtxExecutor(Executor):
            @handler
            async def handle(self, value: int, ctx: WorkflowContext[int, str]) -> None:
                await ctx.send_message(value)
                await ctx.yield_output("done")

        executor = ForwardRefCtxExecutor(id="forward_ref")
        spec = executor._handler_specs[0]
        assert spec["output_types"] == [int]
        assert spec["workflow_output_types"] == [str]


def test_executor_handler_future_annotations(self) -> None:
    class FutureAnnotationExecutor(Executor):
        @handler
        async def handle(self, value: int, ctx: WorkflowContext[int]) -> None:
            await ctx.send_message(value)

    executor = FutureAnnotationExecutor(id="future")
    spec = executor._handler_specs[0]
    assert spec["message_type"] is int
    assert spec["output_types"] == [int]


def test_executor_handler_forward_ref_ctx_annotation(self) -> None:
    class ForwardRefCtxExecutor(Executor):
        @handler
        async def handle(self, value: int, ctx: WorkflowContext[int, str]) -> None:
            await ctx.send_message(value)
            await ctx.yield_output("done")

    executor = ForwardRefCtxExecutor(id="forward_ref")
    spec = executor._handler_specs[0]
    assert spec["output_types"] == [int]
    assert spec["workflow_output_types"] == [str]
