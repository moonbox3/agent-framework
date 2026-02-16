# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler


class TestExecutorFutureAnnotations:
    """Test suite for Executor with from __future__ import annotations."""

    def test_handler_future_annotations(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: int, ctx: WorkflowContext[int]) -> None:
                pass

        executor = FutureExecutor(id="future_executor")
        spec = executor._handler_specs[0]
        assert spec["message_type"] is int
        assert spec["output_types"] == [int]
