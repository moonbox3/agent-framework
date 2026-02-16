# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler


class TestExecutorFutureAnnotations:
    """Test suite for Executor handlers with from __future__ import annotations."""

    def test_handler_future_annotations(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: int, ctx: WorkflowContext[int, str]) -> None:
                pass

        executor = FutureExecutor(id="future")
        assert int in executor._handlers

        spec = executor._handler_specs[0]
        assert spec["message_type"] is int
        assert spec["output_types"] == [int]
        assert spec["workflow_output_types"] == [str]
