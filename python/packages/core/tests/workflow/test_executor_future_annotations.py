# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any

from agent_framework import Executor, WorkflowContext, handler


def test_handler_future_annotations_resolved() -> None:
    class FutureExecutor(Executor):
        @handler
        async def handle(self, message: dict[str, Any], ctx: WorkflowContext[int]) -> None:
            pass

    executor = FutureExecutor(id="future")

    assert dict[str, Any] in executor._handlers
    handler_func = executor._handlers[dict[str, Any]]
    assert handler_func._handler_spec["message_type"] == dict[str, Any]
    assert handler_func._handler_spec["output_types"] == [int]
    assert int in executor.output_types
