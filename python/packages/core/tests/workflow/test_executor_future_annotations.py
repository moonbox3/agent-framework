# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any

from agent_framework import Executor, WorkflowContext, handler


class ForwardRefMessage:
    pass


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


def test_handler_future_annotations_explicit_types_ctx_annotation() -> None:
    class ExplicitFutureExecutor(Executor):
        @handler(input="dict[str, Any]", output="int")
        async def handle(self, message, ctx: WorkflowContext[int]) -> None:  # type: ignore[no-untyped-def]
            pass

    executor = ExplicitFutureExecutor(id="explicit_future")

    assert dict[str, Any] in executor._handlers
    handler_func = executor._handlers[dict[str, Any]]
    assert handler_func._handler_spec["output_types"] == [int]
    assert handler_func._handler_spec["ctx_annotation"] == WorkflowContext[int]
    assert int in executor.output_types


def test_handler_future_annotations_forward_ref_message() -> None:
    class ForwardRefExecutor(Executor):
        @handler
        async def handle(self, message: ForwardRefMessage, ctx: WorkflowContext) -> None:
            pass

    executor = ForwardRefExecutor(id="forward_ref")

    assert ForwardRefMessage in executor._handlers
    handler_func = executor._handlers[ForwardRefMessage]
    assert handler_func._handler_spec["message_type"] is ForwardRefMessage
