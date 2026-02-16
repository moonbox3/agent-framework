# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureMessage:
    value: str


@dataclass
class FutureOutput:
    value: int


@dataclass
class FutureWorkflowOutput:
    value: bool


class FutureAnnotationExecutor(Executor):
    @handler
    async def handle(self, message: FutureMessage, ctx: WorkflowContext[FutureOutput, FutureWorkflowOutput]) -> None:
        pass


def test_handler_future_annotations_resolved() -> None:
    executor = FutureAnnotationExecutor(id="future")

    assert FutureMessage in executor._handlers
    handler_func = executor._handlers[FutureMessage]
    assert handler_func._handler_spec["message_type"] is FutureMessage
    assert handler_func._handler_spec["output_types"] == [FutureOutput]
    assert handler_func._handler_spec["workflow_output_types"] == [FutureWorkflowOutput]
