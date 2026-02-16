# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import get_origin

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


class FutureAnnotatedExecutor(Executor):
    @handler
    async def handle(
        self,
        message: FutureMessage,
        ctx: WorkflowContext[FutureOutput, FutureWorkflowOutput],
    ) -> None:
        pass


class ExplicitTypeExecutor(Executor):
    @handler(input="FutureMessage", output="FutureOutput", workflow_output="FutureWorkflowOutput")
    async def handle(
        self,
        message,
        ctx: WorkflowContext[FutureOutput, FutureWorkflowOutput],
    ) -> None:
        pass


def test_handler_future_annotations_resolved() -> None:
    executor = FutureAnnotatedExecutor(id="future_executor")

    assert FutureMessage in executor._handlers

    handler_spec = executor._handler_specs[0]
    assert handler_spec["message_type"] is FutureMessage
    assert handler_spec["output_types"] == [FutureOutput]
    assert handler_spec["workflow_output_types"] == [FutureWorkflowOutput]

    ctx_annotation = handler_spec["ctx_annotation"]
    assert get_origin(ctx_annotation) is WorkflowContext


def test_handler_explicit_types_resolve_ctx_annotation() -> None:
    executor = ExplicitTypeExecutor(id="explicit_executor")

    assert FutureMessage in executor._handlers

    handler_spec = executor._handler_specs[0]
    assert handler_spec["message_type"] is FutureMessage
    assert handler_spec["output_types"] == [FutureOutput]
    assert handler_spec["workflow_output_types"] == [FutureWorkflowOutput]

    ctx_annotation = handler_spec["ctx_annotation"]
    assert get_origin(ctx_annotation) is WorkflowContext
