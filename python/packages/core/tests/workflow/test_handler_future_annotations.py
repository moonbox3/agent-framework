# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler


class OutputMessage:
    pass


class WorkflowOutput:
    pass


class FutureAnnotationExecutor(Executor):
    @handler
    async def handle_string(self, message: str, ctx: WorkflowContext[OutputMessage, WorkflowOutput]) -> None:
        return None

    @handler
    async def handle_int(self, message: int, ctx: WorkflowContext[OutputMessage, WorkflowOutput]) -> None:
        return None


def test_handler_future_annotations_resolve() -> None:
    executor = FutureAnnotationExecutor(id="future")
    specs = {spec["message_type"]: spec for spec in executor._handler_specs}

    assert specs[str]["output_types"] == [OutputMessage]
    assert specs[str]["workflow_output_types"] == [WorkflowOutput]
    assert specs[int]["output_types"] == [OutputMessage]
    assert specs[int]["workflow_output_types"] == [WorkflowOutput]
