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


class TestExecutorFutureAnnotations:
    def test_executor_handler_future_annotations(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(
                self, message: FutureMessage, ctx: WorkflowContext[FutureOutput, FutureWorkflowOutput]
            ) -> None:
                pass

        executor = FutureExecutor(id="future")
        spec = executor._handler_specs[0]

        assert spec["message_type"] is FutureMessage
        assert spec["output_types"] == [FutureOutput]
        assert spec["workflow_output_types"] == [FutureWorkflowOutput]
