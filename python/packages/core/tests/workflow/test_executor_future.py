# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureTypeA:
    value: str


@dataclass
class FutureTypeB:
    value: int


class TestExecutorFutureAnnotations:
    def test_handler_decorator_future_annotations_resolves_workflow_context(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
                pass

        ex = MyExecutor(id="test")
        assert str in ex._handlers

        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [FutureTypeA]
        assert spec["workflow_output_types"] == [FutureTypeB]
