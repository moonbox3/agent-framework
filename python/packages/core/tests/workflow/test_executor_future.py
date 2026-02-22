# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Never

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class MyTypeA:
    x: int = 1


@dataclass
class MyTypeB:
    y: int = 2


class TestExecutorFutureAnnotations:
    """Regression tests for Executor/@handler with from __future__ import annotations.

    When future annotations are enabled, annotations are stored as strings and must be
    resolved before calling validate_workflow_context_annotation.
    """

    def test_handler_future_annotations_workflow_context_two_params(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        ex = MyExecutor(id="x")

        # Validate handler spec contains resolved ctx annotation and inferred types
        assert len(ex._handler_specs) == 1
        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == [MyTypeB]

    def test_handler_future_annotations_workflow_context_never_and_type(self) -> None:
        class YieldOnlyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[Never, MyTypeB]) -> None:
                pass

        ex = YieldOnlyExecutor(id="y")

        assert len(ex._handler_specs) == 1
        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == [MyTypeB]
