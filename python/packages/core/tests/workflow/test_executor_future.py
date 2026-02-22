# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class MyTypeA:
    value: int


@dataclass
class MyTypeB:
    value: str


class TestExecutorFutureAnnotations:
    """Test suite for Executor/@handler with from __future__ import annotations."""

    def test_handler_decorator_future_annotations_workflow_context_generics(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        exec_instance = MyExecutor(id="future_test")

        assert str in exec_instance._handlers

        spec = exec_instance._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == [MyTypeB]
