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
    """Regression tests for Executor @handler when annotations are stringified.

    When `from __future__ import annotations` is enabled, parameter annotations
    are stored as strings and must be resolved via typing.get_type_hints.
    """

    def test_handler_decorator_future_annotations_workflow_context_generic(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
                pass

        exec_instance = MyExecutor(id="test")
        assert str in exec_instance._handlers

        spec = exec_instance._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [FutureTypeA]
        assert spec["workflow_output_types"] == [FutureTypeB]
