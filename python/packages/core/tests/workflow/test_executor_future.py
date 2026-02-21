# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class MyTypeA:
    value: str


@dataclass
class MyTypeB:
    value: int


class TestExecutorFutureAnnotations:
    """Test suite for Executor with from __future__ import annotations."""

    def test_handler_future_annotations_workflow_context_generic(self):
        """Ensure @handler validation resolves string annotations for WorkflowContext[T, U]."""

        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        ex = MyExecutor(id="test")

        # Handler should be registered for input type str
        assert str in ex._handlers

        # Output types inferred from WorkflowContext[MyTypeA, MyTypeB]
        assert MyTypeA in ex.output_types
        assert ex.workflow_output_types == [MyTypeB]
