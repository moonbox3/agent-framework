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
    def test_handler_decorator_future_annotations_resolves_workflow_context(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        # Instantiation triggers handler discovery/validation.
        executor = MyExecutor(id="test")

        handler_func = executor._handlers[str]
        spec = getattr(handler_func, "_handler_spec", None)

        assert isinstance(spec, dict)
        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == [MyTypeB]
