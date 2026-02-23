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
    """Regression tests for Executor/@handler with postponed (string) annotations."""

    def test_handler_future_annotations_workflow_context_two_params(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        exec_instance = MyExecutor(id="future_exec")

        # Handler is registered for str input
        assert str in exec_instance._handlers

        # Ensure ctx annotation was resolved and output types were inferred
        spec = exec_instance._handler_specs[0]
        assert spec["ctx_annotation"] == WorkflowContext[MyTypeA, MyTypeB]
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == [MyTypeB]

    def test_handler_future_annotations_workflow_context_one_param(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA]) -> None:
                pass

        exec_instance = MyExecutor(id="future_exec_one")

        spec = exec_instance._handler_specs[0]
        assert spec["ctx_annotation"] == WorkflowContext[MyTypeA]
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == []
