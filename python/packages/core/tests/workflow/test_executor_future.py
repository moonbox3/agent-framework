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
    """Tests for Executor @handler signature validation under postponed annotations."""

    def test_handler_future_annotations_workflow_context_two_args(self) -> None:
        """WorkflowContext[T, U] should validate when annotations are stringified."""

        class FutureExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        exec_instance = FutureExecutor(id="future")

        assert str in exec_instance._handlers
        handler_func = exec_instance._handlers[str]
        assert handler_func._handler_spec["output_types"] == [MyTypeA]
        assert handler_func._handler_spec["workflow_output_types"] == [MyTypeB]

    def test_handler_future_annotations_workflow_context_one_arg(self) -> None:
        """WorkflowContext[T] should validate when annotations are stringified."""

        class FutureExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA]) -> None:
                pass

        exec_instance = FutureExecutor(id="future_one")

        assert str in exec_instance._handlers
        handler_func = exec_instance._handlers[str]
        assert handler_func._handler_spec["output_types"] == [MyTypeA]
        assert handler_func._handler_spec["workflow_output_types"] == []

    def test_explicit_handler_types_still_work_without_ctx_annotation(self) -> None:
        """Explicit handler mode should remain unchanged and not require ctx annotation."""

        class ExplicitExecutor(Executor):
            @handler(input=str, output=int, workflow_output=bool)
            async def example(self, input, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

        exec_instance = ExplicitExecutor(id="explicit")

        assert str in exec_instance._handlers
        handler_func = exec_instance._handlers[str]
        assert handler_func._handler_spec["output_types"] == [int]
        assert handler_func._handler_spec["workflow_output_types"] == [bool]
