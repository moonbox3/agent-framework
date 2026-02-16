# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureInput:
    value: str


@dataclass
class FutureOutput:
    value: int


@dataclass
class FutureWorkflowOutput:
    value: bool


def test_handler_future_annotations_are_resolved() -> None:
    class FutureExecutor(Executor):
        @handler
        async def handle(self, message: FutureInput, ctx: WorkflowContext[FutureOutput, FutureWorkflowOutput]) -> None:
            pass

    executor = FutureExecutor(id="future_executor")

    assert FutureInput in executor.input_types
    assert FutureOutput in executor.output_types
    assert FutureWorkflowOutput in executor.workflow_output_types


def test_handler_future_annotations_union_generics() -> None:
    class UnionExecutor(Executor):
        @handler
        async def handle(self, message: int, ctx: WorkflowContext[int | str, bool]) -> None:
            pass

    executor = UnionExecutor(id="union_executor")

    assert int in executor.output_types
    assert str in executor.output_types
    assert len(executor.output_types) == 2
    assert executor.workflow_output_types == [bool]
