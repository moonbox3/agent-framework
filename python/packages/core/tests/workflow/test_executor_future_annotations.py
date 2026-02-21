# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

import inspect
from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class MyTypeA:
    value: str


@dataclass
class MyTypeB:
    value: int


def test_executor_handler_future_annotations_workflow_context_is_resolved():
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
            pass

    ex = MyExecutor(id="test")

    assert str in ex.input_types
    assert MyTypeA in ex.output_types
    assert MyTypeB in ex.workflow_output_types


def test_executor_handler_explicit_types_ctx_unannotated_preserves_behavior():
    class MyExecutor(Executor):
        @handler(input=str, output=int)
        async def example(self, input, ctx) -> None:
            pass

    ex = MyExecutor(id="test")

    assert str in ex.input_types
    assert int in ex.output_types

    spec = ex._handler_specs[0]
    assert spec["ctx_annotation"] is inspect.Parameter.empty
