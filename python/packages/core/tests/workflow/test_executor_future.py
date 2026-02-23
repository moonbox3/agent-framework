# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Never

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class TypeA:
    value: int


@dataclass
class TypeB:
    value: str


class TestExecutorFutureAnnotations:
    """Regression tests for Executor handler validation with postponed (string) annotations."""

    def test_handler_future_annotations_workflow_context_two_args(self) -> None:
        class FutureTwoArgsExecutor(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        ex = FutureTwoArgsExecutor(id="future_two")
        assert set(ex.output_types) == {TypeA}
        assert set(ex.workflow_output_types) == {TypeB}

    def test_handler_future_annotations_workflow_context_union_args(self) -> None:
        class FutureUnionArgsExecutor(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext[TypeA | TypeB, Never]) -> None:
                pass

        ex = FutureUnionArgsExecutor(id="future_union")
        assert set(ex.output_types) == {TypeA, TypeB}
        assert ex.workflow_output_types == []
