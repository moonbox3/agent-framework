# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import Executor, FunctionExecutor, WorkflowContext, WorkflowMessage, executor, handler


@dataclass
class FutureTypeA:
    value: str


@dataclass
class FutureTypeB:
    count: int


class TestFunctionExecutorFutureAnnotations:
    """Test suite for FunctionExecutor with from __future__ import annotations."""

    def test_executor_decorator_future_annotations(self):
        """Test @executor decorator works with stringified annotations."""

        @executor(id="future_test")
        async def process_future(value: int, ctx: WorkflowContext[int]) -> None:
            await ctx.send_message(value * 2)

        assert isinstance(process_future, FunctionExecutor)
        assert process_future.id == "future_test"
        assert int in process_future._handlers

        # Check spec
        spec = process_future._handler_specs[0]
        assert spec["message_type"] is int
        assert spec["output_types"] == [int]

    def test_executor_decorator_future_annotations_complex(self):
        """Test @executor decorator works with complex stringified annotations."""

        @executor
        async def process_complex(data: dict[str, Any], ctx: WorkflowContext[list[str]]) -> None:
            await ctx.send_message(["done"])

        assert isinstance(process_complex, FunctionExecutor)
        spec = process_complex._handler_specs[0]
        assert spec["message_type"] == dict[str, Any]
        assert spec["output_types"] == [list[str]]


class TestExecutorFutureAnnotations:
    """Test suite for Executor handlers with from __future__ import annotations."""

    def test_handler_future_annotations(self) -> None:
        """Ensure class handlers resolve postponed WorkflowContext annotations."""

        class FutureAnnotatedExecutor(Executor):
            @handler
            async def handle(
                self,
                message: FutureTypeA | FutureTypeB,
                ctx: WorkflowContext[list[str] | dict[str, Any], FutureTypeA | FutureTypeB],
            ) -> None:
                pass

        exec_instance = FutureAnnotatedExecutor(id="future_handler")
        assert exec_instance.can_handle(WorkflowMessage(data=FutureTypeA("hi"), source_id="mock"))
        assert exec_instance.can_handle(WorkflowMessage(data=FutureTypeB(1), source_id="mock"))
        assert set(exec_instance.output_types) == {list[str], dict[str, Any]}
        assert set(exec_instance.workflow_output_types) == {FutureTypeA, FutureTypeB}
