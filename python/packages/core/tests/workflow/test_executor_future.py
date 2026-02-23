# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class TypeA:
    value: str


@dataclass
class TypeB:
    value: int


class TestExecutorFutureAnnotations:
    """Test suite for Executor @handler with from __future__ import annotations."""

    def test_handler_future_annotations_workflow_context_generics(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        exec_instance = MyExecutor(id="future")

        handler_func = exec_instance._handlers[str]
        spec = cast(dict[str, Any], handler_func._handler_spec)

        assert spec["message_type"] is str
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]
