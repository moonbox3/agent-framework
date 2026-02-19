# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureTypeA:
    v: str


@dataclass
class FutureTypeB:
    v: int


class TestExecutorFutureAnnotations:
    """Tests for Executor/@handler when `from __future__ import annotations` is enabled."""

    def test_handler_decorator_future_annotations_resolves_ctx_generic(self) -> None:
        """Executor subclass definition should not raise when ctx annotation is stringified."""

        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
                pass

        # Instantiation triggers handler discovery and uses resolved annotations
        exec_instance = MyExecutor(id="future")

        # User-observable behavior: executor has handler registered for `str`
        assert str in exec_instance._handlers

        # And declared workflow output types were inferred
        assert FutureTypeA in exec_instance.output_types
        assert FutureTypeB in exec_instance.workflow_output_types

    def test_handler_decorator_future_annotations_resolves_message_type(self) -> None:
        """Message type should be resolved to actual typing object (not a string)."""

        class ComplexInputExecutor(Executor):
            @handler
            async def example(self, input: dict[str, Any], ctx: WorkflowContext) -> None:
                pass

        exec_instance = ComplexInputExecutor(id="complex")

        spec = exec_instance._handler_specs[0]
        assert spec["message_type"] == dict[str, Any]
