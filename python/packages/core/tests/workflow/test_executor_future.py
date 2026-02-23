# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class TypeA:
    value: str


@dataclass
class TypeB:
    value: int


def _make_executor_from_source(source: str) -> type[Executor]:
    ns: dict[str, object] = {
        "Executor": Executor,
        "WorkflowContext": WorkflowContext,
        "handler": handler,
        "TypeA": TypeA,
        "TypeB": TypeB,
    }
    exec(source, ns, ns)
    return ns["MyExecutor"]  # type: ignore[return-value]


class TestExecutorFutureAnnotations:
    """Regression tests for Executor @handler introspection with future annotations."""

    def test_handler_future_annotations_workflow_context_validation(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        ex = MyExecutor(id="test")

        assert str in ex._handlers

        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]

    def test_handler_future_annotations_message_type_resolved(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: TypeA, ctx: WorkflowContext) -> None:
                pass

        ex = MyExecutor(id="test")

        # Message type key should be the actual class, not a string
        assert TypeA in ex._handlers
        assert "TypeA" not in ex._handlers

        spec = ex._handler_specs[0]
        assert spec["message_type"] is TypeA

    def test_exec_future_annotations_stringized_annotations_resolve(self) -> None:
        MyExecutor = _make_executor_from_source(
            """
from __future__ import annotations

class MyExecutor(Executor):
    @handler
    async def example(self, input: 'TypeA', ctx: WorkflowContext['TypeA', 'TypeB']) -> None:
        pass
"""
        )

        # Ensure the raw signature annotations are actually strings (deterministic future-annotations behavior)
        raw_annotations = MyExecutor.example.__annotations__
        assert raw_annotations["input"] == "TypeA"
        assert raw_annotations["ctx"] == "WorkflowContext['TypeA', 'TypeB']"

        ex = MyExecutor(id="test")
        assert TypeA in ex._handlers

        spec = ex._handler_specs[0]
        assert spec["message_type"] is TypeA
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]

    def test_exec_future_annotations_unresolved_message_annotation_errors(self) -> None:
        MyExecutor = _make_executor_from_source(
            """
from __future__ import annotations

class MyExecutor(Executor):
    @handler
    async def example(self, input: 'DoesNotExist', ctx: WorkflowContext) -> None:
        pass
"""
        )

        with pytest.raises(ValueError, match=r"could not be resolved"):
            MyExecutor(id="test")

    def test_exec_future_annotations_unresolved_workflow_context_annotation_errors(self) -> None:
        MyExecutor = _make_executor_from_source(
            """
from __future__ import annotations

class MyExecutor(Executor):
    @handler
    async def example(self, input: 'TypeA', ctx: WorkflowContext['DoesNotExist']) -> None:
        pass
"""
        )

        with pytest.raises(ValueError, match=r"could not be resolved"):
            MyExecutor(id="test")

    def test_skip_message_annotation_true_does_not_require_message_type(self) -> None:
        class MyExecutor(Executor):
            @handler(input=TypeA)
            async def example(self, input, ctx: WorkflowContext) -> None:
                pass

        ex = MyExecutor(id="test")
        assert TypeA in ex._handlers
