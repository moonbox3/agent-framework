# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agent_framework import Executor, WorkflowContext, handler
from agent_framework._workflows._typing_utils import resolve_type_annotation


class TestExecutorFutureAnnotations:
    """Test suite for class-based Executor with from __future__ import annotations."""

    def test_handler_decorator_future_annotations_ctx_generic_two_params(self):
        class MyTypeA(BaseModel):
            pass

        class MyTypeB(BaseModel):
            pass

        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        ex = MyExecutor(id="ex")

        # Ensure handler registered for message type
        assert str in ex._handlers

        # Ensure ctx annotation was validated and output types inferred correctly
        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == [MyTypeB]

    def test_handler_decorator_future_annotations_ctx_generic_one_param(self):
        class MyTypeA(BaseModel):
            pass

        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA]) -> None:
                pass

        ex = MyExecutor(id="ex")

        assert str in ex._handlers
        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == []

    def test_output_types_properties_are_typed_and_deduped(self):
        class MyTypeA(BaseModel):
            pass

        class MyTypeB(BaseModel):
            pass

        class MyExecutor(Executor):
            @handler
            async def example1(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

            @handler
            async def example2(self, input: int, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        ex = MyExecutor(id="ex")

        # These properties should return concrete lists at runtime and include both types once.
        assert set(ex.output_types) == {MyTypeA}
        assert set(ex.workflow_output_types) == {MyTypeB}

    def test_resolve_type_annotation_unknown_optional_namespace_returns_any(self):
        # Namespace packages like "google" are often optional dependencies.
        # If a forward reference mentions them but they're not installed,
        # we should not crash during type resolution.
        resolved = resolve_type_annotation("google.cloud.storage.Client")
        assert resolved is Any
