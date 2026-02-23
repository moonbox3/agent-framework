# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

import pytest

from agent_framework import Executor, WorkflowContext, handler


class TypeA:
    pass


class TypeB:
    pass


class TestExecutorFutureAnnotations:
    """Test suite for Executor/@handler with from __future__ import annotations."""

    def test_handler_decorator_future_annotations(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        e = MyExecutor(id="test")

        # Ensure handler was registered correctly
        assert str in e._handlers

        spec = e._handler_specs[0]
        assert spec["message_type"] is str
        # OutT should be TypeA; W_OutT should be TypeB
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]

    def test_handler_decorator_future_annotations_unresolvable_forward_ref_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match=r"type annotations could not be resolved"):

            class BadExecutor(Executor):
                @handler
                async def example(self, input: str, ctx: WorkflowContext[DoesNotExist]) -> None:  # type: ignore[name-defined]  # noqa: F821
                    pass


def test_handler_decorator_non_future_annotations_preserve_typing_objects() -> None:
    """Regression test: non-__future__ typing objects must not be stringified/mis-propagated."""

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
            pass

    e = MyExecutor(id="test")
    spec = e._handler_specs[0]

    assert spec["message_type"] is str
    assert spec["ctx_annotation"].__origin__ is WorkflowContext  # type: ignore[attr-defined]
    assert spec["output_types"] == [TypeA]
    assert spec["workflow_output_types"] == [TypeB]


def test_handler_explicit_types_allows_missing_ctx_annotation() -> None:
    class MyExecutor(Executor):
        @handler(input=str, output=int)
        async def example(self, input, ctx) -> None:  # type: ignore[no-untyped-def]
            pass

    e = MyExecutor(id="test")
    spec = e._handler_specs[0]

    assert spec["message_type"] is str
    assert spec["output_types"] == [int]
    assert spec["workflow_output_types"] == []


def test_handler_future_annotations_forward_ref_requires_local_scope_resolves() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: "LocalMessage", ctx: WorkflowContext) -> None:
            pass

    class LocalMessage:  # defined after handler; requires class localns resolution
        pass

    e = MyExecutor(id="test")
    assert e.can_handle(__import__("agent_framework").WorkflowMessage(data=LocalMessage(), source_id="mock"))


def test_handler_future_annotations_missing_name_resolution_failure_is_clear() -> None:
    with pytest.raises(ValueError, match=r"type annotations could not be resolved"):

        class BadExecutor(Executor):
            @handler
            async def example(self, input: "MissingType", ctx: WorkflowContext) -> None:
                pass


def test_handler_future_annotations_message_param_forward_ref_failure_is_clear() -> None:
    with pytest.raises(ValueError, match=r"type annotations could not be resolved"):

        class BadExecutor(Executor):
            @handler
            async def example(self, input: "MissingMsg", ctx: WorkflowContext) -> None:
                pass
