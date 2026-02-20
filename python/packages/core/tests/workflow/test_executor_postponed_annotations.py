# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_framework import Executor, handler


@dataclass
class TypeA:
    value: str


@dataclass
class TypeB:
    value: int


class TestExecutorHandlerPostponedAnnotations:
    def test_handler_introspection_resolves_string_annotations(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, message: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        ex = MyExecutor(id="ex")

        assert str in ex._handlers
        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]

    def test_handler_introspection_invalid_ctx_annotation_errors_cleanly(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, message: str, ctx: str) -> None:
                pass

        with pytest.raises(ValueError, match=r"must be annotated as WorkflowContext"):
            MyExecutor(id="ex")

    def test_handler_explicit_types_still_resolves_ctx_annotation(self) -> None:
        class MyExecutor(Executor):
            @handler(input=str)
            async def example(self, message, ctx: WorkflowContext[TypeA]) -> None:
                pass

        ex = MyExecutor(id="ex")

        assert str in ex._handlers
        spec = ex._handler_specs[0]
        assert spec["message_type"] is str
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == []


def test_handler_introspection_resolves_string_annotations(self) -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, message: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
            pass

    ex = MyExecutor(id="ex")

    assert str in ex._handlers
    spec = ex._handler_specs[0]
    assert spec["message_type"] is str
    assert spec["output_types"] == [TypeA]
    assert spec["workflow_output_types"] == [TypeB]


def test_handler_introspection_invalid_ctx_annotation_errors_cleanly(self) -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, message: str, ctx: str) -> None:
            pass

    with pytest.raises(ValueError, match=r"must be annotated as WorkflowContext"):
        MyExecutor(id="ex")


def test_handler_explicit_types_still_resolves_ctx_annotation(self) -> None:
    class MyExecutor(Executor):
        @handler(input=str)
        async def example(self, message, ctx: WorkflowContext[TypeA]) -> None:
            pass

    ex = MyExecutor(id="ex")

    assert str in ex._handlers
    spec = ex._handler_specs[0]
    assert spec["message_type"] is str
    assert spec["output_types"] == [TypeA]
    assert spec["workflow_output_types"] == []
