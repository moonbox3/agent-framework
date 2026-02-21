# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

import inspect
import typing

import pytest

from agent_framework import Executor, WorkflowContext, handler


class TypeA:
    pass


class TypeB:
    pass


def _ctx_annotation_of(fn: typing.Callable[..., typing.Any]) -> typing.Any:
    sig = inspect.signature(fn)
    return sig.parameters["ctx"].annotation


def test_handler_introspection_resolves_future_annotations_for_workflow_context_generics() -> None:
    # Define the handler in a module that has future-annotations enabled (this file).
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
            return None

    # Under postponed evaluation, inspect.signature will show a string annotation.
    raw = _ctx_annotation_of(MyExecutor.example)
    assert isinstance(raw, str)

    # But type evaluation should resolve to the real typing object.
    resolved = typing.get_type_hints(MyExecutor.example, include_extras=True)["ctx"]
    assert typing.get_origin(resolved) is WorkflowContext
    assert typing.get_args(resolved) == (TypeA, TypeB)


def test_handler_introspection_resolves_explicit_string_literal_annotation_for_ctx() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
            return None

    raw = inspect.signature(MyExecutor.example).parameters["ctx"].annotation
    assert isinstance(raw, str)

    resolved = typing.get_type_hints(MyExecutor.example, include_extras=True)["ctx"]
    assert typing.get_origin(resolved) is WorkflowContext
    assert typing.get_args(resolved) == (TypeA, TypeB)


def test_handler_introspection_unresolvable_forward_ref_raises_clear_error() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(
            self,
            input: str,
            ctx: WorkflowContext[no_such_module.MissingTypeA, no_such_module.MissingTypeB],
        ) -> None:
            return None

    # Use a non-existent module path so resolution fails deterministically without
    # referencing bare undefined names (ruff F821 in this repo).
    with pytest.raises(ModuleNotFoundError):
        typing.get_type_hints(MyExecutor.example, include_extras=True)


def test_handler_introspection_unresolvable_forward_ref_raises_clear_error() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(
            self,
            input: str,
            ctx: WorkflowContext[no_such_module.MissingTypeA, no_such_module.MissingTypeB],
        ) -> None:
            return None

    with pytest.raises(ModuleNotFoundError):
        typing.get_type_hints(MyExecutor.example, include_extras=True)
