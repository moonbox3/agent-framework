# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for handler signature validation with postponed annotations.

This file uses `from __future__ import annotations` so parameter annotations are
stored as strings and must be resolved via typing.get_type_hints during
@handler introspection.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_framework import Executor, WorkflowContext, handler


class MyTypeA(BaseModel):
    pass


class MyTypeB(BaseModel):
    pass


def test_handler_introspection_resolves_postponed_workflow_context_generic_args() -> None:
    """Defining a handler under postponed annotations should not raise."""

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
            return None

    # Instantiation triggers handler discovery; should not error.
    exec_ = MyExecutor(id="e1")
    assert exec_.input_types == [str]


def test_handler_introspection_resolves_nested_forward_refs_in_workflow_context() -> None:
    """WorkflowContext type arguments may be forward refs (string literals) nested in generics."""

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext["MyTypeA", "MyTypeB"]) -> None:
            return None

    exec_ = MyExecutor(id="e2")
    assert exec_.input_types == [str]


def test_handler_introspection_falls_back_to_raw_annotations_when_unresolvable() -> None:
    """If get_type_hints fails due to missing names, we should still raise a clear ValueError."""

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: "WorkflowContext[MissingType]") -> None:  # type: ignore[name-defined]
            return None

    with pytest.raises(ValueError, match="WorkflowContext"):
        MyExecutor(id="e3")
