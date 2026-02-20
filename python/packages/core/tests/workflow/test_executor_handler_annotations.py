# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_framework import Executor, WorkflowContext, handler


class MyTypeA(BaseModel):
    pass


class MyTypeB(BaseModel):
    pass


def test_handler_introspection_resolves_postponed_workflow_context_generic_args() -> None:
    """Postponed WorkflowContext[...] annotations should be resolved and propagated."""

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
            return None

    exec_ = MyExecutor(id="e1")
    assert exec_.input_types == [str]
    assert exec_.output_types == [MyTypeA]
    assert exec_.workflow_output_types == [MyTypeB]


def test_handler_introspection_resolves_nested_forward_refs_in_workflow_context() -> None:
    """WorkflowContext generic args may be forward refs and should resolve under postponed annotations."""

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
            return None

    exec_ = MyExecutor(id="e2")
    assert exec_.input_types == [str]
    assert exec_.output_types == [MyTypeA]
    assert exec_.workflow_output_types == [MyTypeB]


def test_handler_introspection_invalid_ctx_annotation_still_raises() -> None:
    """Under postponed annotations, invalid ctx annotation should still raise ValueError."""

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: int) -> None:
            return None

    with pytest.raises(ValueError, match="must be annotated as WorkflowContext"):
        MyExecutor(id="e3")


def test_handler_explicit_types_allows_missing_ctx_annotation_under_postponed_annotations() -> None:
    """Explicit decorator mode must not require ctx annotation."""

    class MyExecutor(Executor):
        @handler(input=str, output=MyTypeA, workflow_output=MyTypeB)
        async def example(self, input, ctx) -> None:
            return None

    exec_ = MyExecutor(id="e4")
    assert exec_.input_types == [str]
    assert exec_.output_types == [MyTypeA]
    assert exec_.workflow_output_types == [MyTypeB]
