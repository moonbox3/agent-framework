# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_framework import Executor, WorkflowContext, handler


class _TypeA(BaseModel):
    pass


class _TypeB(BaseModel):
    pass


class _FutureExecutor(Executor):
    @handler
    async def example(self, input: str, ctx: WorkflowContext[_TypeA, _TypeB]) -> None:
        return None


def test_executor_handler_future_annotations_resolves_workflow_context_generics() -> None:
    ex = _FutureExecutor(id="ex")
    assert _TypeA in ex.output_types
    assert _TypeB in ex.workflow_output_types


def test_executor_handler_future_annotations_unresolvable_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match=r"type annotation could not be resolved"):

        class _BadExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MissingType, MissingType]) -> None:
                return None
