# Copyright (c) Microsoft. All rights reserved.

"""Tests for the explicit output/intermediate designation contract on WorkflowBuilder.

State A: output_executors=None and intermediate_executors=None -> DeprecationWarning at build
State B: explicit designation with at least one executor -> no warning
State C: explicit designation with no executors -> validation error
"""

from __future__ import annotations

import warnings

import pytest
from typing_extensions import Never

from agent_framework import (
    Message,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowValidationError,
    executor,
)


@executor
async def _emit_one(messages: list[Message], ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output("hello")


def test_designation_unset_emits_deprecation_warning() -> None:
    """State A: WorkflowBuilder built without explicit designation warns."""
    with pytest.warns(DeprecationWarning, match="output_executors or intermediate_executors"):
        WorkflowBuilder(start_executor=_emit_one).build()


@pytest.mark.parametrize(
    ("output_executors", "intermediate_executors"),
    [([_emit_one], None), (None, [_emit_one]), ([], [_emit_one])],
    ids=["output_list", "intermediate_list", "empty_output_with_intermediate"],
)
def test_explicit_designation_with_executor_does_not_warn(output_executors, intermediate_executors) -> None:
    """State B: any explicit designation with at least one executor opts into explicit mode without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        WorkflowBuilder(
            start_executor=_emit_one,
            output_executors=output_executors,
            intermediate_executors=intermediate_executors,
        ).build()


@pytest.mark.parametrize(
    ("output_executors", "intermediate_executors"),
    [([], None), (None, []), ([], [])],
    ids=["empty_output", "empty_intermediate", "both_empty"],
)
def test_empty_explicit_designation_fails(output_executors, intermediate_executors) -> None:
    """State C: explicit mode needs at least one output or intermediate executor."""
    with pytest.raises(WorkflowValidationError, match="at least one output or intermediate executor"):
        WorkflowBuilder(
            start_executor=_emit_one,
            output_executors=output_executors,
            intermediate_executors=intermediate_executors,
        ).build()
