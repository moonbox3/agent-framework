# Copyright (c) Microsoft. All rights reserved.

"""Tests for the explicit output/intermediate selection contract on WorkflowBuilder."""

from __future__ import annotations

import warnings
from typing import Any

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


@executor
async def _start(messages: list[Message], ctx: WorkflowContext[str, str]) -> None:
    await ctx.yield_output("from-start")
    await ctx.send_message("downstream")


@executor
async def _downstream(message: str, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output("from-downstream")


def test_designation_unset_emits_deprecation_warning() -> None:
    """State A: WorkflowBuilder built without explicit designation warns."""
    with pytest.warns(DeprecationWarning, match="output_from or intermediate_output_from"):
        WorkflowBuilder(start_executor=_emit_one).build()


@pytest.mark.asyncio
async def test_designation_unset_preserves_legacy_all_output_behavior() -> None:
    """Omitted designation keeps legacy all-output behavior while warning."""
    with pytest.warns(DeprecationWarning, match="output_from or intermediate_output_from"):
        workflow = WorkflowBuilder(start_executor=_start).add_edge(_start, _downstream).build()

    result = await workflow.run([Message(role="user", contents=["hi"])])

    assert result.get_outputs() == ["from-start", "from-downstream"]
    assert result.get_intermediate_outputs() == []


@pytest.mark.asyncio
async def test_output_from_all_emits_all_outputs_without_legacy_warning() -> None:
    """Explicit all-output designation emits every executor payload without legacy warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        workflow = WorkflowBuilder(start_executor=_start, output_from="all").add_edge(_start, _downstream).build()

    result = await workflow.run([Message(role="user", contents=["hi"])])

    assert result.get_outputs() == ["from-start", "from-downstream"]
    assert result.get_intermediate_outputs() == []


@pytest.mark.asyncio
async def test_output_from_all_with_empty_intermediate_list_is_valid() -> None:
    """Explicit all-output plus an empty intermediate list is a concrete no-intermediate selection."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        workflow = (
            WorkflowBuilder(start_executor=_start, output_from="all", intermediate_output_from=[])
            .add_edge(_start, _downstream)
            .build()
        )

    result = await workflow.run([Message(role="user", contents=["hi"])])

    assert result.get_outputs() == ["from-start", "from-downstream"]
    assert result.get_intermediate_outputs() == []


def test_intermediate_output_from_all_is_rejected() -> None:
    """The all-output literal is only valid for workflow output selection."""
    with pytest.raises(ValueError, match="intermediate_output_from.*all.*output_from"):
        WorkflowBuilder(start_executor=_emit_one, intermediate_output_from="all")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("output_from", "intermediate_output_from"),
    [([_emit_one], None), (None, [_emit_one]), ([], [_emit_one])],
    ids=["output_list", "intermediate_list", "empty_output_with_intermediate"],
)
def test_explicit_designation_with_executor_does_not_warn(output_from, intermediate_output_from) -> None:
    """State B: any explicit designation with at least one executor opts into explicit mode without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        WorkflowBuilder(
            start_executor=_emit_one,
            output_from=output_from,
            intermediate_output_from=intermediate_output_from,
        ).build()


@pytest.mark.parametrize(
    ("output_from", "intermediate_output_from"),
    [([], None), (None, []), ([], [])],
    ids=["empty_output", "empty_intermediate", "both_empty"],
)
def test_empty_explicit_designation_fails(output_from, intermediate_output_from) -> None:
    """State C: explicit mode needs at least one output or intermediate executor."""
    with pytest.raises(WorkflowValidationError, match="at least one output or intermediate executor"):
        WorkflowBuilder(
            start_executor=_emit_one,
            output_from=output_from,
            intermediate_output_from=intermediate_output_from,
        ).build()


def test_passing_both_output_executors_and_output_from_raises_type_error() -> None:
    """State D: supplying a deprecated alias and the canonical kwarg is unambiguous user error."""
    with pytest.raises(TypeError, match="Cannot pass multiple workflow output selection parameters"):
        WorkflowBuilder(
            start_executor=_emit_one,
            output_executors=[_emit_one],
            output_from=[_emit_one],
        )


def test_intermediate_executors_builder_parameter_is_not_public() -> None:
    """The branch-only intermediate_executors builder parameter is not supported."""
    builder_type: Any = WorkflowBuilder
    with pytest.raises(TypeError, match="unexpected keyword argument 'intermediate_executors'"):
        builder_type(
            start_executor=_emit_one,
            intermediate_executors=[_emit_one],
        )
