# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any

import pytest

from agent_framework import Executor, WorkflowContext, handler


class TypeA:
    pass


class TypeB:
    pass


class TypeC:
    pass


class FutureHandlerExecutor(Executor):
    @handler
    async def handle_text(self, message: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
        return None

    @handler
    async def handle_mapping(self, message: dict[str, Any], ctx: WorkflowContext[TypeA | TypeB, TypeC]) -> None:
        return None


class StringForwardRefExecutor(Executor):
    # Force explicit string annotations even without relying on __future__ behavior.
    @handler
    async def handle_str(self, message: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
        return None


def _assert_no_str_types(executor: Executor) -> None:
    all_types: list[object] = []
    all_types.extend(executor.input_types)
    all_types.extend(executor.output_types)
    all_types.extend(executor.workflow_output_types)
    assert all(not isinstance(t, str) for t in all_types)


class TestExecutorPostponedAnnotations:
    def test_handler_registration_with_postponed_annotations(self) -> None:
        executor = FutureHandlerExecutor(id="future-handler")

        # input types
        assert str in executor.input_types
        assert dict in executor.input_types  # dict[str, Any] should normalize to dispatchable dict

        # output types
        assert TypeA in executor.output_types
        assert TypeB in executor.output_types

        # workflow outputs
        assert TypeB in executor.workflow_output_types
        assert TypeC in executor.workflow_output_types

        _assert_no_str_types(executor)

    def test_handler_registration_with_explicit_string_forward_refs(self) -> None:
        executor = StringForwardRefExecutor(id="string-forward-ref")

        assert str in executor.input_types
        assert TypeA in executor.output_types
        assert TypeB in executor.workflow_output_types
        _assert_no_str_types(executor)

    def test_unresolvable_forward_ref_fails_closed(self) -> None:
        # Defining the class should raise during @handler decoration.
        with pytest.raises(ValueError, match=r"unresolvable|Unable to resolve|forward reference"):

            class BadExecutor(Executor):
                @handler
                async def handle_bad(self, message: MissingType, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                    return None

            _ = BadExecutor
