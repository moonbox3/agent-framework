# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest
from typing_extensions import Never

from agent_framework import Executor, WorkflowContext, handler


class MissingType:
    """Sentinel used for forward-ref tests (removed from globals to force NameError)."""


class MissingReturnType:
    """Sentinel return type used to ensure explicit-mode ignores return annotations."""


class TestExecutorFutureAnnotations:
    def test_executor_handler_future_annotations_resolves_workflow_context_and_infers_outputs(self) -> None:
        class FutureAnnotatedExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[int, bool]) -> None:
                return None

        ex = FutureAnnotatedExecutor(id="future")

        # Observable behavior: handler registration succeeds with inferred type mapping
        assert str in ex.input_types
        assert int in ex.output_types
        assert bool in ex.workflow_output_types

    def test_executor_handler_unresolved_ctx_forward_ref_raises_clear_value_error_with_chain(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"parameter 'ctx' annotation could not be resolved/validated",
        ) as excinfo:

            class BadExecutor(Executor):
                @handler
                async def example(self, input: str, ctx: WorkflowContext[MissingType, Never]) -> None:
                    return None

            # Force runtime failure during annotation resolution while keeping Ruff happy:
            # remove the name from the handler function globals so evaluation raises NameError.
            BadExecutor.example.__globals__.pop("MissingType", None)  # type: ignore[attr-defined]

            BadExecutor(id="bad")

        # Deterministic: ctx string eval fails with NameError and is preserved via chaining
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, NameError)

    def test_skip_message_annotation_allows_unannotated_ctx_and_ignores_unresolved_return_type(self) -> None:
        # Explicit-mode handler validation must not attempt to resolve unrelated annotations.
        class ExplicitTypesExecutor(Executor):
            @handler(input=str, output=int)
            async def example(self, input, ctx) -> MissingReturnType:
                return None

        ex = ExplicitTypesExecutor(id="explicit")
        assert str in ex.input_types
        assert int in ex.output_types


def test_executor_handler_unresolved_ctx_forward_ref_raises_clear_value_error_with_chain(self) -> None:
    with pytest.raises(
        ValueError,
        match=r"parameter 'ctx' annotation could not be resolved/validated",
    ) as excinfo:

        class BadExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MissingType, Never]) -> None:
                return None

        # Force runtime failure during annotation resolution while keeping Ruff happy:
        BadExecutor.example.__globals__.pop("MissingType", None)  # type: ignore[attr-defined]

        BadExecutor(id="bad")

    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, NameError)
