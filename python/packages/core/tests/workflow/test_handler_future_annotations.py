# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

from agent_framework import Executor, WorkflowContext, handler


class TypeA:
    pass


class TypeB:
    pass


def test_handler_introspection_supports_future_annotations_workflow_context_generics() -> None:
    """Ensure postponed annotations are resolved for WorkflowContext[T, U] generics."""

    class FutureAnnotationsExecutor(Executor):
        @handler
        async def handle(self, message: int, ctx: WorkflowContext[TypeA, TypeB]) -> None:
            return None

    ex = FutureAnnotationsExecutor(id="future")

    assert int in ex.input_types
    # output_types/workflow_output_types are sets flattened to lists; assert via membership.
    assert TypeA in ex.output_types
    assert TypeB in ex.workflow_output_types


def test_handler_introspection_unresolved_forward_ref_raises_actionable_error() -> None:
    """Unresolvable forward refs should raise a clear ValueError (not a cryptic typing error)."""

    with pytest.raises(ValueError, match="postponed|forward-referenced.*could not be resolved"):

        class UnresolvedForwardRefExecutor(Executor):
            @handler
            async def handle(self, message: int, ctx: WorkflowContext[\"MissingType\"]) -> None:
                return None

        UnresolvedForwardRefExecutor(id="bad")
