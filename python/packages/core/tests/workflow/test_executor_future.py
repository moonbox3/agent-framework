# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any

from agent_framework import Executor, WorkflowContext, handler


class TypeA:
    pass


class TypeB:
    pass


class TypeC:
    pass


class FutureAnnotationsExecutor(Executor):
    """Executor used to validate @handler type introspection with postponed annotations."""

    @handler
    async def example(self, message: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
        # No runtime behavior needed for this test; decoration/registration should succeed.
        return None

    @handler
    async def union_example(self, message: dict[str, Any], ctx: WorkflowContext[TypeA | TypeB, TypeC]) -> None:
        return None


class TestExecutorFutureAnnotations:
    def test_executor_handler_future_annotations_resolve_ctx(self) -> None:
        """Ensure ctx annotations are evaluated (not strings) before validation."""
        ex = FutureAnnotationsExecutor(id="ex")

        # Handler registration should succeed, and output/workflow output inference should work.
        assert TypeA in ex.output_types
        assert TypeB in ex.workflow_output_types

    def test_executor_handler_future_annotations_resolve_unions(self) -> None:
        """Ensure union args inside WorkflowContext are correctly inferred."""
        ex = FutureAnnotationsExecutor(id="ex2")

        # union_example declares OutT = TypeA | TypeB and W_OutT = TypeC
        assert TypeA in ex.output_types
        assert TypeB in ex.output_types
        assert TypeC in ex.workflow_output_types
