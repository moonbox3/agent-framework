# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureTypeA:
    value: str


@dataclass
class FutureTypeB:
    value: int


@dataclass
class FutureMessage:
    text: str


class TestExecutorFutureAnnotations:
    def test_executor_handler_future_annotations_inferred_types(self) -> None:
        """Regression: @handler introspection must resolve stringified annotations."""

        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: FutureMessage, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
                pass

        exec_instance = FutureExecutor(id="future_exec")

        # Ensure handler registration uses resolved class, not string
        assert FutureMessage in exec_instance._handlers

        # Ensure inferred context output types are resolved
        assert exec_instance.output_types == [FutureTypeA]
        assert exec_instance.workflow_output_types == [FutureTypeB]

    def test_executor_handler_future_annotations_missing_type_is_error(self) -> None:
        """Edge case: unresolved forward refs should fail with the existing ctx validation error."""

        try:

            class BadFutureExecutor(Executor):
                @handler
                async def handle(self, message: "MissingMessage", ctx: "WorkflowContext[MissingOut]") -> None:
                    pass

            _ = BadFutureExecutor
        except ValueError as exc:
            # When annotations can't be resolved, ctx_annotation remains a string,
            # and ctx validation should complain about needing WorkflowContext.
            assert "WorkflowContext" in str(exc)
        else:
            raise AssertionError("Expected ValueError due to unresolved annotations")
