# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import typing

import pytest

from agent_framework import Executor, WorkflowContext, handler


class TestExecutorHandlerFutureAnnotations:
    def test_handler_future_annotations_ctx_generic_1_and_2_args(self) -> None:
        class FutureHandlerExecutor(Executor):
            @handler
            async def handle_one(self, message: int, ctx: WorkflowContext[int]) -> None:
                return None

            @handler
            async def handle_two(self, message: str, ctx: WorkflowContext[int, str]) -> None:
                return None

        exec_instance = FutureHandlerExecutor(id="future_handler")

        # Registration succeeds; inferred types are exposed via public properties.
        assert set(exec_instance.input_types) == {int, str}
        assert set(exec_instance.output_types) == {int}
        assert set(exec_instance.workflow_output_types) == {str}

    def test_handler_future_annotations_message_annotation_is_resolved(self) -> None:
        class FutureMessageExecutor(Executor):
            @handler
            async def handle(self, message: int, ctx: WorkflowContext) -> None:
                return None

        exec_instance = FutureMessageExecutor(id="future_message")
        assert set(exec_instance.input_types) == {int}

    def test_handler_future_annotations_invalid_ctx_resolved_preserves_error_semantics(self) -> None:
        # ctx: "int" is resolvable via get_type_hints, so it should raise the existing
        # WorkflowContext mismatch error (not the targeted "could not be resolved" error).
        with pytest.raises(ValueError, match=r"must be annotated as WorkflowContext"):

            class BadCtxExecutor(Executor):
                @handler
                async def handle(self, message: int, ctx: int) -> None:
                    return None

            BadCtxExecutor(id="bad_ctx")

    def test_handler_future_annotations_unresolvable_forward_ref_in_workflowcontext_raises_targeted_error(self) -> None:
        # Use typing.ForwardRef to trigger get_type_hints failure without introducing
        # undefined identifiers that Ruff would flag (F821).
        with pytest.raises(ValueError, match=r"could not be resolved under postponed annotations"):

            class UnresolvableCtxExecutor(Executor):
                @handler
                async def handle(
                    self,
                    message: int,
                    ctx: WorkflowContext[typing.ForwardRef("NotAType")],
                ) -> None:
                    return None

            UnresolvableCtxExecutor(id="unresolvable_ctx")

    def test_handler_future_annotations_unrelated_get_type_hints_failure_does_not_trigger_targeted_ctx_error(
        self,
    ) -> None:
        # Force get_type_hints() to fail for an unrelated reason (unknown message type), while ctx is valid.
        # We should not emit the targeted ctx error because ctx itself is not the reason hints failed.
        with pytest.raises(ValueError, match=r"must have a type annotation for the message parameter"):

            class UnrelatedHintFailureExecutor(Executor):
                @handler
                async def handle(
                    self,
                    message: typing.ForwardRef("UnknownMessageType"),
                    ctx: WorkflowContext[int],
                ) -> None:
                    return None

            UnrelatedHintFailureExecutor(id="unrelated_hint_failure")


class TestExecutorHandlerFutureAnnotations:
    def test_handler_future_annotations_ctx_generic_1_and_2_args(self) -> None:
        class FutureHandlerExecutor(Executor):
            @handler
            async def handle_one(self, message: int, ctx: WorkflowContext[int]) -> None:
                return None

            @handler
            async def handle_two(self, message: str, ctx: WorkflowContext[int, str]) -> None:
                return None

        exec_instance = FutureHandlerExecutor(id="future_handler")
        assert set(exec_instance.input_types) == {int, str}
        assert set(exec_instance.output_types) == {int}
        assert set(exec_instance.workflow_output_types) == {str}


class TestExecutorHandlerFutureAnnotations:
    def test_handler_future_annotations_unresolvable_forward_ref_in_workflowcontext_raises_targeted_error(self) -> None:
        with pytest.raises(ValueError, match=r"could not be resolved under postponed annotations"):

            class UnresolvableCtxExecutor(Executor):
                @handler
                async def handle(
                    self,
                    message: int,
                    ctx: WorkflowContext[typing.ForwardRef("NotAType")],
                ) -> None:
                    return None

            UnresolvableCtxExecutor(id="unresolvable_ctx")


class TestExecutorHandlerFutureAnnotations:
    def test_handler_future_annotations_unrelated_get_type_hints_failure_does_not_trigger_targeted_ctx_error(
        self,
    ) -> None:
        with pytest.raises(ValueError, match=r"must have a type annotation for the message parameter"):

            class UnrelatedHintFailureExecutor(Executor):
                @handler
                async def handle(
                    self,
                    message: typing.ForwardRef("UnknownMessageType"),
                    ctx: WorkflowContext[int],
                ) -> None:
                    return None

            UnrelatedHintFailureExecutor(id="unrelated_hint_failure")
