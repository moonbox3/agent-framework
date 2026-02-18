# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureMessage:
    value: str


@dataclass
class FutureOutput:
    value: int


@dataclass
class FutureWorkflowOutput:
    value: bool


def test_handler_future_annotations_resolve_workflow_context_generics_and_routing_key():
    class FutureAnnotationExecutor(Executor):
        @handler
        async def handle(
            self,
            message: FutureMessage,
            ctx: WorkflowContext[FutureOutput, FutureWorkflowOutput],
        ) -> None:
            pass

    executor = FutureAnnotationExecutor(id="future")

    # User-visible behavior: types are discoverable
    assert FutureMessage in executor.input_types
    assert set(executor.output_types) == {FutureOutput}
    assert set(executor.workflow_output_types) == {FutureWorkflowOutput}

    # Regression guard: handler routing key/spec should not remain as a string under postponed annotations
    assert all(not isinstance(t, str) for t in executor.input_types)
    assert all(not isinstance(spec["message_type"], str) for spec in executor._handler_specs)
    assert all(not isinstance(spec["ctx_annotation"], str) for spec in executor._handler_specs)


from __future__ import annotations

import pytest


def test_handler_unresolved_forward_ref_raises_clear_error_early():
    with pytest.raises(ValueError, match=r"could not be resolved.*parameter\(s\).*ctx"):

        class BadExecutor(Executor):  # type: ignore
            @handler
            async def handle(self, message: int, ctx: WorkflowContext[MissingType]) -> None:
                pass


def test_handler_missing_required_annotations_is_rejected():
    # Missing message annotation
    with pytest.raises(ValueError, match="must have a type annotation for the message parameter"):

        class MissingMessageAnnotationExecutor(Executor):  # type: ignore
            @handler
            async def handle(self, message, ctx: WorkflowContext) -> None:
                pass

    # Missing ctx annotation
    with pytest.raises(ValueError, match=r"parameter 'ctx' must be annotated as WorkflowContext"):

        class MissingCtxAnnotationExecutor(Executor):  # type: ignore
            @handler
            async def handle(self, message: int, ctx) -> None:
                pass


from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExplicitMsg:
    x: int


@dataclass
class ExplicitOut:
    y: int


@dataclass
class ExplicitWorkflowOut:
    z: int


def test_handler_explicit_types_allow_missing_ctx_annotation():
    class ExplicitTypesExecutor(Executor):
        # Explicit types mode: ctx annotation can be omitted
        @handler(input=ExplicitMsg, output=ExplicitOut, workflow_output=ExplicitWorkflowOut)
        async def handle(self, message, ctx) -> None:
            pass

    executor = ExplicitTypesExecutor(id="explicit")

    assert ExplicitMsg in executor.input_types
    assert set(executor.output_types) == {ExplicitOut}
    assert set(executor.workflow_output_types) == {ExplicitWorkflowOut}
