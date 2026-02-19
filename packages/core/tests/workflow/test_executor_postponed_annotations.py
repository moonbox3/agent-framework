# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class OutA:
    value: int


@dataclass
class OutB:
    value: str


@dataclass
class Msg:
    text: str


class TestExecutorHandlerPostponedAnnotations:
    def test_future_annotations_workflow_context_is_resolved(self) -> None:
        """Regression: postponed annotations should not break WorkflowContext validation."""

        class MyExec(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext[OutA, OutB]) -> None:
                pass

        exec_instance = MyExec(id="x")

        # Public, stable outcomes: inferred output types.
        assert OutA in exec_instance.output_types
        assert OutB in exec_instance.workflow_output_types

    def test_quoted_forward_ref_ctx_annotation_is_resolved_and_infers_outputs(self) -> None:
        """Edge case: quoted forward refs should resolve via get_type_hints."""

        class MyExec(Executor):
            @handler
            async def handle(self, message: Msg, ctx: "WorkflowContext[OutA, OutB]") -> None:
                pass

        exec_instance = MyExec(id="x")
        assert OutA in exec_instance.output_types
        assert OutB in exec_instance.workflow_output_types

    def test_unresolvable_forward_ref_raises_actionable_error(self) -> None:
        """Failure path: when ctx annotation can't be resolved, error should be actionable."""

        class MyExec(Executor):
            @handler
            async def handle(
                self, message: str, ctx: "WorkflowContext[not_a_module.MissingOut]"
            ) -> None:
                pass

        with pytest.raises(ValueError, match=r"annotation could not be resolved at runtime"):
            MyExec(id="x")

# Copyright (c) Microsoft. All rights reserved.


from dataclasses import dataclass

import pytest

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class OutA:
    value: int


@dataclass
class OutB:
    value: str


@dataclass
class Msg:
    text: str


class TestExecutorHandlerPostponedAnnotations:
    def test_future_annotations_workflow_context_is_resolved(self) -> None:
        class MyExec(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext[OutA, OutB]) -> None:
                pass

        exec_instance = MyExec(id="x")

        assert OutA in exec_instance.output_types
        assert OutB in exec_instance.workflow_output_types

    def test_quoted_forward_ref_ctx_annotation_is_resolved_and_infers_outputs(self) -> None:
        class MyExec(Executor):
            @handler
            async def handle(self, message: Msg, ctx: "WorkflowContext[OutA, OutB]") -> None:
                pass

        exec_instance = MyExec(id="x")
        assert OutA in exec_instance.output_types
        assert OutB in exec_instance.workflow_output_types

    def test_unresolvable_forward_ref_raises_actionable_error(self) -> None:
        class MyExec(Executor):
            @handler
            async def handle(
                self, message: str, ctx: "WorkflowContext[not_a_module.MissingOut]"
            ) -> None:
                pass

        with pytest.raises(ValueError, match=r"annotation could not be resolved at runtime"):
            MyExec(id="x")
