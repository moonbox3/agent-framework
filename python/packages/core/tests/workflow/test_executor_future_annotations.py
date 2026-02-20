# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

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
    def test_executor_handler_future_annotations_discover_and_infer(self) -> None:
        """Regression: __future__.annotations must not break handler discovery or type inference."""

        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: FutureMessage, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
                pass

        exec_instance = FutureExecutor(id="future_exec")

        # Public assertions: discovery and inferred types
        assert FutureMessage in exec_instance.input_types
        assert exec_instance.output_types == [FutureTypeA]
        assert exec_instance.workflow_output_types == [FutureTypeB]

    def test_executor_handler_future_annotations_unresolvable_annotation_raises_value_error(self) -> None:
        """Edge case: annotation evaluation failure should raise a controlled ValueError.

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

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


def test_executor_handler_future_annotations_discover_and_infer() -> None:
    """Regression: __future__.annotations must not break handler discovery or type inference."""

    class FutureExecutor(Executor):
        @handler
        async def handle(self, message: FutureMessage, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
            pass

    exec_instance = FutureExecutor(id="future_exec")

    # Public assertions: discovery and inferred types
    assert FutureMessage in exec_instance.input_types
    assert exec_instance.output_types == [FutureTypeA]
    assert exec_instance.workflow_output_types == [FutureTypeB]


def test_executor_handler_future_annotations_unresolvable_annotation_raises_value_error() -> None:
    """Edge case: failing to evaluate annotations should raise a controlled ValueError.

    Avoids ruff F821 by only referencing a defined symbol (Present) in the string annotation,
    but forces typing.get_type_hints() evaluation failure by referencing a missing attribute.
    """

    class Present:
        pass

    with pytest.raises(ValueError, match=r"type annotations could not be evaluated"):

        class BadFutureExecutor(Executor):
            @handler
            async def handle(
                self,
                message: "Present.Missing",  # attribute does not exist -> evaluation fails
                ctx: WorkflowContext[FutureTypeA, FutureTypeB],
            ) -> None:
                pass

    """

    class Present:  # noqa: B903
        pass

    with pytest.raises(ValueError, match=r"type annotations could not be evaluated"):

        class BadFutureExecutor(Executor):
            @handler
            async def handle(
                self,
                message: "Present.Missing",  # attribute does not exist -> evaluation fails
                ctx: WorkflowContext[FutureTypeA, FutureTypeB],
            ) -> None:
                pass
