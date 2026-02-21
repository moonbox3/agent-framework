# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

import inspect
from typing import Annotated

import pytest

from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


def test_handler_decorator_accepts_workflow_context_with_future_annotations_and_annotation_is_stringized() -> None:
    class MyTypeA:
        pass

    class MyTypeB:
        pass

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:  # noqa: A002
            return None

    # Precondition: with future annotations enabled in this module, the raw signature
    # annotation should be a string.
    sig = inspect.signature(MyExecutor.example)
    ctx_ann = sig.parameters["ctx"].annotation
    assert isinstance(ctx_ann, str)

    # If decorator validation raised, class definition would fail.
    assert MyExecutor is not None


def test_handler_decorator_accepts_annotated_workflow_context_under_future_annotations() -> None:
    class MyTypeA:
        pass

    class MyTypeB:
        pass


# Copyright (c) Microsoft. All rights reserved.


def test_handler_decorator_accepts_workflow_context_with_future_annotations_and_annotation_is_stringized() -> None:
    class MyTypeA:
        pass

    class MyTypeB:
        pass

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:  # noqa: A002
            return None

    # With future annotations enabled in this module, the raw signature annotation is stringized.
    sig = inspect.signature(MyExecutor.example)
    ctx_ann = sig.parameters["ctx"].annotation
    assert isinstance(ctx_ann, str)

    # If decorator validation raised, class definition would fail.
    assert MyExecutor is not None


def test_handler_decorator_accepts_annotated_workflow_context_under_future_annotations() -> None:
    class MyTypeA:
        pass

    class MyTypeB:
        pass

    class MyExecutor(Executor):
        @handler
        async def example(
            self,
            input: str,
            ctx: Annotated[WorkflowContext[MyTypeA, MyTypeB], "meta"],  # noqa: A002
        ) -> None:
            return None

    assert MyExecutor is not None


# Define the name so Ruff doesn't flag it as undefined (F821), but use an expression
# that will still fail during type-hint evaluation.
DefinitelyPresentName = object()


def test_handler_decorator_future_annotations_unresolved_forward_ref_errors_with_stable_message() -> None:
    # If type-hint evaluation fails, validation should raise a user-visible error.
    with pytest.raises(Exception) as excinfo:

        class MyExecutor(Executor):
            @handler
            async def example(
                self,
                input: str,
                ctx: WorkflowContext[__import__("definitely_missing_module_12345")],  # noqa: A002
            ) -> None:
                return None

    assert "ctx" in str(excinfo.value)
    assert "WorkflowContext" in str(excinfo.value)

    sig = inspect.signature(MyExecutor.example)
    assert isinstance(sig.parameters["ctx"].annotation, str)

    assert MyExecutor is not None


# Copyright (c) Microsoft. All rights reserved.


def test_handler_decorator_accepts_annotated_workflow_context_under_future_annotations() -> None:
    class MyTypeA:
        pass

    class MyTypeB:
        pass

    class MyExecutor(Executor):
        @handler
        async def example(
            self,
            input: str,
            ctx: Annotated[WorkflowContext[MyTypeA, MyTypeB], "meta"],  # noqa: A002
        ) -> None:
            return None

    assert MyExecutor is not None


# Copyright (c) Microsoft. All rights reserved.


def test_handler_decorator_future_annotations_unresolved_forward_ref_errors_with_stable_message() -> None:
    with pytest.raises(Exception) as excinfo:

        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[_ThisNameDoesNotExist]) -> None:  # noqa: A002
                return None

    assert "ctx" in str(excinfo.value)
    assert "WorkflowContext" in str(excinfo.value)


# Copyright (c) Microsoft. All rights reserved.


def test_handler_decorator_accepts_workflow_context_with_future_annotations_and_annotation_is_stringized() -> None:
    class MyTypeA:
        pass

    class MyTypeB:
        pass

    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:  # noqa: A002
            return None

    sig = inspect.signature(MyExecutor.example)
    assert isinstance(sig.parameters["ctx"].annotation, str)


def test_handler_decorator_accepts_annotated_workflow_context_under_future_annotations() -> None:
    class MyTypeA:
        pass

    class MyTypeB:
        pass

    class MyExecutor(Executor):
        @handler
        async def example(
            self,
            input: str,
            ctx: Annotated[WorkflowContext[MyTypeA, MyTypeB], "meta"],  # noqa: A002
        ) -> None:
            return None

    assert MyExecutor is not None


def test_handler_decorator_future_annotations_unresolved_forward_ref_errors_with_stable_message() -> None:
    with pytest.raises(Exception) as excinfo:

        class MyExecutor(Executor):
            @handler
            async def example(
                self,
                input: str,
                ctx: WorkflowContext[__import__("definitely_missing_module_12345")],  # noqa: A002
            ) -> None:
                return None

    assert "ctx" in str(excinfo.value)
    assert "WorkflowContext" in str(excinfo.value)
