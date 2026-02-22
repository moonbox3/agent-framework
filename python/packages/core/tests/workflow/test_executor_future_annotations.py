# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_framework import Executor, WorkflowContext, handler


class MyTypeA(BaseModel):
    pass


class MyTypeB(BaseModel):
    pass


def test_handler_future_annotations_resolves_workflow_context_generics() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
            pass

    # Defining the class should not raise; instantiation triggers handler discovery.
    e = MyExecutor(id="x")
    assert e.output_types == [MyTypeA]
    assert e.workflow_output_types == [MyTypeB]


def test_handler_invalid_ctx_annotation_still_raises() -> None:
    with pytest.raises(ValueError, match="must be annotated as WorkflowContext"):

        class BadExecutor(Executor):
            # Use explicit decorator types so mypy doesn't type-check ctx parameter as ContextT.
            @handler(input=str)
            async def example(self, input: Any, ctx: str) -> None:
                pass

            def __init__(self) -> None:
                super().__init__(id="bad")

        _ = BadExecutor()


def test_handler_future_annotations_get_type_hints_failure_does_not_fallback_to_raw_ctx_annotation() -> None:
    # Force typing.get_type_hints() to raise (NameError) due to unresolved forward ref in ctx annotation.
    # Define the missing type only within the local scope so it's absent from function globals.
    # The handler discovery should raise a targeted ValueError rather than letting a str/ForwardRef reach
    # validate_workflow_context_annotation().
    with pytest.raises(ValueError, match=r"unresolved type annotation.*parameter 'ctx'"):

        class MyMissingType(BaseModel):
            pass

        class BadExecutor(Executor):
            @handler
            async def example(
                self,
                input: str,
                ctx: WorkflowContext[MyMissingType, MyTypeB],  # noqa: F821
            ) -> None:
                pass

        _ = BadExecutor(id="bad")


def test_handler_future_annotations_message_enforcement_unchanged() -> None:
    # Message annotation enforcement unchanged: missing message annotation should still raise.
    with pytest.raises(ValueError, match="must have a type annotation for the message parameter"):

        class BadExecutor(Executor):
            @handler
            async def example(self, input, ctx: WorkflowContext[MyTypeA]) -> None:  # type: ignore[no-untyped-def]
                pass

        _ = BadExecutor(id="bad")

    # But when explicit decorator types are used, message annotation is skipped.
    class OkExecutor(Executor):
        @handler(input=str, output=MyTypeA)
        async def example(self, input, ctx) -> None:  # type: ignore[no-untyped-def]
            pass

    e = OkExecutor(id="ok")
    assert e.output_types == [MyTypeA]
