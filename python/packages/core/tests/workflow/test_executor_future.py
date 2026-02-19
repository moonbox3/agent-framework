# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from agent_framework import Executor, WorkflowContext, handler


class MyTypeA(BaseModel):
    pass


class MyTypeB(BaseModel):
    pass


def test_handler_decorator_supports_future_annotations_workflow_context_generics() -> None:
    """Regression: @handler should accept WorkflowContext[T, U] when annotations are postponed."""

    class FutureCtxExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
            pass

    # Instantiation triggers handler discovery/validation.
    exec_instance = FutureCtxExecutor(id="future_ctx")

    assert str in exec_instance._handlers
    # Ensure output/workflow_output types were correctly inferred.
    assert exec_instance.output_types == [MyTypeA]
    assert exec_instance.workflow_output_types == [MyTypeB]


def test_handler_decorator_supports_future_annotations_forward_ref_message_type() -> None:
    """Regression: message annotations should be resolved so routing doesn't store a raw string."""

    @dataclass
    class MyMessage:
        content: str

    class ForwardRefMsgExecutor(Executor):
        @handler
        async def example(self, message: "MyMessage", ctx: WorkflowContext) -> None:
            pass

    exec_instance = ForwardRefMsgExecutor(id="forward_ref_msg")

    # If message annotation isn't resolved, handler would be registered under a string key,
    # which breaks routing via isinstance checks.
    assert MyMessage in exec_instance._handlers
