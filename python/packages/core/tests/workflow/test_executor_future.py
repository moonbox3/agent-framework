# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class MyTypeA:
    value: str


@dataclass
class MyTypeB:
    value: int


class TestExecutorFutureAnnotations:
    """Regression tests for Executor handler discovery with postponed annotations."""

    def test_handler_future_annotations_resolved_for_context_generics(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                pass

        exec_instance = MyExecutor(id="test")

        handler_func = exec_instance._handlers[str]
        spec = handler_func._handler_spec

        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == [MyTypeB]

        assert MyTypeA in exec_instance.output_types
        assert MyTypeB in exec_instance.workflow_output_types

    def test_handler_future_annotations_unresolvable_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="could not be resolved"):

            class BadExecutor(Executor):
                @handler
                async def example(self, input: str, ctx: "WorkflowContext[MissingType]") -> None:  # type: ignore[name-defined]
                    pass
