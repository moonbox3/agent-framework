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


class FutureHandlerExecutor(Executor):
    @handler
    async def handle_text(self, message: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
        return None

    @handler
    async def handle_mapping(self, message: dict[str, Any], ctx: WorkflowContext[TypeA | TypeB, TypeC]) -> None:
        return None


class TestExecutorPostponedAnnotations:
    def test_handler_registration_with_postponed_annotations(self) -> None:
        executor = FutureHandlerExecutor(id="future-handler")

        assert str in executor.input_types
        assert TypeA in executor.output_types
        assert TypeB in executor.output_types
        assert TypeB in executor.workflow_output_types
        assert TypeC in executor.workflow_output_types
