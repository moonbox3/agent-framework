# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from agent_framework import Executor, WorkflowContext, handler


class TestExecutorOutputTypeProperties:
    def test_output_types_and_workflow_output_types_exposed_as_lists(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, message: str, ctx: WorkflowContext[int, bool]) -> None:
                return None

        ex = MyExecutor(id="x")

        # User-observable behavior: properties return lists and include inferred types.
        assert isinstance(ex.output_types, list)
        assert isinstance(ex.workflow_output_types, list)

        assert int in ex.output_types
        assert bool in ex.workflow_output_types
