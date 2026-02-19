# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

from agent_framework import Executor, WorkflowContext, handler


class TypeA:
    pass


class TypeB:
    pass


class TestHandlerFutureAnnotations:
    def test_handler_introspection_supports_future_annotations_workflow_context_generics(self):
        """@handler introspection should resolve postponed annotations via typing.get_type_hints."""

        class FutureAnnotationsExecutor(Executor):
            @handler
            async def handle(self, message: int, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        # Instantiation triggers handler discovery/signature validation.
        ex = FutureAnnotationsExecutor(id="future")

        assert int in ex._handlers
        spec = ex._handler_specs[0]
        assert spec["message_type"] is int
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]

    def test_handler_introspection_unresolved_forward_ref_raises_actionable_error(self):
        """If get_type_hints can't resolve stringized forward refs, raise an actionable ValueError."""

        with pytest.raises(ValueError, match="postponed or forward-referenced annotations"):

            class UnresolvedForwardRefExecutor(Executor):
                @handler
                async def handle(self, message: int, ctx: WorkflowContext["MissingType"]) -> None:
                    pass

            UnresolvedForwardRefExecutor(id="bad")
