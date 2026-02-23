# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_framework import Executor, WorkflowContext, handler

if TYPE_CHECKING:
    DoesNotExist = object


class TypeA:
    pass


class TypeB:
    pass


class TestExecutorFutureAnnotations:
    """Test suite for Executor/@handler with from __future__ import annotations."""

    def test_handler_decorator_future_annotations(self) -> None:
        class MyExecutor(Executor):
            @handler
            async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
                pass

        e = MyExecutor(id="test")

        # Ensure handler was registered correctly
        assert str in e._handlers

        spec = e._handler_specs[0]
        assert spec["message_type"] is str
        # OutT should be TypeA; W_OutT should be TypeB
        assert spec["output_types"] == [TypeA]
        assert spec["workflow_output_types"] == [TypeB]

    def test_handler_decorator_future_annotations_unresolvable_forward_ref_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match=r"could not be resolved"):

            class BadExecutor(Executor):
                @handler
                async def example(self, input: str, ctx: WorkflowContext[DoesNotExist]) -> None:  # noqa: F821
                    pass
