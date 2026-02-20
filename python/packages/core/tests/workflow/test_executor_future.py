# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Never

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class _Msg:
    value: int


@dataclass
class _Out:
    value: int


class TestExecutorFutureAnnotations:
    """Tests for class-based Executor handler validation with PEP 563 stringized annotations."""

    def test_handler_future_annotations_workflow_context_unsubscripted(self) -> None:
        class FutureCtxExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext) -> None:
                pass

        ex = FutureCtxExecutor(id="future_ctx")
        spec = ex._handler_specs[0]
        assert spec["message_type"] is _Msg
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == []

    def test_handler_future_annotations_workflow_context_one_param(self) -> None:
        class FutureCtxExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out]) -> None:
                pass

        ex = FutureCtxExecutor(id="future_ctx_1")
        spec = ex._handler_specs[0]
        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == []

    def test_handler_future_annotations_workflow_context_two_params(self) -> None:
        class FutureCtxExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out, str]) -> None:
                pass

        ex = FutureCtxExecutor(id="future_ctx_2")
        spec = ex._handler_specs[0]
        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_still_allows_missing_annotations_on_message_and_ctx(self) -> None:
        class ExplicitModeExecutor(Executor):
            @handler(input=_Msg, output=_Out, workflow_output=str)
            async def handle(self, message, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutor(id="explicit_mode")
        spec = ex._handler_specs[0]
        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_with_ctx_annotation_is_validated_and_inferred(self) -> None:
        class ExplicitModeExecutorWithCtxAnno(Executor):
            @handler(input=_Msg)
            async def handle(self, message, ctx: WorkflowContext[Never, str]) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutorWithCtxAnno(id="explicit_mode_ctx")
        spec = ex._handler_specs[0]
        assert spec["message_type"] is _Msg
        # explicit mode: output/workflow_output come from decorator params only
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == []
        # but ctx_annotation should still be accepted by validation and preserved in spec
        assert spec["ctx_annotation"] == WorkflowContext[Never, str]
