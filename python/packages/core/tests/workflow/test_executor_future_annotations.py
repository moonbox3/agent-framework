# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest
from typing_extensions import Never

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class _Msg:
    value: int


@dataclass
class _Out:
    value: int


class TestExecutorFutureAnnotations:
    def test_future_annotations_message_type_and_ctx_annotation_are_resolved(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext) -> None:
                pass

        ex = FutureExecutor(id="future_msg")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["ctx_annotation"] is WorkflowContext

    def test_future_annotations_workflow_context_one_param_is_inferred(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out]) -> None:
                pass

        ex = FutureExecutor(id="future_ctx_1")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == []

    def test_future_annotations_workflow_context_two_params_is_inferred(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out, str]) -> None:
                pass

        ex = FutureExecutor(id="future_ctx_2")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_allows_missing_message_and_ctx_annotations(self) -> None:
        class ExplicitModeExecutor(Executor):
            @handler(input=_Msg, output=_Out, workflow_output=str)
            async def handle(self, message, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutor(id="explicit_mode")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_validates_ctx_annotation_if_provided_but_does_not_infer(self) -> None:
        class ExplicitModeExecutorWithCtxAnno(Executor):
            @handler(input=_Msg)
            async def handle(self, message, ctx: WorkflowContext[Never, str]) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutorWithCtxAnno(id="explicit_mode_ctx")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        # Explicit mode: output/workflow_output types come only from decorator params.
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == []
        # But ctx_annotation should be validated and stored as resolved.
        assert spec["ctx_annotation"] == WorkflowContext[Never, str]

    def test_explicit_mode_rejects_non_workflow_context_ctx_annotation(self) -> None:
        with pytest.raises(ValueError, match="must be annotated as WorkflowContext"):

            class BadExplicitCtxExecutor(Executor):
                @handler(input=_Msg)
                async def handle(self, message, ctx: int) -> None:  # type: ignore[no-untyped-def]
                    pass

            BadExplicitCtxExecutor(id="bad_explicit_ctx")


from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Msg:
    value: int


@dataclass
class _Out:
    value: int


class TestExecutorFutureAnnotations:
    def test_future_annotations_message_type_and_ctx_annotation_are_resolved(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext) -> None:
                pass

        ex = FutureExecutor(id="future_msg")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["ctx_annotation"] is WorkflowContext

    def test_future_annotations_workflow_context_one_param_is_inferred(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out]) -> None:
                pass

        ex = FutureExecutor(id="future_ctx_1")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == []

    def test_future_annotations_workflow_context_two_params_is_inferred(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out, str]) -> None:
                pass

        ex = FutureExecutor(id="future_ctx_2")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_allows_missing_message_and_ctx_annotations(self) -> None:
        class ExplicitModeExecutor(Executor):
            @handler(input=_Msg, output=_Out, workflow_output=str)
            async def handle(self, message, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutor(id="explicit_mode")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_validates_ctx_annotation_if_provided_but_does_not_infer(self) -> None:
        class ExplicitModeExecutorWithCtxAnno(Executor):
            @handler(input=_Msg)
            async def handle(self, message, ctx: WorkflowContext[Never, str]) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutorWithCtxAnno(id="explicit_mode_ctx")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == []
        assert spec["ctx_annotation"] == WorkflowContext[Never, str]

    def test_explicit_mode_rejects_non_workflow_context_ctx_annotation(self) -> None:
        with pytest.raises(ValueError, match="must be annotated as WorkflowContext"):

            class BadExplicitCtxExecutor(Executor):
                @handler(input=_Msg)
                async def handle(self, message, ctx: int) -> None:  # type: ignore[no-untyped-def]
                    pass

            BadExplicitCtxExecutor(id="bad_explicit_ctx")


from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Msg:
    value: int


@dataclass
class _Out:
    value: int


class TestExecutorFutureAnnotations:
    def test_future_annotations_message_type_and_ctx_annotation_are_resolved(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext) -> None:
                pass

        ex = FutureExecutor(id="future_msg")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["ctx_annotation"] is WorkflowContext

    def test_future_annotations_workflow_context_one_param_is_inferred(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out]) -> None:
                pass

        ex = FutureExecutor(id="future_ctx_1")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == []

    def test_future_annotations_workflow_context_two_params_is_inferred(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: _Msg, ctx: WorkflowContext[_Out, str]) -> None:
                pass

        ex = FutureExecutor(id="future_ctx_2")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_allows_missing_message_and_ctx_annotations(self) -> None:
        class ExplicitModeExecutor(Executor):
            @handler(input=_Msg, output=_Out, workflow_output=str)
            async def handle(self, message, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutor(id="explicit_mode")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == [_Out]
        assert spec["workflow_output_types"] == [str]

    def test_explicit_mode_validates_ctx_annotation_if_provided_but_does_not_infer(self) -> None:
        class ExplicitModeExecutorWithCtxAnno(Executor):
            @handler(input=_Msg)
            async def handle(self, message, ctx: WorkflowContext[Never, str]) -> None:  # type: ignore[no-untyped-def]
                pass

        ex = ExplicitModeExecutorWithCtxAnno(id="explicit_mode_ctx")
        spec = ex._handler_specs[0]

        assert spec["message_type"] is _Msg
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == []
        assert spec["ctx_annotation"] == WorkflowContext[Never, str]

    def test_explicit_mode_rejects_non_workflow_context_ctx_annotation(self) -> None:
        with pytest.raises(ValueError, match="must be annotated as WorkflowContext"):

            class BadExplicitCtxExecutor(Executor):
                @handler(input=_Msg)
                async def handle(self, message, ctx: int) -> None:  # type: ignore[no-untyped-def]
                    pass

            BadExplicitCtxExecutor(id="bad_explicit_ctx")
