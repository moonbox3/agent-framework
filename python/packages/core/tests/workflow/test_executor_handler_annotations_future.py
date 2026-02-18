# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

from agent_framework import Executor, WorkflowContext, handler


class OutputA:
    pass


class OutputB:
    pass


def test_executor_handler_future_annotations_resolves_workflow_context_generics() -> None:
    class FutureExecutor(Executor):
        @handler
        async def handle(self, message: int, ctx: WorkflowContext[OutputA, OutputB]) -> None:
            await ctx.send_message(OutputA())
            await ctx.yield_output(OutputB())

    executor = FutureExecutor(id="future")

    assert int in executor.input_types
    assert OutputA in executor.output_types
    assert OutputB in executor.workflow_output_types


def test_executor_handler_quoted_forward_refs_resolve() -> None:
    class QuotedExecutor(Executor):
        @handler
        async def handle(self, message: int, ctx: WorkflowContext[OutputA, OutputB]) -> None:
            await ctx.send_message(OutputA())
            await ctx.yield_output(OutputB())

    executor = QuotedExecutor(id="quoted")

    assert int in executor.input_types
    assert OutputA in executor.output_types
    assert OutputB in executor.workflow_output_types


def test_executor_handler_class_local_types_resolve_in_ctx_annotation() -> None:
    class LocalTypesExecutor(Executor):
        class LocalOut:
            pass

        class LocalWorkflowOut:
            pass

        @handler
        async def handle(
            self,
            message: int,
            ctx: WorkflowContext[LocalTypesExecutor.LocalOut, LocalTypesExecutor.LocalWorkflowOut],
        ) -> None:
            await ctx.send_message(LocalTypesExecutor.LocalOut())
            await ctx.yield_output(LocalTypesExecutor.LocalWorkflowOut())

    executor = LocalTypesExecutor(id="local")

    assert int in executor.input_types
    assert LocalTypesExecutor.LocalOut in executor.output_types
    assert LocalTypesExecutor.LocalWorkflowOut in executor.workflow_output_types


def test_executor_handler_get_type_hints_failure_falls_back_to_explicit_resolution() -> None:
    class BrokenOtherAnnotationExecutor(Executor):
        # Force typing.get_type_hints() to raise by referencing an unknown type in a
        # non-essential annotation (return type), while keeping message/ctx resolvable.
        @handler
        async def handle(self, message: int, ctx: WorkflowContext[OutputA, OutputB]) -> MissingReturn:
            await ctx.send_message(OutputA())
            await ctx.yield_output(OutputB())

    executor = BrokenOtherAnnotationExecutor(id="broken")

    assert int in executor.input_types
    assert OutputA in executor.output_types
    assert OutputB in executor.workflow_output_types


def test_executor_handler_unresolved_forward_reference_raises_clear_error() -> None:
    with pytest.raises(ValueError, match=r"unresolved forward reference"):

        class BadExecutor(Executor):
            @handler
            async def handle(self, message: int, ctx: WorkflowContext[MissingType]) -> None:
                pass


def test_executor_handler_explicit_types_skip_message_annotation_allows_empty_ctx_annotation() -> None:
    class ExplicitTypesExecutor(Executor):
        @handler(input=int, output=OutputA, workflow_output=OutputB)
        async def handle(self, message, ctx) -> None:
            # explicit mode: ctx annotation intentionally omitted
            await ctx.send_message(OutputA())
            await ctx.yield_output(OutputB())

    executor = ExplicitTypesExecutor(id="explicit")

    assert int in executor.input_types
    assert OutputA in executor.output_types
    assert OutputB in executor.workflow_output_types


def test_executor_handler_get_type_hints_failure_falls_back_to_explicit_resolution() -> None:
    class BrokenOtherAnnotationExecutor(Executor):
        @handler
        async def handle(self, message: int, ctx: WorkflowContext[OutputA, OutputB]) -> MissingReturn:
            await ctx.send_message(OutputA())
            await ctx.yield_output(OutputB())

    executor = BrokenOtherAnnotationExecutor(id="broken")

    assert int in executor.input_types
    assert OutputA in executor.output_types
    assert OutputB in executor.workflow_output_types


def test_executor_handler_explicit_types_skip_message_annotation_allows_empty_ctx_annotation() -> None:
    class ExplicitTypesExecutor(Executor):
        @handler(input=int, output=OutputA, workflow_output=OutputB)
        async def handle(self, message, ctx) -> None:
            await ctx.send_message(OutputA())
            await ctx.yield_output(OutputB())

    executor = ExplicitTypesExecutor(id="explicit")

    assert int in executor.input_types
    assert OutputA in executor.output_types
    assert OutputB in executor.workflow_output_types
