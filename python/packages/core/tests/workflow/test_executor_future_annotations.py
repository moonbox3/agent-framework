# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class FutureTypeA:
    value: str


@dataclass
class FutureTypeB:
    value: int


class TestExecutorFutureAnnotations:
    def test_handler_future_annotations_registers_under_resolved_message_type(self) -> None:
        class FutureExecutor(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
                pass

        # Prove the scenario: postponed evaluation stores annotations as strings in the raw signature.
        raw = inspect.signature(FutureExecutor.handle).parameters
        assert isinstance(raw["message"].annotation, str)
        assert isinstance(raw["ctx"].annotation, str)

        executor = FutureExecutor(id="future_parameterized")

        # Key outcome: registration uses concrete type keys, not stringified annotations.
        assert str in executor._handlers
        assert "str" not in executor._handlers

        # And ctx generic parameters are inferred correctly.
        assert set(executor.output_types) == {FutureTypeA}
        assert set(executor.workflow_output_types) == {FutureTypeB}

    def test_handler_future_annotations_union_context(self) -> None:
        class FutureUnionExecutor(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext[FutureTypeA | FutureTypeB]) -> None:
                pass

        raw = inspect.signature(FutureUnionExecutor.handle).parameters
        assert isinstance(raw["ctx"].annotation, str)

        executor = FutureUnionExecutor(id="future_union")
        assert str in executor._handlers
        assert "str" not in executor._handlers

        assert set(executor.output_types) == {FutureTypeA, FutureTypeB}
        assert executor.workflow_output_types == []

    def test_handler_future_annotations_unparameterized_context(self) -> None:
        class FutureBareExecutor(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext) -> None:
                pass

        raw = inspect.signature(FutureBareExecutor.handle).parameters
        assert isinstance(raw["ctx"].annotation, str)

        executor = FutureBareExecutor(id="future_bare")
        assert str in executor._handlers
        assert "str" not in executor._handlers

        assert executor.output_types == []
        assert executor.workflow_output_types == []

    def test_handler_future_annotations_fallback_when_get_type_hints_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force get_type_hints failure in the code under test to exercise the fallback path.
        import agent_framework._workflows._executor as executor_mod

        def _raise_name_error(*_args, **_kwargs):
            raise NameError("forced")

        monkeypatch.setattr(executor_mod, "get_type_hints", _raise_name_error)

        class FallbackExecutor(Executor):
            @handler
            async def handle(self, message: str, ctx: WorkflowContext[FutureTypeA]) -> None:
                pass

        raw = inspect.signature(FallbackExecutor.handle).parameters
        assert isinstance(raw["message"].annotation, str)
        assert isinstance(raw["ctx"].annotation, str)

        executor = FallbackExecutor(id="future_fallback")

        # Fallback contract: message and ctx strings are resolved; routing key is concrete type.
        assert str in executor._handlers
        assert "str" not in executor._handlers
        assert executor.output_types == [FutureTypeA]
        assert executor.workflow_output_types == []


def test_handler_future_annotations_registers_under_resolved_message_type() -> None:
    class FutureExecutor(Executor):
        @handler
        async def handle(self, message: str, ctx: WorkflowContext[FutureTypeA, FutureTypeB]) -> None:
            pass

    raw = inspect.signature(FutureExecutor.handle).parameters
    assert isinstance(raw["message"].annotation, str)
    assert isinstance(raw["ctx"].annotation, str)

    executor = FutureExecutor(id="future_parameterized")

    assert str in executor._handlers
    assert "str" not in executor._handlers

    assert set(executor.output_types) == {FutureTypeA}
    assert set(executor.workflow_output_types) == {FutureTypeB}


def test_handler_future_annotations_union_context() -> None:
    class FutureUnionExecutor(Executor):
        @handler
        async def handle(self, message: str, ctx: WorkflowContext[FutureTypeA | FutureTypeB]) -> None:
            pass

    raw = inspect.signature(FutureUnionExecutor.handle).parameters
    assert isinstance(raw["ctx"].annotation, str)

    executor = FutureUnionExecutor(id="future_union")
    assert str in executor._handlers
    assert "str" not in executor._handlers

    assert set(executor.output_types) == {FutureTypeA, FutureTypeB}
    assert executor.workflow_output_types == []


def test_handler_future_annotations_unparameterized_context() -> None:
    class FutureBareExecutor(Executor):
        @handler
        async def handle(self, message: str, ctx: WorkflowContext) -> None:
            pass

    raw = inspect.signature(FutureBareExecutor.handle).parameters
    assert isinstance(raw["ctx"].annotation, str)

    executor = FutureBareExecutor(id="future_bare")
    assert str in executor._handlers
    assert "str" not in executor._handlers

    assert executor.output_types == []
    assert executor.workflow_output_types == []


def test_handler_future_annotations_fallback_when_get_type_hints_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_framework._workflows._executor as executor_mod

    def _raise_name_error(*_args, **_kwargs):
        raise NameError("forced")

    monkeypatch.setattr(executor_mod, "get_type_hints", _raise_name_error)

    class FallbackExecutor(Executor):
        @handler
        async def handle(self, message: str, ctx: WorkflowContext[FutureTypeA]) -> None:
            pass

    raw = inspect.signature(FallbackExecutor.handle).parameters
    assert isinstance(raw["message"].annotation, str)
    assert isinstance(raw["ctx"].annotation, str)

    executor = FallbackExecutor(id="future_fallback")

    assert str in executor._handlers
    assert "str" not in executor._handlers
    assert executor.output_types == [FutureTypeA]
    assert executor.workflow_output_types == []
