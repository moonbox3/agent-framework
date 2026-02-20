# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest
from typing_extensions import Annotated, Never

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class _TypeA:
    value: str


@dataclass
class _TypeB:
    value: int


class TestExecutorHandlerFutureAnnotations:
    def test_handler_introspection_resolves_workflow_context_annotation(self) -> None:
        class _Exec(Executor):
            @handler
            async def handle(self, message: int, ctx: WorkflowContext[_TypeA, _TypeB]) -> None:  # type: ignore[valid-type]
                return None

        exec_instance = _Exec(id="e1")

        assert _TypeA in exec_instance.output_types
        assert _TypeB in exec_instance.workflow_output_types

    def test_handler_introspection_requires_localns_for_nested_forward_refs(self) -> None:
        def _factory():
            @dataclass
            class LocalOut:
                n: int

            @dataclass
            class LocalWOut:
                s: str

            class LocalExec(Executor):
                @handler
                async def handle(
                    self,
                    message: int,
                    ctx: WorkflowContext["LocalOut", "LocalWOut"],
                ) -> None:
                    return None

            return LocalExec, LocalOut, LocalWOut

        LocalExec, LocalOut, LocalWOut = _factory()
        exec_instance = LocalExec(id="nested")

        assert LocalOut in exec_instance.output_types
        assert LocalWOut in exec_instance.workflow_output_types

    def test_handler_introspection_unresolved_names_raise_clear_error(self) -> None:
        ns: dict[str, object] = {
            "Executor": Executor,
            "WorkflowContext": WorkflowContext,
            "handler": handler,
        }
        code = """
class _Exec(Executor):
    @handler
    async def handle(self, message: int, ctx: WorkflowContext[\"MissingOut\", \"MissingWOut\"]) -> None:
        return None

_Exec(id=\"bad\")
"""

        with pytest.raises(ValueError, match="could not be resolved") as excinfo:
            exec(code, ns, ns)

        assert excinfo.value.__cause__ is not None

    def test_handler_introspection_annotated_ctx_preserves_types(self) -> None:
        class _Exec(Executor):
            @handler
            async def handle(
                self,
                message: str,
                ctx: Annotated[WorkflowContext[Never, int | str], "meta"],
            ) -> None:  # type: ignore[valid-type]
                return None

        exec_instance = _Exec(id="annotated")

        assert exec_instance.output_types == []
        assert int in exec_instance.workflow_output_types
        assert str in exec_instance.workflow_output_types
