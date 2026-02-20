# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Annotated

import pytest
from typing_extensions import Never

from agent_framework import Executor, WorkflowContext, handler


@dataclass
class _TypeA:
    value: str


@dataclass
class _TypeB:
    value: int


def test_handler_introspection_resolves_workflow_context_annotation() -> None:
    class _Exec(Executor):
        @handler
        async def handle(
            self,
            message: int,
            ctx: WorkflowContext[_TypeA, _TypeB],
        ) -> None:  # type: ignore[valid-type]
            return None

    exec_instance = _Exec(id="e1")

    assert int in exec_instance.input_types
    assert _TypeA in exec_instance.output_types
    assert _TypeB in exec_instance.workflow_output_types


def test_handler_introspection_annotated_ctx_preserves_types() -> None:
    class _Exec(Executor):
        @handler
        async def handle(
            self,
            message: str,
            ctx: Annotated[WorkflowContext[Never, int | str], "meta"],
        ) -> None:  # type: ignore[valid-type]
            return None

    exec_instance = _Exec(id="annotated")

    assert str in exec_instance.input_types
    assert exec_instance.output_types == []
    assert int in exec_instance.workflow_output_types
    assert str in exec_instance.workflow_output_types


def test_handler_introspection_get_type_hints_failure_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: object, **kwargs: object) -> dict[str, object]:
        raise NameError("boom")

    monkeypatch.setattr(typing, "get_type_hints", _boom)

    ns: dict[str, object] = {
        "Executor": Executor,
        "WorkflowContext": WorkflowContext,
        "handler": handler,
    }

    code = """
from __future__ import annotations

class _Exec(Executor):
    @handler
    async def handle(self, message: int, ctx: WorkflowContext["MissingOut", "MissingWOut"]) -> None:
        return None

_Exec(id="bad")
"""

    with pytest.raises(ValueError, match="could not be resolved") as excinfo:
        exec(code, ns, ns)

    assert excinfo.value.__cause__ is not None
