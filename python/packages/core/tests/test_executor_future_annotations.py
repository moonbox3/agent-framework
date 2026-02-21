from __future__ import annotations

import sys
import types

import pytest


def _exec_in_temp_module(name: str, code: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    try:
        exec(code, mod.__dict__)
        return mod
    except Exception:
        # Ensure cleanup on failure too
        sys.modules.pop(name, None)
        raise


def test_handler_registers_and_validates_workflow_context_with_future_annotations():
    code = """

from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext

class A: ...
class B: ...

class MyExecutor(Executor):
    @handler
    async def example(self, input: str, ctx: WorkflowContext[A, B]) -> None:
        return None
"""
    name = "tmp_future_ann_exec_ok"
    try:
        mod = _exec_in_temp_module(name, code)
        # Assert observable behavior: handler should be registered/discoverable.
        # This intentionally avoids asserting private internals; we only assert that
        # the decorated method exists and remains callable.
        assert hasattr(mod, "MyExecutor")
        assert hasattr(mod.MyExecutor, "example")
        assert callable(mod.MyExecutor.example)
    finally:
        sys.modules.pop(name, None)


def test_handler_still_rejects_invalid_ctx_annotation_with_future_annotations():
    code = """

from agent_framework._workflows._executor import Executor, handler

class NotCtx: ...

class MyExecutor(Executor):
    @handler
    async def example(self, input: str, ctx: NotCtx) -> None:
        return None
"""
    name = "tmp_future_ann_exec_invalid"
    sys.modules[name] = types.ModuleType(name)
    try:
        with pytest.raises(ValueError):
            exec(code, sys.modules[name].__dict__)
    finally:
        sys.modules.pop(name, None)


def test_explicit_handler_decorator_path_with_future_annotations():
    # Covers @handler(...) explicit configuration path to ensure it also resolves
    # future-annotations correctly.
    code = """

from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext

class A: ...
class B: ...

class MyExecutor(Executor):
    @handler(name="example")
    async def example(self, input: str, ctx: WorkflowContext[A, B]) -> None:
        return None
"""
    name = "tmp_future_ann_exec_explicit"
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    try:
        exec(code, mod.__dict__)
        assert hasattr(mod, "MyExecutor")
        assert hasattr(mod.MyExecutor, "example")
        assert callable(mod.MyExecutor.example)
    finally:
        sys.modules.pop(name, None)
