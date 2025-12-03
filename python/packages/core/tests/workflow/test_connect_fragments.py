# Copyright (c) Microsoft. All rights reserved.

from typing import Any

import pytest

from agent_framework import (
    Executor,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)


class _Source(Executor):
    @handler
    async def start(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)


class _Upper(Executor):
    @handler
    async def elevate(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text.upper())


class _AppendBang(Executor):
    @handler
    async def punctuate(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(f"{text}!")


class _Sink(Executor):
    @handler
    async def finish(self, text: str, ctx: WorkflowContext[Any, str]) -> None:
        await ctx.yield_output(text)


async def test_connect_merges_fragments_and_runs() -> None:
    connection_one = (
        WorkflowBuilder().add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src").as_connection()
    )

    connection_two = (
        WorkflowBuilder()
        .add_edge(_Source(id="forward"), _AppendBang(id="bang"))
        .set_start_executor("forward")
        .as_connection()
    )

    sink = _Sink(id="sink")
    builder = WorkflowBuilder()
    handle_one = builder.add_workflow(connection_one, prefix="f1")
    handle_two = builder.add_workflow(connection_two, prefix="f2")
    builder.connect(handle_one.outputs[0], handle_two.start)
    builder.connect(handle_two.outputs[0], sink)
    builder.set_start_executor(handle_one.start)

    workflow: Workflow = builder.build()
    result = await workflow.run("hello")

    assert result.get_outputs() == ["HELLO!"]
    assert any(exec_id.startswith("f1/") for exec_id in workflow.executors)
    assert any(exec_id.startswith("f2/") for exec_id in workflow.executors)


async def test_connect_detects_id_collision_with_raw_connection() -> None:
    connection = (
        WorkflowBuilder().add_edge(_Source(id="dup"), _Upper(id="dup_upper")).set_start_executor("dup").as_connection()
    )

    builder = WorkflowBuilder()
    builder.add_edge(_Source(id="dup"), _Sink(id="terminal"))
    builder.set_start_executor("dup")

    with pytest.raises(ValueError):
        builder.add_connection(connection)


async def test_workflow_as_connection_round_trip() -> None:
    inner_builder = (
        WorkflowBuilder(name="wrapped")
        .add_edge(_Source(id="inner_src"), _Sink(id="inner_sink"))
        .set_start_executor("inner_src")
    )
    inner = inner_builder.build()

    outer = WorkflowBuilder()
    handle = outer.add_workflow(inner)
    outer.set_start_executor(handle.start)
    workflow = outer.build()

    result = await workflow.run("pipeline")
    assert result.get_outputs() == ["pipeline"]
    assert any(exec_id.startswith("wrapped/") for exec_id in workflow.executors)


# =============================================================================
# MergeResult tests
# =============================================================================


async def test_merge_returns_merge_result_with_id_mapping() -> None:
    """merge() returns MergeResult that maps original IDs to prefixed IDs."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src")

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    # Attribute access
    assert result.src == "p/src"
    assert result.up == "p/up"

    # Dict access
    assert result["src"] == "p/src"
    assert result["up"] == "p/up"

    # Prefix property
    assert result.prefix == "p"


async def test_merge_result_attribute_access_with_valid_identifiers() -> None:
    """MergeResult supports attribute access for valid Python identifiers."""
    fragment = WorkflowBuilder(name="frag").set_start_executor(_Source(id="my_executor"))

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="test")

    assert result.my_executor == "test/my_executor"


async def test_merge_result_dict_access_for_invalid_identifiers() -> None:
    """MergeResult dict access works for IDs that aren't valid Python identifiers."""
    fragment = WorkflowBuilder(name="frag").set_start_executor(_Source(id="my-executor"))

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="test")

    # Dict access works for hyphenated IDs
    assert result["my-executor"] == "test/my-executor"


async def test_merge_result_raises_attribute_error_for_unknown_id() -> None:
    """MergeResult raises AttributeError for unknown executor IDs."""
    fragment = WorkflowBuilder(name="frag").set_start_executor(_Source(id="src"))

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    with pytest.raises(AttributeError, match="No executor with original id 'unknown'"):
        _ = result.unknown


async def test_merge_result_raises_key_error_for_unknown_id() -> None:
    """MergeResult raises KeyError for unknown executor IDs via dict access."""
    fragment = WorkflowBuilder(name="frag").set_start_executor(_Source(id="src"))

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    with pytest.raises(KeyError, match="No executor with original id 'unknown'"):
        _ = result["unknown"]


async def test_merge_result_contains_check() -> None:
    """MergeResult supports 'in' operator for checking ID existence."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src")

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    assert "src" in result
    assert "up" in result
    assert "unknown" not in result


async def test_merge_result_iteration_yields_prefixed_ids() -> None:
    """Iterating MergeResult yields prefixed IDs."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src")

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    prefixed_ids = list(result)
    assert "p/src" in prefixed_ids
    assert "p/up" in prefixed_ids


async def test_merge_result_items_yields_original_and_prefixed_pairs() -> None:
    """MergeResult.items() yields (original_id, prefixed_id) pairs."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src")

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    items = dict(result.items())
    assert items["src"] == "p/src"
    assert items["up"] == "p/up"


async def test_merge_result_keys_yields_original_ids() -> None:
    """MergeResult.keys() yields original (unprefixed) IDs."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src")

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    keys = list(result.keys())
    assert "src" in keys
    assert "up" in keys


async def test_merge_result_len() -> None:
    """MergeResult supports len() to get executor count."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src")

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="p")

    assert len(result) == 2


async def test_merge_result_used_with_add_edge() -> None:
    """MergeResult IDs can be used directly with add_edge()."""
    frag_a = WorkflowBuilder(name="a").set_start_executor(_Source(id="src"))
    frag_b = WorkflowBuilder(name="b").add_edge(_Upper(id="up"), _Sink(id="sink")).set_start_executor("up")

    builder = WorkflowBuilder()
    a = builder.merge(frag_a, prefix="a")
    b = builder.merge(frag_b, prefix="b")

    # Wire using MergeResult IDs
    builder.add_edge(a.src, b.up)
    builder.set_start_executor(a.src)

    workflow = builder.build()
    result = await workflow.run("hello")
    assert result.get_outputs() == ["HELLO"]


async def test_merge_result_used_with_get_executor() -> None:
    """MergeResult IDs can be used with get_executor() for type-safe access."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Sink(id="sink")).set_start_executor("src")

    builder = WorkflowBuilder()
    ids = builder.merge(fragment, prefix="p")

    # Get executor using MergeResult ID
    src_exec = builder.get_executor(ids.src)
    sink_exec = builder.get_executor(ids["sink"])

    assert src_exec.id == "p/src"
    assert sink_exec.id == "p/sink"
    assert src_exec.input_types == [str]


async def test_merge_result_repr() -> None:
    """MergeResult has a useful repr."""
    fragment = WorkflowBuilder(name="frag").add_edge(_Source(id="src"), _Upper(id="up")).set_start_executor("src")

    builder = WorkflowBuilder()
    result = builder.merge(fragment, prefix="test")

    repr_str = repr(result)
    assert "MergeResult" in repr_str
    assert "test" in repr_str
    assert "src" in repr_str
    assert "up" in repr_str
