# Copyright (c) Microsoft. All rights reserved.

import pytest
from typing_extensions import Never

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowExecutor,
    WorkflowValidationError,
    handler,
)
from agent_framework._workflows._checkpoint import InMemoryCheckpointStorage


class StartExecutor(Executor):
    @handler
    async def run(self, message: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(message, target_id="finish")


class FinishExecutor(Executor):
    @handler
    async def finish(self, message: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(message)


def build_sub_workflow(checkpoint_storage: InMemoryCheckpointStorage | None = None) -> WorkflowExecutor:
    sub_workflow = (
        WorkflowBuilder(start_executor="start", checkpoint_storage=checkpoint_storage)
        .register_executor(lambda: StartExecutor(id="start"), name="start")
        .register_executor(lambda: FinishExecutor(id="finish"), name="finish")
        .add_edge("start", "finish")
        .build()
    )
    return WorkflowExecutor(sub_workflow, id="sub")


def test_build_fails_when_parent_has_checkpoint_but_sub_does_not() -> None:
    """Parent has checkpoint_storage, sub-workflow does not -> error at build time."""
    storage = InMemoryCheckpointStorage()

    with pytest.raises(WorkflowValidationError, match="sub-workflow in executor 'sub'") as exc_info:
        WorkflowBuilder(start_executor="start", checkpoint_storage=storage).register_executor(
            lambda: StartExecutor(id="start"), name="start"
        ).register_executor(build_sub_workflow, name="sub").add_edge("start", "sub").build()

    assert exc_info.value.type == "checkpoint_configuration"


def test_build_succeeds_when_both_have_checkpoint() -> None:
    """Both parent and sub-workflow have checkpoint_storage -> no error."""
    storage = InMemoryCheckpointStorage()

    workflow = (
        WorkflowBuilder(start_executor="start", checkpoint_storage=storage)
        .register_executor(lambda: StartExecutor(id="start"), name="start")
        .register_executor(lambda: build_sub_workflow(checkpoint_storage=storage), name="sub")
        .add_edge("start", "sub")
        .build()
    )
    assert workflow is not None


def test_build_succeeds_when_neither_has_checkpoint() -> None:
    """Neither parent nor sub-workflow has checkpoint_storage -> no validation needed."""
    workflow = (
        WorkflowBuilder(start_executor="start")
        .register_executor(lambda: StartExecutor(id="start"), name="start")
        .register_executor(build_sub_workflow, name="sub")
        .add_edge("start", "sub")
        .build()
    )
    assert workflow is not None


async def test_runtime_checkpoint_validates_sub_workflows() -> None:
    """Runtime checkpoint_storage on run() triggers validation of sub-workflows."""
    storage = InMemoryCheckpointStorage()

    # Build without checkpoint_storage on either - succeeds
    workflow = (
        WorkflowBuilder(start_executor="start")
        .register_executor(lambda: StartExecutor(id="start"), name="start")
        .register_executor(build_sub_workflow, name="sub")
        .add_edge("start", "sub")
        .build()
    )

    # Run with runtime checkpoint_storage - should fail because sub has none
    with pytest.raises(WorkflowValidationError, match="sub-workflow in executor 'sub'") as exc_info:
        await workflow.run("hello", checkpoint_storage=storage)

    assert exc_info.value.type == "checkpoint_configuration"


def test_nested_sub_workflows_all_require_checkpoint() -> None:
    """A -> B -> C: if A has checkpoint, B must too, and B's build validates C."""
    storage = InMemoryCheckpointStorage()

    # Inner sub-workflow without checkpoint
    inner_sub = build_sub_workflow()

    # Middle workflow wrapping the inner sub - this should fail because
    # middle has checkpoint but inner doesn't
    with pytest.raises(WorkflowValidationError, match="sub-workflow in executor 'sub'") as exc_info:
        WorkflowBuilder(start_executor="start", checkpoint_storage=storage).register_executor(
            lambda: StartExecutor(id="start"), name="start"
        ).register_executor(lambda: inner_sub, name="sub").add_edge("start", "sub").build()

    assert exc_info.value.type == "checkpoint_configuration"


def test_error_message_identifies_executor() -> None:
    """Error message includes the executor ID of the offending sub-workflow."""
    storage = InMemoryCheckpointStorage()
    custom_id_sub = WorkflowExecutor(
        WorkflowBuilder(start_executor="start")
        .register_executor(lambda: StartExecutor(id="start"), name="start")
        .register_executor(lambda: FinishExecutor(id="finish"), name="finish")
        .add_edge("start", "finish")
        .build(),
        id="my_custom_executor_name",
    )

    with pytest.raises(WorkflowValidationError, match="my_custom_executor_name"):
        WorkflowBuilder(start_executor="start", checkpoint_storage=storage).register_executor(
            lambda: StartExecutor(id="start"), name="start"
        ).register_executor(lambda: custom_id_sub, name="my_custom_executor_name").add_edge(
            "start", "my_custom_executor_name"
        ).build()


def test_sub_workflow_without_checkpoint_parent_without_checkpoint_is_fine() -> None:
    """Sub-workflow has checkpoint but parent doesn't -> no error (sub manages its own checkpoints)."""
    storage = InMemoryCheckpointStorage()

    workflow = (
        WorkflowBuilder(start_executor="start")
        .register_executor(lambda: StartExecutor(id="start"), name="start")
        .register_executor(lambda: build_sub_workflow(checkpoint_storage=storage), name="sub")
        .add_edge("start", "sub")
        .build()
    )
    assert workflow is not None
