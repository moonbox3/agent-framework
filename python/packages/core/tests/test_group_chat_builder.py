# Copyright (c) Microsoft. All rights reserved.

from agent_framework import GroupChatBuilder
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._group_chat import GroupChatOrchestratorExecutor
from agent_framework._workflows._workflow_context import WorkflowContext


class _StubExecutor(Executor):
    """Minimal executor used to satisfy workflow wiring in tests."""

    def __init__(self, id: str) -> None:
        super().__init__(id=id)

    @handler
    async def handle(self, message: object, ctx: WorkflowContext[object]) -> None:
        await ctx.yield_output(message)


def test_set_manager_builds_with_agent_manager() -> None:
    """GroupChatBuilder should build when using an agent-based manager."""

    manager = _StubExecutor("manager_executor")
    participant = _StubExecutor("participant_executor")

    workflow = (
        GroupChatBuilder().set_manager(manager, display_name="Moderator").participants({"worker": participant}).build()
    )

    orchestrator = workflow.get_start_executor()

    assert isinstance(orchestrator, GroupChatOrchestratorExecutor)
    assert orchestrator._is_manager_agent()
