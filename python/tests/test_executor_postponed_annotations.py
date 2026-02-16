from __future__ import annotations

from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class OutputTypeA:
    pass


class OutputTypeB:
    pass


class PostponedAnnotationExecutor(Executor):
    @handler
    async def handle(self, message: str, ctx: WorkflowContext[OutputTypeA, OutputTypeB]) -> None:
        return None


def test_postponed_annotations_are_resolved() -> None:
    executor = PostponedAnnotationExecutor(id="postponed")
    assert set(executor.output_types) == {OutputTypeA}
    assert set(executor.workflow_output_types) == {OutputTypeB}
