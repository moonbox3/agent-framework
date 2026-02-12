# Copyright (c) Microsoft. All rights reserved.

"""Workflow wrapper for AG-UI protocol compatibility."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ag_ui.core import BaseEvent
from agent_framework import Workflow

from ._workflow_run import run_workflow_stream


class AgentFrameworkWorkflow:
    """Base AG-UI workflow wrapper.

    Can wrap a native ``Workflow`` or be subclassed for custom ``run_agent`` behavior.
    """

    def __init__(
        self,
        workflow: Workflow | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.workflow = workflow
        self.name = name if name is not None else getattr(workflow, "name", "workflow")
        self.description = description if description is not None else getattr(workflow, "description", "")

    async def run_agent(self, input_data: dict[str, Any]) -> AsyncGenerator[BaseEvent, None]:
        """Run the wrapped workflow and yield AG-UI events.

        Subclasses may override this to provide custom AG-UI streams.
        """
        if self.workflow is None:
            raise NotImplementedError("No workflow is attached. Override run_agent or pass workflow=...")
        async for event in run_workflow_stream(input_data, self.workflow):
            yield event


# Backward-compatible alias for older imports.
AgentFrameworkWorkflowAgent = AgentFrameworkWorkflow
