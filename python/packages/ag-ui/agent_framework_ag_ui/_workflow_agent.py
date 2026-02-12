# Copyright (c) Microsoft. All rights reserved.

"""Workflow wrapper for AG-UI protocol compatibility."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ag_ui.core import BaseEvent
from agent_framework import Workflow

from ._workflow_run import run_workflow_stream


class AgentFrameworkWorkflowAgent:
    """Wrap Workflow instances with a run_agent interface expected by AG-UI endpoints."""

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.name = getattr(workflow, "name", "workflow")
        self.description = getattr(workflow, "description", "")

    async def run_agent(self, input_data: dict[str, Any]) -> AsyncGenerator[BaseEvent, None]:
        """Run the wrapped workflow and yield AG-UI events."""
        async for event in run_workflow_stream(input_data, self.workflow):
            yield event
