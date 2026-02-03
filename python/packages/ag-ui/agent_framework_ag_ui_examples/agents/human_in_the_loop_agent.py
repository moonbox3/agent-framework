# Copyright (c) Microsoft. All rights reserved.

"""Human-in-the-loop agent demonstrating step customization (Feature 5).

This agent also serves as a test case for the MCP tool double-call bug fix.
To test: call the agent twice with tasks that require approval (e.g., "Build a robot"
then "Build a house"). Before the fix, the second call would fail with:
"An assistant message with 'tool_calls' must be followed by tool messages..."
"""

from datetime import datetime
from enum import Enum
from typing import Any

from agent_framework import ChatAgent, ChatClientProtocol, tool
from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of a task step."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class TaskStep(BaseModel):
    """A single step in a task execution plan."""

    description: str = Field(..., description="The text of the step in imperative form (e.g., 'Dig hole', 'Open door')")
    status: StepStatus = Field(default=StepStatus.ENABLED, description="Whether the step is enabled or disabled")


# Simple tool for quick testing of the double-call bug fix
@tool(
    name="get_current_time",
    description="Get the current date and time. Requires user approval.",
    approval_mode="always_require",
)
def get_current_time() -> str:
    """Get the current date and time.

    This is a simple tool for testing the approval flow. Call it multiple times
    to verify the double-call bug fix works.

    Returns:
        Current date and time string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(
    name="generate_task_steps",
    description="Generate execution steps for a task",
    approval_mode="always_require",
)
def generate_task_steps(steps: list[TaskStep]) -> str:
    """Make up 10 steps (only a couple of words per step) that are required for a task.

    The step should be in imperative form (i.e. Dig hole, Open door, ...).
    Each step will have status='enabled' by default.

    Args:
        steps: An array of 10 step objects, each containing description and status

    Returns:
        Confirmation message
    """
    return f"Generated {len(steps)} execution steps for the task."


def human_in_the_loop_agent(chat_client: ChatClientProtocol[Any]) -> ChatAgent[Any]:
    """Create a human-in-the-loop agent using tool-based approach for predictive state.

    Args:
        chat_client: The chat client to use for the agent

    Returns:
        A configured ChatAgent instance with human-in-the-loop capabilities
    """
    return ChatAgent(
        name="human_in_the_loop_agent",
        instructions="""You are a helpful assistant that can perform any task by breaking it down into steps.
    You can also tell the user the current time.

    FOR TIME REQUESTS:
    When the user asks "what time is it?" or similar, call the `get_current_time` function.
    This is useful for testing the approval flow multiple times quickly.

    FOR TASK PLANNING:
    When asked to perform a task, you MUST call the `generate_task_steps` function with the proper
    number of steps per the request.

    Rules for steps:
    - Each step description should be in imperative form (e.g., "Dig hole", "Open door", "Prepare ingredients")
    - Each step should be brief (only a couple of words)
    - All steps must have status='enabled' initially

    Example steps for "Build a robot":
    1. "Design blueprint"
    2. "Gather components"
    3. "Assemble frame"
    4. "Install motors"
    5. "Wire electronics"
    6. "Program controller"
    7. "Test movements"
    8. "Add sensors"
    9. "Calibrate systems"
    10. "Final testing"

    IMPORTANT: Both tools require user approval before execution.
    Do NOT output any text along with the function call - just call the function.
    After the user approves and the function executes, provide a brief response with the result.
    """,
        chat_client=chat_client,
        tools=[get_current_time, generate_task_steps],
    )
