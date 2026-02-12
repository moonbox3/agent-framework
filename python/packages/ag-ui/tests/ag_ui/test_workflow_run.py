# Copyright (c) Microsoft. All rights reserved.

"""Tests for native workflow AG-UI runner."""

from typing import Any

from agent_framework import (
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    executor,
)
from typing_extensions import Never

from agent_framework_ag_ui._workflow_run import run_workflow_stream


class ProgressEvent(WorkflowEvent):
    """Custom workflow event used to validate CUSTOM mapping."""

    def __init__(self, progress: int) -> None:
        super().__init__("custom_progress", data={"progress": progress})


async def test_workflow_run_maps_custom_and_text_events():
    """Custom workflow events and yielded text are mapped to AG-UI events."""

    @executor(id="start")
    async def start(message: Any, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.add_event(ProgressEvent(10))
        await ctx.yield_output("Hello workflow")

    workflow = WorkflowBuilder(start_executor=start).build()
    input_data = {"messages": [{"role": "user", "content": "go"}]}

    events = [event async for event in run_workflow_stream(input_data, workflow)]

    event_types = [event.type for event in events]
    assert "RUN_STARTED" in event_types
    assert "CUSTOM" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert "STEP_STARTED" in event_types
    assert "STEP_FINISHED" in event_types
    assert "RUN_FINISHED" in event_types

    custom_events = [event for event in events if event.type == "CUSTOM" and event.name == "custom_progress"]
    assert len(custom_events) == 1
    assert custom_events[0].value == {"progress": 10}


async def test_workflow_run_request_info_emits_interrupt_and_resume_works():
    """request_info should emit interrupt metadata and resume should continue run."""

    @executor(id="requester")
    async def requester(message: Any, ctx: WorkflowContext) -> None:
        await ctx.request_info("Need approval", str)

    workflow = WorkflowBuilder(start_executor=requester).build()

    first_run_events = [event async for event in run_workflow_stream({"messages": [{"role": "user", "content": "go"}]}, workflow)]

    run_finished_events = [event for event in first_run_events if event.type == "RUN_FINISHED"]
    assert len(run_finished_events) == 1
    interrupt_payload = run_finished_events[0].model_dump().get("interrupt")
    assert isinstance(interrupt_payload, list)
    assert len(interrupt_payload) == 1

    request_id = str(interrupt_payload[0]["id"])
    assert request_id

    resumed_events = [
        event
        async for event in run_workflow_stream(
            {"messages": [], "resume": {"interrupts": [{"id": request_id, "value": "approved"}]}},
            workflow,
        )
    ]

    resumed_types = [event.type for event in resumed_events]
    assert "RUN_STARTED" in resumed_types
    assert "RUN_FINISHED" in resumed_types
    assert "RUN_ERROR" not in resumed_types


async def test_workflow_run_non_chat_output_maps_to_custom_output_event():
    """Non-chat workflow outputs are emitted as CUSTOM workflow_output events."""

    @executor(id="structured")
    async def structured(message: Any, ctx: WorkflowContext[Never, dict[str, int]]) -> None:
        await ctx.yield_output({"count": 3})

    workflow = WorkflowBuilder(start_executor=structured).build()
    events = [event async for event in run_workflow_stream({"messages": [{"role": "user", "content": "go"}]}, workflow)]

    output_custom = [event for event in events if event.type == "CUSTOM" and event.name == "workflow_output"]
    assert len(output_custom) == 1
    assert output_custom[0].value == {"count": 3}
