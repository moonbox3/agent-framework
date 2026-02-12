# Copyright (c) Microsoft. All rights reserved.

"""Tests for native workflow AG-UI runner."""

from typing import Any

from ag_ui.core import EventType, StateSnapshotEvent
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


async def test_workflow_run_request_info_interrupt_uses_raw_dict_value():
    """Dict request payloads should be surfaced directly in RUN_FINISHED.interrupt.value."""

    @executor(id="requester")
    async def requester(message: Any, ctx: WorkflowContext) -> None:
        await ctx.request_info(
            {
                "message": "Choose a flight",
                "options": [{"airline": "KLM"}],
                "recommendation": {"airline": "KLM"},
                "agent": "flights",
            },
            dict,
            request_id="flights-choice",
        )

    workflow = WorkflowBuilder(start_executor=requester).build()
    events = [event async for event in run_workflow_stream({"messages": [{"role": "user", "content": "go"}]}, workflow)]

    run_finished = [event for event in events if event.type == "RUN_FINISHED"][0].model_dump()
    interrupt_payload = run_finished.get("interrupt")
    assert isinstance(interrupt_payload, list)
    assert interrupt_payload[0]["id"] == "flights-choice"
    assert interrupt_payload[0]["value"]["agent"] == "flights"
    assert interrupt_payload[0]["value"]["message"] == "Choose a flight"


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


async def test_workflow_run_passthroughs_ag_ui_base_events():
    """Workflow outputs that are AG-UI BaseEvent instances should be emitted directly."""

    @executor(id="stateful")
    async def stateful(message: Any, ctx: WorkflowContext[Never, StateSnapshotEvent]) -> None:
        await ctx.yield_output(StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot={"active_agent": "flights"}))

    workflow = WorkflowBuilder(start_executor=stateful).build()
    events = [event async for event in run_workflow_stream({"messages": [{"role": "user", "content": "go"}]}, workflow)]

    snapshots = [event for event in events if event.type == "STATE_SNAPSHOT"]
    assert len(snapshots) == 1
    assert snapshots[0].snapshot["active_agent"] == "flights"


async def test_workflow_run_plain_text_follow_up_does_not_infer_interrupt_response():
    """User follow-up text should not be coerced into request_info responses for workflows."""

    @executor(id="requester")
    async def requester(message: Any, ctx: WorkflowContext) -> None:
        del message
        await ctx.request_info(
            {
                "message": "Choose a flight",
                "options": [{"airline": "KLM"}, {"airline": "United"}],
                "agent": "flights",
            },
            dict,
            request_id="flights-choice",
        )

    workflow = WorkflowBuilder(start_executor=requester).build()
    _ = [event async for event in run_workflow_stream({"messages": [{"role": "user", "content": "go"}]}, workflow)]

    follow_up_events = [
        event
        async for event in run_workflow_stream(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "flights-choice",
                                "type": "function",
                                "function": {"name": "request_info", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "user", "content": "I prefer KLM please"},
                ]
            },
            workflow,
        )
    ]

    follow_up_types = [event.type for event in follow_up_events]
    assert "RUN_ERROR" not in follow_up_types
    assert "TOOL_CALL_START" not in follow_up_types

    run_finished = [event for event in follow_up_events if event.type == "RUN_FINISHED"][0].model_dump()
    interrupt_payload = run_finished.get("interrupt")
    assert isinstance(interrupt_payload, list)
    assert interrupt_payload[0]["id"] == "flights-choice"
    assert interrupt_payload[0]["value"]["agent"] == "flights"


async def test_workflow_run_empty_turn_with_pending_request_preserves_interrupts():
    """An empty turn should still return pending workflow interrupts without errors."""

    @executor(id="requester")
    async def requester(message: Any, ctx: WorkflowContext) -> None:
        del message
        await ctx.request_info({"prompt": "choose"}, dict, request_id="pick-one")

    workflow = WorkflowBuilder(start_executor=requester).build()
    _ = [event async for event in run_workflow_stream({"messages": [{"role": "user", "content": "go"}]}, workflow)]

    events = [event async for event in run_workflow_stream({"messages": []}, workflow)]
    types = [event.type for event in events]
    assert types[0] == "RUN_STARTED"
    assert "RUN_FINISHED" in types
    assert "RUN_ERROR" not in types

    finished = [event for event in events if event.type == "RUN_FINISHED"][0].model_dump()
    interrupts = finished.get("interrupt")
    assert isinstance(interrupts, list)
    assert interrupts[0]["id"] == "pick-one"
