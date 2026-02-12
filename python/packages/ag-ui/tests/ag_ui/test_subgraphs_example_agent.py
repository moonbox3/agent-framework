# Copyright (c) Microsoft. All rights reserved.

"""Tests for the subgraphs example agent used by Dojo."""

from __future__ import annotations

import json
from typing import Any

from agent_framework_ag_ui_examples.agents.subgraphs_agent import subgraphs_agent


async def _run(agent: Any, payload: dict[str, Any]) -> list[Any]:
    return [event async for event in agent.run_agent(payload)]


async def test_subgraphs_example_initial_run_emits_flight_interrupt() -> None:
    """Initial run should publish flight options and pause with an interrupt."""
    agent = subgraphs_agent()

    events = await _run(
        agent,
        {
            "thread_id": "thread-subgraphs-initial",
            "run_id": "run-initial",
            "messages": [{"role": "user", "content": "Help me plan a trip to San Francisco"}],
        },
    )

    event_types = [event.type for event in events]
    assert event_types[0] == "RUN_STARTED"
    assert "STATE_SNAPSHOT" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert "RUN_FINISHED" in event_types

    finished = [event for event in events if event.type == "RUN_FINISHED"][0]
    interrupt_payload = finished.model_dump().get("interrupt")
    assert isinstance(interrupt_payload, list)
    assert interrupt_payload
    assert interrupt_payload[0]["value"]["agent"] == "flights"
    assert len(interrupt_payload[0]["value"]["options"]) == 2
    assert interrupt_payload[0]["value"]["options"][0]["airline"] == "KLM"


async def test_subgraphs_example_resume_flow_reaches_completion() -> None:
    """Flight + hotel resume payloads should complete the itinerary state."""
    agent = subgraphs_agent()
    thread_id = "thread-subgraphs-complete"

    first_events = await _run(
        agent,
        {
            "thread_id": thread_id,
            "run_id": "run-1",
            "messages": [{"role": "user", "content": "I want to visit San Francisco from Amsterdam"}],
        },
    )
    first_interrupt = [event for event in first_events if event.type == "RUN_FINISHED"][0].model_dump()["interrupt"][0]

    second_events = await _run(
        agent,
        {
            "thread_id": thread_id,
            "run_id": "run-2",
            "resume": {
                "interrupts": [
                    {
                        "id": first_interrupt["id"],
                        "value": json.dumps(
                            {
                                "airline": "United",
                                "departure": "Amsterdam (AMS)",
                                "arrival": "San Francisco (SFO)",
                                "price": "$720",
                                "duration": "12h 15m",
                            }
                        ),
                    }
                ]
            },
        },
    )
    second_finished = [event for event in second_events if event.type == "RUN_FINISHED"][0].model_dump()
    second_interrupt = second_finished.get("interrupt")
    assert isinstance(second_interrupt, list)
    assert second_interrupt[0]["value"]["agent"] == "hotels"

    third_events = await _run(
        agent,
        {
            "thread_id": thread_id,
            "run_id": "run-3",
            "resume": {
                "interrupts": [
                    {
                        "id": second_interrupt[0]["id"],
                        "value": json.dumps(
                            {
                                "name": "The Ritz-Carlton",
                                "location": "Nob Hill",
                                "price_per_night": "$550/night",
                                "rating": "4.8 stars",
                            }
                        ),
                    }
                ]
            },
        },
    )

    third_finished = [event for event in third_events if event.type == "RUN_FINISHED"][0].model_dump()
    assert "interrupt" not in third_finished

    snapshots = [event.snapshot for event in third_events if event.type == "STATE_SNAPSHOT"]
    assert snapshots
    final_snapshot = snapshots[-1]
    assert final_snapshot["planning_step"] == "complete"
    assert final_snapshot["active_agent"] == "supervisor"
    assert final_snapshot["itinerary"]["flight"]["airline"] == "United"
    assert final_snapshot["itinerary"]["hotel"]["name"] == "The Ritz-Carlton"
    assert len(final_snapshot["experiences"]) == 4


async def test_subgraphs_example_progresses_with_plain_user_text() -> None:
    """Agent should continue on same thread even when selection arrives as plain user text."""
    agent = subgraphs_agent()
    thread_id = "thread-subgraphs-text"

    first_events = await _run(
        agent,
        {
            "thread_id": thread_id,
            "run_id": "run-a",
            "messages": [{"role": "user", "content": "Plan a trip for me"}],
        },
    )
    first_finished = [event for event in first_events if event.type == "RUN_FINISHED"][0].model_dump()
    assert isinstance(first_finished.get("interrupt"), list)
    assert first_finished["interrupt"][0]["value"]["agent"] == "flights"

    second_events = await _run(
        agent,
        {
            "thread_id": thread_id,
            "run_id": "run-b",
            "messages": [{"role": "user", "content": "Let's do the United flight"}],
        },
    )
    second_finished = [event for event in second_events if event.type == "RUN_FINISHED"][0].model_dump()
    assert isinstance(second_finished.get("interrupt"), list)
    assert second_finished["interrupt"][0]["value"]["agent"] == "hotels"

    second_snapshots = [event.snapshot for event in second_events if event.type == "STATE_SNAPSHOT"]
    assert second_snapshots[-1]["itinerary"]["flight"]["airline"] == "United"

    third_events = await _run(
        agent,
        {
            "thread_id": thread_id,
            "run_id": "run-c",
            "messages": [{"role": "user", "content": "Book the Ritz-Carlton please"}],
        },
    )
    third_finished = [event for event in third_events if event.type == "RUN_FINISHED"][0].model_dump()
    assert "interrupt" not in third_finished

    third_snapshots = [event.snapshot for event in third_events if event.type == "STATE_SNAPSHOT"]
    assert third_snapshots[-1]["itinerary"]["hotel"]["name"] == "The Ritz-Carlton"
    assert third_snapshots[-1]["planning_step"] == "complete"
