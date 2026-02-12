# Copyright (c) Microsoft. All rights reserved.

"""Subgraphs travel planner agent for Dojo compatibility."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Any

from ag_ui.core import (
    BaseEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)

from agent_framework_ag_ui import AgentFrameworkWorkflow

STATIC_FLIGHTS: list[dict[str, str]] = [
    {
        "airline": "KLM",
        "departure": "Amsterdam (AMS)",
        "arrival": "San Francisco (SFO)",
        "price": "$650",
        "duration": "11h 30m",
    },
    {
        "airline": "United",
        "departure": "Amsterdam (AMS)",
        "arrival": "San Francisco (SFO)",
        "price": "$720",
        "duration": "12h 15m",
    },
]

STATIC_HOTELS: list[dict[str, str]] = [
    {
        "name": "Hotel Zephyr",
        "location": "Fisherman's Wharf",
        "price_per_night": "$280/night",
        "rating": "4.2 stars",
    },
    {
        "name": "The Ritz-Carlton",
        "location": "Nob Hill",
        "price_per_night": "$550/night",
        "rating": "4.8 stars",
    },
    {
        "name": "Hotel Zoe",
        "location": "Union Square",
        "price_per_night": "$320/night",
        "rating": "4.4 stars",
    },
]

STATIC_EXPERIENCES: list[dict[str, str]] = [
    {
        "name": "Pier 39",
        "type": "activity",
        "description": "Iconic waterfront destination with shops and sea lions",
        "location": "Fisherman's Wharf",
    },
    {
        "name": "Golden Gate Bridge",
        "type": "activity",
        "description": "World-famous suspension bridge with stunning views",
        "location": "Golden Gate",
    },
    {
        "name": "Swan Oyster Depot",
        "type": "restaurant",
        "description": "Historic seafood counter serving fresh oysters",
        "location": "Polk Street",
    },
    {
        "name": "Tartine Bakery",
        "type": "restaurant",
        "description": "Artisanal bakery famous for bread and pastries",
        "location": "Mission District",
    },
]


def _initial_state() -> dict[str, Any]:
    """Create default travel planner state."""
    return {
        "itinerary": {},
        "experiences": [],
        "flights": [],
        "hotels": [],
        "planning_step": "start",
        "active_agent": "supervisor",
    }


def _emit_text(text: str) -> list[BaseEvent]:
    """Emit a complete assistant text message event sequence."""
    message_id = str(uuid.uuid4())
    return [
        TextMessageStartEvent(message_id=message_id, role="assistant"),
        TextMessageContentEvent(message_id=message_id, delta=text),
        TextMessageEndEvent(message_id=message_id),
    ]


def _extract_resume_value(resume_payload: Any) -> Any | None:
    """Extract the first resume value from AG-UI interrupt payloads."""
    if resume_payload is None:
        return None

    interrupts: list[Any] = []
    if isinstance(resume_payload, list):
        interrupts = resume_payload
    elif isinstance(resume_payload, dict):
        if isinstance(resume_payload.get("interrupts"), list):
            interrupts = resume_payload["interrupts"]
        elif isinstance(resume_payload.get("interrupt"), list):
            interrupts = resume_payload["interrupt"]
        else:
            interrupts = [resume_payload]

    for interrupt in interrupts:
        if not isinstance(interrupt, dict):
            continue
        if "value" not in interrupt:
            continue
        value = interrupt["value"]
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    return None


def _content_to_text(content: Any) -> str:
    """Convert AG-UI message content to plain text for lightweight intent parsing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return " ".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _latest_user_text(messages: Any) -> str | None:
    """Get the latest user text from AG-UI messages."""
    if not isinstance(messages, list):
        return None
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        if role != "user":
            continue
        text = _content_to_text(item.get("content", "")).strip()
        if text:
            return text.lower()
    return None


def _flight_from_user_text(user_text: str | None) -> dict[str, str] | None:
    """Infer a flight choice from plain user text."""
    if not user_text:
        return None
    if "united" in user_text:
        return deepcopy(STATIC_FLIGHTS[1])
    if "klm" in user_text:
        return deepcopy(STATIC_FLIGHTS[0])
    return None


def _hotel_from_user_text(user_text: str | None) -> dict[str, str] | None:
    """Infer a hotel choice from plain user text."""
    if not user_text:
        return None
    if "ritz" in user_text or "carlton" in user_text:
        return deepcopy(STATIC_HOTELS[1])
    if "zephyr" in user_text:
        return deepcopy(STATIC_HOTELS[0])
    if "zoe" in user_text:
        return deepcopy(STATIC_HOTELS[2])
    return None


def _as_flight(value: Any) -> dict[str, str]:
    """Normalize flight selection from resume payload."""
    if isinstance(value, dict) and value.get("airline"):
        return {
            "airline": str(value.get("airline", "")),
            "departure": str(value.get("departure", "")),
            "arrival": str(value.get("arrival", "")),
            "price": str(value.get("price", "")),
            "duration": str(value.get("duration", "")),
        }
    return deepcopy(STATIC_FLIGHTS[0])


def _as_hotel(value: Any) -> dict[str, str]:
    """Normalize hotel selection from resume payload."""
    if isinstance(value, dict) and value.get("name"):
        return {
            "name": str(value.get("name", "")),
            "location": str(value.get("location", "")),
            "price_per_night": str(value.get("price_per_night", "")),
            "rating": str(value.get("rating", "")),
        }
    return deepcopy(STATIC_HOTELS[2])


class SubgraphsTravelAgent(AgentFrameworkWorkflow):
    """Deterministic travel-planning agent with interrupt-driven subgraph flow."""

    def __init__(self) -> None:
        super().__init__(
            name="subgraphs",
            description="Travel planning supervisor with flights/hotels/experiences subgraphs.",
        )
        self._state_by_thread: dict[str, dict[str, Any]] = {}

    async def run_agent(self, input_data: dict[str, Any]) -> AsyncGenerator[BaseEvent, None]:
        """Run one turn of the subgraphs planner."""
        thread_id = str(input_data.get("thread_id") or input_data.get("threadId") or uuid.uuid4())
        run_id = str(input_data.get("run_id") or input_data.get("runId") or uuid.uuid4())

        yield RunStartedEvent(run_id=run_id, thread_id=thread_id)

        state = self._state_by_thread.get(thread_id)

        if state is None:
            state = _initial_state()
            self._state_by_thread[thread_id] = state
            yield StateSnapshotEvent(snapshot=deepcopy(state))

            for event in _emit_text(
                "Supervisor: I will coordinate our specialist agents to plan your San Francisco trip end to end."
            ):
                yield event

            state["active_agent"] = "flights"
            state["planning_step"] = "collecting_flights"
            state["flights"] = deepcopy(STATIC_FLIGHTS)
            yield StateSnapshotEvent(snapshot=deepcopy(state))

            for event in _emit_text(
                "Flights Agent: I found two flight options from Amsterdam to San Francisco. "
                "KLM is recommended for the best value and schedule."
            ):
                yield event

            yield RunFinishedEvent(
                run_id=run_id,
                thread_id=thread_id,
                interrupt=[
                    {
                        "id": "flights-choice",
                        "value": {
                            "message": (
                                "Choose the flight you want. I recommend KLM because it is cheaper and usually on time."
                            ),
                            "options": deepcopy(STATIC_FLIGHTS),
                            "recommendation": deepcopy(STATIC_FLIGHTS[0]),
                            "agent": "flights",
                        },
                    }
                ],
            )
            return

        state = self._state_by_thread.setdefault(thread_id, _initial_state())
        itinerary = state.setdefault("itinerary", {})
        resume_value = _extract_resume_value(input_data.get("resume"))
        user_text = _latest_user_text(input_data.get("messages"))

        if "flight" not in itinerary:
            if resume_value is None:
                resume_value = _flight_from_user_text(user_text)
            selected_flight = _as_flight(resume_value)
            itinerary["flight"] = selected_flight
            state["active_agent"] = "hotels"
            state["planning_step"] = "collecting_hotels"
            state["hotels"] = deepcopy(STATIC_HOTELS)
            yield StateSnapshotEvent(snapshot=deepcopy(state))

            for event in _emit_text(
                f"Flights Agent: Great choice. I will book the {selected_flight['airline']} flight. "
                "Now I am routing you to Hotels Agent for accommodation."
            ):
                yield event

            for event in _emit_text(
                "Hotels Agent: I found three accommodation options in San Francisco. "
                "Hotel Zoe is recommended for the best balance of location, quality, and price."
            ):
                yield event

            yield RunFinishedEvent(
                run_id=run_id,
                thread_id=thread_id,
                interrupt=[
                    {
                        "id": "hotels-choice",
                        "value": {
                            "message": (
                                "Choose your hotel. I recommend Hotel Zoe for the best value in a central location."
                            ),
                            "options": deepcopy(STATIC_HOTELS),
                            "recommendation": deepcopy(STATIC_HOTELS[2]),
                            "agent": "hotels",
                        },
                    }
                ],
            )
            return

        if "hotel" not in itinerary:
            if resume_value is None:
                resume_value = _hotel_from_user_text(user_text)
            selected_hotel = _as_hotel(resume_value)
            itinerary["hotel"] = selected_hotel

            state["active_agent"] = "experiences"
            state["planning_step"] = "curating_experiences"
            state["experiences"] = deepcopy(STATIC_EXPERIENCES)
            yield StateSnapshotEvent(snapshot=deepcopy(state))

            for event in _emit_text(
                f"Hotels Agent: Excellent, {selected_hotel['name']} is booked. "
                "I am routing you to Experiences Agent for activities and restaurants."
            ):
                yield event

            for event in _emit_text(
                "Experiences Agent: I planned activities and restaurants including "
                "Pier 39, Golden Gate Bridge, Swan Oyster Depot, and Tartine Bakery."
            ):
                yield event

            state["active_agent"] = "supervisor"
            state["planning_step"] = "complete"
            yield StateSnapshotEvent(snapshot=deepcopy(state))

            for event in _emit_text("Supervisor: Your travel planning is complete and your itinerary is ready."):
                yield event

            yield RunFinishedEvent(run_id=run_id, thread_id=thread_id)
            return

        yield StateSnapshotEvent(snapshot=deepcopy(state))
        for event in _emit_text("Supervisor: Your itinerary is already complete. Ask me to start a new trip anytime."):
            yield event
        yield RunFinishedEvent(run_id=run_id, thread_id=thread_id)


def subgraphs_agent() -> SubgraphsTravelAgent:
    """Create the deterministic subgraphs travel planner agent."""
    return SubgraphsTravelAgent()
