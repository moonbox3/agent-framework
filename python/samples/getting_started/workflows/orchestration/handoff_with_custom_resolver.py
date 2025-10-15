# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
from collections.abc import AsyncIterable
from typing import Any, Literal, cast

from agent_framework import (
    ChatAgent,
    ChatMessage,
    HandoffBuilder,
    HandoffUserInputRequest,
    RequestInfoEvent,
    WorkflowEvent,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
)
from agent_framework._workflows._agent_executor import AgentExecutorResponse
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field

"""Sample: Handoff workflow with custom resolver using structured outputs.

This sample demonstrates a production-ready approach to custom handoff resolution
using Pydantic models with response_format for guaranteed structured outputs.

Instead of parsing text or relying on instructions to return JSON, this approach:
- Uses response_format to enforce a Pydantic schema
- Gets guaranteed structured output from the model
- Eliminates parsing errors and validation issues
- Makes routing decisions deterministic

The pattern:
1. Define a Pydantic model for the triage agent's response schema
2. Pass the model as response_format when creating the agent
3. Custom resolver reads the parsed .value field from the response
4. Extract routing decision from the structured data

Prerequisites:
    - `az login` (Azure CLI authentication)
    - Environment variables configured for OpenAIChatClient
    - Model must support structured outputs (e.g., gpt-4o, gpt-4o-mini)
"""


class TriageResponse(BaseModel):
    """Structured response from the triage agent.

    This Pydantic model defines the exact schema the triage agent must follow.
    The model enforces this structure via response_format.
    """

    action: Literal["route", "handle"] = Field(description="Whether to route to a specialist or handle directly")
    target: str | None = Field(
        default=None,
        description="Target agent name if action is 'route' (e.g., 'refund_agent', 'cancellation_agent')",
    )
    response: str = Field(description="Natural language response to the user explaining what will happen")


def structured_output_resolver(response: AgentExecutorResponse) -> str | None:
    """Parse handoff target from structured Pydantic model response.

    This resolver expects the triage agent to use response_format with TriageResponse.
    The agent's .value field will contain the parsed Pydantic model.

    Args:
        response: The agent's response after processing user input

    Returns:
        The target agent ID to hand off to, or None if triage handles it

    Example:
        Agent returns: TriageResponse(action="route", target="refund_agent", response="...")
        Resolver extracts: "refund_agent"
    """
    agent_response = response.agent_run_response

    # Check if agent returned structured output via response_format
    if agent_response.value is None:
        print("[Resolver] No structured value in response")
        return None

    # The value should be our TriageResponse Pydantic model
    if not isinstance(agent_response.value, TriageResponse):
        print(f"[Resolver] Unexpected value type: {type(agent_response.value).__name__}")
        return None

    triage_response = agent_response.value

    if triage_response.action == "route":
        target = triage_response.target
        if target:
            print(f"[Resolver] Routing to '{target}' (from structured output)")
            return target.strip()
        print("[Resolver] Action is 'route' but no target specified")
        return None

    if triage_response.action == "handle":
        print("[Resolver] Triage handling directly (action='handle')")
        return None

    print(f"[Resolver] Unknown action: {triage_response.action}")
    return None


def create_agents(chat_client: OpenAIChatClient) -> tuple[ChatAgent, ChatAgent, ChatAgent]:
    """Create triage and specialist agents with structured output configuration.

    The triage agent uses response_format=TriageResponse to guarantee structured output.
    This eliminates the need for JSON parsing or verbose instructions.

    Returns:
        Tuple of (triage_agent, refund_agent, cancellation_agent)
    """
    # Triage agent with response_format for guaranteed structured output
    triage = chat_client.create_agent(
        instructions=(
            "You are a customer service triage agent. Analyze user requests and determine routing.\n\n"
            "Available specialists:\n"
            "- 'refund_agent' for refund requests\n"
            "- 'cancellation_agent' for subscription cancellations\n\n"
            "If you can answer directly, set action='handle'.\n"
            "If a specialist is needed, set action='route' and specify the target agent.\n"
            "Always provide a helpful response explaining what will happen."
        ),
        name="triage_agent",
        response_format=TriageResponse,  # Enforce structured output schema
    )

    refund = chat_client.create_agent(
        instructions="You handle refund requests. Ask for order number and process refunds.",
        name="refund_agent",
    )

    cancellation = chat_client.create_agent(
        instructions="You handle subscription cancellations. Confirm details and process.",
        name="cancellation_agent",
    )

    return triage, refund, cancellation


async def _drain(stream: AsyncIterable[WorkflowEvent]) -> list[WorkflowEvent]:
    """Collect all events from an async stream into a list."""
    return [event async for event in stream]


def _render_message_text(message: ChatMessage, *, truncate: int | None = None) -> str:
    """Render message text, unwrapping structured outputs when available."""
    text = message.text or ""
    if text:
        payload: Any
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            payload_dict = cast(dict[str, Any], payload)
            response_value = payload_dict.get("response")
            if isinstance(response_value, str):
                text = response_value.strip()
    if truncate is not None and len(text) > truncate:
        return text[:truncate] + "..."
    return text


def _handle_events(events: list[WorkflowEvent]) -> list[RequestInfoEvent]:
    """Process workflow events and extract pending user input requests."""
    requests: list[RequestInfoEvent] = []

    for event in events:
        if isinstance(event, WorkflowStatusEvent) and event.state in {
            WorkflowRunState.IDLE,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        }:
            print(f"[status] {event.state.name}")

        elif isinstance(event, WorkflowOutputEvent):
            conversation = cast(list[ChatMessage], event.data)
            if isinstance(conversation, list):
                print("\n=== Final Conversation ===")
                for message in conversation:
                    speaker = message.author_name or message.role.value
                    text = _render_message_text(message, truncate=100)
                    print(f"- {speaker}: {text}")
                print("==========================")

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                _print_handoff_request(event.data)
            requests.append(event)

    return requests


def _print_handoff_request(request: HandoffUserInputRequest) -> None:
    """Display a user input request with conversation context."""
    print("\n=== User Input Requested ===")
    for message in request.conversation:
        speaker = message.author_name or message.role.value
        text = _render_message_text(message, truncate=80)
        print(f"- {speaker}: {text}")
    print("============================")


async def main() -> None:
    """Demonstrate handoff workflow with structured output resolver.

    This sample shows the production-ready pattern for handoff detection:
    1. Define a Pydantic model for the triage agent's response
    2. Use response_format to enforce the schema
    3. Custom resolver reads the structured .value field
    4. No JSON parsing, no text instructions, no ambiguity

    Key Benefits:
    - Guaranteed structured output from the model
    - Type-safe routing decisions
    - No parsing errors or validation issues
    - Clean, maintainable code
    """
    chat_client = OpenAIChatClient()
    triage, refund, cancellation = create_agents(chat_client)

    # Build workflow with structured output resolver
    # The resolver reads the TriageResponse Pydantic model from agent.value
    workflow = (
        HandoffBuilder(
            name="support_with_structured_outputs",
            participants=[triage, refund, cancellation],
        )
        .starting_agent("triage_agent")
        .handoff_resolver(structured_output_resolver)  # Use structured output resolver
        .with_termination_condition(lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 4)
        .build()
    )

    # Scripted responses demonstrating different routing scenarios
    scripted_responses = [
        "Yes, please proceed with the cancellation.",
        "Thank you for your help.",
    ]

    print("\n[Starting workflow with structured output resolver...]")
    print("[Triage agent uses response_format=TriageResponse]")
    print("[Resolver reads parsed Pydantic model from .value]\n")

    # Start workflow
    events = await _drain(workflow.run_stream("I want to cancel my subscription."))
    pending_requests = _handle_events(events)

    response_index = 0

    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print(f"\n[User responding: {user_response}]")

        responses = {req.request_id: user_response for req in pending_requests}
        events = await _drain(workflow.send_responses_streaming(responses))
        pending_requests = _handle_events(events)
        response_index += 1

    """
    Sample Output:

    [Starting workflow with structured output resolver...]
    [Triage agent uses response_format=TriageResponse]
    [Resolver reads parsed Pydantic model from .value]

    [Resolver] Routing to 'cancellation_agent' (from structured output)

    === User Input Requested ===
    - user: I want to cancel my subscription.
    - triage_agent: I understand you'd like to cancel your subscription. I'll connect you with our c...
    - cancellation_agent: I'm here to help you with your cancellation request. To proceed, could you pleas...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User responding: Yes, please proceed with the cancellation.]
    [Resolver] Routing to 'cancellation_agent' (from structured output)

    === User Input Requested ===
    - user: I want to cancel my subscription.
    - triage_agent: I understand you'd like to cancel your subscription. I'll connect you with our c...
    - cancellation_agent: I'm here to help you with your cancellation request. To proceed, could you pleas...
    - user: Yes, please proceed with the cancellation.
    - triage_agent: Thank you for confirming. I will now connect you with our cancellation specialis...
    - cancellation_agent: Thank you for your confirmation. For security and verification purposes, could y...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User responding: Thank you for your help.]
    [Resolver] Triage handling directly (action='handle')

    === User Input Requested ===
    - user: I want to cancel my subscription.
    - triage_agent: I understand you'd like to cancel your subscription. I'll connect you with our c...
    - cancellation_agent: I'm here to help you with your cancellation request. To proceed, could you pleas...
    - user: Yes, please proceed with the cancellation.
    - triage_agent: Thank you for confirming. I will now connect you with our cancellation specialis...
    - cancellation_agent: Thank you for your confirmation. For security and verification purposes, could y...
    - user: Thank you for your help.
    - triage_agent: You're very welcome! If you have any more questions or need further assistance, ...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS
    """


if __name__ == "__main__":
    asyncio.run(main())
