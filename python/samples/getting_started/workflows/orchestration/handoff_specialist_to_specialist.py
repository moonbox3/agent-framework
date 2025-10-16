# Copyright (c) Microsoft. All rights reserved.

"""Sample: Multi-tier handoff workflow with specialist-to-specialist routing.

This sample demonstrates advanced handoff routing where specialist agents can hand off
to other specialists, enabling complex multi-tier workflows. Unlike the simple handoff
pattern (see handoff_simple.py), specialists here can delegate to other specialists
without returning control to the user until the specialist chain completes.

Routing Pattern:
    User → Triage → Specialist A → Specialist B → Back to User

This pattern is useful for complex support scenarios where different specialists need
to collaborate or escalate to each other before returning to the user. For example:
    - Replacement agent needs shipping info → hands off to delivery agent
    - Technical support needs billing info → hands off to billing agent
    - Level 1 support escalates to Level 2 → hands off to escalation agent

Configuration uses `.with_handoffs()` to explicitly define the routing graph.

Prerequisites:
    - `az login` (Azure CLI authentication)
    - Environment variables configured for AzureOpenAIChatClient
"""

import asyncio
from collections.abc import AsyncIterable
from typing import cast

from agent_framework import (
    ChatMessage,
    HandoffBuilder,
    HandoffUserInputRequest,
    RequestInfoEvent,
    WorkflowEvent,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


def create_agents(chat_client: AzureOpenAIChatClient):
    """Create triage and specialist agents with multi-tier handoff capabilities.

    Returns:
        Tuple of (triage_agent, replacement_agent, delivery_agent, billing_agent)
    """
    triage = chat_client.create_agent(
        instructions=(
            "You are a customer support triage agent. Assess the user's issue and route appropriately:\n"
            "- For product replacement issues: call handoff_to_replacement_agent\n"
            "- For delivery/shipping inquiries: call handoff_to_delivery_agent\n"
            "- For billing/payment issues: call handoff_to_billing_agent\n"
            "Be concise and friendly."
        ),
        name="triage_agent",
    )

    replacement = chat_client.create_agent(
        instructions=(
            "You handle product replacement requests. Ask for order number and reason for replacement.\n"
            "If the user also needs shipping/delivery information, call handoff_to_delivery_agent to "
            "get tracking details. Otherwise, process the replacement and confirm with the user.\n"
            "Be concise and helpful."
        ),
        name="replacement_agent",
    )

    delivery = chat_client.create_agent(
        instructions=(
            "You handle shipping and delivery inquiries. Provide tracking information, estimated "
            "delivery dates, and address any delivery concerns.\n"
            "If billing issues come up, call handoff_to_billing_agent.\n"
            "Be concise and clear."
        ),
        name="delivery_agent",
    )

    billing = chat_client.create_agent(
        instructions=(
            "You handle billing and payment questions. Help with refunds, payment methods, "
            "and invoice inquiries. Be concise."
        ),
        name="billing_agent",
    )

    return triage, replacement, delivery, billing


async def _drain(stream: AsyncIterable[WorkflowEvent]) -> list[WorkflowEvent]:
    """Collect all events from an async stream into a list."""
    return [event async for event in stream]


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
                    # Filter out messages with no text (tool calls)
                    if not message.text.strip():
                        continue
                    speaker = message.author_name or message.role.value
                    print(f"- {speaker}: {message.text}")
                print("==========================")

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                _print_handoff_request(event.data)
            requests.append(event)

    return requests


def _print_handoff_request(request: HandoffUserInputRequest) -> None:
    """Display a user input request with conversation context."""
    print("\n=== User Input Requested ===")
    # Filter out messages with no text for cleaner display
    messages_with_text = [msg for msg in request.conversation if msg.text.strip()]
    print(f"Last {len(messages_with_text)} messages in conversation:")
    for message in messages_with_text[-5:]:  # Show last 5 for brevity
        speaker = message.author_name or message.role.value
        text = message.text[:100] + "..." if len(message.text) > 100 else message.text
        print(f"  {speaker}: {text}")
    print("============================")


async def main() -> None:
    """Demonstrate specialist-to-specialist handoffs in a multi-tier support scenario.

    This sample shows:
    1. Triage agent routes to replacement specialist
    2. Replacement specialist hands off to delivery specialist
    3. Delivery specialist can hand off to billing if needed
    4. All transitions are seamless without returning to user until complete

    The workflow configuration explicitly defines which agents can hand off to which others:
    - triage_agent → replacement_agent, delivery_agent, billing_agent
    - replacement_agent → delivery_agent, billing_agent
    - delivery_agent → billing_agent
    """
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    triage, replacement, delivery, billing = create_agents(chat_client)

    # Configure multi-tier handoffs explicitly
    # This allows specialists to hand off to other specialists
    workflow = (
        HandoffBuilder(
            name="multi_tier_support",
            participants=[triage, replacement, delivery, billing],
        )
        .starting_agent("triage_agent")
        .with_handoffs({
            # Triage can route to any specialist
            "triage_agent": ["replacement_agent", "delivery_agent", "billing_agent"],
            # Replacement can delegate to delivery or billing
            "replacement_agent": ["delivery_agent", "billing_agent"],
            # Delivery can escalate to billing if needed
            "delivery_agent": ["billing_agent"],
        })
        .with_termination_condition(lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 4)
        .build()
    )

    # Scripted user responses simulating a multi-tier handoff scenario
    scripted_responses = [
        "I need help with order 12345. I want a replacement and need to know when it will arrive.",
        "The item arrived damaged. I'd like a replacement shipped to the same address.",
        "Great! Can you confirm the shipping cost won't be charged again?",
    ]

    print("\n" + "=" * 80)
    print("SPECIALIST-TO-SPECIALIST HANDOFF DEMONSTRATION")
    print("=" * 80)
    print("\nScenario: Customer needs replacement + shipping info + billing confirmation")
    print("Expected flow: User → Triage → Replacement → Delivery → Billing → User")
    print("=" * 80 + "\n")

    # Start workflow with initial message
    print("[User]: I need help with order 12345. I want a replacement and need to know when it will arrive.\n")
    events = await _drain(
        workflow.run_stream("I need help with order 12345. I want a replacement and need to know when it will arrive.")
    )
    pending_requests = _handle_events(events)

    # Process scripted responses
    response_index = 0
    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print(f"\n[User]: {user_response}\n")

        responses = {req.request_id: user_response for req in pending_requests}
        events = await _drain(workflow.send_responses_streaming(responses))
        pending_requests = _handle_events(events)

        response_index += 1

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey observations:")
    print("1. Triage correctly routed to Replacement agent")
    print("2. Replacement agent delegated to Delivery for shipping info")
    print("3. Delivery agent could escalate to Billing if needed")
    print("4. User only intervened when additional input was required")
    print("5. Agents collaborated seamlessly across tiers")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
