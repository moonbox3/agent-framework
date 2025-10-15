# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncIterable
from typing import cast

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
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

"""Sample: Handoff workflow with context window (rolling history).

This sample demonstrates how to use `.with_context_window(N)` to limit the conversation
history sent to each agent while relying on the auto-registered handoff tools provided
by `HandoffBuilder`. This is useful for:
- Reducing token usage and API costs
- Focusing agents on recent context only
- Managing long conversations that would exceed token limits

Instead of sending the entire conversation history to each agent, only the last N messages
are included. This creates a "rolling window" effect where older messages are dropped.

Prerequisites:
    - `az login` (Azure CLI authentication)
    - Environment variables configured for AzureOpenAIChatClient
"""


def create_agents(chat_client: AzureOpenAIChatClient) -> tuple[ChatAgent, ChatAgent, ChatAgent]:
    """Create triage and specialist agents for the demo.

    Returns:
        Tuple of (triage_agent, technical_agent, billing_agent)
    """
    triage = chat_client.create_agent(
        instructions=(
            "You are a triage agent for customer support. Assess the user's issue and route to "
            "technical_agent for technical problems or billing_agent for billing issues. Provide a concise "
            "response for the user, and when delegation is required call the matching handoff tool "
            "(`handoff_to_technical_agent` or `handoff_to_billing_agent`)."
        ),
        name="triage_agent",
    )

    technical = chat_client.create_agent(
        instructions=(
            "You are a technical support specialist. Help users troubleshoot technical issues. "
            "Ask clarifying questions and provide solutions. Be concise."
        ),
        name="technical_agent",
    )

    billing = chat_client.create_agent(
        instructions=(
            "You are a billing specialist. Help users with payment, invoice, and subscription questions. Be concise."
        ),
        name="billing_agent",
    )

    return triage, technical, billing


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
                print("\n=== Final Conversation (Full History) ===")
                # Filter out messages with no text for cleaner display
                for message in conversation:
                    if not message.text.strip():
                        continue
                    speaker = message.author_name or message.role.value
                    # Truncate long messages for display
                    text = message.text[:100] + "..." if len(message.text) > 100 else message.text
                    print(f"- {speaker}: {text}")
                print("==========================================")

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                _print_handoff_request(event.data)
            requests.append(event)

    return requests


def _print_handoff_request(request: HandoffUserInputRequest) -> None:
    """Display a user input request with conversation context.

    NOTE: This shows the FULL conversation as stored by the workflow,
    but each agent only receives the last N messages (context window).
    Filters out messages with no text (e.g., tool calls) for cleaner display.
    """
    print("\n=== User Input Requested ===")
    # Filter messages to show only those with actual text content
    messages_with_text = [msg for msg in request.conversation if msg.text.strip()]
    print(f"Context available to agents: Last {len(messages_with_text)} messages")
    for i, message in enumerate(messages_with_text, 1):
        speaker = message.author_name or message.role.value
        # Truncate long messages for display
        text = message.text[:80] + "..." if len(message.text) > 80 else message.text
        print(f"{i}. {speaker}: {text}")
    print("============================")


async def main() -> None:
    """Demonstrate handoff workflow with context window limiting conversation history.

    This sample shows how the context window affects what each agent sees:
    - The workflow maintains the FULL conversation history internally
    - Each agent receives only the last N messages (context window)
    - This reduces token usage and focuses agents on recent context

    We use a small context window (4 messages) to make the effect visible in the demo.
    In production, you might use 10-20 messages depending on your needs.
    """
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    triage, technical, billing = create_agents(chat_client)

    # Build workflow with a 4-message context window
    # This means each agent will only see the 4 most recent messages,
    # even though the full conversation may be much longer.
    #
    # Why use a context window?
    # 1. Reduce token costs (fewer tokens per API call)
    # 2. Focus agents on recent context (older messages may be irrelevant)
    # 3. Prevent exceeding token limits in very long conversations
    workflow = (
        HandoffBuilder(
            name="support_with_context_window",
            participants=[triage, technical, billing],
        )
        .starting_agent("triage_agent")
        .with_context_window(4)  # Only send last 4 messages to each agent
        .with_termination_condition(
            # Terminate after 6 user messages to demonstrate longer conversation
            lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 6
        )
        .build()
    )

    # Scripted responses simulating a longer conversation
    # This will generate more than 4 total messages, demonstrating
    # how the context window drops older messages
    scripted_responses = [
        "I'm having trouble connecting to the VPN.",  # Response 1
        "I'm on Windows 11, using the company VPN client.",  # Response 2
        "It says 'Connection timeout' after 30 seconds.",  # Response 3
        "I already tried restarting, same issue.",  # Response 4
        "Thanks for the help!",  # Response 5
    ]

    print("\n[Starting workflow with context window of 4 messages...]")
    print("[Each agent will only see the 4 most recent messages]\n")

    # Start workflow
    events = await _drain(workflow.run_stream("Hello, I need technical support."))
    pending_requests = _handle_events(events)

    response_index = 0
    conversation_length = 1  # Start with 1 (initial message)

    while pending_requests and response_index < len(scripted_responses):
        user_response = scripted_responses[response_index]
        print(f"\n[User responding (message #{conversation_length + 1}): {user_response}]")
        print(f"[Total messages so far: {conversation_length + 1}]")

        # At this point, if conversation_length > 4, agents will only see last 4 messages
        if conversation_length + 1 > 4:
            print(f"[Agents will see only messages {conversation_length + 1 - 3} through {conversation_length + 1}]")

        responses = {req.request_id: user_response for req in pending_requests}
        events = await _drain(workflow.send_responses_streaming(responses))
        pending_requests = _handle_events(events)

        response_index += 1
        conversation_length += 2  # +1 for user message, +1 for agent response (approximate)

    """
    Sample Output:

    [Starting workflow with context window of 4 messages...]
    [Each agent will only see the 4 most recent messages]


    === User Input Requested ===
    Context available to agents: Last 3 messages
    1. user: Hello, I need technical support.
    2. triage_agent: Thank you for contacting support. I will route your request to a technical speci...
    3. technical_agent: Hello! I'm here to help. Could you please describe the issue you're experiencing...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    [User responding (message #2): I'm having trouble connecting to the VPN.]
    [Total messages so far: 2]

    === User Input Requested ===
    Context available to agents: Last 6 messages
    1. user: Hello, I need technical support.
    2. triage_agent: Thank you for contacting support. I will route your request to a technical speci...
    3. technical_agent: Hello! I'm here to help. Could you please describe the issue you're experiencing...
    4. user: I'm having trouble connecting to the VPN.
    5. triage_agent: Thank you for letting us know you're having trouble connecting to the VPN. I wil...
    6. technical_agent: I'm sorry you're having trouble with the VPN connection. To help diagnose the is...
    ============================
    [status] IDLE_WITH_PENDING_REQUESTS

    ...

    1. Are you getting a specific erro...
    - user: It says 'Connection timeout' after 30 seconds.
    - triage_agent: Thank you for providing the error message. I will route your request to a technical specialist for f...
    - technical_agent: Thank you for the error message. A "Connection timeout" typically means your computer can't reach th...
    - user: I already tried restarting, same issue.
    - triage_agent: Thank you for the update. I will route your request to a technical specialist for advanced troublesh...
    - technical_agent: Thank you for letting me know. Let's try a few more steps:

    1. **Can you access the internet (e.g., ...
    - user: Thanks for the help!
    ==========================================
    [status] IDLE
    """  # noqa: E501


if __name__ == "__main__":
    asyncio.run(main())
