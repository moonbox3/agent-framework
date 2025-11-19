# Copyright (c) Microsoft. All rights reserved.

"""
Sample: Handoff Workflow with Checkpoint-Based Pause/Resume

Demonstrates the two-step pattern for resuming a handoff workflow from a checkpoint
and providing user responses to HandoffUserInputRequest.

Scenario:
1. User starts a conversation with the workflow
2. Workflow requests user input (HandoffUserInputRequest is emitted)
3. Workflow pauses and saves a checkpoint
4. Process can exit/restart
5. On resume: Load checkpoint + provide user response
6. Workflow continues from where it left off

Pattern:
- Step 1: workflow.run_stream(checkpoint_id=...) to restore checkpoint
- Step 2: workflow.send_responses_streaming(responses) to provide user input
- This two-step approach is required because send_responses_streaming doesn't accept checkpoint_id

Prerequisites:
- Azure CLI authentication (az login)
- Environment variables configured for AzureOpenAIChatClient
"""

import asyncio
import logging
from pathlib import Path
from typing import cast

from agent_framework import (
    ChatAgent,
    ChatMessage,
    FileCheckpointStorage,
    HandoffBuilder,
    HandoffUserInputRequest,
    RequestInfoEvent,
    Workflow,
    WorkflowOutputEvent,
    WorkflowStatusEvent,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

CHECKPOINT_DIR = Path(__file__).parent / "tmp" / "handoff_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def create_agents(client: AzureOpenAIChatClient) -> tuple[ChatAgent, ChatAgent, ChatAgent]:
    """Create a simple handoff scenario: triage, refund, and order specialists."""

    triage = client.create_agent(
        name="triage_agent",
        instructions=(
            "You are a customer service triage agent. Listen to customer issues and determine "
            "if they need refund help or order tracking. Use handoff_to_refund_agent or "
            "handoff_to_order_agent to transfer them."
        ),
    )

    refund = client.create_agent(
        name="refund_agent",
        instructions=(
            "You are a refund specialist. Help customers with refund requests. "
            "Be empathetic and ask for order numbers if not provided."
        ),
    )

    order = client.create_agent(
        name="order_agent",
        instructions=(
            "You are an order tracking specialist. Help customers track their orders. "
            "Ask for order numbers and provide shipping updates."
        ),
    )

    return triage, refund, order


def create_workflow(checkpoint_storage: FileCheckpointStorage) -> tuple[Workflow, ChatAgent, ChatAgent, ChatAgent]:
    """Build the handoff workflow with checkpointing enabled."""

    client = AzureOpenAIChatClient(credential=AzureCliCredential())
    triage, refund, order = create_agents(client)

    workflow = (
        HandoffBuilder(
            name="checkpoint_handoff_demo",
            participants=[triage, refund, order],
        )
        .set_coordinator("triage_agent")
        .with_checkpointing(checkpoint_storage)
        .with_termination_condition(
            # Terminate after 5 user messages for this demo
            lambda conv: sum(1 for msg in conv if msg.role.value == "user") >= 5
        )
        .build()
    )

    return workflow, triage, refund, order


async def run_until_user_input_needed(
    workflow: Workflow,
    initial_message: str | None = None,
    checkpoint_id: str | None = None,
) -> tuple[list[RequestInfoEvent], str | None]:
    """
    Run the workflow until it needs user input or completes.

    Returns:
        Tuple of (pending_requests, checkpoint_id_to_use_for_resume)
    """
    pending_requests: list[RequestInfoEvent] = []
    latest_checkpoint_id: str | None = checkpoint_id

    if initial_message:
        print(f"\nStarting workflow with: {initial_message}\n")
        event_stream = workflow.run_stream(message=initial_message)  # type: ignore[attr-defined]
    elif checkpoint_id:
        print(f"\nResuming workflow from checkpoint: {checkpoint_id}\n")
        event_stream = workflow.run_stream(checkpoint_id=checkpoint_id)  # type: ignore[attr-defined]
    else:
        raise ValueError("Must provide either initial_message or checkpoint_id")

    async for event in event_stream:
        if isinstance(event, WorkflowStatusEvent):
            print(f"[Status] {event.state}")

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                print(f"\n{'=' * 60}")
                print("WORKFLOW PAUSED - User input needed")
                print(f"Request ID: {event.request_id}")
                print(f"Awaiting agent: {event.data.awaiting_agent_id}")
                print(f"Prompt: {event.data.prompt}")

                # Print conversation history
                print("\nConversation so far:")
                for msg in event.data.conversation[-3:]:  # Show last 3 messages
                    author = msg.author_name or msg.role.value
                    print(f"  {author}: {msg.text[:80]}...")

                print(f"{'=' * 60}\n")
                pending_requests.append(event)

        elif isinstance(event, WorkflowOutputEvent):
            print("\n[Workflow Completed]")
            if event.data:
                print(f"Final conversation length: {len(event.data)} messages")
            return [], None

    # Workflow paused with pending requests
    # The latest checkpoint was created at the end of the last superstep
    # We'll use the checkpoint storage to find it
    return pending_requests, latest_checkpoint_id


async def resume_with_response(
    workflow: Workflow,
    checkpoint_storage: FileCheckpointStorage,
    user_response: str,
) -> tuple[list[RequestInfoEvent], str | None]:
    """
    Two-step resume pattern (answers customer's question):

    Step 1: Restore checkpoint to load pending requests into workflow state
    Step 2: Send user responses using send_responses_streaming

    This is the current pattern required because send_responses_streaming
    doesn't accept a checkpoint_id parameter.
    """
    print(f"\n{'=' * 60}")
    print("RESUMING WORKFLOW WITH USER RESPONSE")
    print(f"User says: {user_response}")
    print(f"{'=' * 60}\n")

    # Get the latest checkpoint
    checkpoints = await checkpoint_storage.list_checkpoints()
    if not checkpoints:
        raise RuntimeError("No checkpoints found to resume from")

    # Sort by timestamp to get latest
    checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)
    latest_checkpoint = checkpoints[0]

    print(f"Step 1: Restoring checkpoint {latest_checkpoint.checkpoint_id}")

    # Step 1: Restore the checkpoint to load pending requests into memory
    # The checkpoint restoration re-emits pending RequestInfoEvents
    pending_request_ids: list[str] = []
    async for event in workflow.run_stream(checkpoint_id=latest_checkpoint.checkpoint_id):  # type: ignore[attr-defined]
        if isinstance(event, RequestInfoEvent) and isinstance(event.data, HandoffUserInputRequest):
            pending_request_ids.append(event.request_id)
            print(f"Found pending request: {event.request_id}")

    if not pending_request_ids:
        raise RuntimeError("No pending requests found after checkpoint restoration")

    print(f"Step 2: Sending user response for {len(pending_request_ids)} request(s)")

    # Step 2: Send the user's response
    responses = {req_id: user_response for req_id in pending_request_ids}

    new_pending_requests: list[RequestInfoEvent] = []

    async for event in workflow.send_responses_streaming(responses):
        if isinstance(event, WorkflowStatusEvent):
            print(f"[Status] {event.state}")

        elif isinstance(event, WorkflowOutputEvent):
            # Workflow completed or paused - show the conversation
            print("\n[Workflow Output Event - Conversation Update]")
            if event.data and isinstance(event.data, list):
                # Cast event.data to list[ChatMessage] for type checking
                conversation = cast(list[ChatMessage], event.data)  # type: ignore
                for msg in conversation[-3:]:  # Show last 3 messages
                    if isinstance(msg, ChatMessage):
                        author = msg.author_name or msg.role.value
                        text = msg.text[:100] + "..." if len(msg.text) > 100 else msg.text
                        print(f"  {author}: {text}")

        elif isinstance(event, RequestInfoEvent):
            if isinstance(event.data, HandoffUserInputRequest):
                print(f"\n{'=' * 60}")
                print("WORKFLOW PAUSED AGAIN - User input needed")
                print(f"Request ID: {event.request_id}")
                print(f"Awaiting agent: {event.data.awaiting_agent_id}")

                # Show recent agent responses (last 3 messages excluding initial user message)
                print("\nRecent conversation:")
                recent_msgs = [m for m in event.data.conversation[-4:] if m.role.value != "user"][-2:]
                for msg in recent_msgs:
                    author = msg.author_name or msg.role.value
                    print(f"\n  [{author}]:")
                    print(f"  {msg.text}")

                print(f"{'=' * 60}")
                new_pending_requests.append(event)

        elif isinstance(event, WorkflowOutputEvent):
            print("\n[Workflow Completed]")
            print(f"Final conversation length: {len(event.data)} messages")  # type: ignore[arg-type]
            return [], None

    return new_pending_requests, latest_checkpoint.checkpoint_id


async def main() -> None:
    """
    Demonstrate the checkpoint-based pause/resume pattern for handoff workflows.

    This sample shows:
    1. Starting a workflow and getting a HandoffUserInputRequest
    2. Pausing (checkpoint is saved automatically)
    3. Resuming from checkpoint with a user response (two-step pattern)
    4. Continuing the conversation until completion
    """

    # Enable INFO logging to see workflow progress
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    # Clean up old checkpoints
    for file in CHECKPOINT_DIR.glob("*.json"):
        file.unlink()

    storage = FileCheckpointStorage(storage_path=CHECKPOINT_DIR)
    workflow, _, _, _ = create_workflow(checkpoint_storage=storage)

    print("=" * 60)
    print("HANDOFF WORKFLOW CHECKPOINT DEMO")
    print("=" * 60)

    # Scenario: User needs help with a damaged order
    initial_request = "Hi, my order 12345 arrived damaged. I need help."

    # Phase 1: Initial run - workflow will pause when it needs user input
    pending_requests, _ = await run_until_user_input_needed(
        workflow,
        initial_message=initial_request,
    )

    if not pending_requests:
        print("Workflow completed without needing user input")
        return

    print("\n>>> Workflow paused. You could exit the process here.")
    print(f">>> Checkpoint was saved. Pending requests: {len(pending_requests)}")
    print("\n>>> Simulating process restart...\n")

    # Simulate process restart - create fresh workflow instance
    workflow2, _, _, _ = create_workflow(checkpoint_storage=storage)

    # Phase 2: Resume with user response
    user_response_1 = "Yes, I'd like a replacement or refund. The packaging was completely destroyed."

    pending_requests, _ = await resume_with_response(
        workflow2,
        storage,
        user_response_1,
    )

    if not pending_requests:
        print("\nWorkflow completed!")
        return

    # Phase 3: Continue conversation - can repeat the pattern
    print("\n>>> Workflow paused again. Another checkpoint saved.")
    print(">>> Simulating another resume...\n")

    workflow3, _, _, _ = create_workflow(checkpoint_storage=storage)
    user_response_2 = "A replacement would be great. Can you ship it to the same address?"

    await resume_with_response(
        workflow3,
        storage,
        user_response_2,
    )

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
