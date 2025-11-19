"""Debug script to replicate the exact checkpoint creation flow."""

import asyncio
import json
from dataclasses import asdict

from agent_framework import ChatMessage
from agent_framework._workflows._checkpoint import WorkflowCheckpoint
from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value
from agent_framework._workflows._events import RequestInfoEvent
from agent_framework._workflows._handoff import HandoffUserInputRequest


async def main():
    # Simulate the exact flow in _runner_context.py _get_serialized_workflow_state

    # Step 1: Create messages like in the workflow
    messages = {
        "gateway": [
            ChatMessage(role="user", text="Hello"),
            ChatMessage(role="assistant", text="Hi there"),
        ]
    }

    # Step 2: Serialize messages like line 462-463 of _runner_context.py
    serialized_messages = {}
    for source_id, message_list in messages.items():
        serialized_messages[source_id] = [msg.to_dict() for msg in message_list]

    print("Messages serialized successfully")
    print(f"Type of serialized_messages['gateway'][0]: {type(serialized_messages['gateway'][0])}")

    # Step 3: Create HandoffUserInputRequest like in _handoff.py line 538
    request_data = HandoffUserInputRequest(
        conversation=[ChatMessage(role="user", text="Test")],
        awaiting_agent_id="agent1",
        prompt="test",
        source_executor_id="source1",
    )

    # Step 4: Create RequestInfoEvent like in _workflow_context.py line 374
    request_info_event = RequestInfoEvent(
        request_id="req-123",
        source_executor_id="gateway",
        request_data=request_data,
        response_type=object,
    )

    # Step 5: Serialize RequestInfoEvent like line 465 of _runner_context.py
    serialized_pending_request_info_events = {"req-123": request_info_event.to_dict()}

    print("Pending requests serialized successfully")

    # Step 6: Create state dict like line 468-473 of _runner_context.py
    state = {
        "messages": serialized_messages,
        "shared_state": encode_checkpoint_value({}),
        "iteration_count": 1,
        "pending_request_info_events": serialized_pending_request_info_events,
    }

    # Step 7: Create WorkflowCheckpoint like line 382-388 of _runner_context.py
    checkpoint = WorkflowCheckpoint(
        workflow_id="test-workflow",
        messages=state["messages"],
        shared_state=state["shared_state"],
        pending_request_info_events=state["pending_request_info_events"],
        iteration_count=state["iteration_count"],
        metadata={},
    )

    print("WorkflowCheckpoint created successfully")

    # Step 8: Convert to dict like line 141 of _checkpoint.py
    checkpoint_dict = asdict(checkpoint)

    print("asdict(checkpoint) completed")

    # Step 9: Check types
    print("\nInspecting checkpoint_dict...")
    print(f"Type of checkpoint_dict['messages']: {type(checkpoint_dict['messages'])}")
    if "gateway" in checkpoint_dict["messages"]:
        msgs = checkpoint_dict["messages"]["gateway"]
        print(f"Type of first message: {type(msgs[0])}")
        if not isinstance(msgs[0], dict):
            print(f"ERROR: Message is not a dict, it's: {msgs[0]}")

    print(
        f"\nType of checkpoint_dict['pending_request_info_events']: {type(checkpoint_dict['pending_request_info_events'])}"
    )
    if "req-123" in checkpoint_dict["pending_request_info_events"]:
        req = checkpoint_dict["pending_request_info_events"]["req-123"]
        print(f"Type of request: {type(req)}")
        if "data" in req:
            print(f"Type of request['data']: {type(req['data'])}")

    # Step 10: Try json.dump like line 147 of _checkpoint.py
    print("\nAttempting json.dump...")
    try:
        json_str = json.dumps(checkpoint_dict, indent=2)
        print(f"SUCCESS! JSON length: {len(json_str)} chars")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()

        # Find the problematic object
        print("\n\nSearching for ChatMessage objects...")

        def find_chat_messages(obj, path="root"):
            if isinstance(obj, ChatMessage):
                print(f"Found ChatMessage at: {path}")
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    find_chat_messages(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_chat_messages(item, f"{path}[{i}]")

        find_chat_messages(checkpoint_dict)


if __name__ == "__main__":
    asyncio.run(main())
