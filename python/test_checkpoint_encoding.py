"""Debug script to test checkpoint encoding with ChatMessage."""

from dataclasses import asdict, dataclass

from agent_framework import ChatMessage
from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value


@dataclass
class HandoffUserInputRequest:
    conversation: list[ChatMessage]
    awaiting_agent_id: str
    prompt: str
    source_executor_id: str


# Create a sample request
msg1 = ChatMessage(role="user", text="Hello")
msg2 = ChatMessage(role="assistant", text="Hi there")
request = HandoffUserInputRequest(
    conversation=[msg1, msg2], awaiting_agent_id="agent1", prompt="test", source_executor_id="source1"
)

print("=" * 60)
print("Test 1: encode_checkpoint_value on HandoffUserInputRequest")
print("=" * 60)
encoded = encode_checkpoint_value(request)
print(f"Type of encoded: {type(encoded)}")
print(f"Keys: {encoded.keys() if isinstance(encoded, dict) else 'N/A'}")
if isinstance(encoded, dict) and "value" in encoded:
    value = encoded["value"]
    print(f"\nType of encoded['value']: {type(value)}")
    print(f"Keys in value: {value.keys() if isinstance(value, dict) else 'N/A'}")
    if isinstance(value, dict) and "conversation" in value:
        conv = value["conversation"]
        print(f"\nType of conversation: {type(conv)}")
        print(f"Length: {len(conv) if isinstance(conv, list) else 'N/A'}")
        if isinstance(conv, list) and len(conv) > 0:
            first = conv[0]
            print(f"Type of first item: {type(first)}")
            print(f"Is it a ChatMessage? {isinstance(first, ChatMessage)}")
            if isinstance(first, dict):
                print(f"Keys in first item: {first.keys()}")

print("\n" + "=" * 60)
print("Test 2: Try json.dumps on encoded value")
print("=" * 60)
import json

try:
    json_str = json.dumps(encoded, indent=2)
    print("SUCCESS! Encoded value is JSON serializable")
    print(f"JSON length: {len(json_str)} chars")
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "=" * 60)
print("Test 3: Create WorkflowCheckpoint with encoded request")
print("=" * 60)
from agent_framework._workflows._checkpoint import WorkflowCheckpoint

checkpoint = WorkflowCheckpoint(workflow_id="test", pending_request_info_events={"req1": {"data": encoded}})

print("Checkpoint created")
print(f"Type of checkpoint.pending_request_info_events: {type(checkpoint.pending_request_info_events)}")

checkpoint_dict = asdict(checkpoint)
print(f"\nType after asdict: {type(checkpoint_dict)}")

# Check what's in pending_request_info_events after asdict
pending = checkpoint_dict.get("pending_request_info_events", {})
if "req1" in pending:
    req1_data = pending["req1"]["data"]
    print(f"Type of checkpoint_dict['pending_request_info_events']['req1']['data']: {type(req1_data)}")
    if isinstance(req1_data, dict) and "value" in req1_data:
        value = req1_data["value"]
        if isinstance(value, dict) and "conversation" in value:
            conv = value["conversation"]
            print(f"Type of conversation after asdict: {type(conv)}")
            if isinstance(conv, list) and len(conv) > 0:
                first = conv[0]
                print(f"Type of first item after asdict: {type(first)}")
                print(f"Is it a ChatMessage? {isinstance(first, ChatMessage)}")

print("\n" + "=" * 60)
print("Test 4: Try json.dumps on asdict(checkpoint)")
print("=" * 60)
try:
    json_str = json.dumps(checkpoint_dict, indent=2)
    print("SUCCESS! checkpoint_dict is JSON serializable")
    print(f"JSON length: {len(json_str)} chars")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback

    traceback.print_exc()
