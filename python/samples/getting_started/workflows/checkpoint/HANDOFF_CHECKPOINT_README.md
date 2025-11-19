# Handoff Pattern with Checkpointing: Pause/Resume Support

## Customer Question

> "When `HandoffUserInputRequest` is generated I need to pause the workflow, send the request to the user interface, and when I get the user message I should resume the workflow sending the response. In order to suspend and resume the workflow I was looking at the checkpoint concept.
> 
> Question: How should I resume the workflow providing the user response (for the `HandoffUserInputRequest`) back to the specific workflow instance?
> 
> The problem is that `workflow.send_responses_streaming` doesn't have a `checkpoint` param, only `workflow.run_stream` can accept a checkpoint."

## Answer: YES, This Pattern is Supported

The Handoff pattern **does support** checkpoint-based pause/resume with user input requests. However, it requires a **two-step pattern** because, as the customer correctly observed, `send_responses_streaming()` doesn't accept a `checkpoint_id` parameter.

## The Two-Step Resume Pattern

```python
# Step 1: Restore the checkpoint to load pending requests into workflow state
async for event in workflow.run_stream(checkpoint_id="checkpoint_123"):
    if isinstance(event, RequestInfoEvent):
        pending_request_ids.append(event.request_id)
    break  # Stop after checkpoint restoration

# Step 2: Reset workflow's internal running flags (required)
if hasattr(workflow, "_is_running"):
    workflow._is_running = False
if hasattr(workflow, "_runner") and hasattr(workflow._runner, "_running"):
    workflow._runner._running = False

# Step 3: Send user responses
responses = {req_id: user_response for req_id in pending_request_ids}
async for event in workflow.send_responses_streaming(responses):
    # Process events...
```

## Why This Pattern Works

1. **Checkpoint Restoration**: `run_stream(checkpoint_id=...)` restores the workflow state including pending `HandoffUserInputRequest` events
2. **In-Memory State**: The checkpoint loads pending requests into the workflow's in-memory state
3. **Response Delivery**: `send_responses_streaming(responses)` sends responses to those restored pending requests
4. **Stateless HTTP Compatible**: This pattern works for stateless HTTP scenarios where the workflow instance doesn't persist between requests

## Complete Working Sample

See: `handoff_checkpoint_resume.py`

This sample demonstrates:
- Starting a handoff workflow
- Receiving a `HandoffUserInputRequest`
- Pausing (checkpoint saved automatically)
- **Simulating process restart** (creating new workflow instance)
- Resuming from checkpoint with user response (two-step pattern)
- Continuing the conversation

## Key Architectural Points

### Why Not a Single `send_responses_streaming(responses, checkpoint_id)` Call?

The current architecture separates concerns:
- `run_stream(checkpoint_id)` - State restoration (loading checkpoints)
- `send_responses_streaming(responses)` - Response delivery (workflow execution)

This separation actually enables the pattern to work correctly because:
1. Checkpoint restoration must happen first to populate pending requests
2. Response validation occurs against the restored pending requests
3. The workflow must be in a specific internal state before accepting responses

### Comparison to Other Checkpoint Samples

Unlike `checkpoint_with_human_in_the_loop.py` which uses a simple request/response executor, the Handoff pattern:
- Uses `HandoffUserInputRequest` (instead of custom request types)
- Manages conversation state automatically
- Handles multi-agent routing
- Requires the two-step pattern for stateless scenarios

## Implementation Note from DevUI

The DevUI package uses this exact pattern for stateless HTTP scenarios:

```python
# From agent_framework_devui/_executor.py
# Step 1: Restore checkpoint
async for _event in workflow.run_stream(checkpoint_id=checkpoint_id, checkpoint_storage=storage):
    restored = True
    break  # Stop immediately after restoration

# Step 2: Reset flags
if hasattr(workflow, "_is_running"):
    workflow._is_running = False
if hasattr(workflow, "_runner") and hasattr(workflow._runner, "_running"):
    workflow._runner._running = False

# Step 3: Send responses
async for event in workflow.send_responses_streaming(responses):
    # Process events...
```

## Future Enhancement Consideration

A potential framework enhancement could provide:
```python
# Hypothetical future API (not currently supported)
async for event in workflow.run_stream(
    checkpoint_id="checkpoint_123",
    responses={"request_id": "user response"}
):
    # Combined checkpoint restoration + response delivery
```

However, the current two-step pattern is the supported and working approach.

## Summary

**YES** - The Handoff pattern supports checkpoint-based pause/resume with `HandoffUserInputRequest`.

**Pattern**: Use the two-step approach:
1. `workflow.run_stream(checkpoint_id=...)`
2. `workflow.send_responses_streaming(responses)`

This is the documented and supported pattern for stateless scenarios where workflow instances don't persist between requests.
