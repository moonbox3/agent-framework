# Copyright (c) Microsoft. All rights reserved.

"""Tests for checkpoint restoration with immediate response delivery."""

from dataclasses import dataclass

from agent_framework import (
    Executor,
    InMemoryCheckpointStorage,
    RequestInfoEvent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    response_handler,
)


@dataclass
class ApprovalRequest:
    """Simple approval request for testing."""

    message: str


class CombinedExecutor(Executor):
    """Executor that both requests and handles responses."""

    def __init__(self, id: str):
        super().__init__(id=id)
        self.received_response: str | None = None

    @handler
    async def process(self, message: str, ctx: WorkflowContext) -> None:
        """Request approval from user."""
        await ctx.request_info(
            request_data=ApprovalRequest(message=message),
            response_type=str,
        )

    @response_handler
    async def on_response(
        self,
        original_request: ApprovalRequest,
        response: str,
        ctx: WorkflowContext,
    ) -> None:
        """Handle user response."""
        self.received_response = response
        await ctx.yield_output(f"Received: {response} for '{original_request.message}'")


async def test_run_stream_with_checkpoint_and_responses():
    """Test run_stream with checkpoint_id and responses in a single call."""
    storage = InMemoryCheckpointStorage()

    executor = CombinedExecutor(id="approver")

    workflow = WorkflowBuilder().set_start_executor(executor).with_checkpointing(checkpoint_storage=storage).build()

    # Step 1: Start workflow and collect request_id when it pauses
    # Don't break - consume all events to let checkpoint be saved
    request_id = None

    async for event in workflow.run_stream("test message"):
        if isinstance(event, RequestInfoEvent):
            request_id = event.request_id

    assert request_id is not None

    # Get the checkpoint that was created
    checkpoints = await storage.list_checkpoints(workflow.id)
    assert len(checkpoints) > 0
    checkpoint_id = checkpoints[-1].checkpoint_id

    # Step 2: Create fresh workflow instance and resume with responses in one call
    executor2 = CombinedExecutor(id="approver")
    workflow2 = WorkflowBuilder().set_start_executor(executor2).build()

    responses = {request_id: "approved"}
    output_received = False

    async for event in workflow2.run_stream(
        checkpoint_id=checkpoint_id,
        checkpoint_storage=storage,
        responses=responses,
    ):
        if hasattr(event, "data") and event.data and "Received:" in str(event.data):
            assert event.data == "Received: approved for 'test message'"
            output_received = True

    assert output_received
    assert executor2.received_response == "approved"


async def test_run_with_checkpoint_and_responses():
    """Test run (non-streaming) with checkpoint_id and responses in a single call."""
    storage = InMemoryCheckpointStorage()

    executor = CombinedExecutor(id="approver")

    workflow = WorkflowBuilder().set_start_executor(executor).with_checkpointing(checkpoint_storage=storage).build()

    # Step 1: Start workflow and collect request when it pauses
    result = await workflow.run("test message")
    request_events = result.get_request_info_events()

    assert len(request_events) == 1
    request_id = request_events[0].request_id

    # Get the checkpoint
    checkpoints = await storage.list_checkpoints(workflow.id)
    assert len(checkpoints) > 0
    checkpoint_id = checkpoints[-1].checkpoint_id

    # Step 2: Create fresh workflow instance and resume with responses in one call
    executor2 = CombinedExecutor(id="approver")
    workflow2 = WorkflowBuilder().set_start_executor(executor2).build()

    responses = {request_id: "approved"}
    result2 = await workflow2.run(
        checkpoint_id=checkpoint_id,
        checkpoint_storage=storage,
        responses=responses,
    )

    outputs = result2.get_outputs()
    assert len(outputs) == 1
    assert outputs[0] == "Received: approved for 'test message'"


async def test_responses_without_checkpoint_id_raises_error():
    """Test that providing responses without checkpoint_id raises ValueError."""
    storage = InMemoryCheckpointStorage()

    executor = CombinedExecutor(id="approver")

    workflow = WorkflowBuilder().set_start_executor(executor).with_checkpointing(checkpoint_storage=storage).build()

    # Try to provide responses without checkpoint_id
    responses = {"some_request_id": "response"}

    try:
        async for _event in workflow.run_stream(
            message="test",
            responses=responses,
        ):
            pass
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "checkpoint_id" in str(e).lower()


async def test_checkpoint_restore_and_respond_workflow():
    """Test complete workflow: start -> pause -> restore+respond -> complete."""
    storage = InMemoryCheckpointStorage()

    executor = CombinedExecutor(id="approver")

    workflow = WorkflowBuilder().set_start_executor(executor).with_checkpointing(checkpoint_storage=storage).build()

    # Phase 1: Start workflow until it requests input
    # Don't break - consume all events to let checkpoint be saved
    request_id = None

    async for event in workflow.run_stream("Please review this"):
        if isinstance(event, RequestInfoEvent):
            request_id = event.request_id
            assert isinstance(event.data, ApprovalRequest)
            assert event.data.message == "Please review this"

    assert request_id is not None

    # Get the checkpoint
    checkpoints = await storage.list_checkpoints(workflow.id)
    assert len(checkpoints) > 0
    checkpoint_id = checkpoints[-1].checkpoint_id

    # Simulate server restart - create completely fresh workflow instance
    executor2 = CombinedExecutor(id="approver")
    fresh_workflow = WorkflowBuilder().set_start_executor(executor2).build()

    # Phase 2: Restore and respond in single call (simulating HTTP request with user response)
    responses = {request_id: "looks good"}
    completed = False

    async for event in fresh_workflow.run_stream(
        checkpoint_id=checkpoint_id,
        checkpoint_storage=storage,
        responses=responses,
    ):
        if hasattr(event, "data") and event.data and "Received:" in str(event.data):
            assert "Received: looks good" in event.data
            completed = True

    assert completed
    assert executor2.received_response == "looks good"
