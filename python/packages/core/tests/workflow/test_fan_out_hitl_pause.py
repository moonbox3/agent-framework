# Copyright (c) Microsoft. All rights reserved.

"""Tests for fan-out HITL pause behavior.

When a workflow fans out to parallel nodes and one node requests HITL (via request_info),
the workflow should pause at the superstep boundary. Messages from other parallel nodes
are preserved and delivered alongside HITL responses in the next run.

See: https://github.com/microsoft/agent-framework/issues/3539
"""

from dataclasses import dataclass

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    WorkflowRunState,
    handler,
    response_handler,
)
from agent_framework._workflows._request_info_mixin import RequestInfoMixin

# region Test Executors


class DispatchExecutor(Executor):
    """Start executor that dispatches a string message to fan-out targets."""

    @handler
    async def handle(self, message: str, ctx: WorkflowContext) -> None:
        await ctx.send_message(message)


class WorkerExecutor(Executor):
    """Executor that processes input and sends a result downstream."""

    @handler
    async def handle(self, data: str, ctx: WorkflowContext) -> None:
        await ctx.send_message(f"worker:{data}")


class HITLExecutor(Executor, RequestInfoMixin):
    """Executor that requests human-in-the-loop input."""

    @handler
    async def handle(self, data: str, ctx: WorkflowContext) -> None:
        await ctx.request_info(f"need_info:{data}", str)

    @response_handler
    async def on_response(self, original: str, response: str, ctx: WorkflowContext) -> None:
        await ctx.send_message(f"hitl:{response}")


class CollectorExecutor(Executor):
    """Executor that collects results as workflow outputs."""

    def __init__(self, id: str):
        super().__init__(id=id)
        self.collected: list[str] = []

    @handler
    async def handle(self, data: str, ctx: WorkflowContext) -> None:
        self.collected.append(data)
        await ctx.yield_output(data)


@dataclass
class TypedRequest:
    """Typed request for HITL."""

    question: str


@dataclass
class TypedResponse:
    """Typed response for HITL."""

    answer: str


class TypedHITLExecutor(Executor, RequestInfoMixin):
    """HITL executor using typed request/response."""

    @handler
    async def handle(self, data: str, ctx: WorkflowContext) -> None:
        await ctx.request_info(TypedRequest(question=data), TypedResponse)

    @response_handler
    async def on_response(self, original: TypedRequest, response: TypedResponse, ctx: WorkflowContext) -> None:
        await ctx.send_message(f"hitl:{response.answer}")


# endregion


class TestFanOutHITLPause:
    """Tests for workflow pause behavior when fan-out nodes request HITL."""

    async def test_fan_out_pauses_when_hitl_requested(self):
        """When one fan-out node requests HITL, the workflow pauses at the superstep
        boundary. Messages from other parallel nodes are held, not delivered.

        Topology: Dispatch -> fan-out -> [Worker, HITL] -> Collector
        """
        dispatch = DispatchExecutor(id="dispatch")
        worker = WorkerExecutor(id="worker")
        hitl = HITLExecutor(id="hitl")
        collector = CollectorExecutor(id="collector")

        workflow = (
            WorkflowBuilder(start_executor=dispatch)
            .add_fan_out_edges(dispatch, [worker, hitl])
            .add_edge(worker, collector)
            .add_edge(hitl, collector)
            .build()
        )

        # First run: Worker processes and sends to Collector, HITL requests info
        request_events: list[WorkflowEvent] = []
        outputs: list[str] = []
        final_state: WorkflowRunState | None = None

        async for event in workflow.run("hello", stream=True):
            if event.type == "request_info":
                request_events.append(event)
            elif event.type == "output":
                outputs.append(event.data)
            elif event.type == "status":
                final_state = event.state

        # Workflow should be idle with pending requests
        assert final_state == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        # HITL request should be surfaced
        assert len(request_events) == 1
        assert request_events[0].data == "need_info:hello"

        # Key assertion: Collector should NOT have received any messages yet.
        # Worker's message is held at the superstep boundary.
        assert len(outputs) == 0
        assert len(collector.collected) == 0

        # Resume with HITL response
        outputs2: list[str] = []
        final_state2: WorkflowRunState | None = None

        async for event in workflow.run(stream=True, responses={request_events[0].request_id: "human_answer"}):
            if event.type == "output":
                outputs2.append(event.data)
            elif event.type == "status":
                final_state2 = event.state

        # Workflow should complete
        assert final_state2 == WorkflowRunState.IDLE

        # Collector should have received BOTH worker and HITL results
        assert len(outputs2) == 2
        assert "worker:hello" in outputs2
        assert "hitl:human_answer" in outputs2

    async def test_fan_out_without_hitl_continues_normally(self):
        """Fan-out without HITL should not be affected by the pause behavior.

        Topology: Dispatch -> fan-out -> [Worker1, Worker2] -> Collector
        """
        dispatch = DispatchExecutor(id="dispatch")
        worker1 = WorkerExecutor(id="worker1")
        worker2 = WorkerExecutor(id="worker2")
        collector = CollectorExecutor(id="collector")

        workflow = (
            WorkflowBuilder(start_executor=dispatch)
            .add_fan_out_edges(dispatch, [worker1, worker2])
            .add_edge(worker1, collector)
            .add_edge(worker2, collector)
            .build()
        )

        result = await workflow.run("hello")

        assert result.get_final_state() == WorkflowRunState.IDLE
        outputs = result.get_outputs()
        assert len(outputs) == 2
        assert "worker:hello" in outputs
        assert "worker:hello" in outputs

    async def test_multiple_hitl_requests_in_fan_out(self):
        """When multiple fan-out nodes request HITL, all requests are surfaced
        before the workflow pauses.

        Topology: Dispatch -> fan-out -> [HITL1, HITL2] -> Collector
        """
        dispatch = DispatchExecutor(id="dispatch")
        hitl1 = HITLExecutor(id="hitl1")
        hitl2 = HITLExecutor(id="hitl2")
        collector = CollectorExecutor(id="collector")

        workflow = (
            WorkflowBuilder(start_executor=dispatch)
            .add_fan_out_edges(dispatch, [hitl1, hitl2])
            .add_edge(hitl1, collector)
            .add_edge(hitl2, collector)
            .build()
        )

        # First run: Both HITL nodes request info
        request_events: list[WorkflowEvent] = []
        outputs: list[str] = []
        final_state: WorkflowRunState | None = None

        async for event in workflow.run("hello", stream=True):
            if event.type == "request_info":
                request_events.append(event)
            elif event.type == "output":
                outputs.append(event.data)
            elif event.type == "status":
                final_state = event.state

        assert final_state == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
        assert len(request_events) == 2
        assert len(outputs) == 0

        # Resume with both responses
        responses = {
            request_events[0].request_id: "answer1",
            request_events[1].request_id: "answer2",
        }
        outputs2: list[str] = []
        final_state2: WorkflowRunState | None = None

        async for event in workflow.run(stream=True, responses=responses):
            if event.type == "output":
                outputs2.append(event.data)
            elif event.type == "status":
                final_state2 = event.state

        assert final_state2 == WorkflowRunState.IDLE
        assert len(outputs2) == 2
        assert "hitl:answer1" in outputs2
        assert "hitl:answer2" in outputs2

    async def test_fan_out_hitl_with_typed_request_response(self):
        """HITL pause works correctly with typed (dataclass) request/response.

        Topology: Dispatch -> fan-out -> [Worker, TypedHITL] -> Collector
        """
        dispatch = DispatchExecutor(id="dispatch")
        worker = WorkerExecutor(id="worker")
        hitl = TypedHITLExecutor(id="typed_hitl")
        collector = CollectorExecutor(id="collector")

        workflow = (
            WorkflowBuilder(start_executor=dispatch)
            .add_fan_out_edges(dispatch, [worker, hitl])
            .add_edge(worker, collector)
            .add_edge(hitl, collector)
            .build()
        )

        # First run
        request_events: list[WorkflowEvent] = []
        outputs: list[str] = []

        async for event in workflow.run("what is 2+2?", stream=True):
            if event.type == "request_info":
                request_events.append(event)
            elif event.type == "output":
                outputs.append(event.data)

        assert len(request_events) == 1
        assert isinstance(request_events[0].data, TypedRequest)
        assert request_events[0].data.question == "what is 2+2?"
        assert len(outputs) == 0  # Worker's message held

        # Resume with typed response
        outputs2: list[str] = []

        async for event in workflow.run(
            stream=True,
            responses={request_events[0].request_id: TypedResponse(answer="4")},
        ):
            if event.type == "output":
                outputs2.append(event.data)

        assert len(outputs2) == 2
        assert "worker:what is 2+2?" in outputs2
        assert "hitl:4" in outputs2

    async def test_non_streaming_fan_out_hitl_pause(self):
        """HITL pause works correctly in non-streaming mode.

        Topology: Dispatch -> fan-out -> [Worker, HITL] -> Collector
        """
        dispatch = DispatchExecutor(id="dispatch")
        worker = WorkerExecutor(id="worker")
        hitl = HITLExecutor(id="hitl")
        collector = CollectorExecutor(id="collector")

        workflow = (
            WorkflowBuilder(start_executor=dispatch)
            .add_fan_out_edges(dispatch, [worker, hitl])
            .add_edge(worker, collector)
            .add_edge(hitl, collector)
            .build()
        )

        # Non-streaming run
        result = await workflow.run("hello")

        assert result.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
        assert len(result.get_outputs()) == 0  # Worker's message held

        # Get request info events
        request_events = result.get_request_info_events()
        assert len(request_events) == 1

        # Resume with response
        result2 = await workflow.run(responses={request_events[0].request_id: "done"})
        assert result2.get_final_state() == WorkflowRunState.IDLE
        outputs = result2.get_outputs()
        assert len(outputs) == 2
        assert "worker:hello" in outputs
        assert "hitl:done" in outputs

    async def test_single_node_hitl_still_works(self):
        """Single-node HITL (no fan-out) should continue to work as before.

        This is a regression test to ensure the pause logic doesn't break
        simple sequential HITL workflows.
        """
        hitl = HITLExecutor(id="hitl")
        collector = CollectorExecutor(id="collector")

        workflow = WorkflowBuilder(start_executor=hitl).add_edge(hitl, collector).build()

        # First run
        request_events: list[WorkflowEvent] = []
        async for event in workflow.run("hello", stream=True):
            if event.type == "request_info":
                request_events.append(event)

        assert len(request_events) == 1

        # Resume
        outputs: list[str] = []
        async for event in workflow.run(stream=True, responses={request_events[0].request_id: "world"}):
            if event.type == "output":
                outputs.append(event.data)

        assert len(outputs) == 1
        assert outputs[0] == "hitl:world"

    async def test_fan_out_hitl_pause_with_fan_in(self):
        """HITL pause works with fan-out/fan-in topology.

        Topology: Dispatch -> fan-out -> [Worker, HITL] -> fan-in -> Aggregator
        """

        class AggregatorExecutor(Executor):
            """Aggregates fan-in results."""

            def __init__(self, id: str):
                super().__init__(id=id)
                self.aggregated: list[str] = []

            @handler
            async def handle(self, data: list[str], ctx: WorkflowContext) -> None:
                self.aggregated = data
                await ctx.yield_output(data)

        dispatch = DispatchExecutor(id="dispatch")
        worker = WorkerExecutor(id="worker")
        hitl = HITLExecutor(id="hitl")
        aggregator = AggregatorExecutor(id="aggregator")

        workflow = (
            WorkflowBuilder(start_executor=dispatch)
            .add_fan_out_edges(dispatch, [worker, hitl])
            .add_fan_in_edges([worker, hitl], aggregator)
            .build()
        )

        # First run: HITL pauses, worker's message is held
        request_events: list[WorkflowEvent] = []
        outputs: list = []

        async for event in workflow.run("hello", stream=True):
            if event.type == "request_info":
                request_events.append(event)
            elif event.type == "output":
                outputs.append(event.data)

        assert len(request_events) == 1
        assert len(outputs) == 0  # Aggregator didn't run

        # Resume: both messages reach the fan-in, aggregator processes them
        outputs2: list = []
        async for event in workflow.run(stream=True, responses={request_events[0].request_id: "human_input"}):
            if event.type == "output":
                outputs2.append(event.data)

        assert len(outputs2) == 1
        aggregated = outputs2[0]
        assert isinstance(aggregated, list)
        assert len(aggregated) == 2
        assert "worker:hello" in aggregated
        assert "hitl:human_input" in aggregated
