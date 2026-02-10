# Copyright (c) Microsoft. All rights reserved.

"""
Sample: Fan-out Async + HITL with response_handlers parameter

Demonstrates automatic HITL request handling in fan-out workflows using the
response_handlers parameter. Execution is sequential in two phases:

1. Phase 1: The workflow runs to idle, collecting any request_info events.
2. Matching handlers are called concurrently via asyncio.gather.
3. Phase 2: Collected responses are submitted back and the workflow runs
   again to process them.

Usage:
    response_handlers = {
        ReviewRequest: handle_review,
        ApprovalRequest: handle_approval,
    }

    result = await workflow.run(
        initial_data,
        response_handlers=response_handlers,
    )
"""

import asyncio
from dataclasses import dataclass

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    response_handler,
)


@dataclass
class AnalysisData:
    """Initial data for analysis."""

    task: str
    data: str


@dataclass
class DataPacket:
    """Data flowing through the async processor."""

    iteration: int
    content: str
    analysis: str = ""


@dataclass
class ReviewRequest:
    """Request sent to human reviewer."""

    packet: DataPacket
    analysis: str
    prompt: str = "Please review the analysis and provide feedback"


class Analyzer(Executor):
    def __init__(self, id: str = "analyzer"):
        super().__init__(id)

    @handler
    async def start_analysis(
        self,
        data: AnalysisData,
        ctx: WorkflowContext[DataPacket, DataPacket],
    ) -> None:
        packet = DataPacket(
            iteration=0,
            content=data.data,
            analysis=f"Starting {data.task}...",
        )
        await ctx.send_message(packet)


class AsyncProcessor(Executor):
    """Async branch: loops and processes data."""

    def __init__(self, id: str = "processor"):
        super().__init__(id)
        self.iteration_count = 0

    @handler
    async def process_data(
        self,
        packet: DataPacket,
        ctx: WorkflowContext[DataPacket, DataPacket],
    ) -> None:
        self.iteration_count += 1
        await asyncio.sleep(0.5)
        analysis = f"Iteration {self.iteration_count}: {packet.content[:50]}... processed"

        if self.iteration_count < 3:
            updated_packet = DataPacket(
                iteration=self.iteration_count,
                content=packet.content,
                analysis=analysis,
            )
            await ctx.send_message(updated_packet)
        else:
            await ctx.send_message(
                DataPacket(
                    iteration=self.iteration_count,
                    content=packet.content,
                    analysis=analysis,
                )
            )


class Reviewer(Executor):
    """HITL branch: requests external feedback."""

    def __init__(self, id: str = "reviewer"):
        super().__init__(id)

    @handler
    async def review_data(
        self,
        packet: DataPacket,
        ctx: WorkflowContext[str, str],
    ) -> None:
        review_request = ReviewRequest(
            packet=packet,
            analysis=packet.analysis or "No analysis yet",
            prompt=f"Please review iteration {packet.iteration}",
        )

        await ctx.request_info(
            request_data=review_request,
            response_type=str,
        )

    @response_handler
    async def handle_review_feedback(
        self,
        original_request: ReviewRequest,
        feedback: str,
        ctx: WorkflowContext[str],
    ) -> None:
        result = f"Review feedback processed: {feedback}"
        await ctx.send_message(result)


class FinalAggregator(Executor):
    def __init__(self, id: str = "aggregator"):
        super().__init__(id)
        self.results: list[str] = []

    @handler
    async def aggregate(
        self,
        message: DataPacket | str,
        ctx: WorkflowContext[str],
    ) -> None:
        self.results.append(str(message))

        if len(self.results) >= 2:
            summary = "=== WORKFLOW COMPLETE ===\n"
            summary += "Results from async path: " + str(self.results[0]) + "\n"
            summary += "Results from HITL path: " + str(self.results[1])
            await ctx.send_message(summary)


# ============================================================================
# Response Handlers (External)
# ============================================================================
# These are registered via response_handlers dict, not as executor methods.
# They run after Phase 1 completes and their responses are submitted in Phase 2.


async def handle_review(request: ReviewRequest) -> str:
    """Handle external review request.

    Called after Phase 1 when the Reviewer emits a request_info with
    ReviewRequest type. The response is submitted back to the workflow
    for Phase 2 processing.
    """
    print(f"\n[Handler] Processing review request for iteration {request.packet.iteration}")

    # Simulate external API call
    await asyncio.sleep(2.0)

    feedback = "Analysis looks good. Continue processing."
    print(f"[Handler] Review feedback: {feedback}\n")

    return feedback


# ============================================================================
# Main
# ============================================================================


async def main() -> None:
    print("=" * 80)
    print("HITL: response_handlers parameter in workflow.run()")
    print("=" * 80)

    # Create executors
    analyzer = Analyzer()
    processor = AsyncProcessor()
    reviewer = Reviewer()
    aggregator = FinalAggregator()

    # Build workflow
    workflow = (
        WorkflowBuilder(start_executor=analyzer)
        .add_edge(analyzer, processor)
        .add_edge(analyzer, reviewer)
        .add_edge(processor, aggregator)
        .add_edge(reviewer, aggregator)
        .build()
    )

    initial_data = AnalysisData(
        task="document analysis",
        data="Sample document to process",
    )

    # ========================================================================
    # THE CLEAN API
    # ========================================================================
    # Define external response handlers (type-based dispatch)
    response_handlers = {
        ReviewRequest: handle_review,
        # Can add more: ApprovalRequest: handle_approval, etc.
    }

    print(f"\n[Start] Task: {initial_data.task}\n")

    # Run workflow with automatic response handling (two-phase execution)
    # Phase 1: workflow runs to idle, collecting request_info events
    # Handlers run concurrently, then responses are submitted for Phase 2
    result = await workflow.run(
        initial_data,
        response_handlers=response_handlers,
    )

    # Display results
    print(f"\nFinal state: {result.get_final_state()}")
    print(f"Total events: {len(result)}")
    outputs = result.get_outputs()
    if outputs:
        print(f"Outputs ({len(outputs)}):")
        for output in outputs:
            print(f"  - {output}")


if __name__ == "__main__":
    asyncio.run(main())
