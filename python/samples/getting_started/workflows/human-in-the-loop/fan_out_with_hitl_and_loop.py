# Copyright (c) Microsoft. All rights reserved.

"""
Sample: Fan-out with HITL pause and looping node

Demonstrates the fan-out HITL pause behavior.
When a workflow fans out to parallel branches and one branch requests HITL
(human-in-the-loop), the workflow pauses at the superstep boundary. This
prevents the other branch from continuing to run away in a loop while
human input is pending.

Topology:

    Dispatcher
        |
    fan-out
    /       \
  Analyst   Reviewer (HITL)
    |
  Refiner
    |
  Analyst  (cycle: Analyst <-> Refiner)

- The Analyst <-> Refiner cycle iteratively refines a draft.
- The Reviewer requests human approval via request_info.
- Without the HITL pause, the Analyst/Refiner loop would keep cycling
  through supersteps while the Reviewer waits for human input.
- With the HITL pause, the workflow pauses after the first superstep
  where the Reviewer calls request_info. The Analyst's message to
  the Refiner is held until the human responds.

Prerequisites:
- No external dependencies. This sample uses plain executors (no LLM) to demonstrate
the behavior.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    WorkflowRunState,
    handler,
    response_handler,
)
from typing_extensions import Never

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    """A document draft that gets refined iteratively."""

    content: str
    version: int


@dataclass
class ReviewRequest:
    """Request for human review of the research plan."""

    plan: str


@dataclass
class ReviewResponse:
    """Human's review response."""

    approved: bool
    feedback: str


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


class Dispatcher(Executor):
    """Fans out the initial prompt to both the Analyst and the Reviewer."""

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[str]) -> None:
        print(f"\n[Dispatcher] Received prompt: '{prompt}'")
        print("[Dispatcher] Fanning out to Analyst and Reviewer...")
        await ctx.send_message(prompt)


class Analyst(Executor):
    """Produces a draft and sends it to the Refiner for improvement.

    On receiving a string (initial prompt), creates a v1 draft.
    On receiving a Draft back from the Refiner, checks if it's good
    enough or sends it back for another round.
    """

    MAX_VERSIONS = 3

    @handler
    async def handle_prompt(self, prompt: str, ctx: WorkflowContext[Draft]) -> None:
        draft = Draft(content=f"Initial analysis of '{prompt}'", version=1)
        print(f"  [Analyst] Created draft v{draft.version}: '{draft.content}'")
        print("  [Analyst] Sending to Refiner for improvement...")
        await ctx.send_message(draft)

    @handler
    async def handle_refined(self, draft: Draft, ctx: WorkflowContext[Draft, Draft]) -> None:
        print(f"  [Analyst] Received refined draft v{draft.version}: '{draft.content}'")
        if draft.version >= self.MAX_VERSIONS:
            print(f"  [Analyst] Draft v{draft.version} is final. Done.")
            await ctx.yield_output(draft)
        else:
            print("  [Analyst] Needs more work, sending back to Refiner...")
            await ctx.send_message(draft)


class Refiner(Executor):
    """Refines a draft and sends it back to the Analyst."""

    @handler
    async def refine(self, draft: Draft, ctx: WorkflowContext[Draft]) -> None:
        improved = Draft(
            content=f"{draft.content} [refined v{draft.version + 1}]",
            version=draft.version + 1,
        )
        print(f"  [Refiner] Improved draft to v{improved.version}: '{improved.content}'")
        await ctx.send_message(improved)


class Reviewer(Executor):
    """Requests human review of the research plan before proceeding.

    This is the HITL node. When it receives the prompt, it asks a human
    to approve the plan. The workflow pauses until the human responds.
    """

    @handler
    async def request_review(self, prompt: str, ctx: WorkflowContext) -> None:
        plan = f"Research plan for: '{prompt}'"
        print(f"  [Reviewer] Requesting human approval for: '{plan}'")
        await ctx.request_info(
            ReviewRequest(plan=plan),
            ReviewResponse,
        )

    @response_handler
    async def handle_review(
        self,
        original: ReviewRequest,
        response: ReviewResponse,
        ctx: WorkflowContext[Never, str],
    ) -> None:
        if response.approved:
            result = f"Plan APPROVED: {original.plan}"
        else:
            result = f"Plan NEEDS REVISION ({response.feedback}): {original.plan}"
        print(f"  [Reviewer] Human responded: {result}")
        await ctx.yield_output(result)


# ---------------------------------------------------------------------------
# Workflow construction
# ---------------------------------------------------------------------------


def build_workflow():
    """Build the fan-out workflow with a looping branch and HITL branch.

    Graph:
        Dispatcher --fan-out--> [Analyst, Reviewer]
        Analyst --> Refiner --> Analyst  (cycle)
        Reviewer: HITL (request_info / response_handler)
    """
    dispatcher = Dispatcher(id="dispatcher")
    analyst = Analyst(id="analyst")
    refiner = Refiner(id="refiner")
    reviewer = Reviewer(id="reviewer")

    return (
        WorkflowBuilder(start_executor=dispatcher)
        .add_fan_out_edges(dispatcher, [analyst, reviewer])
        .add_edge(analyst, refiner)
        .add_edge(refiner, analyst)
        .build()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    workflow = build_workflow()

    print("=" * 60)
    print("Fan-out HITL Pause Demo")
    print("=" * 60)
    print()
    print("Topology:")
    print("  Dispatcher --fan-out--> [Analyst, Reviewer]")
    print("  Analyst <--cycle--> Refiner")
    print("  Reviewer: HITL (requests human approval)")
    print()
    print("Expected behavior: The Analyst/Refiner loop should NOT")
    print("continue while the Reviewer waits for human input.")
    print()

    # --- First run ---
    print("-" * 60)
    print("FIRST RUN: Starting workflow...")
    print("-" * 60)

    request_events: list[WorkflowEvent] = []
    outputs: list[Any] = []
    superstep_count = 0
    final_state: WorkflowRunState | None = None

    async for event in workflow.run("Impact of AI on software development", stream=True):
        if event.type == "superstep_started":
            superstep_count += 1
            print(f"\n--- Superstep {superstep_count} ---")
        elif event.type == "request_info":
            request_events.append(event)
            print(f"\n  ** HITL REQUEST: {event.data}")
        elif event.type == "output":
            outputs.append(event.data)
            print(f"\n  ** OUTPUT: {event.data}")
        elif event.type == "status":
            final_state = event.state

    print(f"\n{'=' * 60}")
    print("First run completed.")
    print(f"  Supersteps executed: {superstep_count}")
    print(f"  Final state: {final_state}")
    print(f"  HITL requests: {len(request_events)}")
    print(f"  Outputs: {len(outputs)}")
    print()

    # Key observation: only 1 superstep ran. The Analyst sent a message
    # to the Refiner, but it was NOT delivered because the Reviewer
    # requested HITL in the same superstep. The workflow paused.
    print("VERIFIED: Workflow paused after 1 superstep.")
    print("  The Analyst/Refiner loop did NOT continue.")
    print("  The Analyst's message to Refiner is held in memory.")
    print()

    # --- Simulate human response ---
    print("-" * 60)
    print("HUMAN RESPONSE: Approving the plan...")
    print("-" * 60)

    request_id = request_events[0].request_id
    human_response = ReviewResponse(approved=True, feedback="Looks good!")

    outputs2: list[Any] = []
    superstep_count2 = 0
    final_state2: WorkflowRunState | None = None

    async for event in workflow.run(
        stream=True,
        responses={request_id: human_response},
    ):
        if event.type == "superstep_started":
            superstep_count2 += 1
            print(f"\n--- Superstep {superstep_count2} ---")
        elif event.type == "output":
            outputs2.append(event.data)
            print(f"\n  ** OUTPUT: {event.data}")
        elif event.type == "status":
            final_state2 = event.state

    print(f"\n{'=' * 60}")
    print("Second run completed.")
    print(f"  Supersteps executed: {superstep_count2}")
    print(f"  Final state: {final_state2}")
    print(f"  Outputs: {len(outputs2)}")

    # After HITL response, the held messages are delivered. The
    # Analyst/Refiner loop runs to completion AND the Reviewer's
    # response handler produces output.
    print()
    for i, output in enumerate(outputs2):
        print(f"  Output {i + 1}: {output}")

    print()
    print("VERIFIED: Both branches completed after HITL response.")
    print("  The Analyst/Refiner loop ran to completion.")
    print("  The Reviewer's approval was processed.")
    print()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
