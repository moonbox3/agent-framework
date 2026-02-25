# Copyright (c) Microsoft. All rights reserved.

"""Human-in-the-loop review pipeline using functional workflows.

Demonstrates request_info() for pausing the workflow and resuming
with external input.

This sample uses @step because HITL workflows re-execute the function
on resume. Without @step, write_draft() would run again even though
it already completed. With @step, it replays instantly from cache.
"""

import asyncio

from agent_framework import RunContext, WorkflowRunState, step, workflow


@step
async def write_draft(topic: str) -> str:
    """Simulate writing a draft document.

    Decorated with @step so it doesn't re-execute when the workflow
    resumes after the human review.
    """
    print(f"  write_draft executing for '{topic}'")
    return f"Draft document about '{topic}': Lorem ipsum dolor sit amet..."


@step
async def revise_draft(draft: str, feedback: str) -> str:
    """Revise the draft based on feedback."""
    return f"Revised: {draft[:50]}... [Applied feedback: {feedback}]"


@workflow
async def review_pipeline(topic: str, ctx: RunContext) -> str:
    """Write a draft, get human review, then revise."""
    draft = await write_draft(topic)

    # Suspends the workflow and emits a request_info event.
    # The caller provides the response via run(responses={...}).
    feedback = await ctx.request_info(
        {"draft": draft, "instructions": "Please review this draft"},
        response_type=str,
        request_id="review_request",
    )

    # Only runs after resume — write_draft replays from cache above.
    final = await revise_draft(draft, feedback)
    await ctx.yield_output(final)
    return final


async def main():
    # Phase 1: Run until HITL interrupt
    print("=== Phase 1: Initial run ===")
    result1 = await review_pipeline.run("AI Safety")

    print(f"State: {result1.get_final_state()}")
    assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

    requests = result1.get_request_info_events()
    print(f"Pending request: {requests[0].request_id}")

    # Phase 2: Resume with human response
    print("\n=== Phase 2: Resume with feedback ===")
    print("(write_draft should NOT execute again)")
    result2 = await review_pipeline.run(responses={"review_request": "Add more details about alignment research"})

    print(f"State: {result2.get_final_state()}")
    print(f"Output: {result2.get_outputs()[0]}")


if __name__ == "__main__":
    asyncio.run(main())
