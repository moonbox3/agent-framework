# Copyright (c) Microsoft. All rights reserved.

"""Introducing @step: caching, checkpointing, and observability.

The previous samples used plain functions — and that's fine for simple cases.
But when your workflow has expensive operations, you may want:

- **Caching**: If the workflow is interrupted (e.g., HITL) and resumed, completed
  steps replay instantly from cache instead of re-executing.
- **Checkpointing**: Each step's result is checkpointed to storage, so a crash
  mid-workflow doesn't lose progress on completed steps.
- **Observability**: Each step emits executor_invoked/executor_completed events,
  giving you visibility into what ran and what it produced.

The @step decorator is opt-in. Use it on functions where these features matter.
Plain functions still work alongside @step functions in the same workflow.
"""

import asyncio

from agent_framework import InMemoryCheckpointStorage, RunContext, step, workflow

# Track call counts to demonstrate caching behavior
fetch_calls = 0
transform_calls = 0


@step
async def fetch_data(url: str) -> dict[str, str | int]:
    """Expensive operation — we want this cached and checkpointed."""
    global fetch_calls
    fetch_calls += 1
    print(f"  fetch_data called (call #{fetch_calls})")
    return {"url": url, "content": f"Data from {url}", "status": 200}


@step
async def transform_data(data: dict[str, str | int]) -> str:
    """Another expensive operation worth caching."""
    global transform_calls
    transform_calls += 1
    print(f"  transform_data called (call #{transform_calls})")
    return f"[{data['status']}] {data['content']}"


async def validate_result(summary: str) -> bool:
    """Cheap validation — no @step needed, just a plain function."""
    return len(summary) > 0 and "[200]" in summary


storage = InMemoryCheckpointStorage()


@workflow(checkpoint_storage=storage)
async def data_pipeline(url: str, ctx: RunContext) -> str:
    """Mix of @step functions (cached) and plain functions (not cached)."""
    raw = await fetch_data(url)
    summary = await transform_data(raw)
    is_valid = await validate_result(summary)  # plain function — always runs

    result = f"{summary} (valid={is_valid})"
    await ctx.yield_output(result)
    return result


async def main():
    # --- Run 1: Everything executes normally ---
    print("=== Run 1: Fresh execution ===")
    result = await data_pipeline.run("https://example.com/api/data")
    print(f"Output: {result.get_outputs()[0]}")
    print(f"fetch_calls={fetch_calls}, transform_calls={transform_calls}")

    # Inspect events — @step functions emit executor events, plain functions don't
    print("\nEvents:")
    for event in result:
        if event.type in ("executor_invoked", "executor_completed"):
            print(f"  {event.type}: {event.executor_id}")

    # --- Run 2: Restore from checkpoint — @step results replay from cache ---
    print("\n=== Run 2: Restored from checkpoint ===")
    latest = await storage.get_latest(workflow_name="data_pipeline")
    assert latest is not None
    ckpt_id = latest.checkpoint_id

    result2 = await data_pipeline.run(checkpoint_id=ckpt_id)
    print(f"Output: {result2.get_outputs()[0]}")
    print(f"fetch_calls={fetch_calls}, transform_calls={transform_calls}")
    print("(call counts unchanged — @step results were replayed from cache)")


if __name__ == "__main__":
    asyncio.run(main())
