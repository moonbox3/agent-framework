# Copyright (c) Microsoft. All rights reserved.

"""Basic sequential pipeline using the functional workflow API.

The simplest possible workflow: plain async functions orchestrated by @workflow.
No @step decorator needed — just write Python.

The @workflow decorator gives you:
- A .run() method that returns a WorkflowRunResult with events and outputs
- Streaming support via .run(stream=True)
- A .as_agent() method to wrap the workflow as an agent
"""

import asyncio

from agent_framework import RunContext, workflow


async def fetch_data(url: str) -> dict[str, str | int]:
    """Simulate fetching data from a URL."""
    return {"url": url, "content": f"Data from {url}", "status": 200}


async def transform_data(data: dict[str, str | int]) -> str:
    """Transform raw data into a summary string."""
    return f"[{data['status']}] {data['content']}"


async def validate_result(summary: str) -> bool:
    """Validate the transformed result."""
    return len(summary) > 0 and "[200]" in summary


@workflow
async def data_pipeline(url: str, ctx: RunContext) -> str:
    """A simple sequential data pipeline.

    These are plain async functions — no decorators, no framework concepts.
    The workflow is just Python control flow.
    """
    raw = await fetch_data(url)
    summary = await transform_data(raw)
    is_valid = await validate_result(summary)

    result = f"{summary} (valid={is_valid})"
    await ctx.yield_output(result)
    return result


async def main():
    result = await data_pipeline.run("https://example.com/api/data")
    print("Output:", result.get_outputs()[0])
    print("State:", result.get_final_state())


if __name__ == "__main__":
    asyncio.run(main())
