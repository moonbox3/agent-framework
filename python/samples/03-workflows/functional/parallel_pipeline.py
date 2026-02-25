# Copyright (c) Microsoft. All rights reserved.

"""Parallel pipeline using asyncio.gather with functional workflows.

Still just plain async functions — no @step needed. Fan-out/fan-in
uses native Python concurrency via asyncio.gather.
"""

import asyncio

from agent_framework import RunContext, workflow


async def research_web(topic: str) -> str:
    """Simulate web research."""
    await asyncio.sleep(0.05)
    return f"Web results for '{topic}': 10 articles found"


async def research_papers(topic: str) -> str:
    """Simulate academic paper search."""
    await asyncio.sleep(0.05)
    return f"Papers on '{topic}': 3 relevant papers"


async def research_news(topic: str) -> str:
    """Simulate news search."""
    await asyncio.sleep(0.05)
    return f"News about '{topic}': 5 recent articles"


async def synthesize(sources: list[str]) -> str:
    """Combine research results into a summary."""
    return "Research Summary:\n" + "\n".join(f"  - {s}" for s in sources)


@workflow
async def research_pipeline(topic: str, ctx: RunContext) -> str:
    """Fan-out to three research tasks, then synthesize results."""
    web, papers, news = await asyncio.gather(
        research_web(topic),
        research_papers(topic),
        research_news(topic),
    )

    summary = await synthesize([web, papers, news])
    await ctx.yield_output(summary)
    return summary


async def main():
    result = await research_pipeline.run("AI agents")
    print(result.get_outputs()[0])


if __name__ == "__main__":
    asyncio.run(main())
