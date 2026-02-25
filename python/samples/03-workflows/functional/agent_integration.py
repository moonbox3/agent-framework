# Copyright (c) Microsoft. All rights reserved.

"""Calling agents inside functional workflows.

Agent calls are typically the most expensive operations in a workflow.
Wrapping them with @step ensures their results are cached and checkpointed,
so they don't re-execute on HITL resume or crash recovery.

This sample also demonstrates .as_agent() to wrap a workflow as an agent.

NOTE: Uses a mock agent for demonstration. Replace with a real
Agent + ChatClient for production use.
"""

import asyncio
from dataclasses import dataclass

from agent_framework import RunContext, step, workflow

# ---------------------------------------------------------------------------
# Mock agent (replace with real Agent + ChatClient for production use)
# ---------------------------------------------------------------------------


class MockAgent:
    """Simulates an agent that returns a fixed response."""

    def __init__(self, name: str):
        self.name = name

    async def run(self, prompt: str) -> "MockResponse":
        return MockResponse(text=f"[{self.name}] Response to: {prompt[:50]}")


@dataclass
class MockResponse:
    text: str


classifier_agent = MockAgent("classifier")
writer_agent = MockAgent("writer")
reviewer_agent = MockAgent("reviewer")


# ---------------------------------------------------------------------------
# Agent calls wrapped in @step for caching and observability
# ---------------------------------------------------------------------------


@step
async def classify_document(doc: str) -> str:
    """Use an agent to classify a document."""
    response = await classifier_agent.run(f"Classify this document: {doc}")
    return response.text


@step
async def generate_summary(doc: str) -> str:
    """Use an agent to generate a summary."""
    response = await writer_agent.run(f"Summarize: {doc}")
    return response.text


@step
async def review_summary(summary: str) -> str:
    """Use an agent to review the summary."""
    response = await reviewer_agent.run(f"Review this summary: {summary}")
    return response.text


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@workflow
async def document_pipeline(document: str, ctx: RunContext) -> str:
    """Process a document through classification, summarization, and review."""
    classification = await classify_document(document)
    summary = await generate_summary(document)
    review = await review_summary(summary)

    result = f"Classification: {classification}\nSummary: {summary}\nReview: {review}"
    await ctx.yield_output(result)
    return result


async def main():
    result = await document_pipeline.run("This is a technical document about machine learning...")
    print(result.get_outputs()[0])
    print()
    print(f"Events emitted: {len(result)}")

    # Wrap the workflow as an agent
    agent = document_pipeline.as_agent(name="doc_processor")
    response = await agent.run("Another document to process")
    print(f"\nAgent response: {response.text[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
