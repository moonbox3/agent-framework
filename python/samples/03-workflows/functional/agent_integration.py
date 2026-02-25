# Copyright (c) Microsoft. All rights reserved.

"""Calling agents inside functional workflows.

Agent calls work inside @workflow with or without @step. The difference:
- Without @step: agent calls re-execute on resume (same prompt, same cost).
- With @step: completed agent calls return their saved result instantly.

Since each agent call hits an LLM API (time + money), @step is almost always
worth it here. Each @step also emits executor events for tracing.

This sample also demonstrates .as_agent() to wrap a workflow as an agent.

Environment variables:
  AZURE_OPENAI_ENDPOINT        — Your Azure OpenAI endpoint
  AZURE_OPENAI_API_VERSION     — API version (e.g. 2025-04-01-preview)
  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME — Model deployment name (e.g. gpt-4o)
"""

import asyncio

from agent_framework import Agent, step, workflow
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Create agents with focused, concise instructions
# ---------------------------------------------------------------------------

classifier_agent = Agent(
    name="ClassifierAgent",
    instructions=(
        "Classify documents into one category: Technical, Legal, Marketing, or Scientific. "
        "Reply with only the category name."
    ),
    client=AzureOpenAIChatClient(),
)

writer_agent = Agent(
    name="WriterAgent",
    instructions="Summarize the given content in one sentence.",
    client=AzureOpenAIChatClient(),
)

reviewer_agent = Agent(
    name="ReviewerAgent",
    instructions="Review the given summary in one sentence. Is it accurate and complete?",
    client=AzureOpenAIChatClient(),
)

# ---------------------------------------------------------------------------
# @step saves each agent result so it won't re-execute on resume.
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
async def document_pipeline(document: str) -> str:
    """Process a document through classification, summarization, and review."""
    classification = await classify_document(document)
    summary = await generate_summary(document)
    review = await review_summary(summary)

    return f"Classification: {classification}\nSummary: {summary}\nReview: {review}"


async def main():
    result = await document_pipeline.run("This is a technical document about machine learning...")
    print(result.get_outputs()[0])

    # .as_agent() wraps the workflow so it can be used anywhere an agent
    # is expected — for example, as a node in a graph workflow.
    agent = document_pipeline.as_agent(name="doc_processor")
    response = await agent.run("A short story about a robot learning to paint.")
    print(f"\nAs agent: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
