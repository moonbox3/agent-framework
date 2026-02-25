# Copyright (c) Microsoft. All rights reserved.

"""
Functional Workflow with Agents — Call agents inside @workflow

This sample shows how to call agents inside a functional workflow.
Agent calls are just regular async function calls — no special wrappers needed.

Environment variables:
  AZURE_OPENAI_ENDPOINT        — Your Azure OpenAI endpoint
  AZURE_OPENAI_API_VERSION     — API version (e.g. 2025-04-01-preview)
  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME — Model deployment name (e.g. gpt-4o)
"""

import asyncio

from agent_framework import Agent, workflow
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv

load_dotenv()

writer = Agent(
    name="WriterAgent",
    instructions="Write a short poem (4 lines max) about the given topic.",
    client=AzureOpenAIChatClient(),
)

reviewer = Agent(
    name="ReviewerAgent",
    instructions="Review the given poem in one sentence. Is it good?",
    client=AzureOpenAIChatClient(),
)


@workflow
async def poem_pipeline(topic: str) -> str:
    """Write a poem, then review it."""
    poem = (await writer.run(f"Write a poem about: {topic}")).text
    review = (await reviewer.run(f"Review this poem: {poem}")).text
    return f"Poem:\n{poem}\n\nReview: {review}"


async def main() -> None:
    result = await poem_pipeline.run("a cat learning to code")
    print(result.get_outputs()[0])


if __name__ == "__main__":
    asyncio.run(main())
