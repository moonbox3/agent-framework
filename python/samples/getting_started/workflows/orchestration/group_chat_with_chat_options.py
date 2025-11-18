# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from typing import cast

from agent_framework import (
    AgentRunUpdateEvent,
    ChatAgent,
    ChatMessage,
    ChatOptions,
    GroupChatBuilder,
    WorkflowOutputEvent,
)
from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient

logging.basicConfig(level=logging.INFO)

"""
Sample: Group Chat with Custom Manager Settings

What it does:
- Demonstrates how to configure the prompt-based manager with ChatOptions
- Shows how to control temperature, seed, and other LLM parameters for reproducible manager decisions
- Uses ChatOptions to fine-tune the manager's decision-making behavior

Prerequisites:
- OpenAI environment variables configured for `OpenAIChatClient` and `OpenAIResponsesClient`.
"""


async def main() -> None:
    researcher = ChatAgent(
        name="Researcher",
        description="Collects relevant background information and data.",
        instructions="Gather concise, factual information that helps answer the question.",
        chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),
    )

    writer = ChatAgent(
        name="Writer",
        description="Synthesizes information into clear, well-structured content.",
        instructions="Create clear and structured content using the information provided.",
        chat_client=OpenAIResponsesClient(),
    )

    analyst = ChatAgent(
        name="Analyst",
        description="Analyzes data and provides insights.",
        instructions="Analyze the gathered information and provide key insights.",
        chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),
    )

    # Configure manager with specific LLM settings for reproducible behavior
    manager_options = ChatOptions(
        temperature=0.3,  # Lower temperature for more deterministic decisions
        seed=42,  # Seed for reproducibility
        max_tokens=500,  # Limit response length
    )

    workflow = (
        GroupChatBuilder()
        .set_prompt_based_manager(
            chat_client=OpenAIChatClient(),
            display_name="Coordinator",
            chat_options=manager_options,
        )
        .participants(researcher=researcher, analyst=analyst, writer=writer)
        .build()
    )

    task = "What are the key benefits of using async/await in Python? Provide a clear explanation with examples."

    print("\nStarting Group Chat with Custom Manager Settings...\n")
    print(f"TASK: {task}\n")
    print(f"Manager settings: temperature={manager_options.temperature}, seed={manager_options.seed}\n")
    print("=" * 80)

    final_conversation: list[ChatMessage] = []
    last_executor_id: str | None = None
    async for event in workflow.run_stream(task):
        if isinstance(event, AgentRunUpdateEvent):
            # Handle the streaming agent update as it's produced
            eid = event.executor_id
            if eid != last_executor_id:
                if last_executor_id is not None:
                    print()
                print(f"\n[{eid}]:", end=" ", flush=True)
                last_executor_id = eid
            print(event.data, end="", flush=True)
        elif isinstance(event, WorkflowOutputEvent):
            final_conversation = cast(list[ChatMessage], event.data)

    if final_conversation and isinstance(final_conversation, list):
        print("\n")
        print("=" * 80)
        print("WORKFLOW COMPLETED")
        print("=" * 80)
        print(f"Total messages in conversation: {len(final_conversation)}")
        print("\nFinal message:")
        if final_conversation:
            final_msg = final_conversation[-1]
            author = getattr(final_msg, "author_name", "Unknown")
            text = getattr(final_msg, "text", str(final_msg))
            print(f"[{author}]: {text}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
