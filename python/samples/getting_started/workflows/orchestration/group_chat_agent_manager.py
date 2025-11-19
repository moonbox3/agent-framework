# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from typing import cast

from agent_framework import (
    AgentRunUpdateEvent,
    ChatAgent,
    ChatMessage,
    GroupChatBuilder,
    ManagerSelectionResponse,
    WorkflowOutputEvent,
)
from agent_framework.openai import OpenAIChatClient

logging.basicConfig(level=logging.INFO)

"""
Sample: Group Chat with Agent-Based Manager

What it does:
- Demonstrates the new set_manager() API for agent-based coordination
- Manager is a full ChatAgent with access to tools, context, and observability
- Coordinates a researcher and writer agent to solve tasks collaboratively

Prerequisites:
- OpenAI environment variables configured for OpenAIChatClient
"""


async def main() -> None:
    # Create coordinator agent with structured output for speaker selection
    coordinator = ChatAgent(
        name="Coordinator",
        description="Coordinates multi-agent collaboration by selecting speakers",
        instructions="""
You coordinate a team conversation to solve the user's task.

Review the conversation history and select the next participant to speak.
Consider each participant's expertise:
- Researcher: Collects background information and facts
- Writer: Synthesizes polished answers from gathered information

Return your decision as JSON with this structure:
{
    "selected_participant": "Researcher",
    "instruction": "Optional instruction for the participant",
    "finish": false
}

When the task is complete after multiple participants have contributed, return:
{
    "finish": true,
    "final_message": "Summary of the conversation results"
}

Guidelines:
- Start with Researcher to gather information
- Then have Writer synthesize the final answer
- Only finish after both have contributed meaningfully
""",
        chat_client=OpenAIChatClient(model_id="gpt-4o"),
        response_format=ManagerSelectionResponse,
    )

    researcher = ChatAgent(
        name="Researcher",
        description="Collects relevant background information",
        instructions="Gather concise facts that help a teammate answer the question.",
        chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),
    )

    writer = ChatAgent(
        name="Writer",
        description="Synthesizes polished answers from gathered information",
        instructions="Compose clear and structured answers using any notes provided.",
        chat_client=OpenAIChatClient(model_id="gpt-4o-mini"),
    )

    workflow = (
        GroupChatBuilder()
        .set_manager(coordinator, display_name="Orchestrator")
        .participants([researcher, writer])
        .build()
    )

    task = "What are the key benefits of using async/await in Python? Provide a concise summary."

    print("\nStarting Group Chat with Agent-Based Manager...\n")
    print(f"TASK: {task}\n")
    print("=" * 80)

    final_conversation: list[ChatMessage] = []
    last_executor_id: str | None = None
    async for event in workflow.run_stream(task):
        if isinstance(event, AgentRunUpdateEvent):
            eid = event.executor_id
            if eid != last_executor_id:
                if last_executor_id is not None:
                    print()
                print(f"{eid}:", end=" ", flush=True)
                last_executor_id = eid
            print(event.data, end="", flush=True)
        elif isinstance(event, WorkflowOutputEvent):
            final_conversation = cast(list[ChatMessage], event.data)

    if final_conversation and isinstance(final_conversation, list):
        print("\n\n" + "=" * 80)
        print("FINAL CONVERSATION")
        print("=" * 80)
        for msg in final_conversation:
            author = getattr(msg, "author_name", "Unknown")
            text = getattr(msg, "text", str(msg))
            print(f"\n[{author}]")
            print(text)
            print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
