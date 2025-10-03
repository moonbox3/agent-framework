import asyncio
from typing import cast

from agent_framework import ConversationHandle, ConversationSnapshot, SequentialBuilder, WorkflowOutputEvent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


async def main() -> None:
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    writer = chat_client.create_agent(
        instructions="Concise marketing copy. Reply with a single line.",
        name="writer",
    )

    reviewer = chat_client.create_agent(
        instructions="Provide a short critique of the previous assistant message.",
        name="reviewer",
    )

    workflow = SequentialBuilder().participants([writer, reviewer]).build()

    initial_prompt = "Write a slogan for a solar-powered backpack."

    first_snapshot: ConversationSnapshot | None = None
    async for event in workflow.run_stream(initial_prompt):
        if isinstance(event, WorkflowOutputEvent):
            first_snapshot = cast(ConversationSnapshot, event.data)

    if first_snapshot is None:
        raise RuntimeError("Workflow produced no output snapshot")

    print("=== Initial Conversation ===")
    for message in first_snapshot.messages:
        name = message.author_name or message.role.value
        print(f"[{name}] {message.text}")

    handle = ConversationHandle(first_snapshot.handle.session_id, revision=first_snapshot.handle.revision)

    second_snapshot: ConversationSnapshot | None = None
    async for event in workflow.run_stream(handle):
        if isinstance(event, WorkflowOutputEvent):
            second_snapshot = cast(ConversationSnapshot, event.data)

    if second_snapshot is None:
        raise RuntimeError("Workflow did not resume")

    print("\n=== Resumed Conversation ===")
    for message in second_snapshot.messages:
        name = message.author_name or message.role.value
        print(f"[{name}] {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
