# Copyright (c) Microsoft. All rights reserved.

"""Demonstrates type adapters for bridging incompatible workflow types.

This sample shows how to use TypeAdapter classes to connect workflows
that have mismatched input/output types. It demonstrates:
1. Using built-in adapters (TextToConversation)
2. Using connect_checked() for type validation
3. Custom adapters via FunctionAdapter
"""

import asyncio
from typing import Any

from agent_framework import (
    ChatMessage,
    Executor,
    FunctionAdapter,
    Role,
    TextToConversation,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
)

# =============================================================================
# Text-based executors (work with str)
# =============================================================================


class TextNormalizer(Executor):
    """Normalizes text input: strips whitespace and lowercases."""

    @handler
    async def normalize(self, text: str, ctx: WorkflowContext[str]) -> None:
        normalized = text.strip().lower()
        await ctx.send_message(normalized)


class TextClassifier(Executor):
    """Classifies text and returns a label string."""

    @handler
    async def classify(self, text: str, ctx: WorkflowContext[str]) -> None:
        if "help" in text:
            label = "support_request"
        elif "buy" in text or "order" in text:
            label = "sales_inquiry"
        else:
            label = "general"
        await ctx.send_message(label)


# =============================================================================
# Conversation-based executors (work with list[ChatMessage])
# =============================================================================


class ConversationAnalyzer(Executor):
    """Analyzes a conversation and produces a summary."""

    @handler
    async def analyze(
        self,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[list[ChatMessage]],
    ) -> None:
        text_content = " ".join(msg.text or "" for msg in conversation)
        word_count = len(text_content.split())
        summary_text = f"Analysis: {word_count} words, {len(conversation)} messages"
        await ctx.send_message([ChatMessage(role=Role.ASSISTANT, text=summary_text)])


class ResponseGenerator(Executor):
    """Generates a response based on conversation context."""

    @handler
    async def generate(
        self,
        conversation: list[ChatMessage],
        ctx: WorkflowContext[list[ChatMessage], list[ChatMessage]],
    ) -> None:
        # Extract the last user message
        user_messages = [m for m in conversation if m.role == Role.USER]
        last_user = user_messages[-1].text if user_messages else "nothing"
        response = ChatMessage(
            role=Role.ASSISTANT,
            text=f"Thank you for your message about: {last_user}",
        )
        result = list(conversation) + [response]
        await ctx.yield_output(result)


async def main() -> None:
    """Demonstrate type adapters in workflow composition."""
    # Fragment 1: Text processing pipeline (str -> str)
    text_pipeline = (
        WorkflowBuilder(name="text_pipeline")
        .add_edge(TextNormalizer(id="normalize"), TextClassifier(id="classify"))
        .set_start_executor("normalize")
    )

    # Fragment 2: Conversation processing (list[ChatMessage] -> list[ChatMessage])
    conversation_pipeline = (
        WorkflowBuilder(name="conversation")
        .add_edge(ConversationAnalyzer(id="analyze"), ResponseGenerator(id="respond"))
        .set_start_executor("analyze")
    )

    # ==========================================================================
    # Option A: Manual adapter insertion via connect()
    # ==========================================================================
    print("Option A: Manual adapter insertion")
    print("-" * 40)

    builder_a = WorkflowBuilder()
    text_handle = builder_a.add_workflow(text_pipeline)
    conv_handle = builder_a.add_workflow(conversation_pipeline)

    # Text pipeline outputs str, but conversation pipeline expects list[ChatMessage]
    # We need to insert an adapter between them
    text_to_conv = TextToConversation(id="adapter", role=Role.USER)

    # Wire: text_pipeline -> adapter -> conversation_pipeline
    # connect() accepts Executor instances and adds them to the graph
    builder_a.connect(text_handle.outputs[0], text_to_conv)
    builder_a.connect(text_to_conv, conv_handle.start)
    builder_a.set_start_executor(text_handle.start)

    workflow_a = builder_a.build()
    async for event in workflow_a.run_stream("  HELP me with my ORDER  "):
        if isinstance(event, WorkflowOutputEvent) and event.data:
            for msg in event.data:
                print(f"  {msg.role.value}: {msg.text}")
    print()

    # ==========================================================================
    # Option B: Type-checked connection with adapter
    # ==========================================================================
    print("Option B: connect_checked() with adapter")
    print("-" * 40)

    builder_b = WorkflowBuilder()
    text_handle_b = builder_b.add_workflow(text_pipeline, prefix="txt")
    conv_handle_b = builder_b.add_workflow(conversation_pipeline, prefix="conv")

    # Use connect_checked to validate types and insert adapter in one step
    # Note: The adapter parameter automatically inserts the adapter between source and target
    try:
        # This would fail without adapter (type mismatch: str vs list[ChatMessage])
        builder_b.connect_checked(text_handle_b.outputs[0], conv_handle_b.start)
    except TypeError as e:
        print(f"  Expected error: {e}")
        print()

    # Now do it correctly with adapter
    builder_b.connect_checked(
        text_handle_b.outputs[0],
        conv_handle_b.start,
        adapter=TextToConversation(id="txt_to_conv"),
    )
    builder_b.set_start_executor(text_handle_b.start)

    workflow_b = builder_b.build()
    print("  Running with adapter:")
    async for event in workflow_b.run_stream("  BUY something please  "):
        if isinstance(event, WorkflowOutputEvent) and event.data:
            for msg in event.data:
                print(f"    {msg.role.value}: {msg.text}")
    print()

    # ==========================================================================
    # Option C: Custom adapter with FunctionAdapter
    # ==========================================================================
    print("Option C: Custom FunctionAdapter")
    print("-" * 40)

    # Create a custom adapter that adds metadata
    def custom_transform(text: str, ctx: Any) -> list[ChatMessage]:
        return [
            ChatMessage(role=Role.SYSTEM, text="[Processed by custom pipeline]"),
            ChatMessage(role=Role.USER, text=text),
        ]

    custom_adapter: FunctionAdapter[str, list[ChatMessage]] = FunctionAdapter(
        id="custom_adapter",
        fn=custom_transform,
        _input_type=str,
        _output_type=list[ChatMessage],  # type: ignore[arg-type]
        name="custom_text_to_conv",
    )

    builder_c = WorkflowBuilder()
    text_handle_c = builder_c.add_workflow(text_pipeline, prefix="txt")
    conv_handle_c = builder_c.add_workflow(conversation_pipeline, prefix="conv")

    # Wire using connect() with executor instances
    builder_c.connect(text_handle_c.outputs[0], custom_adapter)
    builder_c.connect(custom_adapter, conv_handle_c.start)
    builder_c.set_start_executor(text_handle_c.start)

    workflow_c = builder_c.build()
    print("  Running with custom adapter:")
    async for event in workflow_c.run_stream("  General inquiry about pricing  "):
        if isinstance(event, WorkflowOutputEvent) and event.data:
            for msg in event.data:
                print(f"    {msg.role.value}: {msg.text}")


if __name__ == "__main__":
    asyncio.run(main())
