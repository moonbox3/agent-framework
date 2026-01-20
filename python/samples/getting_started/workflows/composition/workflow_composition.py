# Copyright (c) Microsoft. All rights reserved.

"""
Sample: Workflow Composition with add_workflow()

What it does:
- Demonstrates composing high-level orchestration patterns (ConcurrentBuilder, SequentialBuilder)
  using WorkflowBuilder.add_workflow()
- Shows how to chain workflows together with add_edge() using logical workflow IDs
- Demonstrates adding pre/post-processing executors around composed workflows

This new API simplifies workflow composition by allowing you to:
1. Add orchestration patterns as logical units
2. Connect them using the workflow ID directly (no need to know internal executor names)
3. Mix and match high-level patterns with custom executors

Prerequisites:
- Azure OpenAI or OpenAI API credentials (set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY,
  or OPENAI_API_KEY)
"""

import asyncio
import os

from agent_framework import (
    ChatAgent,
    ConcurrentBuilder,
    Executor,
    SequentialBuilder,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from typing_extensions import Never


def create_chat_client():
    """Create a chat client based on available environment variables."""
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        from azure.identity import DefaultAzureCredential

        from agent_framework.azure_ai import AzureOpenAIChatClient

        return AzureOpenAIChatClient(
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            credential=DefaultAzureCredential(),
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        )
    elif os.environ.get("OPENAI_API_KEY"):
        from agent_framework.openai import OpenAIChatClient

        return OpenAIChatClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )
    else:
        raise ValueError(
            "Please set either AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, "
            "or OPENAI_API_KEY environment variables."
        )


# Custom preprocessor executor
class InputFormatter(Executor):
    """Formats the input message before sending to analysis."""

    def __init__(self):
        super().__init__(id="input_formatter")

    @handler
    async def format_input(self, message: str, ctx: WorkflowContext[str]) -> None:
        """Format the input message."""
        print(f"\n{'='*60}")
        print("INPUT FORMATTER")
        print(f"{'='*60}")
        print(f"Received: {message[:100]}...")
        formatted = f"Please analyze the following topic thoroughly:\n\n{message}"
        print("Formatted input for analysis agents.")
        await ctx.send_message(formatted)


# Custom postprocessor executor
class OutputFormatter(Executor):
    """Formats the final output from the workflow."""

    def __init__(self):
        super().__init__(id="output_formatter")

    @handler
    async def format_output(self, messages: list, ctx: WorkflowContext[Never, str]) -> None:
        """Format and yield the final output."""
        print(f"\n{'='*60}")
        print("OUTPUT FORMATTER")
        print(f"{'='*60}")

        # Extract text content from the messages
        output_parts = []
        for msg in messages:
            if hasattr(msg, "content"):
                output_parts.append(str(msg.content))
            elif hasattr(msg, "text"):
                output_parts.append(str(msg.text))
            else:
                output_parts.append(str(msg))

        final_output = "\n\n---\n\n".join(output_parts)
        print(f"Compiled {len(output_parts)} analysis sections into final report.")
        await ctx.yield_output(final_output)


async def example_simple_composition():
    """Example: Compose two high-level patterns together.

    This shows the simplest use case - chaining a ConcurrentBuilder (parallel analysis)
    with a SequentialBuilder (summarization).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Workflow Composition")
    print("=" * 80)
    print("Chaining ConcurrentBuilder -> SequentialBuilder")

    client = create_chat_client()

    # Create agents for parallel analysis
    technical_analyst = ChatAgent(
        name="technical_analyst",
        chat_client=client,
        instructions="You are a technical analyst. Analyze the technical aspects of the topic in 2-3 sentences.",
    )
    business_analyst = ChatAgent(
        name="business_analyst",
        chat_client=client,
        instructions="You are a business analyst. Analyze the business implications in 2-3 sentences.",
    )

    # Create agent for summarization
    summarizer = ChatAgent(
        name="summarizer",
        chat_client=client,
        instructions="You are a summarizer. Combine the analyses into a brief executive summary in 2-3 sentences.",
    )

    # Create high-level orchestration patterns
    analysis = ConcurrentBuilder().participants([technical_analyst, business_analyst])
    summary = SequentialBuilder().participants([summarizer])

    # Compose them together using add_workflow()
    workflow = (
        WorkflowBuilder()
        .add_workflow(analysis, id="analysis")
        .add_workflow(summary, id="summary")
        .add_edge("analysis", "summary")  # Framework resolves to analysis/aggregator -> summary/input-conversation
        .set_start_executor("analysis")  # Framework knows this means analysis/dispatcher
        .build()
    )

    print("\nWorkflow structure:")
    print(f"  Start executor: {workflow.start_executor_id}")
    print(f"  Executors: {list(workflow.executors.keys())}")

    # Run the workflow
    result = await workflow.run("Artificial Intelligence in Healthcare")
    outputs = result.get_outputs()

    print("\n--- Final Output ---")
    for output in outputs:
        if hasattr(output, "__iter__") and not isinstance(output, str):
            for item in output:
                print(item.content if hasattr(item, "content") else item)
        else:
            print(output)


async def example_pre_post_processing():
    """Example: Add pre/post-processing around a high-level pattern.

    This shows how to wrap a ConcurrentBuilder with custom preprocessing
    and postprocessing executors.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Pre/Post Processing with Composed Workflows")
    print("=" * 80)
    print("InputFormatter -> ConcurrentBuilder -> OutputFormatter")

    client = create_chat_client()

    # Create agents for analysis
    technical_analyst = ChatAgent(
        name="technical_analyst",
        chat_client=client,
        instructions="You are a technical analyst. Provide a technical analysis in 2-3 sentences.",
    )
    market_analyst = ChatAgent(
        name="market_analyst",
        chat_client=client,
        instructions="You are a market analyst. Provide market analysis in 2-3 sentences.",
    )

    # Create the concurrent analysis pattern
    analysis = ConcurrentBuilder().participants([technical_analyst, market_analyst])

    # Compose with pre/post-processing
    workflow = (
        WorkflowBuilder()
        .register_executor(InputFormatter, name="input_formatter")
        .add_workflow(analysis, id="analysis")
        .register_executor(OutputFormatter, name="output_formatter")
        .add_edge("input_formatter", "analysis")
        .add_edge("analysis", "output_formatter")
        .set_start_executor("input_formatter")
        .build()
    )

    print("\nWorkflow structure:")
    print(f"  Start executor: {workflow.start_executor_id}")
    print(f"  Executors: {list(workflow.executors.keys())}")

    # Run the workflow
    result = await workflow.run("Electric Vehicle Adoption")
    outputs = result.get_outputs()

    print("\n--- Final Output ---")
    for output in outputs:
        print(output)


async def main():
    """Run all composition examples."""
    print("=" * 80)
    print("WORKFLOW COMPOSITION SAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the new add_workflow() API for composing")
    print("high-level orchestration patterns (ConcurrentBuilder, SequentialBuilder)")
    print("with WorkflowBuilder.\n")

    # Run examples
    await example_simple_composition()
    await example_pre_post_processing()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
