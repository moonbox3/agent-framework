# Copyright (c) Microsoft. All rights reserved.

import os

from agent_framework import WorkflowBuilder
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.devui import serve
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

"""
Sample: Agents in a workflow with streaming

A Writer agent generates content, then a Reviewer agent critiques it.
The workflow uses streaming so you can observe incremental AgentRunUpdateEvent chunks as each agent produces tokens.

Purpose:
Show how to wire chat agents directly into a WorkflowBuilder pipeline where agents are auto wrapped as executors.

Demonstrate:
- Automatic streaming of agent deltas via AgentRunUpdateEvent.
- A simple console aggregator that groups updates by executor id and prints them as they arrive.
- The workflow completes when idle and outputs are available in events.get_outputs().

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run az login before executing the sample.
- Basic familiarity with WorkflowBuilder, edges, events, and streaming runs.
"""

load_dotenv(override=True)

"""Build and run a simple two node agent workflow: Writer then Reviewer."""
# Create the Azure chat client. AzureCliCredential uses your current az login.
chat_client = AzureOpenAIChatClient(
    credential=DefaultAzureCredential(),
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.environ.get("AZURE_OPENAI_VERSION"),
)
# Define two domain specific chat agents. The builder will wrap these as executors.
writer_agent = chat_client.create_agent(
    instructions=(
        "You are an excellent content writer. You create new content and edit contents based on the feedback."
    ),
    name="writer_agent",
)

reviewer_agent = chat_client.create_agent(
    instructions=(
        "You are an excellent content reviewer."
        "Provide actionable feedback to the writer about the provided content."
        "Provide the feedback in the most concise manner possible."
    ),
    name="reviewer_agent",
)

# Build the workflow using the fluent builder.
# Set the start node and connect an edge from writer to reviewer.
workflow = WorkflowBuilder().set_start_executor(writer_agent).add_edge(writer_agent, reviewer_agent).build()
serve(entities=[workflow], auto_open=True, tracing_enabled=True)
