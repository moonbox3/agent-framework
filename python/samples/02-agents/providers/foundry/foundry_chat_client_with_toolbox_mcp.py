# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from collections.abc import Callable
from typing import Any

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.foundry import FoundryChatClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Foundry Toolbox via MAF ``MCPStreamableHTTPTool``

Instead of fetching the toolbox and fanning out individual tool specs, point
MAF's ``MCPStreamableHTTPTool`` at the toolbox's MCP endpoint. The agent
discovers and calls the toolbox's tools over MCP at runtime.

Prerequisites:
- A Microsoft Foundry project with a toolbox configured
- FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL environment variables set
- FOUNDRY_TOOLBOX_ENDPOINT: the toolbox's MCP endpoint URL, e.g.
  ``https://<account>.services.ai.azure.com/api/projects/<project>/toolsets/<name>/mcp?api-version=v1``
- Azure CLI authentication (``az login``)
"""


def make_toolbox_header_provider(credential: TokenCredential) -> Callable[[dict[str, Any]], dict[str, str]]:
    """Build a header_provider that injects a fresh Azure AI bearer token on every MCP request."""
    get_token = get_bearer_token_provider(credential, "https://ai.azure.com/.default")

    def provide(_kwargs: dict[str, Any]) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {get_token()}",
        }

    return provide


async def main() -> None:
    credential = DefaultAzureCredential()

    toolbox_tool = MCPStreamableHTTPTool(
        name="foundry_toolbox",
        description="Tools exposed by the configured Foundry toolbox",
        url=os.environ["FOUNDRY_TOOLBOX_ENDPOINT"],
        header_provider=make_toolbox_header_provider(credential),
        load_prompts=False,
    )

    async with Agent(
        client=FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["FOUNDRY_MODEL"],
            credential=credential,
        ),
        instructions="You are a helpful assistant. Use the available toolbox tools to answer the user.",
        tools=toolbox_tool,
    ) as agent:
        query = "What tools do you have access to?"
        print(f"User: {query}")
        result = await agent.run(query)
        print(f"Assistant: {result}")


if __name__ == "__main__":
    asyncio.run(main())
