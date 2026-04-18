# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from typing import Any

from agent_framework import Agent, AgentSession, ContextProvider, SessionContext
from agent_framework.foundry import FoundryChatClient, select_toolbox_tools
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Foundry Toolbox + Context Provider Example

This sample composes a Foundry toolbox with a ContextProvider so the agent's
tool list is chosen dynamically per-turn based on the latest user message. The
toolbox is fetched once on the first invocation and cached on the provider's
state dict; subsequent turns reuse the cache and apply message-driven filtering
through ``select_toolbox_tools``.

Prerequisites:
- A Microsoft Foundry project
- A toolbox already configured in that project (set TOOLBOX_NAME below)
- FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL environment variables set
- Azure CLI authentication (`az login`)
"""

# Replace with your own Foundry toolbox name and version.
TOOLBOX_NAME = "<your-toolbox-name>"
# Set to None to resolve the toolbox's current default version at fetch time.
TOOLBOX_VERSION: str | None = "<your-toolbox-version>"

# Simple keyword → toolbox-tool-type mapping for message-driven selection.
# Extend or swap this map to match the tools actually configured in your toolbox.
TOOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "code_interpreter": ("code", "compute", "calculate", "python", "fibonacci"),
    "web_search": ("search", "web", "look up", "news", "latest"),
    "file_search": ("file", "document", "doc", "knowledge base"),
    "image_generation": ("image", "picture", "draw", "render"),
    "mcp": ("mcp",),
}


class DynamicToolboxProvider(ContextProvider):
    """Fetches a Foundry toolbox once and picks tools per-turn from the user message."""

    DEFAULT_SOURCE_ID = "foundry_toolbox"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        *,
        client: FoundryChatClient,
        toolbox_name: str,
        toolbox_version: str | None = None,
    ) -> None:
        super().__init__(source_id)
        self._client = client
        self._toolbox_name = toolbox_name
        self._toolbox_version = toolbox_version

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Cache the toolbox on first call, then pick tools by latest user message."""
        toolbox = state.get("toolbox")
        if toolbox is None:
            toolbox = await self._client.get_toolbox(
                self._toolbox_name, version=self._toolbox_version
            )
            state["toolbox"] = toolbox
            print(
                f"[{self.source_id}] Loaded toolbox "
                f"{toolbox.name}@{toolbox.version} ({len(toolbox.tools)} tool(s))"
            )

        user_messages = [
            m for m in context.get_messages(include_input=True)
            if getattr(m, "role", None) == "user"
        ]
        latest_text = user_messages[-1].text.lower() if user_messages else ""

        include_types = self._match_tool_types(latest_text)
        if include_types:
            tools = select_toolbox_tools(toolbox, include_types=list(include_types))
            print(
                f"[{self.source_id}] Turn picks types {sorted(include_types)} — "
                f"surfacing {len(tools)} tool(s)"
            )
        else:
            tools = list(toolbox.tools)
            print(
                f"[{self.source_id}] No keyword match — surfacing all {len(tools)} toolbox tool(s)"
            )

        context.extend_tools(self.source_id, tools)

    @staticmethod
    def _match_tool_types(text: str) -> set[str]:
        matched: set[str] = set()
        for tool_type, keywords in TOOL_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                matched.add(tool_type)
        return matched


async def main() -> None:
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["FOUNDRY_MODEL"],
        credential=AzureCliCredential(),
    )

    toolbox_provider = DynamicToolboxProvider(
        client=client,
        toolbox_name=TOOLBOX_NAME,
        toolbox_version=TOOLBOX_VERSION,
    )

    async with Agent(
        client=client,
        instructions=(
            "You are a helpful assistant. Use the tools available to you on each "
            "turn to answer the user. If no tools are relevant, reply directly."
        ),
        context_providers=[toolbox_provider],
    ) as agent:
        session = agent.create_session()

        # Each query is shaped to trigger a different slice of the toolbox via
        # the keyword-driven selector above.
        queries = [
            "Please search the web for the latest news about quantum computing.",
            "Use Python to calculate the 20th Fibonacci number.",
            "Briefly summarize what you can help me with right now.",
        ]
        for query in queries:
            print(f"\nUser: {query}")
            result = await agent.run(query, session=session)
            print(f"Assistant: {result}")


if __name__ == "__main__":
    asyncio.run(main())
