# Copyright (c) Microsoft. All rights reserved.

"""Test sample for validating MCP tool double-call bug fix.

This sample demonstrates the fix for the bug where calling an MCP tool (or any tool
with `approval_mode="always_require"`) twice would fail with:

    "An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'"

The bug was caused by:
1. Tool results from approved tools ending up in `role="user"` messages instead of `role="tool"`
2. Tool call IDs not being removed from pending tracking after seeing their results

To test this sample:
1. Set environment variables:
   - AZURE_OPENAI_ENDPOINT
   - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME

2. Run the server:
   python -m getting_started.test_mcp_double_call_fix

3. Use the AG-UI frontend or curl to test:
   - First request: "What time is it?" -> Should work, shows approval dialog
   - Approve the request
   - Second request: "What's the date?" -> Before fix: ERROR, After fix: Works!
"""

import logging
import os
from datetime import datetime

from agent_framework import ChatAgent, tool
from agent_framework.ag_ui import AgentFrameworkAgent, add_agent_framework_fastapi_endpoint
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Enable detailed logging to see the fix in action
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Also enable AG-UI specific logging
logging.getLogger("agent_framework_ag_ui").setLevel(logging.INFO)


# Tool with approval_mode="always_require" - simulates MCP tool behavior
@tool(
    name="get_datetime",
    description="Get the current date and time. Requires user approval before execution.",
    approval_mode="always_require",
)
def get_datetime() -> str:
    """Get the current date and time.

    This tool requires user approval before execution, similar to MCP tools.

    Returns:
        The current date and time as a formatted string.
    """
    now = datetime.now()
    result = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"get_datetime tool executed, returning: {result}")
    return result


# Another approval-required tool to test multiple different tools
@tool(
    name="get_system_info",
    description="Get system information. Requires user approval before execution.",
    approval_mode="always_require",
)
def get_system_info() -> str:
    """Get basic system information.

    This tool also requires user approval, to test calling different approval tools.

    Returns:
        Basic system information.
    """
    import platform

    info = f"Python {platform.python_version()} on {platform.system()} {platform.release()}"
    logger.info(f"get_system_info tool executed, returning: {info}")
    return info


def create_test_agent() -> ChatAgent:
    """Create the test agent with approval-required tools."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    if not deployment_name:
        raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME environment variable is required")

    return ChatAgent(
        name="MCPTestAgent",
        instructions="""You are a helpful assistant that can provide date/time and system information.

When the user asks about time, date, or "what time is it", use the get_datetime tool.
When the user asks about system info, use the get_system_info tool.

Important: These tools require user approval before execution. The user will see
an approval dialog and can choose to approve or reject the tool call.

After a tool is approved and executed, provide a brief, friendly response with the result.
""",
        chat_client=AzureOpenAIChatClient(
            endpoint=endpoint,
            deployment_name=deployment_name,
        ),
        tools=[get_datetime, get_system_info],
    )


# Create FastAPI app
app = FastAPI(
    title="MCP Double-Call Bug Test",
    description="Test server for validating the MCP tool double-call bug fix",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the agent
agent = create_test_agent()

# Register the AG-UI endpoint with require_confirmation=True (like the bug report)
add_agent_framework_fastapi_endpoint(
    app=app,
    agent=AgentFrameworkAgent(
        agent=agent,
        name="MCPTestAgent",
        description="Test agent for MCP double-call bug fix validation",
        require_confirmation=True,
    ),
    path="/",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "MCP double-call test server is running"}


def main():
    """Run the test server."""
    import uvicorn

    port = int(os.getenv("PORT", "8889"))
    host = os.getenv("HOST", "127.0.0.1")

    print("\n" + "=" * 70)
    print("MCP Double-Call Bug Fix Test Server")
    print("=" * 70)
    print(f"\nServer starting on http://{host}:{port}")
    print("\nTo test the fix:")
    print("1. Send: 'What time is it?' -> Approve -> Should work")
    print("2. Send: 'What time is it?' again -> Approve -> Should also work!")
    print("   (Before the fix, this second call would fail)")
    print("\nEndpoints:")
    print(f"  - AG-UI: POST http://{host}:{port}/")
    print(f"  - Health: GET http://{host}:{port}/health")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
