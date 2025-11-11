# Copyright (c) Microsoft. All rights reserved.

"""AG-UI server example with server-side tools."""

import os

from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv
from fastapi import FastAPI

from agent_framework_ag_ui import add_agent_framework_fastapi_endpoint

load_dotenv()

# Read required configuration
endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
if not deployment_name:
    raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME environment variable is required")


# Server-side tool (executes on server)
@ai_function
def get_time_zone(location: str) -> str:
    """Get the time zone for a location.

    Args:
        location: The city or location name
    """
    print(f"[SERVER] get_time_zone tool called with location: {location}")
    timezone_data = {
        "seattle": "Pacific Time (UTC-8)",
        "san francisco": "Pacific Time (UTC-8)",
        "new york": "Eastern Time (UTC-5)",
        "london": "Greenwich Mean Time (UTC+0)",
    }
    result = timezone_data.get(location.lower(), f"Time zone data not available for {location}")
    print(f"[SERVER] get_time_zone returning: {result}")
    return result


# Server-side weather tool (for fallback if client doesn't handle it)
@ai_function
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location name
    """
    print(f"[SERVER] get_weather tool called with location: {location}")
    weather_data = {
        "seattle": "Rainy, 55°F (SERVER DATA)",
        "san francisco": "Foggy, 62°F (SERVER DATA)",
        "new york": "Sunny, 68°F (SERVER DATA)",
        "london": "Cloudy, 52°F (SERVER DATA)",
    }
    result = weather_data.get(location.lower(), f"Weather data not available for {location}")
    print(f"[SERVER] get_weather returning: {result}")
    return result


# Create the AI agent with server-side tools
# Note: Client will also send its own tools which could execute client-side
# When both client and server have the same tool, the client-side one should execute
# (via @use_function_invocation decorator on AGUIChatClient)
agent = ChatAgent(
    name="AGUIAssistant",
    instructions="You are a helpful assistant. Use get_weather for weather and get_time_zone for time zones.",
    chat_client=AzureOpenAIChatClient(
        endpoint=endpoint,
        deployment_name=deployment_name,
    ),
    tools=[get_weather, get_time_zone],
)

# Create FastAPI app
app = FastAPI(title="AG-UI Server")

# Register the AG-UI endpoint
add_agent_framework_fastapi_endpoint(app, agent, "/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5100)
