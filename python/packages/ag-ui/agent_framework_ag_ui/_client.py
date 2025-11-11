# Copyright (c) Microsoft. All rights reserved.

"""AG-UI Chat Client implementation."""

import json
import uuid
from collections.abc import AsyncIterable, MutableSequence
from typing import Any

import httpx
from agent_framework import BaseChatClient, ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, DataContent
from agent_framework._middleware import use_chat_middleware
from agent_framework._tools import use_function_invocation
from agent_framework.observability import use_observability

from ._event_converters import AGUIEventConverter
from ._http_service import AGUIHttpService
from ._utils import convert_tools_to_agui_format


@use_function_invocation
@use_observability
@use_chat_middleware
class AGUIChatClient(BaseChatClient):
    """Chat client for communicating with AG-UI compliant servers.

    This client implements the BaseChatClient interface and automatically handles:
    - Thread ID management for conversation continuity
    - State synchronization between client and server
    - Server-Sent Events (SSE) streaming
    - Event conversion to Agent Framework types

    Important: Message History Management
        This client sends exactly the messages it receives to the server. It does NOT
        automatically maintain conversation history. The server must handle history via thread_id.

        For stateless servers: Use ChatAgent wrapper which will send full message history on each
        request. However, even with ChatAgent, the server must echo back all context for the
        agent to maintain history across turns.

    Important: Tool Handling
        This client has @use_function_invocation decorator, enabling HYBRID tool execution:

        1. Client tool metadata (name, description, schema) is sent to server for planning
        2. Server can have additional tools that execute server-side
        3. When server requests a client tool, @use_function_invocation intercepts and executes locally
        4. Both client and server tools work together simultaneously (hybrid pattern)

        This matches the .NET AG-UI implementation behavior.

    Examples:
        Direct usage (server manages thread history):

        .. code-block:: python

            from agent_framework_ag_ui import AGUIChatClient

            client = AGUIChatClient(endpoint="http://localhost:8888/")

            # First message - thread ID auto-generated
            response = await client.get_response("Hello!")
            thread_id = response.additional_properties.get("thread_id")

            # Second message - server retrieves history using thread_id
            response2 = await client.get_response(
                "How are you?",
                metadata={"thread_id": thread_id}
            )

        Recommended usage with ChatAgent (client manages history):

        .. code-block:: python

            from agent_framework import ChatAgent
            from agent_framework_ag_ui import AGUIChatClient

            client = AGUIChatClient(endpoint="http://localhost:8888/")
            agent = ChatAgent(name="assistant", client=client)
            thread = await agent.get_new_thread()

            # ChatAgent automatically maintains history and sends full context
            response = await agent.run("Hello!", thread=thread)
            response2 = await agent.run("How are you?", thread=thread)

        Streaming usage:

        .. code-block:: python

            async for update in client.get_streaming_response("Tell me a story"):
                if update.contents:
                    for content in update.contents:
                        if hasattr(content, "text"):
                            print(content.text, end="", flush=True)

        Context manager:

        .. code-block:: python

            async with AGUIChatClient(endpoint="http://localhost:8888/") as client:
                response = await client.get_response("Hello!")
                print(response.messages[0].text)
    """

    OTEL_PROVIDER_NAME = "agui"

    def __init__(
        self,
        *,
        endpoint: str,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        additional_properties: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AG-UI chat client.

        Args:
            endpoint: The AG-UI server endpoint URL (e.g., "http://localhost:8888/")
            http_client: Optional httpx.AsyncClient instance. If None, one will be created.
            timeout: Request timeout in seconds (default: 60.0)
            additional_properties: Additional properties to store
            **kwargs: Additional arguments passed to BaseChatClient
        """
        super().__init__(additional_properties=additional_properties, **kwargs)
        self._http_service = AGUIHttpService(
            endpoint=endpoint,
            http_client=http_client,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http_service.close()

    async def __aenter__(self) -> "AGUIChatClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        await self.close()

    def _extract_state_from_messages(
        self, messages: MutableSequence[ChatMessage]
    ) -> tuple[list[ChatMessage], dict[str, Any] | None]:
        """Extract state from last message if present.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (messages_without_state, state_dict)
        """
        if not messages:
            return list(messages), None

        last_message = messages[-1]

        for content in last_message.contents:
            if isinstance(content, DataContent) and content.media_type == "application/json":
                try:
                    uri = content.uri
                    if uri.startswith("data:application/json;base64,"):
                        import base64

                        encoded_data = uri.split(",", 1)[1]
                        decoded_bytes = base64.b64decode(encoded_data)
                        state = json.loads(decoded_bytes.decode("utf-8"))

                        messages_without_state = list(messages[:-1]) if len(messages) > 1 else []
                        return messages_without_state, state
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    from agent_framework._logging import get_logger

                    logger = get_logger()
                    logger.warning(f"Failed to extract state from message: {e}")

        return list(messages), None

    def _convert_messages_to_agui_format(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert Agent Framework messages to AG-UI format.

        Args:
            messages: List of ChatMessage objects

        Returns:
            List of AG-UI formatted message dictionaries
        """
        agui_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

            message_dict: dict[str, Any] = {
                "role": role,
                "content": msg.text,
            }

            if msg.message_id:
                message_dict["id"] = msg.message_id

            agui_messages.append(message_dict)

        return agui_messages

    def _get_thread_id(self, chat_options: ChatOptions) -> str:
        """Get or generate thread ID from chat options.

        Args:
            chat_options: Chat options containing metadata

        Returns:
            Thread ID string
        """
        thread_id = None
        if chat_options.metadata:
            thread_id = chat_options.metadata.get("thread_id")

        if not thread_id:
            thread_id = f"thread_{uuid.uuid4().hex}"

        return thread_id

    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        """Internal method to get non-streaming response.

        Keyword Args:
            messages: List of chat messages
            chat_options: Chat options for the request
            **kwargs: Additional keyword arguments

        Returns:
            ChatResponse object
        """
        updates: list[ChatResponseUpdate] = []

        async for update in self._inner_get_streaming_response(
            messages=messages,
            chat_options=chat_options,
            **kwargs,
        ):
            updates.append(update)

        return ChatResponse.from_chat_response_updates(updates)

    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """Internal method to get streaming response.

        Keyword Args:
            messages: List of chat messages
            chat_options: Chat options for the request
            **kwargs: Additional keyword arguments

        Yields:
            ChatResponseUpdate objects
        """
        messages_to_send, state = self._extract_state_from_messages(messages)

        thread_id = self._get_thread_id(chat_options)
        run_id = f"run_{uuid.uuid4().hex}"

        agui_messages = self._convert_messages_to_agui_format(messages_to_send)

        # Convert client tools to AG-UI format
        # Send tool metadata (name, description, JSON schema) to server for planning
        # The @use_function_invocation decorator handles client-side execution when
        # the server requests a function. This matches .NET implementation behavior.
        agui_tools = convert_tools_to_agui_format(chat_options.tools)

        converter = AGUIEventConverter()

        async for event in self._http_service.post_run(
            thread_id=thread_id,
            run_id=run_id,
            messages=agui_messages,
            state=state,
            tools=agui_tools,
        ):
            update = converter.convert_event(event)
            if update is not None:
                yield update
