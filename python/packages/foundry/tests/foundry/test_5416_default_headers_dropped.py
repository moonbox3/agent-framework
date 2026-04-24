# Copyright (c) Microsoft. All rights reserved.
# Regression tests for issue #5416:
# FoundryChatClient and FoundryAgent silently drop default_headers
# (get_openai_client is called without forwarding the headers).

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from agent_framework_foundry import FoundryChatClient
from agent_framework_foundry._agent import RawFoundryAgentChatClient

_TEST_ENDPOINT = "https://test-project.services.ai.azure.com/"
_TEST_MODEL = "test-gpt-4o"


def _make_mock_openai_client() -> MagicMock:
    client = MagicMock()
    client.default_headers = {}
    client._custom_headers = {}
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    client.responses.parse = AsyncMock()
    return client


def _make_mock_project_client() -> MagicMock:
    project_client = MagicMock()
    project_client.get_openai_client.return_value = _make_mock_openai_client()
    return project_client


def test_foundry_chat_client_default_headers_forwarded_to_get_openai_client() -> None:
    """default_headers must be forwarded into get_openai_client() so the
    underlying AsyncOpenAI client actually sends them on every request.

    Regression for #5416: get_openai_client() was called with no arguments, so
    headers were only stored on self.default_headers (a dead field) and never
    reached the HTTP layer.
    """
    custom_headers = {"x-custom-header": "repro-value"}
    project_client = _make_mock_project_client()

    FoundryChatClient(
        project_client=project_client,
        model=_TEST_MODEL,
        default_headers=custom_headers,
    )

    project_client.get_openai_client.assert_called_once_with(default_headers=custom_headers)


def test_foundry_chat_client_no_default_headers_calls_get_openai_client_without_headers() -> None:
    """When no default_headers are provided, get_openai_client() is called with None."""
    project_client = _make_mock_project_client()

    FoundryChatClient(
        project_client=project_client,
        model=_TEST_MODEL,
    )

    project_client.get_openai_client.assert_called_once_with(default_headers=None)


def test_foundry_agent_chat_client_default_headers_forwarded_to_get_openai_client() -> None:
    """default_headers must be forwarded into get_openai_client() for FoundryAgent.

    Regression for #5416: same bug as FoundryChatClient, in _agent.py.
    """
    custom_headers = {"x-custom-header": "repro-value"}
    project_client = _make_mock_project_client()

    RawFoundryAgentChatClient(
        project_client=project_client,
        agent_name="test-agent",
        default_headers=custom_headers,
    )

    project_client.get_openai_client.assert_called_once_with(default_headers=custom_headers)


def test_foundry_agent_chat_client_no_default_headers_calls_get_openai_client_without_headers() -> None:
    """When no default_headers are provided, get_openai_client() is called with None."""
    project_client = _make_mock_project_client()

    RawFoundryAgentChatClient(
        project_client=project_client,
        agent_name="test-agent",
    )

    project_client.get_openai_client.assert_called_once_with(default_headers=None)
