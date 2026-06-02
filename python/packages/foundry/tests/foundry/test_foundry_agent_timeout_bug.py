# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for #6241: FoundryAgent ConnectTimeout on multi-turn conversations.

Root cause: RawFoundryAgentChatClient called project_client.get_openai_client() without
applying any timeout, so the openai SDK default of httpx.Timeout(connect=5.0) was used.
Under load or when connections are recycled between turns, the 5s connect timeout fires.

Fix: expose a ``timeout`` parameter on RawFoundryAgentChatClient, _FoundryAgentChatClient,
RawFoundryAgent, FoundryAgent, and RawOpenAIChatClient that is applied to the underlying
AsyncOpenAI client.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
from agent_framework_openai._chat_client import RawOpenAIChatClient
from openai import AsyncOpenAI

from agent_framework_foundry._agent import (
    FoundryAgent,
    RawFoundryAgent,
    RawFoundryAgentChatClient,
    _FoundryAgentChatClient,
)

_FOUNDRY_AGENT_ENV_VARS = (
    "FOUNDRY_PROJECT_ENDPOINT",
    "FOUNDRY_AGENT_NAME",
    "FOUNDRY_AGENT_VERSION",
)


@pytest.fixture(autouse=True)
def clear_foundry_agent_settings_env(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Prevent unit tests from inheriting Foundry agent settings from the shell."""

    if request.node.get_closest_marker("integration") is not None:
        return

    for env_var in _FOUNDRY_AGENT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _make_mock_project() -> MagicMock:
    mock_openai_client = MagicMock(spec=AsyncOpenAI)
    mock_openai_client.timeout = 5.0
    mock_project = MagicMock()
    mock_project.get_openai_client.return_value = mock_openai_client
    return mock_project


# ---------------------------------------------------------------------------
# RawFoundryAgentChatClient.timeout
# ---------------------------------------------------------------------------


def test_raw_foundry_agent_chat_client_has_timeout_parameter() -> None:
    """timeout is an explicit keyword-only parameter on RawFoundryAgentChatClient."""

    sig = inspect.signature(RawFoundryAgentChatClient.__init__)
    assert "timeout" in sig.parameters
    assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_raw_foundry_agent_chat_client_timeout_none_leaves_client_unchanged() -> None:
    """When timeout is None, the openai client timeout is not modified."""

    mock_project = _make_mock_project()
    mock_project.get_openai_client.return_value.timeout = 5.0

    RawFoundryAgentChatClient(
        project_client=mock_project,
        agent_name="test-agent",
        timeout=None,
    )

    assert mock_project.get_openai_client.return_value.timeout == 5.0


def test_raw_foundry_agent_chat_client_timeout_is_applied_to_openai_client() -> None:
    """When timeout is specified, it is set on the underlying AsyncOpenAI client."""

    mock_project = _make_mock_project()
    openai_client_mock = mock_project.get_openai_client.return_value

    RawFoundryAgentChatClient(
        project_client=mock_project,
        agent_name="test-agent",
        timeout=60.0,
    )

    assert openai_client_mock.timeout == 60.0


def test_raw_foundry_agent_chat_client_timeout_applied_with_preview_enabled() -> None:
    """Timeout is applied even when allow_preview=True (hosted agent path)."""

    mock_project = _make_mock_project()
    openai_client_mock = mock_project.get_openai_client.return_value

    RawFoundryAgentChatClient(
        project_client=mock_project,
        agent_name="hosted-agent",
        allow_preview=True,
        timeout=120.0,
    )

    assert openai_client_mock.timeout == 120.0


# ---------------------------------------------------------------------------
# _FoundryAgentChatClient.timeout
# ---------------------------------------------------------------------------


def test_foundry_agent_chat_client_has_timeout_parameter() -> None:
    """timeout is an explicit keyword-only parameter on _FoundryAgentChatClient."""

    sig = inspect.signature(_FoundryAgentChatClient.__init__)
    assert "timeout" in sig.parameters
    assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_foundry_agent_chat_client_timeout_propagated_to_raw_client() -> None:
    """_FoundryAgentChatClient passes timeout down to RawFoundryAgentChatClient."""

    mock_project = _make_mock_project()
    openai_client_mock = mock_project.get_openai_client.return_value

    _FoundryAgentChatClient(
        project_client=mock_project,
        agent_name="test-agent",
        timeout=45.0,
    )

    assert openai_client_mock.timeout == 45.0


# ---------------------------------------------------------------------------
# RawFoundryAgent / FoundryAgent.timeout
# ---------------------------------------------------------------------------


def test_raw_foundry_agent_has_timeout_parameter() -> None:
    """timeout is an explicit keyword-only parameter on RawFoundryAgent."""

    sig = inspect.signature(RawFoundryAgent.__init__)
    assert "timeout" in sig.parameters


def test_foundry_agent_has_timeout_parameter() -> None:
    """timeout is an explicit keyword-only parameter on FoundryAgent."""

    sig = inspect.signature(FoundryAgent.__init__)
    assert "timeout" in sig.parameters


def test_foundry_agent_timeout_propagated_to_openai_client() -> None:
    """FoundryAgent passes timeout all the way to the underlying AsyncOpenAI client."""

    mock_project = _make_mock_project()
    openai_client_mock = mock_project.get_openai_client.return_value

    FoundryAgent(
        project_client=mock_project,
        agent_name="test-agent",
        timeout=90.0,
    )

    assert openai_client_mock.timeout == 90.0


def test_foundry_agent_timeout_none_does_not_alter_default() -> None:
    """FoundryAgent with timeout=None leaves the openai client timeout at its default."""

    mock_project = _make_mock_project()
    openai_client_mock = mock_project.get_openai_client.return_value
    original_timeout = openai_client_mock.timeout

    FoundryAgent(
        project_client=mock_project,
        agent_name="test-agent",
        timeout=None,
    )

    assert openai_client_mock.timeout == original_timeout


# ---------------------------------------------------------------------------
# RawOpenAIChatClient.timeout
# ---------------------------------------------------------------------------


def test_raw_openai_chat_client_has_timeout_parameter() -> None:
    """timeout is an explicit keyword-only parameter on RawOpenAIChatClient."""

    sig = inspect.signature(RawOpenAIChatClient.__init__)
    assert "timeout" in sig.parameters
    assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_raw_openai_chat_client_accepts_timeout_with_preconfigured_client() -> None:
    """timeout parameter is accepted without error when async_client is pre-provided."""

    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.timeout = 5.0

    client = RawOpenAIChatClient(async_client=mock_client, timeout=30.0)
    assert client is not None
