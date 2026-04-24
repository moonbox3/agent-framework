# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from azure.core.credentials import TokenCredential
from azure.core.credentials_async import AsyncTokenCredential

from agent_framework_openai._shared import (
    AZURE_OPENAI_TOKEN_SCOPE,
    _ensure_async_token_provider,
    _resolve_azure_credential_to_token_provider,
    load_openai_service_settings,
)


class _AsyncTokenCredentialStub(AsyncTokenCredential):
    async def get_token(self, *scopes: str, **kwargs: object):
        raise NotImplementedError


class _TokenCredentialStub(TokenCredential):
    def get_token(self, *scopes: str, **kwargs: object):
        raise NotImplementedError


def test_resolve_azure_async_credential_wraps_provider() -> None:
    credential = _AsyncTokenCredentialStub()
    token_provider = MagicMock()

    with patch("azure.identity.aio.get_bearer_token_provider", return_value=token_provider) as mock_provider:
        resolved = _resolve_azure_credential_to_token_provider(credential)

    assert resolved is token_provider
    mock_provider.assert_called_once_with(credential, AZURE_OPENAI_TOKEN_SCOPE)


def test_resolve_azure_sync_credential_wraps_provider() -> None:
    credential = _TokenCredentialStub()
    token_provider = MagicMock()

    with patch("azure.identity.get_bearer_token_provider", return_value=token_provider) as mock_provider:
        resolved = _resolve_azure_credential_to_token_provider(credential)

    assert resolved is token_provider
    mock_provider.assert_called_once_with(credential, AZURE_OPENAI_TOKEN_SCOPE)


def test_resolve_azure_callable_token_provider_passthrough() -> None:
    token_provider = MagicMock()

    assert _resolve_azure_credential_to_token_provider(token_provider) is token_provider


def test_resolve_azure_invalid_credential_raises() -> None:
    with pytest.raises(ValueError, match="credential"):
        _resolve_azure_credential_to_token_provider(object())  # type: ignore[arg-type]


async def test_ensure_async_token_provider_wraps_sync_provider() -> None:
    def sync_provider() -> str:
        return "sync-token"

    wrapper = _ensure_async_token_provider(sync_provider)
    result = await wrapper()

    assert result == "sync-token"


async def test_ensure_async_token_provider_wraps_async_provider() -> None:
    async def async_provider() -> str:
        return "async-token"

    wrapper = _ensure_async_token_provider(async_provider)
    result = await wrapper()

    assert result == "async-token"


def test_load_openai_service_settings_applies_default_headers_to_prebuilt_client() -> None:
    """When a pre-built client is provided, default_headers must be applied to it.

    Regression for #5416: load_openai_service_settings used to early-return the
    pre-built client without applying merged_headers, silently dropping any
    custom headers the caller passed.
    """
    pre_built = MagicMock()
    new_client = MagicMock()
    pre_built.with_options.return_value = new_client

    _, client, _ = load_openai_service_settings(
        model="gpt-4o",
        api_key=None,
        credential=None,
        org_id=None,
        base_url=None,
        endpoint=None,
        api_version=None,
        default_azure_api_version="2024-05-01-preview",
        default_headers={"x-custom-header": "test-value"},
        client=pre_built,
        env_file_path=None,
        env_file_encoding=None,
    )

    pre_built.with_options.assert_called_once()
    call_kwargs = pre_built.with_options.call_args.kwargs
    assert call_kwargs.get("default_headers", {}).get("x-custom-header") == "test-value"
    assert client is new_client


def test_load_openai_service_settings_no_headers_still_applies_app_info() -> None:
    """Even with no default_headers, APP_INFO telemetry headers are applied via with_options."""
    pre_built = MagicMock()
    new_client = MagicMock()
    pre_built.with_options.return_value = new_client

    _, client, _ = load_openai_service_settings(
        model="gpt-4o",
        api_key=None,
        credential=None,
        org_id=None,
        base_url=None,
        endpoint=None,
        api_version=None,
        default_azure_api_version="2024-05-01-preview",
        default_headers=None,
        client=pre_built,
        env_file_path=None,
        env_file_encoding=None,
    )

    pre_built.with_options.assert_called_once()
    assert client is new_client
