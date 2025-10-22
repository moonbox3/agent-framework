# Copyright (c) Microsoft. All rights reserved.

"""Helpers for managing chat conversation history.

These utilities operate on standard `list[ChatMessage]` collections and simple
dictionary snapshots so orchestrators can share logic without new mixins.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .._types import ChatMessage, Role


def clone_conversation(messages: Iterable[ChatMessage]) -> list[ChatMessage]:
    """Return a shallow copy of `messages` as a list."""
    return list(messages)


def append_messages(conversation: list[ChatMessage], messages: Iterable[ChatMessage]) -> None:
    """Extend `conversation` with `messages` in order."""
    conversation.extend(messages)


def latest_user_message(conversation: Sequence[ChatMessage]) -> ChatMessage:
    """Return the most recent user-authored message from `conversation`."""
    for message in reversed(conversation):
        role_value = getattr(message.role, "value", message.role)
        if str(role_value).lower() == "user":
            return message
    if not conversation:
        raise ValueError("Conversation is empty; cannot determine user message.")
    return conversation[-1]


def ensure_author(message: ChatMessage, fallback: str) -> ChatMessage:
    """Attach `fallback` author if message is missing `author_name`."""
    author = getattr(message, "author_name", None)
    if author:
        return message
    if hasattr(message, "to_dict") and callable(message.to_dict):  # type: ignore[attr-defined]
        payload = message.to_dict()  # type: ignore[attr-defined]
    else:
        payload = getattr(message, "__dict__", {}).copy()
    payload["author_name"] = fallback
    if hasattr(ChatMessage, "from_dict") and callable(getattr(ChatMessage, "from_dict", None)):
        return ChatMessage.from_dict(payload)  # type: ignore[attr-defined,return-value]
    return ChatMessage(
        role=getattr(message, "role", Role.ASSISTANT), text=payload.get("text", ""), author_name=fallback
    )


def snapshot_state(conversation: Sequence[ChatMessage]) -> dict[str, Any]:
    """Build an immutable snapshot for checkpoint storage."""
    if hasattr(conversation, "to_dict"):
        result = conversation.to_dict()  # type: ignore[attr-defined]
        if isinstance(result, dict):
            return result  # type: ignore[return-value]
        if isinstance(result, Mapping):
            return dict(result)  # type: ignore[arg-type]
    serialisable: list[dict[str, Any]] = []
    for message in conversation:
        if hasattr(message, "to_dict") and callable(message.to_dict):  # type: ignore[attr-defined]
            msg_dict = message.to_dict()  # type: ignore[attr-defined]
            serialisable.append(dict(msg_dict) if isinstance(msg_dict, Mapping) else msg_dict)  # type: ignore[arg-type]
        elif hasattr(message, "to_json") and callable(message.to_json):  # type: ignore[attr-defined]
            json_payload = message.to_json()  # type: ignore[attr-defined]
            parsed = json.loads(json_payload) if isinstance(json_payload, str) else json_payload
            serialisable.append(dict(parsed) if isinstance(parsed, Mapping) else parsed)  # type: ignore[arg-type]
        else:
            serialisable.append(dict(getattr(message, "__dict__", {})))  # type: ignore[arg-type]
    return {"messages": serialisable}
