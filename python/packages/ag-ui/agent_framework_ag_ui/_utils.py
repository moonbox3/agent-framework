# Copyright (c) Microsoft. All rights reserved.

"""Utility functions for AG-UI integration."""

import copy
import uuid
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

from agent_framework import AIFunction, ToolProtocol


def generate_event_id() -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())


def merge_state(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge state updates.

    Args:
        current: Current state dictionary
        update: Update to apply

    Returns:
        Merged state
    """
    result = copy.deepcopy(current)
    result.update(update)
    return result


def make_json_safe(obj: Any) -> Any:  # noqa: ANN401
    """Make an object JSON serializable.

    Args:
        obj: Object to make JSON safe

    Returns:
        JSON-serializable version of the object
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)  # type: ignore[arg-type]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[no-any-return]
    if hasattr(obj, "dict"):
        return obj.dict()  # type: ignore[no-any-return]
    if hasattr(obj, "__dict__"):
        return {key: make_json_safe(value) for key, value in vars(obj).items()}  # type: ignore[misc]
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]  # type: ignore[misc]
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}  # type: ignore[misc]
    return str(obj)


def convert_tools_to_agui_format(
    tools: (
        ToolProtocol
        | Callable[..., Any]
        | MutableMapping[str, Any]
        | Sequence[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]]
        | None
    ),
) -> list[dict[str, Any]] | None:
    """Convert tools to AG-UI format.

    This sends only the metadata (name, description, JSON schema) to the server.
    The actual executable implementation stays on the client side.
    The @use_function_invocation decorator handles client-side execution when
    the server requests a function.

    Args:
        tools: Tools to convert (single tool or sequence of tools)

    Returns:
        List of tool specifications in AG-UI format, or None if no tools provided
    """
    if not tools:
        return None

    # Normalize to list
    if not isinstance(tools, list):
        tool_list: list[ToolProtocol | Callable[..., Any] | MutableMapping[str, Any]] = [tools]  # type: ignore[list-item]
    else:
        tool_list = tools  # type: ignore[assignment]

    results: list[dict[str, Any]] = []

    for tool in tool_list:
        if isinstance(tool, dict):
            # Already in dict format, pass through
            results.append(tool)  # type: ignore[arg-type]
        elif isinstance(tool, AIFunction):
            # Convert AIFunction to AG-UI tool format
            results.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters(),
                }
            )
        elif callable(tool):
            # Convert callable to AIFunction first, then to AG-UI format
            from agent_framework import ai_function

            ai_func = ai_function(tool)
            results.append(
                {
                    "name": ai_func.name,
                    "description": ai_func.description,
                    "parameters": ai_func.parameters(),
                }
            )
        elif isinstance(tool, ToolProtocol):
            # Handle other ToolProtocol implementations
            # For now, we'll skip non-AIFunction tools as they may not have
            # the parameters() method. This matches .NET behavior which only
            # converts AIFunctionDeclaration instances.
            continue

    return results if results else None
