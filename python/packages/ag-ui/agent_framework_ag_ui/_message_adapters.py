# Copyright (c) Microsoft. All rights reserved.

"""Message format conversion between AG-UI and Agent Framework."""

import json
from typing import Any, cast

from agent_framework import (
    ChatMessage,
    FunctionApprovalResponseContent,
    FunctionCallContent,
    FunctionResultContent,
    Role,
    TextContent,
    prepare_function_call_results,
)

# Role mapping constants
_AGUI_TO_FRAMEWORK_ROLE = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "system": Role.SYSTEM,
}

_FRAMEWORK_TO_AGUI_ROLE = {
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.SYSTEM: "system",
}


def agui_messages_to_agent_framework(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert AG-UI messages to Agent Framework format.

    Args:
        messages: List of AG-UI messages

    Returns:
        List of Agent Framework ChatMessage objects
    """

    def _update_tool_call_arguments(
        raw_messages: list[dict[str, Any]],
        tool_call_id: str,
        modified_args: dict[str, Any],
    ) -> None:
        for raw_msg in raw_messages:
            tool_calls = raw_msg.get("tool_calls") or raw_msg.get("toolCalls")
            if not isinstance(tool_calls, list):
                continue
            tool_calls_list = cast(list[Any], tool_calls)
            for tool_call in tool_calls_list:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_dict = cast(dict[str, Any], tool_call)
                if str(tool_call_dict.get("id", "")) != tool_call_id:
                    continue
                function_payload = tool_call_dict.get("function")
                if not isinstance(function_payload, dict):
                    return
                function_payload_dict = cast(dict[str, Any], function_payload)
                existing_args = function_payload_dict.get("arguments")
                if isinstance(existing_args, str):
                    function_payload_dict["arguments"] = json.dumps(modified_args)
                else:
                    function_payload_dict["arguments"] = modified_args
                return

    result: list[ChatMessage] = []
    for msg in messages:
        # Handle standard tool result messages early (role="tool") to preserve provider invariants
        # This path maps AG‑UI tool messages to FunctionResultContent with the correct tool_call_id
        role_str = msg.get("role", "user")
        if role_str == "tool":
            # Prefer explicit tool_call_id fields; fall back to backend fields only if necessary
            tool_call_id = msg.get("tool_call_id") or msg.get("toolCallId")

            # If no explicit tool_call_id, treat as backend tool rendering payloads where
            # AG‑UI may send actionExecutionId/actionName. This must still map to the
            # assistant's tool call id to satisfy provider requirements.
            if not tool_call_id:
                tool_call_id = msg.get("actionExecutionId") or ""

            # Extract raw content text
            result_content = msg.get("content")
            if result_content is None:
                result_content = msg.get("result", "")

            # Distinguish approval payloads from actual tool results
            parsed: dict[str, Any] | None = None
            if isinstance(result_content, str) and result_content:
                try:
                    parsed_candidate = json.loads(result_content)
                except Exception:
                    parsed_candidate = None
                if isinstance(parsed_candidate, dict):
                    parsed = cast(dict[str, Any], parsed_candidate)
            elif isinstance(result_content, dict):
                parsed = cast(dict[str, Any], result_content)

            is_approval = parsed is not None and "accepted" in parsed

            if is_approval:
                # Look for the matching function call in previous messages to create
                # a proper FunctionApprovalResponseContent. This enables the agent framework
                # to execute the approved tool (fix for GitHub issue #3034).
                accepted = parsed.get("accepted", False) if parsed is not None else False
                approval_payload_text = result_content if isinstance(result_content, str) else json.dumps(parsed)

                # Log the full approval payload to debug modified arguments
                import logging

                logger = logging.getLogger(__name__)
                logger.info(f"Approval payload received: {parsed}")

                # Find the function call that matches this tool_call_id
                matching_func_call = None
                for prev_msg in result:
                    role_val = prev_msg.role.value if hasattr(prev_msg.role, "value") else str(prev_msg.role)
                    if role_val != "assistant":
                        continue
                    for content in prev_msg.contents or []:
                        if isinstance(content, FunctionCallContent):
                            if content.call_id == tool_call_id and content.name != "confirm_changes":
                                matching_func_call = content
                                break

                if matching_func_call:
                    # Remove any existing tool result for this call_id since the framework
                    # will re-execute the tool after approval. Keeping old results causes
                    # OpenAI API errors ("tool message must follow assistant with tool_calls").
                    result = [
                        m
                        for m in result
                        if not (
                            (m.role.value if hasattr(m.role, "value") else str(m.role)) == "tool"
                            and any(
                                isinstance(c, FunctionResultContent) and c.call_id == tool_call_id
                                for c in (m.contents or [])
                            )
                        )
                    ]

                    # Check if the approval payload contains modified arguments
                    # The UI sends back the modified state (e.g., deselected steps) in the approval payload
                    modified_args = {k: v for k, v in parsed.items() if k != "accepted"} if parsed else {}
                    state_args: dict[str, Any] | None = None
                    if modified_args:
                        original_args = matching_func_call.parse_arguments() or {}
                        merged_args: dict[str, Any]
                        if isinstance(original_args, dict) and original_args:
                            merged_args = {**original_args, **modified_args}
                        else:
                            merged_args = dict(modified_args)

                        if isinstance(modified_args.get("steps"), list):
                            original_steps = original_args.get("steps") if isinstance(original_args, dict) else None
                            if isinstance(original_steps, list):
                                approved_steps_list: list[Any] = list(modified_args.get("steps") or [])
                                approved_by_description: dict[str, dict[str, Any]] = {}
                                for step_item in approved_steps_list:
                                    if isinstance(step_item, dict):
                                        step_item_dict = cast(dict[str, Any], step_item)
                                        desc = step_item_dict.get("description")
                                        if desc:
                                            approved_by_description[str(desc)] = step_item_dict
                                merged_steps: list[Any] = []
                                original_steps_list = cast(list[Any], original_steps)
                                for orig_step in original_steps_list:
                                    if not isinstance(orig_step, dict):
                                        merged_steps.append(orig_step)
                                        continue
                                    orig_step_dict = cast(dict[str, Any], orig_step)
                                    description = str(orig_step_dict.get("description", ""))
                                    approved_step = approved_by_description.get(description)
                                    status: str = (
                                        str(approved_step.get("status"))
                                        if approved_step is not None and approved_step.get("status")
                                        else "disabled"
                                    )
                                    updated_step: dict[str, Any] = orig_step_dict.copy()
                                    updated_step["status"] = status
                                    merged_steps.append(updated_step)
                                merged_args["steps"] = merged_steps
                        state_args = merged_args

                        # Keep the original tool call and AG-UI snapshot in sync with approved args.
                        updated_args = (
                            json.dumps(merged_args) if isinstance(matching_func_call.arguments, str) else merged_args
                        )
                        matching_func_call.arguments = updated_args
                        _update_tool_call_arguments(messages, str(tool_call_id), merged_args)
                        # Create a new FunctionCallContent with the modified arguments
                        func_call_for_approval = FunctionCallContent(
                            call_id=matching_func_call.call_id,
                            name=matching_func_call.name,
                            arguments=json.dumps(modified_args),
                        )
                        logger.info(f"Using modified arguments from approval: {modified_args}")
                    else:
                        # No modified arguments - use the original function call
                        func_call_for_approval = matching_func_call

                    # Create FunctionApprovalResponseContent for the agent framework
                    approval_response = FunctionApprovalResponseContent(
                        approved=accepted,
                        id=str(tool_call_id),
                        function_call=func_call_for_approval,
                        additional_properties={"ag_ui_state_args": state_args} if state_args else None,
                    )
                    chat_msg = ChatMessage(
                        role=Role.USER,
                        contents=[approval_response],
                    )
                else:
                    # No matching function call found - this is likely a confirm_changes approval
                    # Keep the old behavior for backwards compatibility
                    chat_msg = ChatMessage(
                        role=Role.USER,
                        contents=[TextContent(text=approval_payload_text)],
                        additional_properties={"is_tool_result": True, "tool_call_id": str(tool_call_id or "")},
                    )
                if "id" in msg:
                    chat_msg.message_id = msg["id"]
                result.append(chat_msg)
                continue

            # Cast result_content to acceptable type for FunctionResultContent
            func_result: str | dict[str, Any] | list[Any]
            if isinstance(result_content, str):
                func_result = result_content
            elif isinstance(result_content, dict):
                func_result = cast(dict[str, Any], result_content)
            elif isinstance(result_content, list):
                func_result = cast(list[Any], result_content)
            else:
                func_result = str(result_content)
            chat_msg = ChatMessage(
                role=Role.TOOL,
                contents=[FunctionResultContent(call_id=str(tool_call_id), result=func_result)],
            )
            if "id" in msg:
                chat_msg.message_id = msg["id"]
            result.append(chat_msg)
            continue

        # Backend tool rendering payloads without an explicit role
        # Prefer standard tool mapping above; this block only covers legacy/minimal payloads
        if "actionExecutionId" in msg or "actionName" in msg:
            # Prefer toolCallId if present; otherwise fall back to actionExecutionId
            tool_call_id = msg.get("toolCallId") or msg.get("tool_call_id") or msg.get("actionExecutionId", "")
            result_content = msg.get("result", msg.get("content", ""))

            chat_msg = ChatMessage(
                role=Role.TOOL,
                contents=[FunctionResultContent(call_id=str(tool_call_id), result=result_content)],
            )
            if "id" in msg:
                chat_msg.message_id = msg["id"]
            result.append(chat_msg)
            continue

        # If assistant message includes tool calls, convert to FunctionCallContent(s)
        tool_calls = msg.get("tool_calls") or msg.get("toolCalls")
        if tool_calls:
            contents: list[Any] = []
            # Include any assistant text content if present
            content_text = msg.get("content")
            if isinstance(content_text, str) and content_text:
                contents.append(TextContent(text=content_text))
            # Convert each tool call entry
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                # Cast to typed dict for proper type inference
                tc_dict = cast(dict[str, Any], tc)
                tc_type = tc_dict.get("type")
                if tc_type == "function":
                    func_data = tc_dict.get("function", {})
                    func_dict = cast(dict[str, Any], func_data) if isinstance(func_data, dict) else {}

                    call_id = str(tc_dict.get("id", ""))
                    name = str(func_dict.get("name", ""))
                    arguments = func_dict.get("arguments")

                    contents.append(
                        FunctionCallContent(
                            call_id=call_id,
                            name=name,
                            arguments=arguments,
                        )
                    )
            chat_msg = ChatMessage(role=Role.ASSISTANT, contents=contents)
            if "id" in msg:
                chat_msg.message_id = msg["id"]
            result.append(chat_msg)
            continue

        # No special handling required for assistant/plain messages here

        role = _AGUI_TO_FRAMEWORK_ROLE.get(role_str, Role.USER)

        # Check if this message contains function approvals
        if "function_approvals" in msg and msg["function_approvals"]:
            # Convert function approvals to FunctionApprovalResponseContent
            approval_contents: list[Any] = []
            for approval in msg["function_approvals"]:
                # Create FunctionCallContent with the modified arguments
                func_call = FunctionCallContent(
                    call_id=approval.get("call_id", ""),
                    name=approval.get("name", ""),
                    arguments=approval.get("arguments", {}),
                )

                # Create the approval response
                approval_response = FunctionApprovalResponseContent(
                    approved=approval.get("approved", True),
                    id=approval.get("id", ""),
                    function_call=func_call,
                )
                approval_contents.append(approval_response)

            chat_msg = ChatMessage(role=role, contents=approval_contents)  # type: ignore[arg-type]
        else:
            # Regular text message
            content = msg.get("content", "")
            if isinstance(content, str):
                chat_msg = ChatMessage(role=role, contents=[TextContent(text=content)])
            else:
                chat_msg = ChatMessage(role=role, contents=[TextContent(text=str(content))])

        if "id" in msg:
            chat_msg.message_id = msg["id"]

        result.append(chat_msg)

    return result


def agent_framework_messages_to_agui(messages: list[ChatMessage] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Agent Framework messages to AG-UI format.

    Args:
        messages: List of Agent Framework ChatMessage objects or AG-UI dicts (already converted)

    Returns:
        List of AG-UI message dictionaries
    """
    from ._utils import generate_event_id

    result: list[dict[str, Any]] = []
    for msg in messages:
        # If already a dict (AG-UI format), ensure it has an ID and normalize keys for Pydantic
        if isinstance(msg, dict):
            # Always work on a copy to avoid mutating input
            normalized_msg = msg.copy()
            # Ensure ID exists
            if "id" not in normalized_msg:
                normalized_msg["id"] = generate_event_id()
            # Normalize tool_call_id to toolCallId for Pydantic's alias_generator=to_camel
            if normalized_msg.get("role") == "tool":
                if "tool_call_id" in normalized_msg:
                    normalized_msg["toolCallId"] = normalized_msg["tool_call_id"]
                    del normalized_msg["tool_call_id"]
                elif "toolCallId" not in normalized_msg:
                    # Tool message missing toolCallId - add empty string to satisfy schema
                    normalized_msg["toolCallId"] = ""
            # Always append the normalized copy, not the original
            result.append(normalized_msg)
            continue

        # Convert ChatMessage to AG-UI format
        role = _FRAMEWORK_TO_AGUI_ROLE.get(msg.role, "user")

        content_text = ""
        tool_calls: list[dict[str, Any]] = []
        tool_result_call_id: str | None = None

        for content in msg.contents:
            if isinstance(content, TextContent):
                content_text += content.text
            elif isinstance(content, FunctionCallContent):
                tool_calls.append(
                    {
                        "id": content.call_id,
                        "type": "function",
                        "function": {
                            "name": content.name,
                            "arguments": content.arguments,
                        },
                    }
                )
            elif isinstance(content, FunctionResultContent):
                # Tool result content - extract call_id and result
                tool_result_call_id = content.call_id
                # Serialize result to string using core utility
                content_text = prepare_function_call_results(content.result)

        agui_msg: dict[str, Any] = {
            "id": msg.message_id if msg.message_id else generate_event_id(),  # Always include id
            "role": role,
            "content": content_text,
        }

        if tool_calls:
            agui_msg["tool_calls"] = tool_calls

        # If this is a tool result message, add toolCallId (using camelCase for Pydantic)
        if tool_result_call_id:
            agui_msg["toolCallId"] = tool_result_call_id
            # Tool result messages should have role="tool"
            agui_msg["role"] = "tool"

        result.append(agui_msg)

    return result


def extract_text_from_contents(contents: list[Any]) -> str:
    """Extract text from Agent Framework contents.

    Args:
        contents: List of content objects

    Returns:
        Concatenated text
    """
    text_parts: list[str] = []
    for content in contents:
        if isinstance(content, TextContent):
            text_parts.append(content.text)
        elif hasattr(content, "text"):
            text_parts.append(content.text)
    return "".join(text_parts)


def agui_messages_to_snapshot_format(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize AG-UI messages for MessagesSnapshotEvent.

    Converts AG-UI input format (with 'input_text' type) to snapshot format (with 'text' type).

    Args:
        messages: List of AG-UI messages in input format

    Returns:
        List of normalized messages suitable for MessagesSnapshotEvent
    """
    from ._utils import generate_event_id

    result: list[dict[str, Any]] = []
    for msg in messages:
        normalized_msg = msg.copy()

        # Ensure ID exists
        if "id" not in normalized_msg:
            normalized_msg["id"] = generate_event_id()

        # Normalize content field
        content = normalized_msg.get("content")
        if isinstance(content, list):
            # Convert content array format to simple string
            text_parts: list[str] = []
            content_list = cast(list[Any], content)
            for item in content_list:
                if isinstance(item, dict):
                    item_dict = cast(dict[str, Any], item)
                    # Convert 'input_text' to 'text' type
                    if item_dict.get("type") == "input_text":
                        text_parts.append(str(item_dict.get("text", "")))
                    elif item_dict.get("type") == "text":
                        text_parts.append(str(item_dict.get("text", "")))
                    else:
                        # Other types - just extract text field if present
                        text_parts.append(str(item_dict.get("text", "")))
            normalized_msg["content"] = "".join(text_parts)
        elif content is None:
            normalized_msg["content"] = ""

        tool_calls = normalized_msg.get("tool_calls") or normalized_msg.get("toolCalls")
        if isinstance(tool_calls, list):
            tool_calls_list = cast(list[Any], tool_calls)
            for tool_call in tool_calls_list:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_dict = cast(dict[str, Any], tool_call)
                function_payload = tool_call_dict.get("function")
                if not isinstance(function_payload, dict):
                    continue
                function_payload_dict = cast(dict[str, Any], function_payload)
                if "arguments" not in function_payload_dict:
                    continue
                arguments = function_payload_dict.get("arguments")
                if arguments is None:
                    function_payload_dict["arguments"] = ""
                elif not isinstance(arguments, str):
                    function_payload_dict["arguments"] = json.dumps(arguments)

        # Normalize tool_call_id to toolCallId for tool messages
        if normalized_msg.get("role") == "tool":
            if "tool_call_id" in normalized_msg:
                normalized_msg["toolCallId"] = normalized_msg["tool_call_id"]
                del normalized_msg["tool_call_id"]
            elif "toolCallId" not in normalized_msg:
                normalized_msg["toolCallId"] = ""

        result.append(normalized_msg)

    return result


__all__ = [
    "agui_messages_to_agent_framework",
    "agent_framework_messages_to_agui",
    "agui_messages_to_snapshot_format",
    "extract_text_from_contents",
]
