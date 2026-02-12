# Copyright (c) Microsoft. All rights reserved.

"""Native AG-UI orchestration for MAF Workflow streams."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from ag_ui.core import (
    ActivitySnapshotEvent,
    BaseEvent,
    CustomEvent,
    RunErrorEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from agent_framework import AgentResponse, AgentResponseUpdate, Content, Message, Workflow, WorkflowRunState

from ._message_adapters import normalize_agui_input_messages
from ._run import (
    FlowState,
    _build_run_finished_event,
    _emit_content,
    _has_only_tool_calls,
    _normalize_resume_interrupts,
)
from ._utils import generate_event_id, make_json_safe

logger = logging.getLogger(__name__)


_TERMINAL_STATES: set[str] = {
    WorkflowRunState.IDLE.value,
    WorkflowRunState.IDLE_WITH_PENDING_REQUESTS.value,
    WorkflowRunState.CANCELLED.value,
}

_WORKFLOW_EVENT_BASE_FIELDS: set[str] = {
    "type",
    "data",
    "origin",
    "state",
    "details",
    "executor_id",
    "_request_id",
    "_source_executor_id",
    "_request_type",
    "_response_type",
    "iteration",
}


def _extract_responses_from_messages(messages: list[Message]) -> dict[str, Any]:
    """Extract request-info responses from incoming tool/function-result messages."""
    responses: dict[str, Any] = {}
    for message in messages:
        for content in message.contents:
            if content.type != "function_result" or not content.call_id:
                continue
            value: Any = content.result
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            responses[str(content.call_id)] = value
    return responses


def _resume_to_workflow_responses(resume_payload: Any) -> dict[str, Any]:
    """Convert AG-UI resume payloads into workflow responses."""
    responses: dict[str, Any] = {}
    for interrupt in _normalize_resume_interrupts(resume_payload):
        responses[str(interrupt["id"])] = interrupt.get("value")
    return responses


def _workflow_payload_to_contents(payload: Any) -> list[Content] | None:
    """Best-effort conversion from workflow payloads to chat content fragments."""
    if payload is None:
        return None
    if isinstance(payload, Content):
        return [payload]
    if isinstance(payload, str):
        return [Content.from_text(text=payload)]
    if isinstance(payload, Message):
        return list(payload.contents or [])
    if isinstance(payload, AgentResponseUpdate):
        return list(payload.contents or [])
    if isinstance(payload, AgentResponse):
        contents: list[Content] = []
        for message in payload.messages:
            contents.extend(list(message.contents or []))
        return contents if contents else None
    if isinstance(payload, list):
        contents: list[Content] = []
        for item in payload:
            item_contents = _workflow_payload_to_contents(item)
            if item_contents is None:
                return None
            contents.extend(item_contents)
        return contents if contents else None
    return None


def _event_name(event: Any) -> str:
    event_type = getattr(event, "type", None)
    if isinstance(event_type, str) and event_type:
        return event_type
    return event.__class__.__name__


def _custom_event_value(event: Any) -> Any:
    if getattr(event, "data", None) is not None:
        return make_json_safe(getattr(event, "data"))

    event_dict = cast(dict[str, Any], getattr(event, "__dict__", {}) or {})
    custom_fields = {
        key: make_json_safe(value)
        for key, value in event_dict.items()
        if key not in _WORKFLOW_EVENT_BASE_FIELDS and not key.startswith("_")
    }
    return custom_fields if custom_fields else None


def _details_message(details: Any) -> str:
    if details is None:
        return "Workflow execution failed."
    if hasattr(details, "message"):
        message = getattr(details, "message")
        if isinstance(message, str) and message:
            return message
    return str(details)


def _details_code(details: Any) -> str | None:
    if details is None:
        return None
    if hasattr(details, "error_type"):
        error_type = getattr(details, "error_type")
        if isinstance(error_type, str) and error_type:
            return error_type
    return None


async def run_workflow_stream(
    input_data: dict[str, Any],
    workflow: Workflow,
) -> AsyncGenerator[BaseEvent]:
    """Run a Workflow and emit AG-UI protocol events."""
    thread_id = input_data.get("thread_id") or input_data.get("threadId") or str(uuid.uuid4())
    run_id = input_data.get("run_id") or input_data.get("runId") or str(uuid.uuid4())
    available_interrupts = input_data.get("available_interrupts") or input_data.get("availableInterrupts")
    if available_interrupts:
        logger.debug("Received available interrupts metadata: %s", available_interrupts)

    raw_messages = list(cast(list[dict[str, Any]], input_data.get("messages", []) or []))
    messages, _ = normalize_agui_input_messages(raw_messages)

    flow = FlowState()
    interrupts: list[dict[str, Any]] = []
    run_started_emitted = False
    terminal_emitted = False
    run_error_emitted = False

    responses = _resume_to_workflow_responses(input_data.get("resume"))
    responses.update(_extract_responses_from_messages(messages))

    if not responses and not messages:
        yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
        yield _build_run_finished_event(run_id=run_id, thread_id=thread_id)
        return

    try:
        if responses:
            event_stream = workflow.run(responses=responses, stream=True)
        else:
            event_stream = workflow.run(message=messages, stream=True)

        async for event in event_stream:
            event_type = getattr(event, "type", None)

            if event_type == "started":
                if not run_started_emitted:
                    yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
                    run_started_emitted = True
                continue

            if not run_started_emitted:
                yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
                run_started_emitted = True

            if event_type == "failed":
                details = getattr(event, "details", None)
                yield RunErrorEvent(message=_details_message(details), code=_details_code(details))
                run_error_emitted = True
                terminal_emitted = True
                continue

            if event_type == "status":
                state = getattr(event, "state", None)
                state_value = state.value if hasattr(state, "value") else str(state)
                if state_value in _TERMINAL_STATES and not terminal_emitted:
                    yield _build_run_finished_event(run_id=run_id, thread_id=thread_id, interrupts=interrupts)
                    terminal_emitted = True
                elif state_value not in _TERMINAL_STATES:
                    yield CustomEvent(name="status", value={"state": state_value})
                continue

            if event_type == "superstep_started":
                iteration = getattr(event, "iteration", None)
                yield StepStartedEvent(step_name=f"superstep:{iteration}")
                continue

            if event_type == "superstep_completed":
                iteration = getattr(event, "iteration", None)
                yield StepFinishedEvent(step_name=f"superstep:{iteration}")
                continue

            if event_type in {"executor_invoked", "executor_completed", "executor_failed"}:
                executor_id = getattr(event, "executor_id", None)
                status = {
                    "executor_invoked": "in_progress",
                    "executor_completed": "completed",
                    "executor_failed": "failed",
                }[event_type]
                payload: dict[str, Any] = {
                    "executor_id": executor_id,
                    "status": status,
                }
                if event_type == "executor_failed":
                    payload["details"] = make_json_safe(getattr(event, "details", None))
                else:
                    payload["data"] = make_json_safe(getattr(event, "data", None))

                yield ActivitySnapshotEvent(
                    message_id=f"executor:{executor_id}" if executor_id else generate_event_id(),
                    activity_type="executor",
                    content=payload,
                )
                continue

            if event_type == "request_info":
                request_id = getattr(event, "request_id", None)
                if not request_id:
                    continue

                request_type = getattr(event, "request_type", None)
                response_type = getattr(event, "response_type", None)
                request_payload = {
                    "request_id": request_id,
                    "source_executor_id": getattr(event, "source_executor_id", None),
                    "request_type": getattr(request_type, "__name__", str(request_type) if request_type else None),
                    "response_type": getattr(response_type, "__name__", str(response_type) if response_type else None),
                    "data": make_json_safe(getattr(event, "data", None)),
                }
                interrupts.append({"id": str(request_id), "value": request_payload})
                args_delta = json.dumps(request_payload)

                yield ToolCallStartEvent(tool_call_id=str(request_id), tool_call_name="request_info")
                yield ToolCallArgsEvent(tool_call_id=str(request_id), delta=args_delta)
                yield ToolCallEndEvent(tool_call_id=str(request_id))
                yield CustomEvent(name="request_info", value=request_payload)
                continue

            if event_type in {"output", "data"}:
                payload = getattr(event, "data", None)
                contents = _workflow_payload_to_contents(payload)
                if contents:
                    if not flow.message_id and _has_only_tool_calls(contents):
                        flow.message_id = generate_event_id()
                        yield TextMessageStartEvent(message_id=flow.message_id, role="assistant")
                    for content in contents:
                        for out_event in _emit_content(content, flow, predictive_handler=None, skip_text=False):
                            yield out_event
                else:
                    yield CustomEvent(name="workflow_output", value=make_json_safe(payload))
                continue

            # Fall back to custom events for diagnostics, orchestration events, and custom workflow events.
            yield CustomEvent(name=_event_name(event), value=_custom_event_value(event))

    except Exception as exc:
        logger.exception("Workflow AG-UI stream failed: %s", exc)
        if not run_started_emitted:
            yield RunStartedEvent(run_id=run_id, thread_id=thread_id)
            run_started_emitted = True
        yield RunErrorEvent(message=str(exc), code=type(exc).__name__)
        run_error_emitted = True
        terminal_emitted = True

    if flow.message_id:
        yield TextMessageEndEvent(message_id=flow.message_id)

    if not run_started_emitted:
        yield RunStartedEvent(run_id=run_id, thread_id=thread_id)

    if not terminal_emitted and not run_error_emitted:
        yield _build_run_finished_event(run_id=run_id, thread_id=thread_id, interrupts=interrupts)
