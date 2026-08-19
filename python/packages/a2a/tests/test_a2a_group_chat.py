# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import pytest
from a2a.types import Artifact, Part, StreamResponse, Task, TaskState, TaskStatus
from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Content,
    Message,
    ResponseStream,
)
from agent_framework.exceptions import AgentInvalidRequestException
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState

from agent_framework_a2a import A2AAgent


class RecordingA2AClient:
    """Minimal A2A transport that records real remote invocations."""

    def __init__(self) -> None:
        self.call_count = 0

    async def send_message(self, request: Any) -> AsyncIterator[StreamResponse]:
        self.call_count += 1
        yield StreamResponse(
            task=Task(
                id=f"task-{self.call_count}",
                context_id="group-chat-context",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[Artifact(artifact_id="answer", parts=[Part(text="Remote answer")])],
            )
        )


class TextlessAgent(BaseAgent):
    """Participant whose response projects to no Group Chat messages."""

    def __init__(self) -> None:
        super().__init__(name="textless", description="Returns framework control content only")
        self.call_count = 0

    def run(  # type: ignore[override]
        self,
        messages: str | Content | Message | Sequence[str | Content | Message] | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        function_call = Content.from_function_call(call_id="control-1", name="internal_control")
        if stream:

            async def _stream() -> AsyncIterator[AgentResponseUpdate]:
                yield AgentResponseUpdate(contents=[function_call], role="assistant", author_name=self.name)

            return ResponseStream(_stream(), finalizer=AgentResponse.from_updates)

        async def _run() -> AgentResponse[Any]:
            return AgentResponse(messages=[Message("assistant", [function_call], author_name=self.name)])

        return _run()


class SessionBackedAgent(BaseAgent):
    """Non-A2A participant that supports empty turns through its session."""

    def __init__(self) -> None:
        super().__init__(name="session-backed", description="Continues from session state")
        self.invocations: list[Any] = []

    def run(  # type: ignore[override]
        self,
        messages: str | Content | Message | Sequence[str | Content | Message] | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Any:
        assert session is not None
        self.invocations.append(messages)
        turn = int(session.state.get("turn", 0)) + 1
        session.state["turn"] = turn
        text = f"Session turn {turn}"
        if stream:

            async def _stream() -> AsyncIterator[AgentResponseUpdate]:
                yield AgentResponseUpdate(
                    contents=[Content.from_text(text=text)],
                    role="assistant",
                    author_name=self.name,
                )

            return ResponseStream(_stream(), finalizer=AgentResponse.from_updates)

        async def _run() -> AgentResponse[Any]:
            return AgentResponse(messages=[Message("assistant", [text], author_name=self.name)])

        return _run()


@pytest.mark.parametrize("stream", [False, True])
async def test_consecutive_a2a_selection_rejects_empty_invocation_without_remote_call(stream: bool) -> None:
    """A consecutive A2A turn fails instead of inventing continuation input."""
    client = RecordingA2AClient()
    remote = A2AAgent(name="remote", client=cast(Any, client), http_client=None)

    def select_remote(state: GroupChatState) -> str:
        return "remote"

    workflow = GroupChatBuilder(
        participants=[remote],
        selection_func=select_remote,
        max_rounds=2,
    ).build()

    with pytest.raises(
        AgentInvalidRequestException,
        match="A2A agent 'remote' requires a real message or an explicit continuation token",
    ):
        if stream:
            async for _ in workflow.run("Investigate the incident", stream=True):
                pass
        else:
            await workflow.run("Investigate the incident", stream=False)

    assert client.call_count == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_a2a_reselection_after_textless_peer_rejects_empty_invocation(stream: bool) -> None:
    """An intervening response with no projected messages cannot activate A2A."""
    client = RecordingA2AClient()
    remote = A2AAgent(name="remote", client=cast(Any, client), http_client=None)
    textless = TextlessAgent()
    speakers = ["remote", "textless", "remote"]

    def select_in_sequence(state: GroupChatState) -> str:
        return speakers[state.current_round]

    workflow = GroupChatBuilder(
        participants=[remote, textless],
        selection_func=select_in_sequence,
        max_rounds=3,
    ).build()

    with pytest.raises(
        AgentInvalidRequestException,
        match="A2A agent 'remote' requires a real message or an explicit continuation token",
    ):
        if stream:
            async for _ in workflow.run("Investigate the incident", stream=True):
                pass
        else:
            await workflow.run("Investigate the incident", stream=False)

    assert client.call_count == 1
    assert textless.call_count == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_consecutive_session_backed_participant_still_receives_empty_turn(stream: bool) -> None:
    """Group Chat preserves valid empty-input behavior for non-A2A agents."""
    participant = SessionBackedAgent()
    selection_count = 0

    def select_participant(state: GroupChatState) -> str:
        nonlocal selection_count
        selection_count += 1
        return "session-backed"

    workflow = GroupChatBuilder(
        participants=[participant],
        selection_func=select_participant,
        max_rounds=2,
    ).build()

    if stream:
        async for _ in workflow.run("Investigate the incident", stream=True):
            pass
    else:
        await workflow.run("Investigate the incident", stream=False)

    assert selection_count == 2
    assert len(participant.invocations) == 2
    assert participant.invocations[1] == []
