import pytest

from agent_framework import (
    ChatMessage,
    ConcurrentBuilder,
    ConversationSnapshot,
    Role,
    SequentialBuilder,
)
from agent_framework._threads import AgentThread
from agent_framework._types import AgentRunResponse, AgentRunResponseUpdate
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class FakeAgent:
    """Minimal AgentProtocol implementation for testing."""

    def __init__(self, name: str) -> None:
        self._id = f"agent-{name}"
        self._name = name
        self._description = f"Test agent {name}"

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def run(self, messages=None, *, thread=None, **_: object) -> AgentRunResponse:
        count = len(messages or [])
        reply = ChatMessage(Role.ASSISTANT, text=f"{self._name}:{count}", author_name=self._name)
        return AgentRunResponse(messages=[reply])

    async def run_stream(self, messages=None, *, thread=None, **_: object):
        response = await self.run(messages, thread=thread)
        update = AgentRunResponseUpdate(text=response.text, role=Role.ASSISTANT, author_name=self._name)
        yield update

    def get_new_thread(self, **_: object) -> AgentThread:
        return AgentThread()


class CaptureExecutor(Executor):
    """Recorder executor that echoes conversations downstream."""

    def __init__(self) -> None:
        super().__init__(id="capture")
        self.captured: list[list[ChatMessage]] = []

    @handler
    async def capture(self, messages: list[ChatMessage], ctx: WorkflowContext[list[ChatMessage]]) -> None:  # type: ignore[name-defined]
        self.captured.append(list(messages))
        await ctx.send_message(list(messages))


@pytest.mark.asyncio
async def test_sequential_workflow_conversation_session_roundtrip() -> None:
    writer = FakeAgent("writer")
    reviewer = FakeAgent("reviewer")
    capture = CaptureExecutor()

    workflow = SequentialBuilder().participants([writer, capture, reviewer]).build()

    result = await workflow.run("hello sequential")
    outputs = result.get_outputs()
    assert len(outputs) == 1
    snapshot = outputs[0]
    assert isinstance(snapshot, ConversationSnapshot)
    assert snapshot.handle.session_id
    assert snapshot.handle.revision is not None and snapshot.handle.revision > 0

    # Conversations recorded by capture executor should reflect shared transcript
    assert capture.captured, "Capture executor did not observe any conversation snapshots"
    first_capture = capture.captured[0]
    assert first_capture[0].text == "hello sequential"
    assert first_capture[-1].author_name == "writer"

    # Resume using conversation handle should reproduce the same transcript
    resumed = await workflow.run(snapshot.handle)
    resumed_snapshot = resumed.get_outputs()[0]
    assert isinstance(resumed_snapshot, ConversationSnapshot)
    assert resumed_snapshot.handle.session_id == snapshot.handle.session_id
    original_texts = [m.text for m in snapshot.messages]
    resumed_texts = [m.text for m in resumed_snapshot.messages]
    assert resumed_texts[: len(original_texts)] == original_texts
    assert resumed_texts[-1].startswith("reviewer")


@pytest.mark.asyncio
async def test_concurrent_workflow_legacy_path_preserved() -> None:
    agents = [FakeAgent("alpha"), FakeAgent("beta")]
    workflow = ConcurrentBuilder().participants(agents).build()

    result = await workflow.run("hello concurrent")
    outputs = result.get_outputs()
    assert len(outputs) == 1
    messages = outputs[0]
    assert isinstance(messages, list)
    assert any(msg.author_name == "alpha" for msg in messages if hasattr(msg, "author_name"))
    assert any(msg.author_name == "beta" for msg in messages if hasattr(msg, "author_name"))
