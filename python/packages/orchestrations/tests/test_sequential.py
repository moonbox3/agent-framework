# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterable, Awaitable
from typing import Any, Literal, overload

import pytest
from agent_framework import (
    AgentExecutorResponse,
    AgentResponse,
    AgentResponseUpdate,
    AgentRunInputs,
    AgentSession,
    BaseAgent,
    Content,
    Executor,
    Message,
    ResponseStream,
    TypeCompatibilityError,
    WorkflowContext,
    WorkflowRunState,
    handler,
)
from agent_framework._workflows._checkpoint import InMemoryCheckpointStorage
from agent_framework.orchestrations import SequentialBuilder


class _EchoAgent(BaseAgent):
    """Simple agent that appends a single assistant message with its name."""

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        if stream:

            async def _stream() -> AsyncIterable[AgentResponseUpdate]:
                yield AgentResponseUpdate(contents=[Content.from_text(text=f"{self.name} reply")])

            return ResponseStream(_stream(), finalizer=AgentResponse.from_updates)

        async def _run() -> AgentResponse:
            return AgentResponse(messages=[Message("assistant", [f"{self.name} reply"])])

        return _run()


class _SummarizerExec(Executor):
    """Custom executor that summarizes by appending a short assistant message."""

    @handler
    async def summarize(self, agent_response: AgentExecutorResponse, ctx: WorkflowContext[list[Message]]) -> None:
        conversation = agent_response.full_conversation or []
        user_texts = [m.text for m in conversation if m.role == "user"]
        agents = [m.author_name or m.role for m in conversation if m.role == "assistant"]
        summary = Message("assistant", [f"Summary of users:{len(user_texts)} agents:{len(agents)}"])
        await ctx.send_message(list(conversation) + [summary])


class _InvalidExecutor(Executor):
    """Invalid executor that does not have a handler that accepts a list of chat messages"""

    @handler
    async def summarize(self, conversation: list[str], ctx: WorkflowContext[list[Message]]) -> None:
        pass


def test_sequential_builder_rejects_empty_participants() -> None:
    with pytest.raises(ValueError):
        SequentialBuilder(participants=[])


def test_sequential_builder_validation_rejects_invalid_executor() -> None:
    """Test that adding an invalid executor to the builder raises an error."""
    with pytest.raises(TypeCompatibilityError):
        SequentialBuilder(participants=[_EchoAgent(id="agent1", name="A1"), _InvalidExecutor(id="invalid")]).build()


async def test_sequential_streaming_yields_only_last_agent_updates() -> None:
    """Streaming mode surfaces only the last agent's AgentResponseUpdate chunks as outputs.

    Intermediate agents do NOT emit `output` events when intermediate_outputs=False (default);
    only the last agent (the workflow's output_executor) emits chunks of the final answer.
    """
    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    wf = SequentialBuilder(participants=[a1, a2]).build()

    completed = False
    update_events: list[AgentResponseUpdate] = []
    async for ev in wf.run("hello sequential", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            completed = True
        elif ev.type == "output":
            update_events.append(ev.data)  # type: ignore[arg-type]
        if completed:
            break

    assert completed
    # Only the last agent's streaming chunks surface as `output` events.
    assert update_events, "Expected at least one streaming update from the last agent"
    for upd in update_events:
        assert isinstance(upd, AgentResponseUpdate)
    combined_text = "".join(u.text for u in update_events if hasattr(u, "text"))
    assert "A2 reply" in combined_text
    assert "A1 reply" not in combined_text


async def test_sequential_non_streaming_yields_only_last_agent_response() -> None:
    """Non-streaming mode emits a single `output` event with the last agent's AgentResponse."""
    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    wf = SequentialBuilder(participants=[a1, a2]).build()

    output_events = [ev for ev in await wf.run("hello sequential") if ev.type == "output"]
    assert len(output_events) == 1
    response = output_events[0].data
    assert isinstance(response, AgentResponse)
    assert all(m.role == "assistant" for m in response.messages)
    combined = " ".join(m.text for m in response.messages)
    assert "A2 reply" in combined
    assert "A1 reply" not in combined


async def test_sequential_intermediate_outputs_emits_data_events() -> None:
    """When intermediate_outputs=True, intermediate agents surface as `data` events.

    The single `output` event still carries the last agent's AgentResponse; intermediate
    agents are emitted as `data` events (not output events) so consumers can clearly tell
    them apart.
    """
    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    wf = SequentialBuilder(participants=[a1, a2], intermediate_outputs=True).build()

    output_events = []
    data_events = []
    for ev in await wf.run("hello"):
        if ev.type == "output":
            output_events.append(ev)
        elif ev.type == "data":
            data_events.append(ev)

    # One output event = the final answer (last agent).
    assert len(output_events) == 1
    final = output_events[0].data
    assert isinstance(final, AgentResponse)
    assert "A2 reply" in " ".join(m.text for m in final.messages)

    # Intermediate agents emit data events (not output events).
    assert len(data_events) == 1
    intermediate = data_events[0].data
    assert isinstance(intermediate, AgentResponse)
    assert "A1 reply" in " ".join(m.text for m in intermediate.messages)
    # Executor id derives from the agent's name (resolve_agent_id behavior).
    assert data_events[0].executor_id == "A1"


async def test_sequential_as_agent_returns_only_last_agent_response() -> None:
    """`workflow.as_agent().run(prompt)` returns ONLY the last agent's messages — not the user
    input or earlier agents' replies. This is the core fix for the orchestration-as-agent
    output contract."""
    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    agent = SequentialBuilder(participants=[a1, a2]).build().as_agent()
    response = await agent.run("hello as_agent")

    assert isinstance(response, AgentResponse)
    # Only the last agent's reply — no user prompt, no agent1 messages.
    combined = " ".join(m.text for m in response.messages)
    assert "A2 reply" in combined
    assert "A1 reply" not in combined
    assert "hello as_agent" not in combined


async def test_sequential_as_agent_with_intermediate_outputs_includes_chain() -> None:
    """With `intermediate_outputs=True`, `as_agent()` surfaces intermediate agent responses
    (via `data` events) followed by the final answer."""
    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    agent = SequentialBuilder(participants=[a1, a2], intermediate_outputs=True).build().as_agent()
    response = await agent.run("hello as_agent")

    assert isinstance(response, AgentResponse)
    combined_text = " ".join(m.text for m in response.messages)
    assert "A1 reply" in combined_text
    assert "A2 reply" in combined_text
    # Final agent's reply should appear last in the message ordering.
    last_text = response.messages[-1].text
    assert "A2 reply" in last_text


async def test_sequential_with_custom_executor_summary() -> None:
    a1 = _EchoAgent(id="agent1", name="A1")
    summarizer = _SummarizerExec(id="summarizer")

    wf = SequentialBuilder(participants=[a1, summarizer]).build()

    completed = False
    output: list[Message] | None = None
    async for ev in wf.run("topic X", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            completed = True
        elif ev.type == "output":
            output = ev.data
        if completed and output is not None:
            break

    assert completed
    assert output is not None
    msgs: list[Message] = output
    # Expect: [user, A1 reply, summary]
    assert len(msgs) == 3
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant" and "A1 reply" in msgs[1].text
    assert msgs[2].role == "assistant" and msgs[2].text.startswith("Summary of users:")


async def test_sequential_checkpoint_resume_round_trip() -> None:
    storage = InMemoryCheckpointStorage()

    initial_agents = (_EchoAgent(id="agent1", name="A1"), _EchoAgent(id="agent2", name="A2"))
    wf = SequentialBuilder(participants=list(initial_agents), checkpoint_storage=storage).build()

    baseline_updates: list[AgentResponseUpdate] = []
    async for ev in wf.run("checkpoint sequential", stream=True):
        if ev.type == "output":
            baseline_updates.append(ev.data)  # type: ignore[arg-type]
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    assert baseline_updates

    checkpoints = await storage.list_checkpoints(workflow_name=wf.name)
    assert checkpoints
    checkpoints.sort(key=lambda cp: cp.timestamp)
    resume_checkpoint = checkpoints[0]

    resumed_agents = (_EchoAgent(id="agent1", name="A1"), _EchoAgent(id="agent2", name="A2"))
    wf_resume = SequentialBuilder(participants=list(resumed_agents), checkpoint_storage=storage).build()

    resumed_updates: list[AgentResponseUpdate] = []
    async for ev in wf_resume.run(checkpoint_id=resume_checkpoint.checkpoint_id, stream=True):
        if ev.type == "output":
            resumed_updates.append(ev.data)  # type: ignore[arg-type]
        if ev.type == "status" and ev.state in (
            WorkflowRunState.IDLE,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        ):
            break

    assert resumed_updates
    baseline_text = "".join(u.text for u in baseline_updates if hasattr(u, "text"))
    resumed_text = "".join(u.text for u in resumed_updates if hasattr(u, "text"))
    assert baseline_text == resumed_text


async def test_sequential_checkpoint_runtime_only() -> None:
    """Test checkpointing configured ONLY at runtime, not at build time."""
    storage = InMemoryCheckpointStorage()

    agents = (_EchoAgent(id="agent1", name="A1"), _EchoAgent(id="agent2", name="A2"))
    wf = SequentialBuilder(participants=list(agents)).build()

    baseline_updates: list[AgentResponseUpdate] = []
    async for ev in wf.run("runtime checkpoint test", checkpoint_storage=storage, stream=True):
        if ev.type == "output":
            baseline_updates.append(ev.data)  # type: ignore[arg-type]
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    assert baseline_updates

    checkpoints = await storage.list_checkpoints(workflow_name=wf.name)
    assert checkpoints
    checkpoints.sort(key=lambda cp: cp.timestamp)
    resume_checkpoint = checkpoints[0]

    resumed_agents = (_EchoAgent(id="agent1", name="A1"), _EchoAgent(id="agent2", name="A2"))
    wf_resume = SequentialBuilder(participants=list(resumed_agents)).build()

    resumed_updates: list[AgentResponseUpdate] = []
    async for ev in wf_resume.run(
        checkpoint_id=resume_checkpoint.checkpoint_id, checkpoint_storage=storage, stream=True
    ):
        if ev.type == "output":
            resumed_updates.append(ev.data)  # type: ignore[arg-type]
        if ev.type == "status" and ev.state in (
            WorkflowRunState.IDLE,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        ):
            break

    assert resumed_updates
    baseline_text = "".join(u.text for u in baseline_updates if hasattr(u, "text"))
    resumed_text = "".join(u.text for u in resumed_updates if hasattr(u, "text"))
    assert baseline_text == resumed_text


async def test_sequential_checkpoint_runtime_overrides_buildtime() -> None:
    """Test that runtime checkpoint storage overrides build-time configuration."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
        from agent_framework._workflows._checkpoint import FileCheckpointStorage

        buildtime_storage = FileCheckpointStorage(temp_dir1)
        runtime_storage = FileCheckpointStorage(temp_dir2)

        agents = (_EchoAgent(id="agent1", name="A1"), _EchoAgent(id="agent2", name="A2"))
        wf = SequentialBuilder(participants=list(agents), checkpoint_storage=buildtime_storage).build()

        baseline_output: list[Message] | None = None
        async for ev in wf.run("override test", checkpoint_storage=runtime_storage, stream=True):
            if ev.type == "output":
                baseline_output = ev.data  # type: ignore[assignment]
            if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
                break

        assert baseline_output is not None

        buildtime_checkpoints = await buildtime_storage.list_checkpoints(workflow_name=wf.name)
        runtime_checkpoints = await runtime_storage.list_checkpoints(workflow_name=wf.name)

        assert len(runtime_checkpoints) > 0, "Runtime storage should have checkpoints"
        assert len(buildtime_checkpoints) == 0, "Build-time storage should have no checkpoints when overridden"


async def test_sequential_builder_reusable_after_build_with_participants() -> None:
    """Test that the builder can be reused to build multiple identical workflows with participants()."""
    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    builder = SequentialBuilder(participants=[a1, a2])

    # Build first workflow
    builder.build()

    assert builder._participants[0] is a1  # type: ignore
    assert builder._participants[1] is a2  # type: ignore


# ---------------------------------------------------------------------------
# chain_only_agent_responses tests
# ---------------------------------------------------------------------------


class _CapturingAgent(BaseAgent):
    """Agent that records the messages it received and returns a configurable reply."""

    def __init__(self, *, reply_text: str = "reply", **kwargs: Any):
        super().__init__(**kwargs)
        self.reply_text = reply_text
        self.last_messages: list[Message] = []

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        captured: list[Message] = []
        if messages:
            for m in messages:  # type: ignore[union-attr]
                if isinstance(m, Message):
                    captured.append(m)
                elif isinstance(m, str):
                    captured.append(Message("user", [m]))
        self.last_messages = captured

        if stream:

            async def _stream() -> AsyncIterable[AgentResponseUpdate]:
                yield AgentResponseUpdate(contents=[Content.from_text(text=self.reply_text)])

            return ResponseStream(_stream(), finalizer=AgentResponse.from_updates)

        async def _run() -> AgentResponse:
            return AgentResponse(messages=[Message("assistant", [self.reply_text])])

        return _run()


async def test_chain_only_agent_responses_false_passes_full_conversation() -> None:
    """Default (chain_only_agent_responses=False) passes full conversation to the second agent."""
    a1 = _CapturingAgent(id="agent1", name="A1", reply_text="A1 reply")
    a2 = _CapturingAgent(id="agent2", name="A2", reply_text="A2 reply")

    wf = SequentialBuilder(participants=[a1, a2], chain_only_agent_responses=False).build()

    async for ev in wf.run("hello", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    # Second agent should see full conversation: [user("hello"), assistant("A1 reply")]
    seen = a2.last_messages
    assert len(seen) == 2
    assert seen[0].role == "user" and "hello" in (seen[0].text or "")
    assert seen[1].role == "assistant" and "A1 reply" in (seen[1].text or "")


async def test_chain_only_agent_responses_true_passes_only_agent_messages() -> None:
    """chain_only_agent_responses=True passes only the previous agent's response messages."""
    a1 = _CapturingAgent(id="agent1", name="A1", reply_text="A1 reply")
    a2 = _CapturingAgent(id="agent2", name="A2", reply_text="A2 reply")

    wf = SequentialBuilder(participants=[a1, a2], chain_only_agent_responses=True).build()

    async for ev in wf.run("hello", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    # Second agent should see only the assistant message: [assistant("A1 reply")]
    seen = a2.last_messages
    assert len(seen) == 1
    assert seen[0].role == "assistant" and "A1 reply" in (seen[0].text or "")


async def test_chain_only_agent_responses_three_agents() -> None:
    """chain_only_agent_responses=True with three agents: each sees only the prior agent's reply."""
    a1 = _CapturingAgent(id="agent1", name="A1", reply_text="A1 reply")
    a2 = _CapturingAgent(id="agent2", name="A2", reply_text="A2 reply")
    a3 = _CapturingAgent(id="agent3", name="A3", reply_text="A3 reply")

    wf = SequentialBuilder(participants=[a1, a2, a3], chain_only_agent_responses=True).build()

    async for ev in wf.run("hello", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    # a2 should see only A1's reply
    assert len(a2.last_messages) == 1
    assert a2.last_messages[0].role == "assistant" and "A1 reply" in (a2.last_messages[0].text or "")

    # a3 should see only A2's reply
    assert len(a3.last_messages) == 1
    assert a3.last_messages[0].role == "assistant" and "A2 reply" in (a3.last_messages[0].text or "")


# ---------------------------------------------------------------------------
# with_request_info tests
# ---------------------------------------------------------------------------


async def test_sequential_request_info_last_participant_emits_output() -> None:
    """When the last participant is wrapped via with_request_info(), the workflow
    still emits a terminal output event after approval.

    This exercises the _EndWithConversation.end_with_agent_executor_response path
    that converts the AgentApprovalExecutor's forwarded AgentExecutorResponse into
    the workflow's final AgentResponse output.
    """
    from agent_framework_orchestrations._orchestration_request_info import AgentRequestInfoResponse

    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    wf = SequentialBuilder(participants=[a1, a2]).with_request_info().build()

    # First run: collect request_info events for both agents
    request_events: list[Any] = []
    async for ev in wf.run("hello with approval", stream=True):
        if ev.type == "request_info" and isinstance(ev.data, AgentExecutorResponse):
            request_events.append(ev)

    # Approve each agent in sequence until the workflow completes
    while request_events:
        responses = {req.request_id: AgentRequestInfoResponse.approve() for req in request_events}
        request_events = []
        output_events: list[Any] = []
        async for ev in wf.run(stream=True, responses=responses):
            if ev.type == "request_info" and isinstance(ev.data, AgentExecutorResponse):
                request_events.append(ev)
            elif ev.type == "output":
                output_events.append(ev)

    # The workflow must produce a terminal output with the last agent's response.
    assert len(output_events) == 1
    response = output_events[0].data
    assert isinstance(response, AgentResponse)
    assert any("A2 reply" in m.text for m in response.messages)


async def test_sequential_request_info_with_intermediate_outputs_emits_data_events() -> None:
    """With both with_request_info() and intermediate_outputs=True, intermediate
    agents' responses are surfaced as data events while the final output is an
    AgentResponse from the last agent.

    This verifies that WorkflowExecutor correctly forwards data events from the
    inner AgentExecutor through the AgentApprovalExecutor wrapper.
    """
    from agent_framework_orchestrations._orchestration_request_info import AgentRequestInfoResponse

    a1 = _EchoAgent(id="agent1", name="A1")
    a2 = _EchoAgent(id="agent2", name="A2")

    wf = (
        SequentialBuilder(participants=[a1, a2], intermediate_outputs=True)
        .with_request_info()
        .build()
    )

    # Run and approve all request_info events until the workflow completes
    all_data_events: list[Any] = []
    all_output_events: list[Any] = []
    request_events: list[Any] = []

    async for ev in wf.run("hello intermediate", stream=True):
        if ev.type == "request_info" and isinstance(ev.data, AgentExecutorResponse):
            request_events.append(ev)
        elif ev.type == "data":
            all_data_events.append(ev)
        elif ev.type == "output":
            all_output_events.append(ev)

    while request_events:
        responses = {req.request_id: AgentRequestInfoResponse.approve() for req in request_events}
        request_events = []
        async for ev in wf.run(stream=True, responses=responses):
            if ev.type == "request_info" and isinstance(ev.data, AgentExecutorResponse):
                request_events.append(ev)
            elif ev.type == "data":
                all_data_events.append(ev)
            elif ev.type == "output":
                all_output_events.append(ev)

    # The first (intermediate) agent should emit a data event.
    assert len(all_data_events) >= 1
    intermediate_texts = set()
    for dev in all_data_events:
        if isinstance(dev.data, AgentResponse):
            for m in dev.data.messages:
                intermediate_texts.add(m.text)
    assert "A1 reply" in intermediate_texts

    # The final output should contain the last agent's response.
    assert len(all_output_events) >= 1
    final = all_output_events[-1].data
    assert isinstance(final, AgentResponse)
    assert any("A2 reply" in m.text for m in final.messages)
