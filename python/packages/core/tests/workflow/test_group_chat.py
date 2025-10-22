# Copyright (c) Microsoft. All rights reserved.

from collections.abc import AsyncIterable, Callable
from typing import Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    GroupChatBuilder,
    GroupChatStateSnapshot,
    MagenticAgentMessageEvent,
    MagenticBuilder,
    MagenticContext,
    MagenticManagerBase,
    MagenticOrchestratorMessageEvent,
    MagenticProgressLedger,
    MagenticProgressLedgerItem,
    MagenticStartMessage,
    Role,
    TextContent,
    Workflow,
    WorkflowOutputEvent,
)


class StubAgent(BaseAgent):
    def __init__(self, agent_name: str, reply_text: str, **kwargs: Any) -> None:
        super().__init__(name=agent_name, description=f"Stub agent {agent_name}", **kwargs)
        self._reply_text = reply_text

    async def run(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        response = ChatMessage(role=Role.ASSISTANT, text=self._reply_text, author_name=self.name)
        return AgentRunResponse(messages=[response])

    def run_stream(  # type: ignore[override]
        self,
        messages: str | ChatMessage | list[str] | list[ChatMessage] | None = None,
        *,
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        async def _stream() -> AsyncIterable[AgentRunResponseUpdate]:
            yield AgentRunResponseUpdate(
                contents=[TextContent(text=self._reply_text)], role=Role.ASSISTANT, author_name=self.name
            )

        return _stream()


def make_sequence_selector() -> Callable[[GroupChatStateSnapshot], Any]:
    state_counter = {"value": 0}

    async def _selector(state: GroupChatStateSnapshot) -> str | None:
        participants = list(state["participants"].keys())
        step = state_counter["value"]
        if step == 0:
            state_counter["value"] = step + 1
            return participants[0]
        if step == 1 and len(participants) > 1:
            state_counter["value"] = step + 1
            return participants[1]
        return None

    _selector.name = "manager"  # type: ignore[attr-defined]
    return _selector


class StubMagenticManager(MagenticManagerBase):
    def __init__(self) -> None:
        super().__init__(max_stall_count=3, max_round_count=5)
        self._round = 0

    async def plan(self, magentic_context: MagenticContext) -> ChatMessage:
        return ChatMessage(role=Role.ASSISTANT, text="plan", author_name="magentic_manager")

    async def replan(self, magentic_context: MagenticContext) -> ChatMessage:
        return await self.plan(magentic_context)

    async def create_progress_ledger(self, magentic_context: MagenticContext) -> MagenticProgressLedger:
        participants = list(magentic_context.participant_descriptions.keys())
        target = participants[0] if participants else "agent"
        if self._round == 0:
            self._round += 1
            return MagenticProgressLedger(
                is_request_satisfied=MagenticProgressLedgerItem(reason="", answer=False),
                is_in_loop=MagenticProgressLedgerItem(reason="", answer=False),
                is_progress_being_made=MagenticProgressLedgerItem(reason="", answer=True),
                next_speaker=MagenticProgressLedgerItem(reason="", answer=target),
                instruction_or_question=MagenticProgressLedgerItem(reason="", answer="respond"),
            )
        return MagenticProgressLedger(
            is_request_satisfied=MagenticProgressLedgerItem(reason="", answer=True),
            is_in_loop=MagenticProgressLedgerItem(reason="", answer=False),
            is_progress_being_made=MagenticProgressLedgerItem(reason="", answer=True),
            next_speaker=MagenticProgressLedgerItem(reason="", answer=target),
            instruction_or_question=MagenticProgressLedgerItem(reason="", answer=""),
        )

    async def prepare_final_answer(self, magentic_context: MagenticContext) -> ChatMessage:
        return ChatMessage(role=Role.ASSISTANT, text="final", author_name="magentic_manager")


async def test_group_chat_builder_basic_flow() -> None:
    selector = make_sequence_selector()
    alpha = StubAgent("alpha", "ack from alpha")
    beta = StubAgent("beta", "ack from beta")

    workflow = (
        GroupChatBuilder()
        .set_speaker_selector(selector, display_name="manager", final_message="done")
        .participants(alpha=alpha, beta=beta)
        .build()
    )

    outputs: list[ChatMessage] = []
    async for event in workflow.run_stream("coordinate task"):
        if isinstance(event, WorkflowOutputEvent):
            data = event.data
            if isinstance(data, ChatMessage):
                outputs.append(data)

    assert len(outputs) == 1
    assert outputs[0].text == "done"
    assert outputs[0].author_name == "manager"


async def test_magentic_builder_returns_workflow_and_runs() -> None:
    manager = StubMagenticManager()
    agent = StubAgent("writer", "first draft")

    workflow = MagenticBuilder().participants(writer=agent).with_standard_manager(manager=manager).build()

    assert isinstance(workflow, Workflow)

    outputs: list[ChatMessage] = []
    orchestrator_events: list[MagenticOrchestratorMessageEvent] = []
    agent_events: list[MagenticAgentMessageEvent] = []
    start_message = MagenticStartMessage.from_string("compose summary")
    async for event in workflow.run_stream(start_message):
        if isinstance(event, MagenticOrchestratorMessageEvent):
            orchestrator_events.append(event)
        if isinstance(event, MagenticAgentMessageEvent):
            agent_events.append(event)
        if isinstance(event, WorkflowOutputEvent):
            msg = event.data
            if isinstance(msg, ChatMessage):
                outputs.append(msg)

    assert outputs, "Expected a final output message"
    final = outputs[-1]
    assert final.text == "final"
    assert final.author_name == "magentic_manager"
    assert orchestrator_events, "Expected orchestrator events to be emitted"
    assert agent_events, "Expected agent message events to be emitted"


async def test_group_chat_as_agent_accepts_conversation() -> None:
    selector = make_sequence_selector()
    alpha = StubAgent("alpha", "ack from alpha")
    beta = StubAgent("beta", "ack from beta")

    workflow = (
        GroupChatBuilder()
        .set_speaker_selector(selector, display_name="manager", final_message="done")
        .participants(alpha=alpha, beta=beta)
        .build()
    )

    agent = workflow.as_agent(name="group-chat-agent")
    conversation = [
        ChatMessage(role=Role.USER, text="kickoff", author_name="user"),
        ChatMessage(role=Role.ASSISTANT, text="noted", author_name="alpha"),
    ]
    response = await agent.run(conversation)

    assert response.messages, "Expected agent conversation output"


async def test_magentic_as_agent_accepts_conversation() -> None:
    manager = StubMagenticManager()
    writer = StubAgent("writer", "draft")

    workflow = MagenticBuilder().participants(writer=writer).with_standard_manager(manager=manager).build()

    agent = workflow.as_agent(name="magentic-agent")
    conversation = [
        ChatMessage(role=Role.SYSTEM, text="Guidelines", author_name="system"),
        ChatMessage(role=Role.USER, text="Summarize the findings", author_name="requester"),
    ]
    response = await agent.run(conversation)

    assert isinstance(response, AgentRunResponse)
