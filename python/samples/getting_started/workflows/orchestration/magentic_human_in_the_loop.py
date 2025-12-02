# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from typing import cast

from agent_framework import (
    MAGENTIC_EVENT_TYPE_AGENT_DELTA,
    MAGENTIC_EVENT_TYPE_ORCHESTRATOR,
    AgentRunUpdateEvent,
    ChatAgent,
    ChatMessage,
    MagenticBuilder,
    MagenticHumanInterventionDecision,
    MagenticHumanInterventionKind,
    MagenticHumanInterventionReply,
    MagenticHumanInterventionRequest,
    RequestInfoEvent,
    WorkflowOutputEvent,
)
from agent_framework.openai import OpenAIChatClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Sample: Magentic Orchestration with Human-in-the-Loop

This sample demonstrates the unified human intervention pattern in Magentic workflows.
All HITL scenarios use a single request/reply type with a `kind` field:

1. PLAN_REVIEW: Human reviews and approves/revises the initial plan before execution
2. TOOL_APPROVAL: Agents request approval for tool/function calls (always surfaced)
3. STALL: Human intervention when workflow detects agents are not making progress

Key types:
- MagenticHumanInterventionRequest: Unified request with `kind` field
- MagenticHumanInterventionReply: Unified reply with `decision` field
- MagenticHumanInterventionKind: PLAN_REVIEW, TOOL_APPROVAL, STALL
- MagenticHumanInterventionDecision: APPROVE, REVISE, CONTINUE, REPLAN, GUIDANCE

Key behaviors demonstrated:
- with_plan_review(): Enables plan review before starting
- with_human_input_on_stall(): Enables human intervention when workflow stalls
- Tool approval requests are always surfaced (foundational behavior)
- Single unified handler for all intervention types

Use cases:
- Plan review: Validating the orchestrator's understanding of requirements
- Tool approval: High-stakes decisions requiring human approval
- Stall intervention: Complex tasks where human guidance helps agents get back on track

Prerequisites:
- OpenAI credentials configured for `OpenAIChatClient`.
"""


async def main() -> None:
    # Create a research agent that may need human clarification
    researcher_agent = ChatAgent(
        name="ResearcherAgent",
        description="Specialist in research and information gathering",
        instructions=(
            "You are a Researcher. When you encounter ambiguous or unclear aspects "
            "of a research task, ask for clarification before proceeding. "
            "You find information without additional computation or quantitative analysis."
        ),
        chat_client=OpenAIChatClient(model_id="gpt-4o"),
    )

    # Create an analyst agent that processes and summarizes information
    analyst_agent = ChatAgent(
        name="AnalystAgent",
        description="Data analyst who processes and summarizes research findings",
        instructions=(
            "You are an Analyst. You take research findings and create clear, "
            "structured summaries with actionable insights. When the analysis "
            "direction is unclear, request guidance from the user."
        ),
        chat_client=OpenAIChatClient(model_id="gpt-4o"),
    )

    # Create a manager agent with specific model options for deterministic planning
    manager_agent = ChatAgent(
        name="MagenticManager",
        description="Orchestrator that coordinates the research and analysis workflow",
        instructions="You coordinate a team to complete research and analysis tasks efficiently.",
        chat_client=OpenAIChatClient(model_id="gpt-4o"),
    )

    print("\nBuilding Magentic Workflow with HITL capabilities...")

    # Build workflow with plan review and stall intervention enabled
    # Note: Tool approval (user_input_requests) is always surfaced - no need to enable
    workflow = (
        MagenticBuilder()
        .participants(researcher=researcher_agent, analyst=analyst_agent)
        .with_standard_manager(
            agent=manager_agent,  # Use agent for control over model behavior
            max_round_count=10,
            max_stall_count=3,
            max_reset_count=2,
        )
        .with_plan_review()  # Enable plan review before execution
        .with_human_input_on_stall()  # Enable human intervention when workflow stalls
        .build()
    )

    task = (
        "Research the latest developments in sustainable aviation fuel (SAF) technology. "
        "Focus on production methods, current adoption rates, and major challenges. "
        "Then analyze the findings and provide recommendations for airline companies "
        "considering SAF adoption."
    )

    print(f"\nTask: {task}")
    print("\nStarting workflow execution...")
    print("=" * 60)

    try:
        pending_request: RequestInfoEvent | None = None
        pending_responses: dict[str, object] | None = None
        completed = False
        workflow_output: str | None = None

        last_stream_agent_id: str | None = None
        stream_line_open: bool = False

        while not completed:
            # Use streaming for both initial run and response sending
            if pending_responses is not None:
                stream = workflow.send_responses_streaming(pending_responses)
            else:
                stream = workflow.run_stream(task)

            # Collect events from the stream
            async for event in stream:
                if isinstance(event, AgentRunUpdateEvent):
                    props = event.data.additional_properties if event.data else None
                    event_type = props.get("magentic_event_type") if props else None

                    if event_type == MAGENTIC_EVENT_TYPE_ORCHESTRATOR:
                        kind = props.get("orchestrator_message_kind", "") if props else ""
                        text = event.data.text if event.data else ""
                        if stream_line_open:
                            print()
                            stream_line_open = False
                        print(f"\n[ORCHESTRATOR: {kind}]\n{text}\n{'-' * 40}")
                    elif event_type == MAGENTIC_EVENT_TYPE_AGENT_DELTA:
                        agent_id = props.get("agent_id", "unknown") if props else "unknown"
                        if last_stream_agent_id != agent_id or not stream_line_open:
                            if stream_line_open:
                                print()
                            print(f"\n[{agent_id}]: ", end="", flush=True)
                            last_stream_agent_id = agent_id
                            stream_line_open = True
                        if event.data and event.data.text:
                            print(event.data.text, end="", flush=True)

                # Handle Plan Review Request
                elif isinstance(event, RequestInfoEvent) and event.request_type is MagenticHumanInterventionRequest:
                    if stream_line_open:
                        print()
                        stream_line_open = False
                    pending_request = event
                    req = cast(MagenticHumanInterventionRequest, event.data)

                    if req.kind == MagenticHumanInterventionKind.PLAN_REVIEW:
                        print("\n" + "=" * 60)
                        print("PLAN REVIEW REQUEST")
                        print("=" * 60)
                        if req.plan_text:
                            print(f"\nProposed Plan:\n{req.plan_text}")
                        print()

                    elif req.kind == MagenticHumanInterventionKind.TOOL_APPROVAL:
                        print("\n" + "=" * 60)
                        print("TOOL APPROVAL REQUESTED")
                        print("=" * 60)
                        print(f"\nAgent: {req.agent_id}")
                        print(f"Request: {req.prompt}")
                        if req.context:
                            print(f"Context: {req.context}")
                        print()

                    elif req.kind == MagenticHumanInterventionKind.STALL:
                        print("\n" + "=" * 60)
                        print("STALL INTERVENTION REQUESTED")
                        print("=" * 60)
                        print(f"\nWorkflow appears stalled after {req.stall_count} rounds")
                        print(f"Reason: {req.stall_reason}")
                        if req.last_agent:
                            print(f"Last active agent: {req.last_agent}")
                        if req.plan_text:
                            print(f"\nCurrent plan:\n{req.plan_text}")
                        print()

                elif isinstance(event, WorkflowOutputEvent):
                    if stream_line_open:
                        print()
                        stream_line_open = False
                    workflow_output = event.data if event.data else None
                    completed = True

            if stream_line_open:
                print()
                stream_line_open = False
            pending_responses = None

            # Handle pending requests
            if pending_request is not None:
                req = cast(MagenticHumanInterventionRequest, pending_request.data)
                reply: MagenticHumanInterventionReply | None = None

                if req.kind == MagenticHumanInterventionKind.PLAN_REVIEW:
                    # Handle plan review
                    print("Plan review options:")
                    print("1. approve - Approve the plan as-is")
                    print("2. approve with comments - Approve with feedback")
                    print("3. revise - Request revision with feedback")
                    print("4. exit - Exit the workflow")

                    while True:
                        choice = input("Enter your choice (1-4): ").strip().lower()  # noqa: ASYNC250
                        if choice in ["approve", "1"]:
                            reply = MagenticHumanInterventionReply(decision=MagenticHumanInterventionDecision.APPROVE)
                            break
                        if choice in ["approve with comments", "2"]:
                            comments = input("Enter your comments: ").strip()  # noqa: ASYNC250
                            reply = MagenticHumanInterventionReply(
                                decision=MagenticHumanInterventionDecision.APPROVE,
                                comments=comments if comments else None,
                            )
                            break
                        if choice in ["revise", "3"]:
                            comments = input("Enter feedback for revision: ").strip()  # noqa: ASYNC250
                            reply = MagenticHumanInterventionReply(
                                decision=MagenticHumanInterventionDecision.REVISE,
                                comments=comments if comments else None,
                            )
                            break
                        if choice in ["exit", "4"]:
                            print("Exiting workflow...")
                            return
                        print("Invalid choice. Please enter a number 1-4.")

                elif req.kind == MagenticHumanInterventionKind.TOOL_APPROVAL:
                    # Handle tool approval request
                    print("Tool approval options:")
                    print("1. approve - Allow the tool call")
                    print("2. deny - Reject the tool call")
                    print("3. guidance - Provide guidance instead")
                    print("4. exit - Exit the workflow")

                    while True:
                        choice = input("Enter your choice (1-4): ").strip().lower()  # noqa: ASYNC250
                        if choice in ["approve", "1"]:
                            reply = MagenticHumanInterventionReply(decision=MagenticHumanInterventionDecision.APPROVE)
                            break
                        if choice in ["deny", "2"]:
                            reply = MagenticHumanInterventionReply(decision=MagenticHumanInterventionDecision.REJECT)
                            break
                        if choice in ["guidance", "3"]:
                            guidance = input("Enter your guidance: ").strip()  # noqa: ASYNC250
                            reply = MagenticHumanInterventionReply(
                                decision=MagenticHumanInterventionDecision.GUIDANCE,
                                comments=guidance if guidance else None,
                            )
                            break
                        if choice in ["exit", "4"]:
                            print("Exiting workflow...")
                            return
                        print("Invalid choice. Please enter a number 1-4.")

                elif req.kind == MagenticHumanInterventionKind.STALL:
                    # Handle stall intervention request
                    print("Stall intervention options:")
                    print("1. continue - Continue with current plan (reset stall counter)")
                    print("2. replan - Trigger automatic replanning")
                    print("3. guidance - Provide guidance to help agents get back on track")
                    print("4. exit - Exit the workflow")

                    while True:
                        choice = input("Enter your choice (1-4): ").strip().lower()  # noqa: ASYNC250
                        if choice in ["continue", "1"]:
                            reply = MagenticHumanInterventionReply(decision=MagenticHumanInterventionDecision.CONTINUE)
                            break
                        if choice in ["replan", "2"]:
                            reply = MagenticHumanInterventionReply(decision=MagenticHumanInterventionDecision.REPLAN)
                            break
                        if choice in ["guidance", "3"]:
                            guidance = input("Enter your guidance: ").strip()  # noqa: ASYNC250
                            reply = MagenticHumanInterventionReply(
                                decision=MagenticHumanInterventionDecision.GUIDANCE,
                                comments=guidance if guidance else None,
                            )
                            break
                        if choice in ["exit", "4"]:
                            print("Exiting workflow...")
                            return
                        print("Invalid choice. Please enter a number 1-4.")

                if reply is not None:
                    pending_responses = {pending_request.request_id: reply}
                pending_request = None

        # Show final result
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETED")
        print("=" * 60)
        if workflow_output:
            # workflow_output is a list[ChatMessage]
            messages = cast(list[ChatMessage], workflow_output)
            if messages:
                final_msg = messages[-1]
                print(f"\nFinal Result:\n{final_msg.text}")

    except Exception as e:
        print(f"Workflow execution failed: {e}")
        logger.exception("Workflow exception", exc_info=e)


if __name__ == "__main__":
    asyncio.run(main())
