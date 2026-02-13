# Copyright (c) Microsoft. All rights reserved.

"""AG-UI handoff workflow demo backend.

This demo exposes a deterministic HandoffBuilder workflow through AG-UI.
It intentionally includes two interrupt styles:

1. Tool approval (`function_approval_request`) for `submit_refund`
2. Follow-up human input (`HandoffAgentUserRequest`) from the order specialist

Run this server and pair it with the frontend in `../frontend`.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import random
import re
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from typing import Any

import uvicorn
from agent_framework import (
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
    Workflow,
    tool,
)
from agent_framework.ag_ui import AgentFrameworkWorkflow, add_agent_framework_fastapi_endpoint
from agent_framework.orchestrations import HandoffBuilder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@tool(approval_mode="always_require")
def submit_refund(refund_description: str, amount: str, order_id: str) -> str:
    """Capture a refund request for manual review before processing."""
    return f"refund recorded for order {order_id} (amount: {amount}) with details: {refund_description}"


@tool(approval_mode="never_require")
def lookup_order_details(order_id: str) -> dict[str, str]:
    """Return synthetic order details for a given order ID."""
    normalized_order_id = "".join(ch for ch in order_id if ch.isdigit()) or order_id
    rng = random.Random(normalized_order_id)
    catalog = [
        "Wireless Headphones",
        "Mechanical Keyboard",
        "Gaming Mouse",
        "27-inch Monitor",
        "USB-C Dock",
        "Bluetooth Speaker",
        "Laptop Stand",
    ]
    item_name = catalog[rng.randrange(len(catalog))]
    amount = f"${rng.randint(39, 349)}.{rng.randint(0, 99):02d}"
    purchase_date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return {
        "order_id": normalized_order_id,
        "item_name": item_name,
        "amount": amount,
        "currency": "USD",
        "purchase_date": purchase_date,
        "status": "delivered",
    }


class DeterministicHandoffChatClient(ChatMiddlewareLayer[Any], FunctionInvocationLayer[Any], BaseChatClient[Any]):
    """Deterministic client used to make the demo fully reproducible.

    Each agent has a small scripted state machine based on call index.
    """

    def __init__(self, *, agent_name: str) -> None:
        ChatMiddlewareLayer.__init__(self)
        FunctionInvocationLayer.__init__(self)
        BaseChatClient.__init__(self)
        self._agent_name = agent_name
        self._call_index = 0

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        del kwargs
        del options

        contents = self._next_contents(messages)

        if stream:
            return self._build_streaming_response(contents)

        async def _get() -> ChatResponse:
            return ChatResponse(
                messages=[
                    Message(
                        role="assistant",
                        contents=contents,
                        author_name=self._agent_name,
                    )
                ],
                response_id=f"{self._agent_name}-{self._call_index}",
            )

        return _get()

    def _build_streaming_response(
        self,
        contents: list[Content],
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        async def _stream() -> AsyncIterable[ChatResponseUpdate]:
            yield ChatResponseUpdate(
                contents=contents,
                role="assistant",
                author_name=self._agent_name,
                finish_reason="stop",
            )

        def _finalize(updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
            return ChatResponse.from_updates(updates)

        return ResponseStream(_stream(), finalizer=_finalize)

    def _next_contents(self, messages: Sequence[Message]) -> list[Content]:
        call_index = self._call_index
        self._call_index += 1

        if self._agent_name == "triage_agent":
            if call_index == 0:
                return [
                    Content.from_function_call(
                        call_id=f"triage-handoff-{call_index}",
                        name="handoff_to_refund_agent",
                        arguments={"context": "Refund workflow requested by customer."},
                    ),
                    Content.from_text(
                        text=("Triage Agent: I am routing you to the Refund Agent to process the damaged-order refund.")
                    ),
                ]

            return [
                Content.from_text(
                    text=(
                        "Triage Agent: Your refund and replacement requests are complete. "
                        "Let me know if you need anything else."
                    )
                )
            ]

        if self._agent_name == "refund_agent":
            if call_index == 0:
                return [
                    Content.from_text(
                        text=(
                            "Refund Agent: I can help with that. "
                            "Please share your order number so I can locate the purchase."
                        )
                    )
                ]

            if call_index == 1:
                order_id = self._extract_order_id(messages)
                if not order_id:
                    # Stay on the same stage until an order number is provided.
                    self._call_index = call_index
                    return [
                        Content.from_text(
                            text=(
                                "Refund Agent: I still need the order number before I can submit a refund. "
                                "Please send a numeric order ID (for example: 12345)."
                            )
                        )
                    ]

                amount = self._refund_amount_for_order(order_id)
                return [
                    Content.from_function_call(
                        call_id="refund-tool-call-1",
                        name="submit_refund",
                        arguments={
                            "order_id": order_id,
                            "amount": amount,
                            "refund_description": "Headphones arrived cracked and unusable.",
                        },
                    ),
                    Content.from_text(
                        text=(
                            f"Refund Agent: I prepared the refund request for order {order_id} ({amount}). "
                            "Please approve the tool call so I can submit it."
                        )
                    ),
                ]

            if call_index == 2:
                return [
                    Content.from_function_call(
                        call_id=f"refund-handoff-{call_index}",
                        name="handoff_to_order_agent",
                        arguments={"context": "Refund approved. Continue with replacement handling."},
                    ),
                    Content.from_text(
                        text=(
                            "Refund Agent: Refund has been recorded. "
                            "I am routing you to the Order Agent for replacement shipping."
                        )
                    ),
                ]

            return [
                Content.from_text(text="Refund Agent: Refund flow complete on my side."),
            ]

        if self._agent_name == "order_agent":
            if call_index == 0:
                return [
                    Content.from_text(
                        text=(
                            "Order Agent: I can submit a replacement shipment now. "
                            "Would you like standard or expedited shipping?"
                        )
                    )
                ]

            user_preference = self._latest_user_text(messages) or "standard"
            return [
                Content.from_function_call(
                    call_id=f"order-handoff-{call_index}",
                    name="handoff_to_triage_agent",
                    arguments={"context": "Replacement workflow complete."},
                ),
                Content.from_text(
                    text=(
                        "Order Agent: Got it. I submitted the replacement request with "
                        f"{user_preference} shipping. Case complete."
                    )
                ),
            ]

        return [Content.from_text(text=f"{self._agent_name}: ready.")]

    @staticmethod
    def _latest_user_text(messages: Sequence[Message]) -> str | None:
        for message in reversed(messages):
            if message.role != "user":
                continue
            if message.text:
                return message.text
        return None

    @staticmethod
    def _extract_order_id(messages: Sequence[Message]) -> str | None:
        """Extract a numeric order ID from the latest user text."""
        latest_text = DeterministicHandoffChatClient._latest_user_text(messages)
        if not latest_text:
            return None

        match = re.search(r"\b(\d{4,12})\b", latest_text)
        return match.group(1) if match else None

    @staticmethod
    def _refund_amount_for_order(order_id: str) -> str:
        """Derive a deterministic pseudo-amount from an order ID."""
        digit_sum = sum(int(ch) for ch in order_id if ch.isdigit())
        dollars = 25 + (digit_sum % 95)
        cents = (digit_sum * 7) % 100
        return f"${dollars}.{cents:02d}"


def create_agents() -> tuple[Agent, Agent, Agent]:
    """Create triage, refund, and order agents for the handoff workflow."""

    from agent_framework.azure import AzureOpenAIResponsesClient
    from azure.identity import AzureCliCredential

    client = AzureOpenAIResponsesClient(
        project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        deployment_name=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=AzureCliCredential(),
    )

    triage = Agent(
        id="triage_agent",
        name="triage_agent",
        instructions=(
            "You are the customer support triage agent. Route refund requests to refund_agent using handoff tools."
        ),
        client=client,
    )

    refund = Agent(
        id="refund_agent",
        name="refund_agent",
        instructions=(
            "You are the refund specialist.\n"
            "Workflow policy:\n"
            "1. If order_id is missing, ask only for order_id.\n"
            "2. Once order_id is available, call lookup_order_details(order_id) to retrieve item and amount.\n"
            "3. Do not ask the customer how much they paid unless lookup_order_details fails.\n"
            "4. Gather a short refund reason from user context.\n"
            "5. Call submit_refund with order_id, amount (from lookup), and refund_description.\n"
            "6. After approval and successful submission, handoff to order_agent."
        ),
        client=client,
        tools=[lookup_order_details, submit_refund],
    )

    order = Agent(
        id="order_agent",
        name="order_agent",
        instructions=("You are the order specialist. Ask the user for shipping preference and complete replacement."),
        client=client,
    )

    return triage, refund, order


def _termination_condition(conversation: list[Message]) -> bool:
    """Stop after the order specialist confirms replacement completion."""

    for message in reversed(conversation):
        if message.role != "assistant":
            continue
        if message.author_name != "order_agent":
            continue
        if "case complete" in (message.text or "").lower():
            return True
    return False


def create_handoff_workflow() -> Workflow:
    """Build the demo HandoffBuilder workflow."""

    triage, refund, order = create_agents()
    return (
        HandoffBuilder(
            name="ag_ui_handoff_workflow_demo",
            participants=[triage, refund, order],
            termination_condition=_termination_condition,
        )
        .with_start_agent(triage)
        .build()
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="AG-UI Handoff Workflow Demo")

    cors_origins = [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173").split(",") if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    demo_workflow = AgentFrameworkWorkflow(
        workflow_factory=lambda _thread_id: create_handoff_workflow(),
        name="ag_ui_handoff_workflow_demo",
        description="Deterministic handoff workflow demo with tool approvals and request_info resumes.",
    )

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=demo_workflow,
        path="/handoff_demo",
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    return app


app = create_app()


def main() -> None:
    """Run the AG-UI demo backend."""

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Add file handler for persistent logging
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ag_ui_handoff_demo.log")
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5  # 10MB max size, keep 5 backups
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
        print(f"Logging to file: {log_file}")
    except Exception as e:
        print(f"Warning: Failed to set up file logging: {e}")

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8891"))

    print(f"AG-UI handoff demo backend running at http://{host}:{port}")
    print("AG-UI endpoint: POST /handoff_demo")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
