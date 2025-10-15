# Test specialist-to-specialist handoffs

import asyncio

from agent_framework import HandoffBuilder
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


async def main():
    """Test specialist-to-specialist handoffs."""
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

    # Create three agents: triage, replacement, and delivery
    triage = chat_client.create_agent(
        instructions=("You are a triage agent. Route replacement issues to 'replacement_agent'. Be concise."),
        name="triage_agent",
    )

    replacement = chat_client.create_agent(
        instructions=(
            "You handle product replacements. If you need delivery/shipping info, "
            "hand off to 'delivery_agent' by calling handoff_to_delivery_agent. "
            "Be concise."
        ),
        name="replacement_agent",
    )

    delivery = chat_client.create_agent(
        instructions=("You handle delivery and shipping inquiries. Provide tracking info. Be concise."),
        name="delivery_agent",
    )

    # Build workflow with specialist-to-specialist handoffs
    workflow = (
        HandoffBuilder(
            name="multi_tier_support",
            participants=[triage, replacement, delivery],
        )
        .starting_agent("triage_agent")
        .with_handoffs({
            "triage_agent": ["replacement_agent", "delivery_agent"],
            "replacement_agent": ["delivery_agent"],  # Replacement can hand off to delivery
        })
        .with_termination_condition(lambda conv: sum(1 for m in conv if m.role.value == "user") >= 3)
        .build()
    )

    print("\n[Starting workflow with specialist-to-specialist handoffs enabled]\n")

    # Start workflow
    events = []
    async for event in workflow.run_stream("I need a replacement for my damaged item and want to check shipping"):
        events.append(event)
        print(f"Event: {type(event).__name__}")

    print(f"\nTotal events: {len(events)}")
    print("\nWorkflow complete!")


if __name__ == "__main__":
    asyncio.run(main())
