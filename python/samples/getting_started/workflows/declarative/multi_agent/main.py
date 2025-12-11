# Copyright (c) Microsoft. All rights reserved.

"""
Run the multi-agent workflow sample.

Usage:
    python main.py

Demonstrates orchestrating multiple agents in a workflow.

Note: This sample uses mock agent responses for demonstration.
In production, configure actual Azure OpenAI endpoints.
"""

import asyncio
from pathlib import Path

from agent_framework._workflows import TextOutputEvent
from agent_framework_declarative._workflows import WorkflowFactory
from agent_framework_declarative._workflows._actions_agents import ExternalLoopEvent


# Mock agent responses for demonstration
MOCK_RESPONSES = {
    "research_agent": """
Based on extensive research, here are the key findings about artificial intelligence:

1. **Machine Learning Foundation**: AI systems learn from data using algorithms
   that identify patterns and make predictions without explicit programming.

2. **Neural Networks**: Deep learning uses layered neural networks inspired by
   the human brain, enabling breakthroughs in image recognition and NLP.

3. **Current Applications**: AI is widely used in healthcare diagnostics,
   autonomous vehicles, financial analysis, and customer service chatbots.

4. **Ethical Considerations**: Important concerns include bias in training data,
   privacy implications, and the impact on employment.

5. **Future Trends**: Emerging areas include AGI research, AI safety,
   multimodal models, and AI-human collaboration frameworks.
""",
    "summary_agent": """
- AI learns from data patterns without explicit programming
- Deep learning neural networks enable breakthroughs in vision and language
- Current applications span healthcare, vehicles, finance, and customer service
- Key ethical concerns: bias, privacy, and employment impact
- Future focus: AGI, safety, multimodal models, and human collaboration
""",
}


async def main():
    """Run the multi-agent workflow with mock responses."""
    # Load the workflow from YAML
    workflow_path = Path(__file__).parent / "workflow.yaml"
    workflow = WorkflowFactory.from_yaml(workflow_path)

    print("=== Multi-Agent Research Workflow Demo ===")
    print("(Using mock agent responses for demonstration)")
    print()

    # Track which agent we're calling
    agent_call_count = 0
    agent_names = ["research_agent", "summary_agent"]

    # Run the workflow
    async for event in workflow.run_async(
        inputs={"topic": "Artificial Intelligence and its applications"}
    ):
        if isinstance(event, TextOutputEvent):
            print(f"[Workflow]: {event.text}")
            print()
        elif isinstance(event, ExternalLoopEvent):
            # In a real scenario, you would:
            # 1. Extract the prompt from the event
            # 2. Send it to the actual AI agent
            # 3. Resume the workflow with the response

            if agent_call_count < len(agent_names):
                agent_name = agent_names[agent_call_count]
                print(f"[System] Agent invocation detected: {agent_name}")
                print(f"[System] Prompt: {event.prompt[:100]}...")
                print()
                print(f"[{agent_name}] Response:")
                print(MOCK_RESPONSES[agent_name])
                print()
                agent_call_count += 1

    print("=== Workflow Complete ===")
    print()
    print("Note: This demo uses mock responses. In production,")
    print("configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT")
    print("environment variables for actual agent calls.")


if __name__ == "__main__":
    asyncio.run(main())
