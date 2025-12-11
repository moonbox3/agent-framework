# Multi-Agent Workflow Sample

This sample demonstrates how to orchestrate multiple agents in a workflow using the `InvokeAzureAgent` action.

## What This Sample Shows

- Invoking multiple agents in sequence
- Passing data between agent calls
- Using agent responses in subsequent actions
- Handling external loop patterns for agent responses

## Files

- `workflow.yaml` - The declarative workflow definition
- `main.py` - Python script that loads and runs the workflow with mock agents

## Running the Sample

1. Ensure you have the package installed:
   ```bash
   cd python
   pip install -e packages/agent-framework-declarative
   ```

2. Run the sample:
   ```bash
   python main.py
   ```

## How It Works

The workflow demonstrates a research assistant pattern:

1. **Topic Input**: Receives a research topic from the user
2. **Research Agent**: Calls an agent to research the topic
3. **Summary Agent**: Takes the research and creates a summary
4. **Output**: Provides the final summarized research

## Key Concepts

### InvokeAzureAgent

The `InvokeAzureAgent` action is used to call AI agents during workflow execution:

```yaml
- kind: InvokeAzureAgent
  id: call_research_agent
  model:
    configuration:
      api_type: azure_openai
  prompt: "Research the following topic: {topic}"
  outputFormat:
    variableName: turn.research_result
```

### External Loop Pattern

For agent invocations, the workflow may yield an `ExternalLoopEvent` that signals the runner to:
1. Send the prompt to an actual agent
2. Collect the agent's response
3. Resume the workflow with the response

This allows integration with real AI services while keeping the workflow declarative.

## Note

This sample uses mock agent responses for demonstration. In a production scenario, you would configure actual Azure AI agents or other LLM services.
