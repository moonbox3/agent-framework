# Declarative Workflows Samples

This directory contains samples demonstrating how to use declarative YAML workflows with the Python Agent Framework.

## Overview

Declarative workflows allow you to define multi-agent orchestration patterns in YAML, including:
- Variable manipulation and state management
- Control flow (loops, conditionals, branching)
- Agent invocations
- Human-in-the-loop patterns

## Samples

| Sample | Description |
|--------|-------------|
| [simple_workflow](./simple_workflow/) | Basic workflow with variable setting, conditionals, and loops |
| [conditional_workflow](./conditional_workflow/) | Nested conditional branching based on user input |
| [human_in_loop](./human_in_loop/) | Interactive workflows that request user input |
| [multi_agent](./multi_agent/) | Orchestrating multiple AI agents in sequence |
| [student_teacher](./student_teacher/) | Iterative two-agent conversation with loop control |
| [marketing](./marketing/) | Sequential multi-agent pipeline (Analyst, Writer, Editor) |
| [function_tools](./function_tools/) | Agent with function tools in interactive loop |

## Prerequisites

```bash
pip install agent-framework-declarative
```

## Running Samples

Each sample directory contains:
- `workflow.yaml` - The declarative workflow definition
- `main.py` - Python code to load and execute the workflow
- `README.md` - Sample-specific documentation

To run a sample:

```bash
cd <sample_directory>
python main.py
```

## Workflow Structure

A basic workflow YAML file looks like:

```yaml
name: my-workflow
description: A simple workflow example

actions:
  - kind: SetValue
    path: turn.greeting
    value: Hello, World!
    
  - kind: SendActivity
    activity:
      text: =turn.greeting
```

## Action Types

### Variable Actions
- `SetValue` - Set a variable in state
- `SetVariable` - Set a variable (.NET style naming)
- `AppendValue` - Append to a list
- `ResetVariable` - Clear a variable

### Control Flow
- `If` - Conditional branching
- `Switch` - Multi-way branching
- `Foreach` - Iterate over collections
- `RepeatUntil` - Loop until condition
- `GotoAction` - Jump to labeled action

### Output
- `SendActivity` - Send text/attachments to user
- `EmitEvent` - Emit custom events

### Agent Invocation
- `InvokeAzureAgent` - Call an Azure AI agent
- `InvokePromptAgent` - Call a local prompt agent

### Human-in-Loop
- `Question` - Request user input
- `WaitForInput` - Pause for external input

## Learn More

- [Design Document](../../../../docs/design/declarative-python-workflows.md)
- [Workflow Samples (YAML only)](../../../../workflow-samples/)
