# Agent Framework Orchestrations

Orchestration patterns for Microsoft Agent Framework. This package provides high-level builders for common multi-agent workflow patterns.

## Installation

```bash
pip install agent-framework-orchestrations --pre
```

## Orchestration Patterns

### SequentialBuilder

Chain agents/executors in sequence, passing conversation context along:

```python
from agent_framework.orchestrations import SequentialBuilder

workflow = SequentialBuilder(participants=[agent1, agent2, agent3]).build()

# Preserve agent1 and agent2 as visible progress, while agent3 remains terminal output.
workflow = SequentialBuilder(
    participants=[agent1, agent2, agent3],
    intermediate_output_from=[agent1, agent2],
).build()
```

### ConcurrentBuilder

Fan-out to multiple agents in parallel, then aggregate results:

```python
from agent_framework.orchestrations import ConcurrentBuilder

workflow = ConcurrentBuilder(participants=[agent1, agent2, agent3]).build()
```

### HandoffBuilder

Decentralized agent routing where agents decide handoff targets:

```python
from agent_framework.orchestrations import HandoffBuilder

workflow = (
    HandoffBuilder()
    .participants([triage, billing, support])
    .with_start_agent(triage)
    .build()
)
```

### GroupChatBuilder

Orchestrator-directed multi-agent conversations:

```python
from agent_framework.orchestrations import GroupChatBuilder

workflow = GroupChatBuilder(
    participants=[agent1, agent2],
    selection_func=my_selector,
    intermediate_output_from=[agent1, agent2],
).build()
```

### MagenticBuilder

Sophisticated multi-agent orchestration using the Magentic One pattern:

```python
from agent_framework.orchestrations import MagenticBuilder

workflow = MagenticBuilder(
    participants=[researcher, writer, reviewer],
    manager_agent=manager_agent,
    intermediate_output_from=[researcher, writer, reviewer],
).build()
```

## Output Designation

Orchestration builders expose workflow output selection using participant names:

- `final_output_from` designates participant emissions as terminal workflow `output` events.
- `intermediate_output_from` designates participant emissions as visible workflow `intermediate` events.
- Unlisted participant emissions are hidden in explicit designation mode.

If neither list is provided, each builder uses its default terminal contract. Sequential emits the final participant;
Concurrent, GroupChat, and Magentic emit their final aggregator/orchestrator/manager output; Handoff emits
participants. Explicit designation is validated for empty lists, duplicates, output/intermediate overlap, and unknown
participants.

When an orchestration is wrapped with `workflow.as_agent()`, terminal workflow output becomes normal response text.
Intermediate workflow output becomes `text_reasoning` content so callers can inspect progress without changing
terminal `.text` behavior.

## Documentation

For more information, see the [Agent Framework documentation](https://aka.ms/agent-framework).
