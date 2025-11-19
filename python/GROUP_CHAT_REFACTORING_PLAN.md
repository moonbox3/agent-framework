# Group Chat Architecture Refactoring Plan

## Progress Update

**Status: Phase 4 Complete - Core Implementation Done**

Completed phases:
- ✅ Phase 1: Manager Communication Protocol (ManagerSelectionRequest/Response dataclasses)
- ✅ Phase 2: GroupChatBuilder API (set_manager() method, validation)
- ✅ Phase 3: Workflow Assembly (manager wired as participant node)
- ✅ Phase 4: Orchestrator Logic (manager query routing, response handling, parsing)
- 🔄 Phase 5: Documentation and Samples (in progress - created initial test sample)
- ⏳ Phase 6: Deprecation and Cleanup (pending)
- ⏳ Phase 7: Testing and Validation (pending)

Key accomplishments:
1. Created `ManagerSelectionRequest` and `ManagerSelectionResponse` dataclasses for manager communication
2. Implemented `set_manager()` method accepting AgentProtocol/Executor instances
3. Updated workflow assembly to wire manager as bidirectional participant node
4. Implemented manager response parsing with fallback strategies (structured output, JSON text)
5. Updated orchestrator to route manager queries through workflow graph
6. Added helper methods: `_is_manager_agent()`, `_handle_manager_response()`, `_parse_manager_selection()`
7. Created test sample demonstrating new agent-based manager API

Next steps:
- Update all existing samples to use new API
- Remove `_PromptBasedGroupChatManager` and related code
- Comprehensive testing and validation

## Executive Summary

Remove the internal `_PromptBasedGroupChatManager` class and refactor `set_prompt_based_manager()` to accept **AgentProtocol instances** instead of ChatClient. The manager/coordinator will be wired as a full participant node in the workflow graph with bidirectional edges to all other participants.

## Current Architecture (Before)

### Flow
```
User Input → Orchestrator → Manager (callable/function) → Orchestrator → Participant → Orchestrator
                   ↑__________________________________________________|
```

### Key Components
- `_PromptBasedGroupChatManager`: Internal class that wraps ChatClient + instructions
- `set_prompt_based_manager(chat_client, instructions, ...)`: Builder method accepting ChatClient
- Manager is a **callable** (`_GroupChatManagerFn`) invoked directly by orchestrator
- Manager returns `GroupChatDirective` with next speaker selection
- Orchestrator state machine queries manager, routes to participants, collects responses

### Problems
1. Manager is not a first-class workflow participant (just a callable)
2. Manager cannot leverage AgentProtocol infrastructure (tools, context, observability)
3. ChatClient-based approach limits flexibility (no streaming, no complex agent logic)
4. Tight coupling between orchestrator and manager implementation
5. Manager decisions are opaque to workflow graph visualization

## Target Architecture (After)

### Flow
```
User Input → Orchestrator → Manager Agent (node) → Orchestrator → Participant → Orchestrator
                                     ↓                                  |
                              All Participants ←-----------------------+
                                     ↑
                                     |
                              All Participants
```

### Key Changes

1. **Manager becomes a full workflow node**
   - Manager is an AgentProtocol instance wrapped as AgentExecutor
   - Wired into the workflow graph like any other participant
   - Bidirectional edges: Orchestrator ↔ Manager ↔ All Participants

2. **API signature change**
   ```python
   # OLD
   .set_prompt_based_manager(
       chat_client: ChatClientProtocol,
       instructions: str | None = None,
       display_name: str | None = None,
       chat_options: ChatOptions | None = None,
   )
   
   # NEW
   .set_manager(
       manager: AgentProtocol | Executor,
       display_name: str | None = None,
   )
   ```

3. **Remove internal manager implementations**
   - Delete `_PromptBasedGroupChatManager` class
   - Delete `ManagerDirectiveModel` Pydantic model
   - Delete `DEFAULT_MANAGER_INSTRUCTIONS` and `DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT`
   - Keep `_SpeakerSelectorAdapter` for `select_speakers()` compatibility

4. **Manager agent responsibilities**
   - Receive full conversation context from orchestrator
   - Analyze state and select next participant
   - Return decision as structured output (via tools or response format)
   - Surface decision rationale through standard agent observability

## Detailed Implementation Plan

### Phase 1: Define Manager Communication Protocol

#### 1.1 Manager Request/Response Messages
```python
@dataclass
class ManagerSelectionRequest:
    """Request sent to manager agent for next speaker selection."""
    task: ChatMessage
    participants: dict[str, str]  # name → description
    conversation: list[ChatMessage]
    round_index: int
    metadata: dict[str, Any] | None = None
```

```python
@dataclass
class ManagerSelectionResponse:
    """Response from manager agent with selection decision."""
    selected_participant: str | None  # None = finish
    instruction: str | None = None  # Optional instruction for participant
    finish: bool = False
    final_message: ChatMessage | None = None
    metadata: dict[str, Any] | None = None
```

#### 1.2 Manager Agent Contract
- Manager agent MUST produce structured output matching `ManagerSelectionResponse`
- Manager agent SHOULD use `response_format` for reliable parsing
- Manager agent receives conversation via standard AgentExecutorRequest
- Manager agent returns selection via standard AgentExecutorResponse

### Phase 2: Update GroupChatBuilder API

#### 2.1 Replace `set_prompt_based_manager()`
```python
def set_manager(
    self,
    manager: AgentProtocol | Executor,
    *,
    display_name: str | None = None,
) -> "GroupChatBuilder":
    """Configure the manager/coordinator agent for group chat orchestration.
    
    The manager coordinates participants by selecting who speaks next based on
    conversation state and task requirements. The manager is a full workflow
    participant with access to all agent infrastructure.
    
    Args:
        manager: Agent or executor responsible for speaker selection and coordination
        display_name: Optional name for manager messages in conversation history
    
    Returns:
        Self for fluent chaining
    
    Example:
        coordinator = ChatAgent(
            name="Coordinator",
            description="Coordinates multi-agent collaboration",
            instructions="Select the next speaker based on expertise and task needs.",
            chat_client=OpenAIChatClient(),
            response_format=ManagerSelectionResponse,
        )
        
        workflow = (
            GroupChatBuilder()
            .set_manager(coordinator)
            .participants([researcher, writer])
            .build()
        )
    """
```

#### 2.2 Update Internal State
```python
class GroupChatBuilder:
    def __init__(self, ...):
        self._participants: dict[str, AgentProtocol | Executor] = {}
        self._manager_participant: AgentProtocol | Executor | None = None  # NEW
        self._manager_name: str = "manager"
        self._manager: _GroupChatManagerFn | None = None  # Keep for select_speakers()
        # ... rest of fields
```

#### 2.3 Validation Logic
- Manager cannot be in participants list (enforce separation)
- Either `set_manager()` OR `select_speakers()` must be called (mutually exclusive)
- Manager instance must be AgentProtocol or Executor

### Phase 3: Update Workflow Assembly

#### 3.1 Modify `assemble_group_chat_workflow()`

**Current wiring:**
```
Orchestrator (start) → Participant1 → Orchestrator
                    → Participant2 → Orchestrator
                    → ...
```

**New wiring with manager:**
```
Orchestrator (start) → Manager → Orchestrator
Manager → Participant1 → Orchestrator
Manager → Participant2 → Orchestrator
Manager → ...
```

Implementation changes:
```python
def assemble_group_chat_workflow(
    *,
    wiring: _GroupChatConfig,
    participant_factory: ...,
    orchestrator_factory: ...,
    interceptors: ...,
    checkpoint_storage: ...,
    builder: ...,
    return_builder: bool = False,
) -> Workflow | tuple[WorkflowBuilder, Executor]:
    """Build the workflow graph for group chat orchestration."""
    
    orchestrator = wiring.orchestrator or orchestrator_factory(wiring)
    workflow_builder = builder or WorkflowBuilder()
    workflow_builder = workflow_builder.set_start_executor(orchestrator)
    
    # NEW: Wire manager as participant if present
    if wiring.manager_participant is not None:
        manager_pipeline = participant_factory(
            GroupChatParticipantSpec(
                name=wiring.manager_name,
                participant=wiring.manager_participant,
                description="Coordination manager",
            ),
            wiring,
        )
        manager_entry = manager_pipeline[0]
        manager_exit = manager_pipeline[-1]
        
        # Register manager
        orchestrator.register_participant_entry(
            wiring.manager_name,
            entry_id=manager_entry.id,
            is_agent=not isinstance(wiring.manager_participant, Executor),
        )
        
        # Orchestrator → Manager → Orchestrator
        workflow_builder = workflow_builder.add_edge(orchestrator, manager_entry)
        if manager_exit is not orchestrator:
            workflow_builder = workflow_builder.add_edge(manager_exit, orchestrator)
    
    # Wire regular participants (existing logic)
    for name, spec in wiring.participants.items():
        pipeline = list(participant_factory(spec, wiring))
        # ... existing wiring logic
```

#### 3.2 Update `_GroupChatConfig`
```python
@dataclass
class _GroupChatConfig:
    manager: _GroupChatManagerFn | None  # Keep for select_speakers()
    manager_participant: AgentProtocol | Executor | None  # NEW
    manager_name: str
    participants: Mapping[str, GroupChatParticipantSpec]
    max_rounds: int | None = None
    orchestrator: Executor | None = None
    participant_aliases: dict[str, str] = field(default_factory=dict)
    participant_executors: dict[str, Executor] = field(default_factory=dict)
```

### Phase 4: Update Orchestrator Logic

#### 4.1 Manager Invocation Pattern
```python
class GroupChatOrchestratorExecutor(BaseGroupChatOrchestrator):
    
    async def _query_manager(
        self,
        ctx: WorkflowContext[...],
    ) -> GroupChatDirective:
        """Query manager for next speaker selection."""
        
        if self._manager is not None:
            # Legacy path: callable manager (select_speakers)
            return await self._manager(self._build_state())
        
        # NEW: Agent-based manager
        # Build manager request with conversation context
        manager_request = self._build_manager_request()
        
        # Route to manager agent via workflow graph
        await self._route_to_participant(
            participant_name=self._manager_name,
            conversation=manager_request.conversation,
            ctx=ctx,
            instruction="",
            task=manager_request.task,
            metadata=manager_request.to_dict(),
        )
        
        # Manager response will arrive via handle_agent_executor_response
        # Return immediately (async state machine pattern)
        return None  # Signal: waiting for manager response
```

#### 4.2 Handle Manager Response
```python
@handler
async def handle_agent_executor_response(
    self,
    response: AgentExecutorResponse,
    ctx: WorkflowContext[...],
) -> None:
    """Handle responses from both manager and regular participants."""
    
    participant_name = self._registry.get_participant_name(response.executor_id)
    if participant_name is None:
        return
    
    # Check if response is from manager
    if participant_name == self._manager_name:
        await self._handle_manager_response(response, ctx)
    else:
        await self._handle_participant_response(participant_name, response, ctx)

async def _handle_manager_response(
    self,
    response: AgentExecutorResponse,
    ctx: WorkflowContext[...],
) -> None:
    """Process manager's speaker selection decision."""
    
    # Extract selection from manager response
    selection = self._parse_manager_selection(response)
    
    if selection.finish:
        # Manager decided to complete conversation
        final_message = selection.final_message or self._create_completion_message(
            text="Conversation completed.",
            reason="manager_finish",
        )
        self._conversation.append(final_message)
        await ctx.yield_output(list(self._conversation))
        return
    
    # Manager selected next participant
    selected = selection.selected_participant
    if not selected or selected not in self._participants:
        raise ValueError(f"Manager selected invalid participant: {selected}")
    
    # Route to selected participant
    await self._route_to_participant(
        participant_name=selected,
        conversation=list(self._conversation),
        ctx=ctx,
        instruction=selection.instruction or "",
        task=self._task_message,
        metadata=selection.metadata,
    )
```

#### 4.3 Manager Response Parsing
```python
def _parse_manager_selection(
    self,
    response: AgentExecutorResponse,
) -> ManagerSelectionResponse:
    """Extract manager selection from agent response."""
    
    # Try response.value first (structured output)
    if response.value is not None:
        if isinstance(response.value, ManagerSelectionResponse):
            return response.value
        if isinstance(response.value, dict):
            return ManagerSelectionResponse(**response.value)
        if isinstance(response.value, str):
            data = json.loads(response.value)
            return ManagerSelectionResponse(**data)
    
    # Fallback: parse from message text
    messages = response.agent_run_response.messages or []
    if messages:
        last_msg = messages[-1]
        text = last_msg.text or ""
        # Attempt JSON parsing
        try:
            data = json.loads(text)
            return ManagerSelectionResponse(**data)
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Error: cannot parse manager decision
    raise RuntimeError("Manager response did not contain valid selection data")
```

### Phase 5: Update Documentation and Examples

#### 5.1 Update Docstrings
- `GroupChatBuilder` class docstring
- `set_manager()` method documentation
- Remove references to `set_prompt_based_manager()`
- Update code examples throughout

#### 5.2 Create Migration Guide
```markdown
# Migrating from set_prompt_based_manager to set_manager

## Old Pattern
```python
workflow = (
    GroupChatBuilder()
    .set_prompt_based_manager(
        chat_client=OpenAIChatClient(),
        instructions="Select the next speaker based on expertise",
        display_name="Coordinator",
    )
    .participants([researcher, writer])
    .build()
)
```

## New Pattern
```python
coordinator = ChatAgent(
    name="Coordinator",
    description="Coordinates team collaboration",
    instructions="""
    You coordinate a team conversation. Review the conversation history
    and select the next participant to speak. Return your decision as:
    {
        "selected_participant": "Researcher",
        "instruction": "Optional instruction for the participant",
        "finish": false
    }
    """,
    chat_client=OpenAIChatClient(),
    response_format=ManagerSelectionResponse,
)

workflow = (
    GroupChatBuilder()
    .set_manager(coordinator)
    .participants([researcher, writer])
    .build()
)
```
```

#### 5.3 Update Sample Files

**Files to update:**
- `samples/getting_started/workflows/orchestration/group_chat_prompt_based_manager.py`
- `samples/getting_started/workflows/orchestration/group_chat_with_chat_options.py`
- `samples/getting_started/workflows/agents/group_chat_workflow_as_agent.py`
- `samples/autogen-migration/orchestrations/02_selector_group_chat.py`
- `samples/semantic-kernel-migration/orchestrations/group_chat.py`

**Sample transformation:**
```python
# NEW sample structure
coordinator = ChatAgent(
    name="Coordinator",
    description="Coordinates multi-agent collaboration",
    instructions="""
    Select the next participant based on conversation context.
    Return JSON: {"selected_participant": "name", "finish": false}
    When task is complete, return: {"finish": true, "final_message": "summary"}
    """,
    chat_client=OpenAIChatClient(model_id="gpt-4o"),
    response_format=ManagerSelectionResponse,
)

workflow = (
    GroupChatBuilder()
    .set_manager(coordinator, display_name="Orchestrator")
    .participants([researcher, writer])
    .build()
)
```

### Phase 6: Deprecation and Cleanup

#### 6.1 Remove Old Code
- **Delete:** `_PromptBasedGroupChatManager` class
- **Delete:** `ManagerDirectiveModel` Pydantic model
- **Delete:** `DEFAULT_MANAGER_INSTRUCTIONS` constant
- **Delete:** `DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT` constant
- **Delete:** `set_prompt_based_manager()` method

#### 6.2 Keep for Compatibility
- `select_speakers()` method (different use case)
- `_SpeakerSelectorAdapter` (wraps callable selectors)
- `GroupChatDirective` dataclass (still used internally)
- `_GroupChatManagerFn` type alias (used by select_speakers)

#### 6.3 Update Exports
```python
# _workflows/__init__.py

# Remove from __all__:
# "DEFAULT_MANAGER_INSTRUCTIONS"
# "DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT"

# Add to __all__:
"ManagerSelectionRequest",
"ManagerSelectionResponse",
```

### Phase 7: Testing Strategy

#### 7.1 Unit Tests
- Test `set_manager()` API with AgentProtocol instances
- Test `set_manager()` API with Executor instances
- Test mutual exclusivity: `set_manager()` vs `select_speakers()`
- Test manager response parsing (structured output, JSON, fallbacks)
- Test workflow graph wiring (manager edges to all participants)

#### 7.2 Integration Tests
- End-to-end group chat with agent-based manager
- Manager selecting different participants based on context
- Manager finishing conversation with final message
- Checkpointing and resumption with agent manager
- Streaming manager decisions

#### 7.3 Migration Validation
- Convert all existing samples to new API
- Verify all samples run successfully
- Confirm observability/streaming still works
- Performance comparison (old vs new architecture)

## Migration Checklist

### Code Changes
- [x] Define `ManagerSelectionRequest` and `ManagerSelectionResponse` dataclasses
- [x] Add `set_manager()` method to `GroupChatBuilder`
- [x] Update `_GroupChatConfig` to include `manager_participant`
- [x] Modify `assemble_group_chat_workflow()` to wire manager as participant
- [x] Update `GroupChatOrchestratorExecutor` to query manager via graph
- [x] Implement `_handle_manager_response()` in orchestrator
- [x] Implement `_parse_manager_selection()` for response parsing
- [x] Add `_is_manager_agent()` helper method
- [x] Update `handle_agent_executor_response()` to route manager responses
- [x] Update `_handle_task_message()` to query manager correctly
- [x] Update `_ingest_participant_message()` to query manager correctly
- [x] Add `is_participant_registered()` to ParticipantRegistry
- [x] Update `__init__.py` exports to include new dataclasses
- [ ] Remove `_PromptBasedGroupChatManager` class
- [ ] Remove `set_prompt_based_manager()` method
- [ ] Remove `DEFAULT_MANAGER_INSTRUCTIONS` and related constants

### Documentation
- [x] Update `GroupChatBuilder` class docstring
- [x] Document `set_manager()` method with examples
- [ ] Create migration guide document
- [ ] Update README files in sample directories
- [ ] Add architecture diagram showing new flow

### Samples
- [x] Create new sample `group_chat_agent_manager.py` demonstrating set_manager()
- [ ] Update `group_chat_prompt_based_manager.py`
- [ ] Update `group_chat_with_chat_options.py`
- [ ] Update `group_chat_workflow_as_agent.py`
- [ ] Update `02_selector_group_chat.py` (autogen migration)
- [ ] Update `group_chat.py` (semantic-kernel migration)

### Testing
- [ ] Unit tests for `set_manager()` API
- [ ] Unit tests for manager response parsing
- [ ] Integration tests for agent-based manager
- [ ] Verify all samples execute successfully
- [ ] Performance benchmarks
- [ ] Backward compatibility verification for `select_speakers()`

### Validation
- [ ] Run full test suite
- [ ] Run all sample files
- [ ] Check for linting/type errors
- [ ] Verify observability features work
- [ ] Confirm checkpointing still functional

## Benefits of New Architecture

### 1. Flexibility
- Manager can be any AgentProtocol implementation (ChatAgent, custom agent, etc.)
- Manager has access to tools, context providers, middleware
- Manager can use different LLM backends, response formats, etc.

### 2. Observability
- Manager decisions visible in standard agent events
- Streaming manager reasoning/thoughts
- Unified observability across all workflow participants

### 3. Consistency
- Manager treated like any other participant in the graph
- No special-case logic for manager invocation
- Consistent checkpointing/resumption semantics

### 4. Extensibility
- Users can implement custom manager logic using full agent framework
- Manager can leverage complex decision-making patterns
- Easy to add manager-specific tools or capabilities

### 5. Simplicity
- Remove internal manager implementation
- Reduce framework surface area
- Push complexity to user-space where it's more controllable

## Risks and Mitigation

### Risk 1: Breaking Changes
**Impact:** All existing `set_prompt_based_manager()` users must migrate

**Mitigation:**
- Provide clear migration guide with side-by-side examples
- Update all framework samples before release
- Include deprecation warnings if phased rollout desired
- Consider providing compatibility shim for one release cycle

### Risk 2: Performance Overhead
**Impact:** Manager as full agent node may add latency

**Mitigation:**
- Benchmark before/after performance
- Optimize agent executor overhead if needed
- Document performance characteristics
- Consider caching manager executor instance

### Risk 3: Increased Complexity for Simple Cases
**Impact:** Users now must create full agent for manager

**Mitigation:**
- Provide helper function to create default manager agent
- Include comprehensive examples for common patterns
- Keep `select_speakers()` for simple functional cases
- Document when to use each approach

### Risk 4: Response Parsing Failures
**Impact:** Manager may not return valid selection format

**Mitigation:**
- Strongly encourage `response_format` usage
- Implement robust fallback parsing logic
- Provide clear error messages
- Include validation in samples

## Success Criteria

1. **Functional:** All group chat workflows support agent-based managers
2. **Performance:** No significant performance regression (<10% overhead)
3. **Migration:** All samples converted and running successfully
4. **Testing:** >90% test coverage for new manager logic
5. **Documentation:** Complete migration guide and updated API docs
6. **Compatibility:** `select_speakers()` continues to work unchanged

## Timeline Estimate

- **Phase 1-2:** ~~2-3 days~~ **COMPLETED** (protocol + API design)
- **Phase 3-4:** ~~3-4 days~~ **COMPLETED** (workflow assembly + orchestrator logic)
- **Phase 5:** 2-3 days **IN PROGRESS** (documentation + samples)
- **Phase 6:** 1 day **PENDING** (cleanup + deprecation)
- **Phase 7:** 2-3 days **PENDING** (testing + validation)

**Total: 10-14 days** | **Completed: ~4 days** | **Remaining: ~6-8 days**

## Implementation Notes

### Design Decisions Made

1. **ManagerSelectionResponse as dataclass**: Chose dataclass over Pydantic for consistency with existing message primitives (`GroupChatDirective`, `_GroupChatRequestMessage`, etc.)

2. **Backward compatibility**: Kept `select_speakers()` fully functional. Both callable managers and agent managers are supported simultaneously through conditional logic in orchestrator.

3. **Manager routing**: Manager is wired as a full participant node with bidirectional edges (orchestrator ↔ manager). This enables:
   - Full observability of manager decisions
   - Streaming manager reasoning
   - Consistent checkpointing semantics
   - Tool usage by manager if needed

4. **Response parsing strategy**: Implemented multi-tier parsing:
   - Primary: `response.value` (structured output via `response_format`)
   - Fallback: JSON parsing from message text
   - Clear error messages when parsing fails

5. **Validation**: Added check to prevent manager name conflicts with participant names in the `participants()` method.

6. **Manager identification**: Added `_is_manager_agent()` helper that checks if manager is registered as a participant (vs callable manager).

### Known Limitations

1. **No automatic manager creation**: Users must create their own manager agent. Consider adding a helper function in future for common patterns.

2. **Manager must use structured output**: For reliable parsing, manager agents should use `response_format=ManagerSelectionResponse`. Text-based parsing is fallback only.

3. **Breaking change**: `set_prompt_based_manager()` remains but will be deprecated. Users need to migrate to `set_manager()`.

## Questions for Review

1. ~~Should we keep `set_prompt_based_manager()` as deprecated for one release?~~ **Decision: Yes, keep for now but document deprecation path**
2. ~~Should `ManagerSelectionResponse` be a Pydantic model or dataclass?~~ **Decision: Dataclass for consistency**
3. Should we provide a helper function to create standard manager agents? **Recommended for future enhancement**
4. ~~How should we handle manager errors (invalid selections, parsing failures)?~~ **Decision: Raise RuntimeError with clear message**
5. ~~Should manager be excluded from `participants` count in logging/metrics?~~ **Decision: Manager is tracked separately**
