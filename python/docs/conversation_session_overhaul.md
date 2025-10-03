# Conversation Session Architecture Overhaul

## Overview
Unify conversation handling across orchestration workflows by introducing a shared session abstraction that coordinates AgentThread variants, enforces chat transcript integrity, and supplies developers with explicit control over conversational context.

## Objectives
- Provide a single authoritative conversation record per workflow run while preserving ChatMessage-based interfaces.
- Support heterogeneous agent implementations (service-managed threads, local stores) without diverging orchestration code paths.
- Expose developers to durable handles for seeding, resuming, and persisting conversations across orchestrations.
- Maintain deterministic, debuggable workflows for sequential and concurrent compositions without breaking existing public contracts.

## Scope Constraints
- Retain AgentProtocol.run / run_stream signatures and existing WorkflowEvent wire formats.
- Avoid new external dependencies or runtime services.
- Enable incremental rollout beside legacy behavior with compatibility shims.

## Current State Assessment
- AgentExecutor caches list[ChatMessage] inputs per hop, ignoring upstream thread state beyond local buffers.
- SequentialBuilder and ConcurrentBuilder propagate raw lists through adapters (_ResponseToConversation) with no canonical ownership.
- AgentThread provides either service_thread_id or ChatMessageStore but orchestration never reconciles multiple thread instances.
- Service-managed threads and local message stores drift out of sync when chaining agents, duplicating transcript replay logic.

## Key Issues
- Transcript reconstruction is ad hoc; full_conversation fields duplicate state and omit thread metadata.
- Developers cannot seed or persist shared context except by manually crafting message lists.
- Concurrent orchestration lacks isolation guarantees when agents mutate shared caches simultaneously.
- No generalized persistence story for conversation state, impeding checkpointing and resumability.

## Solution Alternatives

### Option 1: ConversationSession Orchestrator (Preferred)
- **Description**: Introduce a workflow-scoped ConversationSession managed by ConversationManager, supplying deterministic projections and synchronizing AgentThread bindings (detailed below).
- **Strengths**: Normalizes transcript ownership, enables resumable handles, and simplifies heterogeneous thread reconciliation with a single integration point. Extends naturally to persistence and policy enforcement.
- **Risks**: Requires refactoring AgentExecutor and builders; adds orchestration-layer complexity; mandates careful concurrency control for parallel workflows.

```python
session = conversation_manager.ensure_session(ctx)
view = conversation_manager.view_for(participant="writer", policy="full")
binding = conversation_manager.resolve_thread(participant="writer")
conversation_manager.replay_to_thread(binding, view.delta_since_binding)
response = await agent.run(view.messages, thread=binding.thread, **view.chat_options)
conversation_manager.merge_response(binding, response)
await ctx.send_message(response)
```

### Option 2: Thread Federation with Lazy Transcript Stitching
- **Description**: Keep current per-agent caches but layer a TranscriptBroker that, on demand, merges AgentThread message stores and service-managed transcripts into workflow outputs.
- **Strengths**: Minimal changes to AgentExecutor internals; leverages existing AgentThread APIs; defers orchestration rewrite.
- **Risks**: TranscriptBroker becomes a reactive reconciler prone to race conditions; lacks single source of truth; resumability still requires bespoke stitching logic; developers cannot easily inject custom context policies.

```python
broker = TranscriptBroker(binding_registry, service_transport)
messages = await broker.materialize(participant="reviewer", include_tools=True)
thread = binding_registry.thread_for("reviewer")
response = await agent.run(messages, thread=thread)
await broker.record_response("reviewer", response.messages, response.metadata)
workflow_conversation = await broker.compose_workflow_transcript()
```

### Option 3: Context Relay Bus per Workflow Edge
- **Description**: Introduce typed ContextEnvelope messages transported on workflow edges carrying deltas, metadata, and optional handles. Each executor manages its own partial transcript but publishes envelopes for subscribers.
- **Strengths**: Aligns with existing messaging semantics; fine-grained delta propagation; encourages executor-specific optimizations.
- **Risks**: Elevated cognitive load for developers; duplicated state across executors; requires contract renegotiation for every consumer; persistence story fragmented.

```python
delta = local_transcript.produce_delta()
await context_bus.publish("dispatcher", ContextEnvelope(delta=delta, annotations=meta))
envelope = await context_bus.consume("writer")
messages = envelope.apply_to(local_transcript)
response = await agent.run(messages, thread=local_threads["writer"])
await context_bus.publish("writer", envelope.ack(response.messages, response.tool_calls))
```

### Selection Rationale
Option 1 centralizes responsibility for transcript integrity, thread synchronization, and persistence while exposing a developer-friendly ConversationHandle. Options 2 and 3 both retain fragmented ownership of conversation state, making resumability, policy enforcement, and concurrent coordination brittle. Therefore Option 1 is preferred despite higher up-front refactor cost because it unlocks a clean control plane for heterogeneous agents and future persistence investments.

## Target Architecture

### ConversationSession
Central record containing:
- `session_id`: UUID for workflow run correlation.
- `transcript`: append-only ChatMessage snapshots.
- `participant_profiles`: executor/agent metadata, policies.
- `thread_bindings`: per-participant ThreadBinding state (AgentThread instance, service_thread_id, message_store ref, replay cursor).
- `attachments`: optional structured payloads (tool outputs, files).
- `revision`: monotonic counter enabling optimistic concurrency in concurrent workflows.
- `agent_thread_bridge`: maintains direct references to the AgentThread abstractions contributed by each participant. This preserves existing single-agent state handling while elevating orchestration concerns (cross-agent ordering, resumability, policy enforcement) to the workflow layer where they belong.

### ConversationManager
Controller responsible for:
- Session creation via ConversationSessionFactory at workflow entry.
- Issuing ConversationView projections (full transcript, deltas, role-filtered slices) to executors.
- Resolving or instantiating AgentThread objects with preserved bindings.
- Replaying transcript deltas to threads and coordinating service-managed conversation resumption.
- Merging AgentRunResponse updates into the session and advancing cursors.
- Enforcing ConversationPolicy (retention, truncation, role filters) during view generation and merges.
- Ensures AgentThread semantics remain intact: per-agent context providers, message stores, and service-managed ids are honored through ThreadBinding while conversation-wide invariants (e.g., monotonic transcript) are governed at the workflow scope, preventing individual agents from shouldering orchestration logic.

### Thread Binding Coordination
- `resolve_thread(participant_key)` fetches or creates bound AgentThread with stored overrides.
- `replay_to_thread(binding, transcript_delta)` injects new messages via thread.on_new_messages or agent-specific resume hooks for service-managed threads.
- `merge_response(binding, response)` appends assistant output to transcript and advances replay cursor.

### Workflow Integration
- Introduce `ConversationEntryExecutor` to normalize initial inputs into a ConversationSession; accepts str, ChatMessage, list[ChatMessage], or ConversationHandle to resume existing sessions.
- Refactor `AgentExecutor` to request ConversationView instead of managing _cache; after agent invocation, commit responses through ConversationManager.
- Replace `_ResponseToConversation` adapters with `ConversationProjectionExecutor` configured to emit desired view (default full transcript).
- ConcurrentBuilder dispatcher clones logical views per participant; ConversationManager handles revision gating to ensure deterministic merges; aggregator reads final transcript snapshot.
- Internal plumbing executors (entry/projection) are tagged with workflow metadata `visibility=internal`. Visualization and default telemetry filters omit those nodes so developer-authored graphs remain uncluttered; diagnostics can opt-in via `show_internal=true` when deep inspection is required.

## Ensuring Agent Context Fidelity
- **Conversation Entry**: Workflow start normalizes caller-provided inputs (string, ChatMessage sequence, or ConversationHandle) into a ConversationSession, capturing initial user prompts and any previously persisted transcript.
- **View Assembly**: Before invoking `agent.run` or `agent.run_stream`, AgentExecutor requests a ConversationView configured for that participant. The view merges transcript history, policy-constrained projections, context provider injections, and any tool results destined for the agent.
- **Thread Replay**: ConversationManager resolves the participant's ThreadBinding, replays transcript deltas through `thread.on_new_messages` (or service resume hooks) to guarantee the AgentThread state matches the ConversationView that will be sent to the agent APIs.
- **Invocation Contract**: `agent.run` / `run_stream` receive the ConversationView's ChatMessage list, ensuring the agent is presented with identical context to the synchronized thread. Additional kwargs (ChatOptions, tool metadata) are derived from ConversationPolicy and thread context providers, keeping agent invocations deterministic.
- **Response Merge**: After the agent yields messages, ConversationManager appends them to the transcript, advances ThreadBinding cursors, and records any new service conversation identifiers. Downstream executors always consume the updated session, preventing context drift.
- **Streaming Guardrails**: For streaming runs, incremental updates are staged in the binding until the response commits, so downstream consumers only see consistent transcript states and threads are updated in lockstep with emitted updates.

## Developer-Facing Surface
- `ConversationHandle`: opaque carrier of session identity and persistence token. `Workflow.run` and `workflow.run_stream` accept optional handle for resuming runs; `WorkflowOutputEvent` returns handle for further orchestration.
- `ConversationSnapshot`: workflow output containing the current transcript (`messages`) paired with the resumable `ConversationHandle`, replacing ad hoc list[ChatMessage] payloads.
- Advanced configuration:
  - Attach per-participant ConversationPolicy overrides.
  - Preload attachments or context metadata.
  - Register serialization hooks for ConversationSession persistence.
- Samples updated:
  - `sequential_agents.py` prints the final `ConversationSnapshot` handle and transcript.
  - `resume_conversation.py` demonstrates rehydrating a workflow directly from an existing `ConversationHandle` without providing a new user prompt.

## Persistence Strategy
- `ConversationStore` interface (load/save/prune). Default in-memory implementation preserves legacy behavior.
- Optional durable stores (filesystem, Redis, Cosmos) implement interface.
- Workflow checkpointing stores ConversationSession revision within checkpoint payload. Resume path rehydrates session before continuing.
- AgentThread serialization integrated into session persistence to capture service conversation IDs or message-store state.

## Compatibility Plan
- Legacy workflows instantiate shim ConversationSession behind feature flag, retaining current list[ChatMessage] outputs.
- `AgentExecutorRequest` continues accepting list[ChatMessage]; ConversationManager seeds temporary session and discards post-run.
- Existing builder adapters remain accessible under deprecation flag; logging warns when shim path used.
- Documentation updates clarify ConversationHandle as preferred orchestration artifact.

## Implementation Roadmap
1. **Foundation**: Implement ConversationSession, ConversationManager, ConversationHandle, ConversationStore behind feature flag. Unit tests for session lifecycle, policy enforcement, and thread binding.
2. **AgentExecutor Refactor**: Remove _cache, integrate ConversationManager for message replay and response commit. Update SequentialBuilder/ConcurrentBuilder wiring to use ConversationEntryExecutor and ConversationProjectionExecutor.
3. **Nested Workflows**: Adapt WorkflowAgent to operate on ConversationSession, routing request_info payloads through session attachments and ensuring nested workflows share handles correctly.
4. **Deprecation & Cleanup**: Remove legacy adapters once parity confirmed. Update integration tests, documentation, and migration guides.

## Validation Plan
- Unit coverage for ConversationManager replay across service-managed and local message stores with simulated agents.
- Deterministic concurrency tests verifying conflict resolution during parallel agent execution.
- Streaming regression tests ensuring partial updates append correctly and handles resume subsequent runs.
- Performance benchmarks assessing session overhead under long transcripts with ConversationPolicy truncation enabled.
