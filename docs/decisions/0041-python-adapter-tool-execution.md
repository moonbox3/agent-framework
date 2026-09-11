---
status: proposed
contact: moonbox3
date: 2026-09-11
deciders: moonbox3
---

# Keep Python adapter tool execution within core provider preparation

## Context and Problem Statement

AG-UI approval resume and A2UI mixed planner batches need to execute local tools while also maintaining
transport-specific approval and rendering state. Reconstructing function middleware from Agent and client
attributes misses middleware that a context provider contributes through its public `before_run` hook.
Requiring an additional private provider hook creates a second, incomplete policy contract.

The execution context must remain core-owned without moving AG-UI occurrence registration, claims,
retention, or protocol projection into the core package.

## Decision Drivers

- Preserve public context-provider behavior and fail-closed function policy across execution routes.
- Prepare providers once for an approval execution and its continuation.
- Preserve existing approval transitions and grouped results.
- Keep executable A2UI siblings inside the normal function loop and share their call budget with rendering.
- Avoid a new public execution API or a general execution-engine redesign.

## Considered Options

- **Continue reconstructing middleware in adapters.** Smallest code change, but it duplicates the provider
  contract and remains incomplete whenever a provider contributes middleware through the supported public hook.
- **Run providers independently before each adapter execution and again on continuation.** Reuses the public hook,
  but repeats provider side effects and can give the tool and its continuation different policy contexts.
- **Core-prepared approval execution and core-owned mixed-batch execution.** Introduces a small private preparation
  handoff and a narrowly scoped internal mixed-declaration option, while preserving existing lifecycle ownership.

## Decision Outcome

Use core preparation for local approval execution. A transient preparation record binds the concrete Agent,
AgentSession, prepared SessionContext, and provider options. The continuation consumes that preparation once;
the record is neither serialized into a session nor exposed as a public API.

For A2UI, internally opt into executing executable siblings while returning declaration-only calls unresolved.
The normal core loop supplies function middleware, context, approval handling, and execution accounting.
A2UI handles rendering and transport projection, shares the budget across planner rounds, and does not
re-execute server results returned by core. Compatibility execution for non-core agents is restricted to
agents without unprepared context providers; unsupported combinations fail explicitly.

AG-UI retains its approval lifecycle. A disabled early return releases all unstarted claimed owners with
their original pending-retention boundary while leaving terminal non-grant outcomes intact.

The extra private handoff requires coverage for context reuse and wrong-agent/session rejection. The
mixed-declaration option must remain opt-in so ordinary function-loop behavior does not change.
Responsible engineering-manager and architect review is required before this proposed ADR is accepted.
