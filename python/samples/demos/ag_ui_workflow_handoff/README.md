# AG-UI Handoff Workflow Demo

This demo is a full custom AG-UI application built on top of the new workflow abstractions in `agent_framework_ag_ui`.

It includes:

- A **backend** FastAPI AG-UI endpoint serving a **HandoffBuilder workflow** with:
  - `triage_agent`
  - `refund_agent`
  - `order_agent`
- A required **tool approval checkpoint** (`submit_refund` with `approval_mode="always_require"`)
- A second **request-info resume** step (order agent asks for shipping preference)
- A **frontend** React app that consumes AG-UI SSE events, renders workflow cards, and sends `resume.interrupts` payloads.

The backend is deterministic by design so you can test AG-UI behavior without external model credentials.

## Folder Layout

- `backend/server.py` - FastAPI + AG-UI endpoint + Handoff workflow
- `frontend/` - Vite + React AG-UI client UI

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## 1) Run Backend

From the Python repo root:

```bash
cd /Users/evmattso/git/agent-framework/python
uv sync
uv run python samples/demos/ag_ui_workflow_handoff/backend/server.py
```

Backend default URL:

- `http://127.0.0.1:8891`
- AG-UI endpoint: `POST http://127.0.0.1:8891/handoff_demo`

## 2) Install Frontend Packages (npm)

```bash
cd /Users/evmattso/git/agent-framework/python/samples/demos/ag_ui_workflow_handoff/frontend
npm install
```

## 3) Run Frontend Locally

```bash
npm run dev
```

Frontend default URL:

- `http://127.0.0.1:5173`

If you changed backend host/port, run with:

```bash
VITE_BACKEND_URL=http://127.0.0.1:8891 npm run dev
```

## 4) Demo Flow to Verify

1. Click one of the starter prompts (or type a refund request).
2. Refund Agent asks for an order number; reply with a numeric ID (for example: `987654`).
3. Wait for the `submit_refund` reviewer interrupt (built from your provided order ID).
4. In the **HITL Reviewer Console** modal, click **Approve Tool Call**.
5. Wait for the Order agent prompt asking for shipping preference.
6. Reply in the chat input (for example: `expedited`).
7. Confirm the case snapshot updates and workflow completion.

## What This Validates

- `add_agent_framework_fastapi_endpoint(...)` with `AgentFrameworkWorkflow(workflow_factory=...)`
- Thread-scoped workflow state across turns
- `RUN_FINISHED.interrupt` pause behavior
- `resume.interrupts` continuation behavior
- JSON resume payload coercion for `Content` and `list[Message]` workflow response types
