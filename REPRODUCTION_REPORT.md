# Reproduction Report: Issue #5416

**Issue**: [Python: [Bug]:  FoundryChatClient(default_headers=...) and FoundryAgent(default_headers=...) silently drop custom headers](https://github.com/microsoft/agent-framework/issues/5416)
**Repository**: microsoft/agent-framework
**Investigated**: 2026-04-24 06:16:08 UTC
**Worktree Path**: `/repos/agent-framework/.worktrees/agent/fix-5416-1`
**Gate Result**: ✅ PASSED

---

## Reproduction Verdict

| Field | Value |
|-------|-------|
| Status | `reproduced` |
| Failure observed | `yes` |
| Evidence strength | `high` |
| Confidence | `high` |
| Failing test | `python/packages/foundry/tests/foundry/test_5416_default_headers_dropped.py::test_foundry_chat_client_default_headers_forwarded_to_get_openai_client` |
| Gate passed | ✅ PASSED |

## Finding

`FoundryChatClient.__init__` calls `project_client.get_openai_client()` with no arguments (line 209), so `default_headers` is stored on `self` but never passed to the underlying AsyncOpenAI client; the same gap exists in `_agent.py` line 201. The new test fails asserting `get_openai_client(default_headers=...)` vs the actual `get_openai_client()`, confirming the bug exactly as reported.

## Files Examined

python/packages/foundry/agent_framework_foundry/_chat_client.py, python/packages/foundry/agent_framework_foundry/_agent.py, python/packages/openai/agent_framework_openai/_shared.py, python/packages/foundry/tests/foundry/test_foundry_chat_client.py, python/packages/foundry/tests/foundry/conftest.py

## Tests Run

python/packages/foundry/tests/foundry/test_5416_default_headers_dropped.py::test_foundry_chat_client_default_headers_forwarded_to_get_openai_client

## Full Analysis

The bugs are clearly visible. Let me now check the existing tests and write a repro test:Now I have all the evidence. Let me write the failing test to confirm the bug:The existing test only checks `client.default_headers` (stored on instance), not whether `get_openai_client` was called with the headers. Let me write a failing test:The test fails exactly as described. The output confirms:
- **Expected**: `get_openai_client(default_headers={'x-custom-header': 'repro-value'})`
- **Actual**: `get_openai_client()`

Now let me write the report:**Bug confirmed reproduced.** The failing test shows exactly what the issue describes:

- `get_openai_client()` is called with **no arguments** instead of `get_openai_client(default_headers={...})`
- The headers end up silently stored on `self.default_headers` but never reach the `AsyncOpenAI` instance that makes HTTP calls

There are 3 bug sites: `_chat_client.py:209`, `_agent.py:201`, and the `_shared.py:201–202` short-circuit that would still discard headers even if the first two were patched alone.

---

---

## Next Steps

- Reproduction confirmed — proceed to investigation and fix phase
- Review the failing test or evidence above to understand the root cause
- The issue branch is checked out in the worktree: `/repos/agent-framework/.worktrees/agent/fix-5416-1`