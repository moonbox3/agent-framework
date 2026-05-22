# Reproduction Report — microsoft/agent-framework#5733

## Issue

- Title: Python: Core observability cannot safely serialize workflow request_info payloads
- Worktree: `/repos/agent-framework/.worktrees/agent/fix-5733-1`

## Reused DevFlow triage reproduction

A prior trusted DevFlow triage run already reproduced this issue, so the fix workflow skipped active reproduction.

- Source: https://github.com/microsoft/agent-framework/issues/5733#issuecomment-4515369380
- Failing test: `python/packages/core/tests/core/test_observability_serialization_bug.py`
- Files examined: python/packages/core/agent_framework/observability.py, python/packages/core/agent_framework/_types.py, python/packages/core/tests/core/test_observability.py
- Tests run: python/packages/core/tests/core/test_observability_serialization_bug.py

## Triage notes for fix agent

Repro: _capture_messages() in python/packages/core/agent_framework/observability.py:2176 calls json.dumps(otel_messages, ensure_ascii=False) without a default= handler; _to_otel_part() at line 2214 passes content.arguments raw into the dict. Trigger: create Content.from_function_call with arguments={"payload": <dataclass_instance>} (simulating workflow request_info), wrap in Message, call _capture_messages — raises TypeError. Minimal repro: see test_observability_serialization_bug.py::TestIssue5733::test_capture_messages_raises_typeerror_for_dataclass_arguments.
