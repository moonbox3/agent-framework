# Reproduction Report — Issue #4500
**Python: [Bug]: State updates not propagated to the next activity with durable function and declarative workflow**

*Generated: 2026-03-06 04:28:49 UTC*

## Result

| Field | Value |
|-------|-------|
| Status | `reproduced` |
| Failure observed | `yes` |
| Evidence strength | `high` |
| Confidence | `high` |
| Failing test | `/Users/evmattso/git/agent-framework-2/.worktrees/agent/fix-4500-1/python/packages/azurefunctions/tests/test_shallow_copy_bug.py` |
| Gate passed | Yes |

## Files Examined

python/packages/azurefunctions/agent_framework_azurefunctions/_app.py, python/packages/core/agent_framework/_workflows/_state.py, python/packages/azurefunctions/agent_framework_azurefunctions/_serialization.py, python/packages/azurefunctions/tests/test_app.py

## Tests Run

python/packages/azurefunctions/tests/test_shallow_copy_bug.py

## Finding

The bug is confirmed: `dict()` at line 303 of `_app.py` creates a shallow copy that shares nested object references with the state. When an executor mutates nested values in place, both `original_snapshot` and `current_state` reflect the same mutation, producing an empty diff and causing state updates to not propagate to downstream activities. The test suite demonstrates this with nested dicts, lists, and contrasts with `copy.deepcopy()` which correctly preserves the original snapshot.

## Full Analysis

Now I can see the bug clearly. Line 303 uses `dict()` (shallow copy). Let me find the `State` class and existing tests.Now I understand the bug. Let me look at `deserialize_value` and the `export_state` method to understand the full flow, then write a test.Now I have a clear understanding of the bug. The issue is at line 303: `original_snapshot = dict(deserialized_state)` creates a shallow copy. When the activity mutates a nested value (like a dict or list) in place, the mutation is reflected in both `deserialized_state` and `original_snapshot`, making the diff at line 334 see no changes.

Let me write a focused test to reproduce this.Now I clearly see the issue. The `original_snapshot` at line 303 shares references to nested objects with `deserialized_state`, which is then imported into `shared_state` at line 304. When the executor mutates nested state values, the mutations are visible in `original_snapshot` too, making the diff at line 334 produce no updates.

Let me write a minimal test to demonstrate this.Now let me write a minimal, focused test that demonstrates the shallow copy bug.All 5 tests pass, confirming the bug. The key tests (`test_shallow_copy_loses_nested_mutations`, `test_shallow_copy_loses_new_key_in_nested_dict`, `test_shallow_copy_loses_list_mutations`) pass by asserting `updates == {}` — proving that the shallow copy causes nested mutations to be invisible in the diff. The `test_deep_copy_preserves_nested_mutations` test shows that `copy.deepcopy()` correctly detects changes.
