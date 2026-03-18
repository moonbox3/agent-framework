# Reproduction Report: Issue #4616

**Issue**: [Python: [Feature]: Support the `detail` field in OpenAI Chat API `image_url` payload](https://github.com/microsoft/agent-framework/issues/4616)
**Repository**: microsoft/agent-framework
**Investigated**: 2026-03-18 06:05:17 UTC
**Worktree Path**: `/repos/agent-framework/.worktrees/agent/fix-4616-2`
**Gate Result**: ✅ PASSED

---

## Reproduction Verdict

| Field | Value |
|-------|-------|
| Status | `reproduced` |
| Failure observed | `yes` |
| Evidence strength | `high` |
| Confidence | `high` |
| Failing test | `python/packages/core/tests/openai/test_image_url_detail_bug.py` |
| Gate passed | ✅ PASSED |

## Finding

The `_prepare_content_for_openai` method at line 718 constructs `image_url` payloads as `{"url": content.uri}` without reading the `detail` field from `additional_properties`. A test setting `additional_properties={"detail": "high"}` confirms the value is silently dropped — the returned dict contains only `{'url': '...'}`.

## Files Examined

python/packages/core/agent_framework/openai/_chat_client.py, python/packages/core/agent_framework/_types.py, python/packages/core/tests/openai/test_openai_chat_client.py

## Tests Run

python/packages/core/tests/openai/test_image_url_detail_bug.py::test_prepare_content_for_openai_image_url_detail_field

## Full Analysis

I can see the bug clearly. Line 718 hard-codes `{"url": content.uri}` without any `detail` field, and `Content.from_uri` accepts `additional_properties`. Let me write and run a failing test.Bug reproduced. The test fails with `assert 'detail' in {'url': 'https://example.com/img.png'}` — confirming the `detail` field is silently dropped.Bug **reproduced** with a failing test. Line 718 of `_chat_client.py` hard-codes `{"url": content.uri}` and never reads the `detail` field from `additional_properties`, even though the pattern for doing so already exists for `filename` in the same function.

---

## Next Steps

- Reproduction confirmed — proceed to investigation and fix phase
- Review the failing test or evidence above to understand the root cause
- The issue branch is checked out in the worktree: `/repos/agent-framework/.worktrees/agent/fix-4616-2`