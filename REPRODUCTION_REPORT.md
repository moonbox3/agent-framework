# Reproduction Report — microsoft/agent-framework#6241

## Issue

- Title: Python: [Bug]: FoundryAgent causing ConnectTimeout on multi-turn conversations
- Worktree: `/repos/agent-framework/.worktrees/agent/fix-6241-1`

## Reused DevFlow triage reproduction

A prior trusted DevFlow triage run already reproduced this issue, so the fix workflow skipped active reproduction.

- Source: https://github.com/microsoft/agent-framework/issues/6241#issuecomment-4597628195
- Failing test: `python/packages/foundry/tests/foundry/test_foundry_agent_timeout_bug.py`
- Files examined: python/packages/foundry/agent_framework_foundry/_agent.py, python/packages/openai/agent_framework_openai/_chat_client.py, python/packages/openai/agent_framework_openai/_shared.py, python/packages/core/agent_framework/_types.py, python/packages/core/agent_framework/_clients.py, python/packages/foundry/pyproject.toml
- Tests run: python/packages/foundry/tests/foundry/test_foundry_agent_timeout_bug.py (6 tests, all pass)

## Triage notes for fix agent

Repro: RawFoundryAgentChatClient in python/packages/foundry/agent_framework_foundry/_agent.py::__init__ (line 258-264) calls self.project_client.get_openai_client() without passing a timeout parameter, inheriting the openai library default of httpx.Timeout(timeout=600.0, connect=5.0). When an APITimeoutError occurs during streaming (line 697-698 of python/packages/openai/agent_framework_openai/_chat_client.py), _handle_request_error wraps it as ChatClientException. Fix requires adding a timeout parameter to RawFoundryAgentChatClient, _FoundryAgentChatClient, FoundryAgent, and RawOpenAIChatClient constructors, and passing it through to the AsyncOpenAI client creation in _shared.py::load_openai_service_settings.
