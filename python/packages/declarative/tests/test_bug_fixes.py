# Copyright (c) Microsoft. All rights reserved.

"""Tests for bug fixes:
- #3817: PowerFx conditional import (package should import without dotnet)
- #3523: response_format passed via default_options to providers
- #3562: Declarative workflows forward kwargs to agent tools
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework_declarative._workflows._handlers import ActionContext, WorkflowEvent, get_action_handler
from agent_framework_declarative._workflows._state import WorkflowState

# ---------------------------------------------------------------------------
# #3817 - PowerFx conditional import
# ---------------------------------------------------------------------------


class TestPowerFxConditionalImport:
    """The declarative_base module should import even when powerfx is unavailable."""

    def test_import_succeeds_when_powerfx_unavailable(self):
        """Simulating powerfx raising RuntimeError at import time should not
        crash the _declarative_base module import."""
        import importlib
        import sys

        mod_name = "agent_framework_declarative._workflows._declarative_base"
        # Remove cached module so we can re-import
        saved = sys.modules.pop(mod_name, None)
        # Also remove powerfx so the import triggers fresh
        saved_powerfx = sys.modules.pop("powerfx", None)

        try:
            with patch.dict(sys.modules, {"powerfx": None}):
                # When powerfx cannot be imported, Engine should be None
                mod = importlib.import_module(mod_name)
                assert mod.Engine is None

                # DeclarativeWorkflowState should still be importable
                assert hasattr(mod, "DeclarativeWorkflowState")
        finally:
            # Restore original modules
            if saved is not None:
                sys.modules[mod_name] = saved
            if saved_powerfx is not None:
                sys.modules["powerfx"] = saved_powerfx

    def test_eval_raises_when_powerfx_unavailable(self):
        """When Engine is None, eval() should raise RuntimeError for
        expressions that require PowerFx, not silently return a string."""
        from agent_framework_declarative._workflows._declarative_base import (
            DeclarativeWorkflowState,
        )

        mock_state = MagicMock()
        mock_state._data = {}
        mock_state.get = MagicMock(side_effect=lambda k, d=None: mock_state._data.get(k, d))
        mock_state.set = MagicMock(side_effect=lambda k, v: mock_state._data.__setitem__(k, v))

        state = DeclarativeWorkflowState(mock_state)
        state.initialize({"name": "test"})

        with (
            patch("agent_framework_declarative._workflows._declarative_base.Engine", None),
            pytest.raises(RuntimeError, match="PowerFx is not available"),
        ):
            state.eval("=Local.counter + 1")

    def test_eval_returns_plain_strings_without_powerfx(self):
        """Non-PowerFx strings (no leading '=') should work even without Engine."""
        from agent_framework_declarative._workflows._declarative_base import (
            DeclarativeWorkflowState,
        )

        mock_state = MagicMock()
        mock_state._data = {}
        mock_state.get = MagicMock(side_effect=lambda k, d=None: mock_state._data.get(k, d))
        mock_state.set = MagicMock(side_effect=lambda k, v: mock_state._data.__setitem__(k, v))

        state = DeclarativeWorkflowState(mock_state)
        state.initialize()

        with patch("agent_framework_declarative._workflows._declarative_base.Engine", None):
            # Plain strings should pass through unchanged
            assert state.eval("hello world") == "hello world"
            assert state.eval("") == ""
            assert state.eval(42) == 42


# ---------------------------------------------------------------------------
# #3523 - response_format via default_options
# ---------------------------------------------------------------------------


class TestResponseFormatViaDefaultOptions:
    """response_format from outputSchema must be passed inside default_options,
    not as a direct kwarg to provider.create_agent()."""

    @staticmethod
    def _make_mock_prompt_agent(*, with_output_schema: bool = False) -> MagicMock:
        """Create a mock PromptAgent to avoid serialization complexity."""
        mock_model = MagicMock()
        mock_model.id = "gpt-4"
        mock_model.connection = None

        agent = MagicMock()
        agent.name = "test-agent"
        agent.description = "test"
        agent.instructions = "be helpful"
        agent.model = mock_model
        agent.tools = None

        if with_output_schema:
            mock_schema = MagicMock()
            mock_schema.to_json_schema.return_value = {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            }
            agent.outputSchema = mock_schema
        else:
            agent.outputSchema = None

        return agent

    @staticmethod
    def _make_mock_provider() -> tuple[MagicMock, AsyncMock]:
        """Create a mock provider class and its instance."""
        mock_agent = MagicMock()
        mock_provider_instance = AsyncMock()
        mock_provider_instance.create_agent = AsyncMock(return_value=mock_agent)
        mock_provider_class = MagicMock(return_value=mock_provider_instance)
        return mock_provider_class, mock_provider_instance

    @pytest.mark.asyncio
    async def test_response_format_passed_in_default_options(self):
        """_create_agent_with_provider should wrap response_format in default_options."""
        import builtins

        from agent_framework_declarative._loader import AgentFactory

        prompt_agent = self._make_mock_prompt_agent(with_output_schema=True)
        mock_provider_class, mock_provider_instance = self._make_mock_provider()

        mapping = {"package": "some_module", "name": "SomeProvider"}
        factory = AgentFactory()

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "some_module":
                mod = MagicMock()
                mod.SomeProvider = mock_provider_class
                return mod
            return original_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            patch.object(factory, "_parse_tools", return_value=None),
        ):
            await factory._create_agent_with_provider(prompt_agent, mapping)

        mock_provider_instance.create_agent.assert_called_once()
        call_kwargs = mock_provider_instance.create_agent.call_args.kwargs

        # response_format must NOT be a direct kwarg
        assert "response_format" not in call_kwargs, (
            "response_format should not be passed as a direct kwarg to provider.create_agent()"
        )

        # It should be inside default_options
        default_options = call_kwargs.get("default_options")
        assert default_options is not None, "default_options should be passed to provider.create_agent()"
        assert "response_format" in default_options, "response_format should be inside default_options"

    @pytest.mark.asyncio
    async def test_no_default_options_when_no_output_schema(self):
        """When there's no outputSchema, default_options should be None."""
        import builtins

        from agent_framework_declarative._loader import AgentFactory

        prompt_agent = self._make_mock_prompt_agent(with_output_schema=False)
        mock_provider_class, mock_provider_instance = self._make_mock_provider()

        mapping = {"package": "some_module", "name": "SomeProvider"}
        factory = AgentFactory()

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "some_module":
                mod = MagicMock()
                mod.SomeProvider = mock_provider_class
                return mod
            return original_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=mock_import),
            patch.object(factory, "_parse_tools", return_value=None),
        ):
            await factory._create_agent_with_provider(prompt_agent, mapping)

        call_kwargs = mock_provider_instance.create_agent.call_args.kwargs
        assert call_kwargs.get("default_options") is None


# ---------------------------------------------------------------------------
# #3562 - Declarative workflows forward kwargs to agent tools
# ---------------------------------------------------------------------------


def _create_action_context(
    action: dict[str, Any],
    state: WorkflowState | None = None,
    agents: dict[str, Any] | None = None,
    run_kwargs: dict[str, Any] | None = None,
) -> ActionContext:
    """Helper to create ActionContext with kwargs support."""
    if state is None:
        state = WorkflowState()

    async def execute_actions(actions: list[dict[str, Any]], state: WorkflowState) -> AsyncGenerator[WorkflowEvent]:
        for nested_action in actions:
            handler = get_action_handler(nested_action.get("kind"))
            if handler:
                ctx = ActionContext(
                    state=state,
                    action=nested_action,
                    execute_actions=execute_actions,
                    agents=agents or {},
                    bindings={},
                    run_kwargs=run_kwargs or {},
                )
                async for event in handler(ctx):
                    yield event

    return ActionContext(
        state=state,
        action=action,
        execute_actions=execute_actions,
        agents=agents or {},
        bindings={},
        run_kwargs=run_kwargs or {},
    )


class TestDeclarativeKwargsForwarding:
    """kwargs passed to workflow.run() should reach agent.run() calls."""

    @pytest.mark.asyncio
    async def test_executor_path_forwards_kwargs(self):
        """InvokeAzureAgentExecutor should forward run_kwargs to agent.run()."""
        from unittest.mock import MagicMock

        from agent_framework._workflows._const import WORKFLOW_RUN_KWARGS_KEY
        from agent_framework._workflows._state import State

        from agent_framework_declarative._workflows._declarative_base import (
            DeclarativeWorkflowState,
        )
        from agent_framework_declarative._workflows._executors_agents import (
            InvokeAzureAgentExecutor,
        )

        # Create a mock State with kwargs stored
        mock_state = MagicMock(spec=State)
        state_data: dict[str, Any] = {}

        def mock_get(key, default=None):
            return state_data.get(key, default)

        def mock_set(key, value):
            state_data[key] = value

        mock_state.get = MagicMock(side_effect=mock_get)
        mock_state.set = MagicMock(side_effect=mock_set)

        # Store kwargs in state like Workflow.run() does
        test_kwargs = {"user_token": "abc123", "service_config": {"endpoint": "http://test"}}
        state_data[WORKFLOW_RUN_KWARGS_KEY] = test_kwargs

        # Initialize declarative state
        dws = DeclarativeWorkflowState(mock_state)
        dws.initialize({"input": "hello"})

        # Create a mock agent
        mock_response = MagicMock()
        mock_response.text = "response text"
        mock_response.messages = []
        mock_response.tool_calls = []
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_response)

        # Create a mock workflow context
        mock_ctx = MagicMock()
        mock_ctx.get_state = MagicMock(side_effect=mock_get)
        mock_ctx.yield_output = AsyncMock()

        # Call the method directly
        await dws._invoke_agent_and_store_results_with_ctx(mock_agent, mock_ctx) if hasattr(
            dws, "_invoke_agent_and_store_results_with_ctx"
        ) else None

        # Instead, test the _invoke_agent_and_store_results on the executor
        # We need to instantiate the executor and call the method
        executor = InvokeAzureAgentExecutor.__new__(InvokeAzureAgentExecutor)
        executor._agents = {"test_agent": mock_agent}

        await executor._invoke_agent_and_store_results(
            agent=mock_agent,
            agent_name="test_agent",
            input_text="hello",
            state=dws,
            ctx=mock_ctx,
            messages_var=None,
            response_obj_var=None,
            result_property=None,
            auto_send=True,
        )

        # Verify agent.run was called with kwargs
        mock_agent.run.assert_called_once()
        call_kwargs = mock_agent.run.call_args

        # Check options contains additional_function_arguments
        assert "options" in call_kwargs.kwargs
        assert call_kwargs.kwargs["options"]["additional_function_arguments"] == test_kwargs

        # Check direct kwargs were passed
        assert call_kwargs.kwargs.get("user_token") == "abc123"
        assert call_kwargs.kwargs.get("service_config") == {"endpoint": "http://test"}

    @pytest.mark.asyncio
    async def test_action_context_carries_run_kwargs(self):
        """ActionContext should store and expose run_kwargs."""
        kwargs = {"user_token": "test123"}
        ctx = _create_action_context(
            action={"kind": "SetValue", "path": "Local.x", "value": "1"},
            run_kwargs=kwargs,
        )
        assert ctx.run_kwargs == kwargs

    @pytest.mark.asyncio
    async def test_action_context_defaults_to_empty_kwargs(self):
        """ActionContext.run_kwargs should default to empty dict."""
        ctx = _create_action_context(
            action={"kind": "SetValue", "path": "Local.x", "value": "1"},
        )
        assert ctx.run_kwargs == {}

    @pytest.mark.asyncio
    async def test_action_handler_forwards_kwargs_to_agent_run(self):
        """handle_invoke_azure_agent should forward ctx.run_kwargs to agent.run()."""
        import agent_framework_declarative._workflows._actions_agents  # noqa: F401

        # Create a mock agent
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_response.messages = []
        mock_response.tool_calls = []

        mock_agent = AsyncMock()
        # Make streaming raise TypeError so it falls back to non-streaming
        mock_agent.run = AsyncMock(return_value=mock_response)

        # Make the agent not support streaming by removing the async iterator
        async def non_streaming_run(*args, **kwargs):
            if kwargs.get("stream"):
                raise TypeError("no streaming")
            return mock_response

        mock_agent.run = AsyncMock(side_effect=non_streaming_run)

        test_kwargs = {"user_token": "secret", "api_key": "key123"}

        state = WorkflowState()
        state.add_conversation_message(MagicMock(role="user", text="hello"))

        ctx = _create_action_context(
            action={
                "kind": "InvokeAzureAgent",
                "agent": "my_agent",
            },
            state=state,
            agents={"my_agent": mock_agent},
            run_kwargs=test_kwargs,
        )

        handler = get_action_handler("InvokeAzureAgent")
        _ = [e async for e in handler(ctx)]

        # agent.run should have been called (streaming attempt + non-streaming fallback)
        assert mock_agent.run.call_count >= 1

        # Find the non-streaming call (the one without stream=True, or with stream=True that errored + fallback)
        for call in mock_agent.run.call_args_list:
            call_kw = call.kwargs
            if not call_kw.get("stream"):
                # This is the non-streaming fallback call
                assert call_kw.get("user_token") == "secret"
                assert call_kw.get("api_key") == "key123"
                assert call_kw.get("options") == {"additional_function_arguments": test_kwargs}
                break
        else:
            # All calls were streaming — check the streaming call
            call_kw = mock_agent.run.call_args_list[0].kwargs
            assert call_kw.get("user_token") == "secret"
            assert call_kw.get("api_key") == "key123"
