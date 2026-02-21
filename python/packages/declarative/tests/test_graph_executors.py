# Copyright (c) Microsoft. All rights reserved.

"""Tests for the graph-based declarative workflow executors."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework_declarative._workflows import (
    ALL_ACTION_EXECUTORS,
    DECLARATIVE_STATE_KEY,
    ActionComplete,
    ActionTrigger,
    DeclarativeWorkflowBuilder,
    DeclarativeWorkflowState,
    ForeachInitExecutor,
    LoopIterationResult,
    SendActivityExecutor,
    SetValueExecutor,
)


class TestDeclarativeWorkflowState:
    """Tests for DeclarativeWorkflowState."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock shared state with async get/set methods."""
        mock_state = MagicMock()
        mock_state._data = {}

        def mock_get(key, default=None):
            return mock_state._data.get(key, default)

        def mock_set(key, value):
            mock_state._data[key] = value

        mock_state.get = MagicMock(side_effect=mock_get)
        mock_state.set = MagicMock(side_effect=mock_set)

        return mock_state

    @pytest.mark.asyncio
    async def test_initialize_state(self, mock_state):
        """Test initializing the workflow state."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize({"query": "test"})

        # Verify state was set
        mock_state.set.assert_called_once()
        call_args = mock_state.set.call_args
        assert call_args[0][0] == DECLARATIVE_STATE_KEY
        state_data = call_args[0][1]
        assert state_data["Inputs"] == {"query": "test"}
        assert state_data["Outputs"] == {}
        assert state_data["Local"] == {}

    @pytest.mark.asyncio
    async def test_get_and_set_values(self, mock_state):
        """Test getting and setting values."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize()

        # Set a turn value
        state.set("Local.counter", 5)

        # Get the value
        result = state.get("Local.counter")
        assert result == 5

    @pytest.mark.asyncio
    async def test_get_inputs(self, mock_state):
        """Test getting workflow inputs."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize({"name": "Alice", "age": 30})

        # Get via path
        name = state.get("Workflow.Inputs.name")
        assert name == "Alice"

        # Get all inputs
        inputs = state.get("Workflow.Inputs")
        assert inputs == {"name": "Alice", "age": 30}

    @pytest.mark.asyncio
    async def test_append_value(self, mock_state):
        """Test appending values to a list."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize()

        # Append to non-existent list creates it
        state.append("Local.items", "first")
        result = state.get("Local.items")
        assert result == ["first"]

        # Append to existing list
        state.append("Local.items", "second")
        result = state.get("Local.items")
        assert result == ["first", "second"]

    @pytest.mark.asyncio
    async def test_eval_expression(self, mock_state):
        """Test evaluating expressions."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize()

        # Non-expression returns as-is
        result = state.eval("plain text")
        assert result == "plain text"

        # Boolean literals
        result = state.eval("=true")
        assert result is True

        result = state.eval("=false")
        assert result is False

        # String literals
        result = state.eval('="hello"')
        assert result == "hello"

        # Numeric literals
        result = state.eval("=42")
        assert result == 42


class TestDeclarativeActionExecutor:
    """Tests for DeclarativeActionExecutor subclasses."""

    @pytest.fixture
    def mock_context(self, mock_state):
        """Create a mock workflow context."""
        ctx = MagicMock()
        ctx.state = mock_state
        ctx.send_message = AsyncMock()
        ctx.yield_output = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_state(self):
        """Create a mock shared state."""
        mock_state = MagicMock()
        mock_state._data = {}

        def mock_get(key, default=None):
            return mock_state._data.get(key, default)

        def mock_set(key, value):
            mock_state._data[key] = value

        mock_state.get = MagicMock(side_effect=mock_get)
        mock_state.set = MagicMock(side_effect=mock_set)

        return mock_state

    @pytest.mark.asyncio
    async def test_set_value_executor(self, mock_context, mock_state):
        """Test SetValueExecutor."""
        # Initialize state
        state = DeclarativeWorkflowState(mock_state)
        state.initialize()

        action_def = {
            "kind": "SetValue",
            "path": "Local.result",
            "value": "test value",
        }
        executor = SetValueExecutor(action_def)

        # Execute
        await executor.handle_action(ActionTrigger(), mock_context)

        # Verify action complete was sent
        mock_context.send_message.assert_called_once()
        message = mock_context.send_message.call_args[0][0]
        assert isinstance(message, ActionComplete)

    @pytest.mark.asyncio
    async def test_send_activity_executor(self, mock_context, mock_state):
        """Test SendActivityExecutor."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize()

        action_def = {
            "kind": "SendActivity",
            "activity": {"text": "Hello, world!"},
        }
        executor = SendActivityExecutor(action_def)

        # Execute
        await executor.handle_action(ActionTrigger(), mock_context)

        # Verify output was yielded
        mock_context.yield_output.assert_called_once_with("Hello, world!")

    # Note: ConditionEvaluatorExecutor tests removed - conditions are now evaluated on edges

    @pytest.mark.asyncio
    async def test_foreach_init_with_items(self, mock_context, mock_state):
        """Test ForeachInitExecutor with items."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize()
        state.set("Local.items", ["a", "b", "c"])

        action_def = {
            "kind": "Foreach",
            "itemsSource": "=Local.items",
            "iteratorVariable": "Local.item",
        }
        executor = ForeachInitExecutor(action_def)

        # Execute
        await executor.handle_action(ActionTrigger(), mock_context)

        # Verify result
        mock_context.send_message.assert_called_once()
        message = mock_context.send_message.call_args[0][0]
        assert isinstance(message, LoopIterationResult)
        assert message.has_next is True
        assert message.current_index == 0
        assert message.current_item == "a"

    @pytest.mark.asyncio
    async def test_foreach_init_empty(self, mock_context, mock_state):
        """Test ForeachInitExecutor with empty items list."""
        state = DeclarativeWorkflowState(mock_state)
        state.initialize()

        # Use a literal empty list - no expression evaluation needed
        action_def = {
            "kind": "Foreach",
            "itemsSource": [],  # Direct empty list, not an expression
            "iteratorVariable": "Local.item",
        }
        executor = ForeachInitExecutor(action_def)

        # Execute
        await executor.handle_action(ActionTrigger(), mock_context)

        # Verify result
        mock_context.send_message.assert_called_once()
        message = mock_context.send_message.call_args[0][0]
        assert isinstance(message, LoopIterationResult)
        assert message.has_next is False


class TestDeclarativeWorkflowBuilder:
    """Tests for DeclarativeWorkflowBuilder."""

    def test_all_action_executors_available(self):
        """Test that all expected action types have executors."""
        expected_actions = [
            "SetValue",
            "SetVariable",
            "SendActivity",
            "EmitEvent",
            "EndWorkflow",
            "InvokeAzureAgent",
            "Question",
        ]

        for action in expected_actions:
            assert action in ALL_ACTION_EXECUTORS, f"Missing executor for {action}"

    def test_build_empty_workflow(self):
        """Test building a workflow with no actions raises an error."""
        yaml_def = {"name": "empty_workflow", "actions": []}
        builder = DeclarativeWorkflowBuilder(yaml_def)

        with pytest.raises(ValueError, match="Cannot build workflow with no actions"):
            builder.build()

    def test_build_simple_workflow(self):
        """Test building a workflow with simple sequential actions."""
        yaml_def = {
            "name": "simple_workflow",
            "actions": [
                {"kind": "SendActivity", "id": "greet", "activity": {"text": "Hello!"}},
                {"kind": "SetValue", "id": "set_count", "path": "Local.count", "value": 1},
            ],
        }
        builder = DeclarativeWorkflowBuilder(yaml_def)
        workflow = builder.build()

        assert workflow is not None
        # Verify executors were created
        assert "greet" in builder._executors
        assert "set_count" in builder._executors

    def test_build_workflow_with_if(self):
        """Test building a workflow with If control flow."""
        yaml_def = {
            "name": "conditional_workflow",
            "actions": [
                {
                    "kind": "If",
                    "id": "check_flag",
                    "condition": "=Local.flag",
                    "then": [
                        {"kind": "SendActivity", "id": "say_yes", "activity": {"text": "Yes!"}},
                    ],
                    "else": [
                        {"kind": "SendActivity", "id": "say_no", "activity": {"text": "No!"}},
                    ],
                },
            ],
        }
        builder = DeclarativeWorkflowBuilder(yaml_def)
        workflow = builder.build()

        assert workflow is not None
        # Verify branch executors were created
        # Note: No join executors - branches wire directly to successor
        assert "say_yes" in builder._executors
        assert "say_no" in builder._executors
        # Entry node is created when If is first action
        assert "_workflow_entry" in builder._executors

    def test_build_workflow_with_foreach(self):
        """Test building a workflow with Foreach loop."""
        yaml_def = {
            "name": "loop_workflow",
            "actions": [
                {
                    "kind": "Foreach",
                    "id": "process_items",
                    "itemsSource": "=Local.items",
                    "iteratorVariable": "Local.item",
                    "actions": [
                        {"kind": "SendActivity", "id": "show_item", "activity": {"text": "=Local.item"}},
                    ],
                },
            ],
        }
        builder = DeclarativeWorkflowBuilder(yaml_def)
        workflow = builder.build()

        assert workflow is not None
        # Verify loop executors were created
        assert "process_items_init" in builder._executors
        assert "process_items_next" in builder._executors
        assert "process_items_exit" in builder._executors
        assert "show_item" in builder._executors

    def test_build_workflow_with_switch(self):
        """Test building a workflow with Switch control flow."""
        yaml_def = {
            "name": "switch_workflow",
            "actions": [
                {
                    "kind": "Switch",
                    "id": "check_status",
                    "conditions": [
                        {
                            "condition": '=Local.status = "active"',
                            "actions": [
                                {"kind": "SendActivity", "id": "say_active", "activity": {"text": "Active"}},
                            ],
                        },
                        {
                            "condition": '=Local.status = "pending"',
                            "actions": [
                                {"kind": "SendActivity", "id": "say_pending", "activity": {"text": "Pending"}},
                            ],
                        },
                    ],
                    "else": [
                        {"kind": "SendActivity", "id": "say_unknown", "activity": {"text": "Unknown"}},
                    ],
                },
            ],
        }
        builder = DeclarativeWorkflowBuilder(yaml_def)
        workflow = builder.build()

        assert workflow is not None
        # Verify switch executors were created
        # Note: No join executors - branches wire directly to successor
        assert "say_active" in builder._executors
        assert "say_pending" in builder._executors
        assert "say_unknown" in builder._executors
        # Entry node is created when Switch is first action
        assert "_workflow_entry" in builder._executors
