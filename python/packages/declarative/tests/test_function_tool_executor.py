# Copyright (c) Microsoft. All rights reserved.

"""Tests for InvokeFunctionTool executor.

These tests verify:
- Basic function invocation (sync and async)
- Expression evaluation for functionName and arguments
- Output formatting (messages and result)
- Error handling (function not found, execution errors)
- WorkflowFactory registration
"""

import pytest

from agent_framework_declarative._workflows import (
    DeclarativeWorkflowBuilder,
    InvokeFunctionToolExecutor,
    ToolApprovalRequest,
    ToolApprovalResponse,
    ToolApprovalState,
    ToolInvocationResult,
    WorkflowFactory,
)


class TestInvokeFunctionToolExecutor:
    """Tests for InvokeFunctionToolExecutor."""

    @pytest.mark.asyncio
    async def test_basic_sync_function_invocation(self):
        """Test invoking a simple synchronous function."""

        def get_weather(location: str, unit: str = "F") -> dict:
            return {"temp": 72, "unit": unit, "location": location}

        yaml_def = {
            "name": "function_tool_test",
            "actions": [
                {"kind": "SetValue", "id": "set_location", "path": "Local.city", "value": "Seattle"},
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_weather",
                    "functionName": "get_weather",
                    "arguments": {"location": "=Local.city", "unit": "C"},
                    "output": {"result": "Local.weatherData"},
                },
                # Use SendActivity to output the result so we can check it
                {"kind": "SendActivity", "id": "output_location", "activity": {"text": "=Local.weatherData.location"}},
                {"kind": "SendActivity", "id": "output_unit", "activity": {"text": "=Local.weatherData.unit"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"get_weather": get_weather})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        # Verify the function was called with correct arguments
        assert "Seattle" in outputs  # location
        assert "C" in outputs  # unit

    @pytest.mark.asyncio
    async def test_async_function_invocation(self):
        """Test invoking an async function."""

        async def fetch_data(url: str) -> dict:
            return {"url": url, "status": "success"}

        yaml_def = {
            "name": "async_function_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "fetch",
                    "functionName": "fetch_data",
                    "arguments": {"url": "https://example.com/api"},
                    "output": {"result": "Local.response"},
                },
                {"kind": "SendActivity", "id": "output_url", "activity": {"text": "=Local.response.url"}},
                {"kind": "SendActivity", "id": "output_status", "activity": {"text": "=Local.response.status"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"fetch_data": fetch_data})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert "https://example.com/api" in outputs
        assert "success" in outputs

    @pytest.mark.asyncio
    async def test_expression_function_name(self):
        """Test dynamic function name via expression."""

        def tool_a() -> str:
            return "result_a"

        def tool_b() -> str:
            return "result_b"

        yaml_def = {
            "name": "dynamic_function_name_test",
            "actions": [
                {"kind": "SetValue", "id": "set_tool", "path": "Local.toolName", "value": "tool_b"},
                {
                    "kind": "InvokeFunctionTool",
                    "id": "dynamic_call",
                    "functionName": "=Local.toolName",
                    "arguments": {},
                    "output": {"result": "Local.result"},
                },
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.result"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"tool_a": tool_a, "tool_b": tool_b})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert "result_b" in outputs

    @pytest.mark.asyncio
    async def test_function_not_found(self):
        """Test error handling when function is not in registry."""
        yaml_def = {
            "name": "function_not_found_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_missing",
                    "functionName": "nonexistent_function",
                    "arguments": {},
                    "output": {"result": "Local.result"},
                },
                # Check if error is stored
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.result.error"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={})  # Empty registry
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        # Result should contain error info
        assert "not found" in outputs[0].lower()

    @pytest.mark.asyncio
    async def test_function_execution_error(self):
        """Test error handling when function raises exception."""

        def failing_function() -> str:
            raise ValueError("Intentional test error")

        yaml_def = {
            "name": "function_error_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_failing",
                    "functionName": "failing_function",
                    "arguments": {},
                    "output": {"result": "Local.result"},
                },
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.result.error"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"failing_function": failing_function})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        # Result should contain error info
        assert "Intentional test error" in outputs[0]

    @pytest.mark.asyncio
    async def test_function_with_no_output_config(self):
        """Test that function works even without output configuration."""

        counter = {"value": 0}

        def increment() -> int:
            counter["value"] += 1
            return counter["value"]

        yaml_def = {
            "name": "no_output_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "increment_call",
                    "functionName": "increment",
                    "arguments": {},
                    # No output configuration
                },
                {"kind": "SendActivity", "id": "done", "activity": {"text": "Done"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"increment": increment})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        # Workflow should complete
        assert "Done" in outputs
        # Function should have been called
        assert counter["value"] == 1


class TestInvokeFunctionToolWithWorkflowFactory:
    """Tests for InvokeFunctionTool with WorkflowFactory registration."""

    @pytest.mark.asyncio
    async def test_register_tool_method(self):
        """Test registering tools via WorkflowFactory.register_tool()."""

        def multiply(a: int, b: int) -> int:
            return a * b

        yaml_content = """
name: factory_tool_test
actions:
  - kind: InvokeFunctionTool
    id: multiply_call
    functionName: multiply
    arguments:
      a: 6
      b: 7
    output:
      result: Local.product
  - kind: SendActivity
    id: output
    activity:
      text: =Local.product
"""
        factory = WorkflowFactory().register_tool("multiply", multiply)
        workflow = factory.create_workflow_from_yaml(yaml_content)

        events = await workflow.run({})
        outputs = events.get_outputs()

        # PowerFx outputs integers as floats, so we check for 42 or 42.0
        assert any("42" in out for out in outputs)

    @pytest.mark.asyncio
    async def test_fluent_registration(self):
        """Test fluent chaining for tool registration."""

        def add(a: int, b: int) -> int:
            return a + b

        def subtract(a: int, b: int) -> int:
            return a - b

        yaml_content = """
name: fluent_test
actions:
  - kind: InvokeFunctionTool
    id: add_call
    functionName: add
    arguments:
      a: 10
      b: 5
    output:
      result: Local.sum
  - kind: InvokeFunctionTool
    id: subtract_call
    functionName: subtract
    arguments:
      a: 10
      b: 5
    output:
      result: Local.diff
  - kind: SendActivity
    id: output_sum
    activity:
      text: =Local.sum
  - kind: SendActivity
    id: output_diff
    activity:
      text: =Local.diff
"""
        factory = WorkflowFactory().register_tool("add", add).register_tool("subtract", subtract)

        workflow = factory.create_workflow_from_yaml(yaml_content)

        events = await workflow.run({})
        outputs = events.get_outputs()

        # PowerFx outputs integers as floats, so we check for 15 or 15.0
        assert any("15" in out for out in outputs)  # sum
        assert any("5" in out for out in outputs)  # diff


class TestToolInvocationResult:
    """Tests for ToolInvocationResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = ToolInvocationResult(
            success=True,
            result={"data": "value"},
            messages=[],
        )
        assert result.success is True
        assert result.result == {"data": "value"}
        assert result.rejected is False
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolInvocationResult(
            success=False,
            error="Function failed",
        )
        assert result.success is False
        assert result.error == "Function failed"
        assert result.result is None

    def test_rejected_result(self):
        """Test creating a rejected result."""
        result = ToolInvocationResult(
            success=False,
            rejected=True,
            rejection_reason="User denied approval",
        )
        assert result.success is False
        assert result.rejected is True
        assert result.rejection_reason == "User denied approval"


class TestToolApprovalTypes:
    """Tests for approval-related dataclasses."""

    def test_approval_request(self):
        """Test creating an approval request."""
        request = ToolApprovalRequest(
            request_id="test-123",
            function_name="dangerous_operation",
            arguments={"target": "production"},
            conversation_id="conv-456",
        )
        assert request.request_id == "test-123"
        assert request.function_name == "dangerous_operation"
        assert request.arguments == {"target": "production"}
        assert request.conversation_id == "conv-456"

    def test_approval_response_approved(self):
        """Test creating an approved response."""
        response = ToolApprovalResponse(approved=True)
        assert response.approved is True
        assert response.reason is None

    def test_approval_response_rejected(self):
        """Test creating a rejected response."""
        response = ToolApprovalResponse(approved=False, reason="Not authorized")
        assert response.approved is False
        assert response.reason == "Not authorized"

    def test_approval_state(self):
        """Test creating approval state for yield/resume."""
        state = ToolApprovalState(
            function_name="delete_user",
            arguments={"user_id": "123"},
            output_messages_var="Local.messages",
            output_result_var="Local.result",
            conversation_id="conv-789",
        )
        assert state.function_name == "delete_user"
        assert state.arguments == {"user_id": "123"}
        assert state.output_messages_var == "Local.messages"
        assert state.output_result_var == "Local.result"
        assert state.conversation_id == "conv-789"


class TestInvokeFunctionToolEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_function_name_field_raises_validation_error(self):
        """Test that missing functionName raises validation error at build time."""
        yaml_def = {
            "name": "missing_function_name_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "no_name",
                    # Missing functionName field
                    "arguments": {},
                },
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={})

        # Should raise validation error
        with pytest.raises(ValueError, match="missing required field 'functionName'"):
            builder.build()

    @pytest.mark.asyncio
    async def test_empty_function_name_expression(self):
        """Test handling when functionName expression evaluates to empty."""
        yaml_def = {
            "name": "empty_function_name_test",
            "actions": [
                {"kind": "SetValue", "id": "set_empty", "path": "Local.toolName", "value": ""},
                {
                    "kind": "InvokeFunctionTool",
                    "id": "empty_name",
                    "functionName": "=Local.toolName",
                    "arguments": {},
                },
                {"kind": "SendActivity", "id": "done", "activity": {"text": "Completed"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        # Should complete without crashing
        assert "Completed" in outputs

    @pytest.mark.asyncio
    async def test_messages_output_configuration(self):
        """Test that messages output stores ChatMessage list."""

        def simple_func(x: int) -> int:
            return x * 2

        yaml_def = {
            "name": "messages_output_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_func",
                    "functionName": "simple_func",
                    "arguments": {"x": 5},
                    "output": {
                        "messages": "Local.toolMessages",
                        "result": "Local.result",
                    },
                },
                {"kind": "SendActivity", "id": "output_result", "activity": {"text": "=Local.result"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"simple_func": simple_func})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        # Result should be doubled
        assert any("10" in out for out in outputs)

    @pytest.mark.asyncio
    async def test_function_returning_none(self):
        """Test handling function that returns None."""

        def returns_none() -> None:
            pass

        yaml_def = {
            "name": "returns_none_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_none",
                    "functionName": "returns_none",
                    "arguments": {},
                    "output": {"result": "Local.result"},
                },
                {"kind": "SendActivity", "id": "done", "activity": {"text": "Completed"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"returns_none": returns_none})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert "Completed" in outputs

    @pytest.mark.asyncio
    async def test_function_with_complex_return_type(self):
        """Test function returning complex nested data."""

        def complex_return() -> dict:
            return {
                "nested": {
                    "array": [1, 2, 3],
                    "string": "test",
                },
                "boolean": True,
                "number": 42.5,
            }

        yaml_def = {
            "name": "complex_return_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_complex",
                    "functionName": "complex_return",
                    "arguments": {},
                    "output": {"result": "Local.data"},
                },
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.data.nested.string"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"complex_return": complex_return})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert "test" in outputs

    @pytest.mark.asyncio
    async def test_function_with_list_argument(self):
        """Test passing list as argument."""

        def sum_list(numbers: list) -> int:
            return sum(numbers)

        yaml_def = {
            "name": "list_argument_test",
            "actions": [
                {"kind": "SetValue", "id": "set_list", "path": "Local.numbers", "value": [1, 2, 3, 4, 5]},
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_sum",
                    "functionName": "sum_list",
                    "arguments": {"numbers": "=Local.numbers"},
                    "output": {"result": "Local.total"},
                },
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.total"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"sum_list": sum_list})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert any("15" in out for out in outputs)

    @pytest.mark.asyncio
    async def test_conversation_id_expression(self):
        """Test conversationId field with expression."""

        def echo_id(msg: str) -> str:
            return msg

        yaml_def = {
            "name": "conversation_id_test",
            "actions": [
                {"kind": "SetValue", "id": "set_conv_id", "path": "Local.convId", "value": "conv-123"},
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_with_conv_id",
                    "functionName": "echo_id",
                    "conversationId": "=Local.convId",
                    "arguments": {"msg": "hello"},
                    "output": {"result": "Local.result"},
                },
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.result"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"echo_id": echo_id})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert "hello" in outputs

    @pytest.mark.asyncio
    async def test_function_with_only_result_output(self):
        """Test output config with only result, no messages."""

        def double(x: int) -> int:
            return x * 2

        yaml_def = {
            "name": "result_only_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_double",
                    "functionName": "double",
                    "arguments": {"x": 21},
                    "output": {"result": "Local.doubled"},
                },
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.doubled"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"double": double})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert any("42" in out for out in outputs)

    @pytest.mark.asyncio
    async def test_function_with_only_messages_output(self):
        """Test output config with only messages, no result."""

        def simple() -> str:
            return "done"

        yaml_def = {
            "name": "messages_only_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_simple",
                    "functionName": "simple",
                    "arguments": {},
                    "output": {"messages": "Local.msgs"},
                },
                {"kind": "SendActivity", "id": "done", "activity": {"text": "Completed"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"simple": simple})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert "Completed" in outputs

    @pytest.mark.asyncio
    async def test_function_string_return(self):
        """Test function that returns a simple string."""

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        yaml_def = {
            "name": "string_return_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "call_greet",
                    "functionName": "greet",
                    "arguments": {"name": "World"},
                    "output": {"result": "Local.greeting"},
                },
                {"kind": "SendActivity", "id": "output", "activity": {"text": "=Local.greeting"}},
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"greet": greet})
        workflow = builder.build()

        events = await workflow.run({})
        outputs = events.get_outputs()

        assert "Hello, World!" in outputs


class TestInvokeFunctionToolBuilder:
    """Tests for InvokeFunctionTool executor registration in builder."""

    def test_executor_registered_in_all_executors(self):
        """Test that InvokeFunctionTool is registered in ALL_ACTION_EXECUTORS."""
        from agent_framework_declarative._workflows import ALL_ACTION_EXECUTORS

        assert "InvokeFunctionTool" in ALL_ACTION_EXECUTORS
        assert ALL_ACTION_EXECUTORS["InvokeFunctionTool"] == InvokeFunctionToolExecutor

    def test_builder_creates_tool_executor(self):
        """Test that builder creates InvokeFunctionToolExecutor for InvokeFunctionTool actions."""

        def dummy() -> str:
            return "test"

        yaml_def = {
            "name": "builder_test",
            "actions": [
                {
                    "kind": "InvokeFunctionTool",
                    "id": "my_tool",
                    "functionName": "dummy",
                    "arguments": {},
                },
            ],
        }

        builder = DeclarativeWorkflowBuilder(yaml_def, tools={"dummy": dummy})
        _ = builder.build()

        # Verify the executor was created
        assert "my_tool" in builder._executors
        executor = builder._executors["my_tool"]
        assert isinstance(executor, InvokeFunctionToolExecutor)
