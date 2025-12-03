# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework import (
    ChatMessage,
    ConversationToText,
    FunctionAdapter,
    Role,
    TextToConversation,
    TypeAdapter,
    WorkflowContext,
)


@dataclass
class IntToStrAdapter(TypeAdapter[int, str]):
    """Test adapter that converts int to str."""

    input_type: type[int] = int
    output_type: type[str] = str

    async def adapt(self, value: int, ctx: WorkflowContext) -> str:
        return str(value * 2)


class TestTypeAdapterBase:
    """Tests for TypeAdapter base class."""

    async def test_custom_adapter_transforms_value(self) -> None:
        """Test that a custom adapter correctly transforms values."""
        adapter = IntToStrAdapter(id="int_to_str")
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt(42, ctx)

        assert result == "84"
        assert isinstance(result, str)

    async def test_adapter_has_correct_input_types(self) -> None:
        """Test that adapter reports correct input types."""
        adapter = IntToStrAdapter(id="test")

        assert adapter.input_types == [int]

    async def test_adapter_has_correct_output_types(self) -> None:
        """Test that adapter reports correct output types."""
        adapter = IntToStrAdapter(id="test")

        assert adapter.output_types == [str]

    async def test_adapter_validates_input_type(self) -> None:
        """Test that adapter validates input type when strict_validation is True."""
        adapter = IntToStrAdapter(id="test", strict_validation=True)

        with pytest.raises(TypeError, match="expected input of type"):
            adapter.validate_input("not an int")

    async def test_adapter_validates_output_type(self) -> None:
        """Test that adapter validates output type when strict_validation is True."""
        adapter = IntToStrAdapter(id="test", strict_validation=True)

        with pytest.raises(TypeError, match="produced output of type"):
            adapter.validate_output(123)  # Expected str, got int

    async def test_adapter_skips_validation_when_disabled(self) -> None:
        """Test that adapter skips validation when strict_validation is False."""
        adapter = IntToStrAdapter(id="test", strict_validation=False)

        # Should not raise even with wrong types
        result_in = adapter.validate_input("not an int")
        result_out = adapter.validate_output(123)

        assert result_in == "not an int"
        assert result_out == 123

    async def test_adapter_auto_generates_id(self) -> None:
        """Test that adapter auto-generates ID if not provided."""
        adapter = IntToStrAdapter()

        assert adapter.id is not None
        assert adapter.id.startswith("adapter-")


class TestTextToConversation:
    """Tests for TextToConversation adapter."""

    async def test_converts_text_to_conversation(self) -> None:
        """Test basic text to conversation conversion."""
        adapter = TextToConversation(id="t2c")
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt("Hello, world!", ctx)

        assert len(result) == 1
        assert result[0].text == "Hello, world!"
        assert result[0].role == Role.USER

    async def test_uses_custom_role(self) -> None:
        """Test that custom role is applied."""
        adapter = TextToConversation(id="t2c", role=Role.ASSISTANT)
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt("Response text", ctx)

        assert result[0].role == Role.ASSISTANT

    async def test_uses_string_role(self) -> None:
        """Test that string role is converted properly."""
        adapter = TextToConversation(id="t2c", role="system")
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt("System message", ctx)

        assert result[0].role == Role.SYSTEM

    async def test_includes_author_name(self) -> None:
        """Test that author_name is included."""
        adapter = TextToConversation(id="t2c", author_name="TestUser")
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt("Message", ctx)

        assert result[0].author_name == "TestUser"

    async def test_output_types_returns_parameterized_list(self) -> None:
        """Test that output_types returns list[ChatMessage]."""
        adapter = TextToConversation(id="t2c")

        output_types = adapter.output_types

        assert len(output_types) == 1
        assert output_types[0] == list[ChatMessage]


class TestConversationToText:
    """Tests for ConversationToText adapter."""

    async def test_converts_conversation_to_text(self) -> None:
        """Test basic conversation to text conversion."""
        adapter = ConversationToText(id="c2t")
        ctx = MagicMock(spec=WorkflowContext)
        messages = [
            ChatMessage(role=Role.USER, text="Hello"),
            ChatMessage(role=Role.ASSISTANT, text="Hi there!"),
        ]

        result = await adapter.adapt(messages, ctx)

        assert result == "Hello\n\nHi there!"

    async def test_uses_custom_separator(self) -> None:
        """Test that custom separator is used."""
        adapter = ConversationToText(id="c2t", separator=" | ")
        ctx = MagicMock(spec=WorkflowContext)
        messages = [
            ChatMessage(role=Role.USER, text="A"),
            ChatMessage(role=Role.ASSISTANT, text="B"),
        ]

        result = await adapter.adapt(messages, ctx)

        assert result == "A | B"

    async def test_includes_roles(self) -> None:
        """Test that roles are included when requested."""
        adapter = ConversationToText(id="c2t", include_roles=True)
        ctx = MagicMock(spec=WorkflowContext)
        messages = [
            ChatMessage(role=Role.USER, text="Hello"),
        ]

        result = await adapter.adapt(messages, ctx)

        assert result == "user: Hello"

    async def test_last_only_extracts_last_message(self) -> None:
        """Test that last_only returns only the last message."""
        adapter = ConversationToText(id="c2t", last_only=True)
        ctx = MagicMock(spec=WorkflowContext)
        messages = [
            ChatMessage(role=Role.USER, text="First"),
            ChatMessage(role=Role.ASSISTANT, text="Last"),
        ]

        result = await adapter.adapt(messages, ctx)

        assert result == "Last"

    async def test_handles_empty_conversation(self) -> None:
        """Test that empty conversation returns empty string."""
        adapter = ConversationToText(id="c2t")
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt([], ctx)

        assert result == ""

    async def test_input_types_returns_parameterized_list(self) -> None:
        """Test that input_types returns list[ChatMessage]."""
        adapter = ConversationToText(id="c2t")

        input_types = adapter.input_types

        assert len(input_types) == 1
        assert input_types[0] == list[ChatMessage]


class TestFunctionAdapter:
    """Tests for FunctionAdapter."""

    async def test_sync_function_transforms_value(self) -> None:
        """Test that sync function adapter works."""
        adapter: FunctionAdapter[str, int] = FunctionAdapter(
            id="str_to_int",
            fn=lambda s, ctx: len(s),
            _input_type=str,
            _output_type=int,
        )
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt("hello", ctx)

        assert result == 5

    async def test_async_function_transforms_value(self) -> None:
        """Test that async function adapter works."""

        async def async_transform(s: str, ctx: Any) -> int:
            return len(s) * 2

        adapter: FunctionAdapter[str, int] = FunctionAdapter(
            id="async_str_to_int",
            fn=async_transform,
            _input_type=str,
            _output_type=int,
        )
        ctx = MagicMock(spec=WorkflowContext)

        result = await adapter.adapt("hello", ctx)

        assert result == 10

    async def test_requires_fn_parameter(self) -> None:
        """Test that FunctionAdapter requires fn parameter."""
        with pytest.raises(ValueError, match="requires a transformation function"):
            FunctionAdapter(id="no_fn", _input_type=str, _output_type=int)

    async def test_inherits_id_from_type_adapter(self) -> None:
        """Test that FunctionAdapter properly inherits id handling."""
        adapter: FunctionAdapter[str, str] = FunctionAdapter(
            id="custom_id",
            fn=lambda s, ctx: s.upper(),
            _input_type=str,
            _output_type=str,
        )

        assert adapter.id == "custom_id"

    async def test_auto_generates_id(self) -> None:
        """Test that FunctionAdapter auto-generates id if not provided."""
        adapter: FunctionAdapter[str, str] = FunctionAdapter(
            fn=lambda s, ctx: s.upper(),
            _input_type=str,
            _output_type=str,
        )

        assert adapter.id is not None
        assert adapter.id.startswith("adapter-")


class TestAdapterHandler:
    """Tests for adapter handler integration."""

    async def test_handler_calls_adapt(self) -> None:
        """Test that the handler method calls adapt correctly."""
        adapter = IntToStrAdapter(id="test")
        ctx = MagicMock(spec=WorkflowContext)
        ctx.send_message = AsyncMock()

        await adapter.handle_input(42, ctx)

        ctx.send_message.assert_called_once_with("84")

    async def test_handler_unwraps_single_element_sequence(self) -> None:
        """Test that handler unwraps single-element sequences."""
        adapter = IntToStrAdapter(id="test")
        ctx = MagicMock(spec=WorkflowContext)
        ctx.send_message = AsyncMock()

        await adapter.handle_input([42], ctx)

        ctx.send_message.assert_called_once_with("84")

    async def test_handler_passes_sequence_for_multiple_elements(self) -> None:
        """Test that handler passes full sequence for multiple elements."""

        @dataclass
        class ListToStrAdapter(TypeAdapter[list[int], str]):
            input_type: type[list[int]] = list
            output_type: type[str] = str

            async def adapt(self, value: list[int], ctx: WorkflowContext) -> str:
                return ",".join(str(v) for v in value)

        adapter = ListToStrAdapter(id="test", strict_validation=False)
        ctx = MagicMock(spec=WorkflowContext)
        ctx.send_message = AsyncMock()

        await adapter.handle_input([1, 2, 3], ctx)

        ctx.send_message.assert_called_once_with("1,2,3")
