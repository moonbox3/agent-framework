# Copyright (c) Microsoft. All rights reserved.

"""Type adapter utilities for workflow composition.

This module provides infrastructure for transforming data between incompatible types
when composing workflows. When two workflows cannot be directly connected because their
input/output types do not align, a type adapter bridges the gap by performing an
explicit, type-safe transformation.

Design Philosophy
-----------------
The adapter pattern here follows the Gang of Four "Adapter" pattern, adapted for
dataflow graphs. Key principles:

1. **Explicit over Implicit**: Rather than performing silent coercion (which hides bugs),
   adapters make type transformations visible in the workflow graph.

2. **Composable**: Adapters are themselves executors, meaning they appear as nodes in
   the workflow graph and can be inspected, debugged, and traced.

3. **Bidirectional When Possible**: Some adapters provide both forward and reverse
   transformations, enabling reuse across composition boundaries.

4. **Zero Overhead for Compatible Types**: When source and target types are already
   compatible, no adapter is needed - this module only handles mismatches.

Usage Patterns
--------------
There are three levels of adapter usage:

**Level 1: Built-in Adapters (90% of cases)**

For common transformations like str <-> list[ChatMessage], use the provided adapters:

.. code-block:: python

    from agent_framework._workflows._type_adapters import TextToConversation

    adapter = TextToConversation()
    # Use in workflow:
    builder.add_executor("adapt", adapter)
    builder.add_edge(["text_producer"], "adapt")
    builder.add_edge(["adapt"], "chat_consumer")

**Level 2: Custom Adapters (9% of cases)**

For domain-specific transformations, subclass TypeAdapter:

.. code-block:: python

    from agent_framework._workflows._type_adapters import TypeAdapter


    class CustomerToProfile(TypeAdapter[Customer, UserProfile]):
        input_type = Customer
        output_type = UserProfile

        async def adapt(self, value: Customer, ctx: WorkflowContext) -> UserProfile:
            return UserProfile(
                name=value.full_name,
                email=value.contact_email,
                tier=value.subscription_level,
            )

**Level 3: Inline Lambda Adapters (1% of cases)**

For one-off transformations, use FunctionAdapter:

.. code-block:: python

    from agent_framework._workflows._type_adapters import FunctionAdapter

    adapter = FunctionAdapter(
        input_type=dict,
        output_type=str,
        fn=lambda d, ctx: json.dumps(d),
    )

Type Safety Guarantees
----------------------
Adapters provide compile-time (via type checkers) and runtime type safety:

- Input types are validated before the adapter runs
- Output types are validated after the adapter runs (in debug mode)
- Type mismatches produce clear error messages with source/target info

See Also:
--------
- _workflow_builder.connect : The method that may require adapters
- _typing_utils.is_type_compatible : Used for type compatibility checking
- Option 3 in the composability ADR : Detailed design rationale
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast, get_origin

from agent_framework import ChatMessage, Role

from ._executor import Executor, handler
from ._workflow_context import WorkflowContext

__all__ = [
    "ConversationToText",
    "FunctionAdapter",
    "IdentityAdapter",
    "ItemToList",
    "ListToItem",
    "MessageWrapper",
    "SingleMessageExtractor",
    "TextToConversation",
    "TypeAdapter",
    "dict_to_struct_adapter",
    "find_adapter",
    "json_deserializer",
    "json_serializer",
    "struct_to_dict_adapter",
]


# Type variables for generic adapter signatures
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass
class TypeAdapter(ABC, Executor, Generic[TInput, TOutput]):
    """Abstract base class for type adapters used in workflow composition.

    A TypeAdapter is a specialized Executor that transforms data from one type to another.
    It serves as the explicit, type-safe bridge between workflow components that have
    incompatible input/output signatures.

    Subclass Contract
    -----------------
    Subclasses MUST:
    - Define `input_type` class attribute with the expected input type
    - Define `output_type` class attribute with the produced output type
    - Implement `adapt()` to perform the transformation

    Subclasses MAY:
    - Override `validate_input()` for custom input validation
    - Override `validate_output()` for custom output validation
    - Define additional fields for configuration

    Thread Safety
    -------------
    TypeAdapter instances should be stateless or use immutable configuration.
    The `adapt()` method may be called concurrently from multiple workflow branches.

    Example:
        .. code-block:: python

            class TemperatureConverter(TypeAdapter[Celsius, Fahrenheit]):
                input_type = Celsius
                output_type = Fahrenheit

                async def adapt(self, value: Celsius, ctx: WorkflowContext) -> Fahrenheit:
                    return Fahrenheit(value.degrees * 9 / 5 + 32)

    Attributes:
        id: Unique identifier for this adapter (inherited from Executor)
        input_type: Class attribute defining the expected input type
        output_type: Class attribute defining the produced output type
        name: Optional human-readable name for debugging
        strict_validation: If True, perform runtime type checks (default: True)
    """

    id: str = field(default_factory=lambda: f"adapter-{id(object())}")
    input_type: type[TInput] = field(default=object, init=False, repr=False)  # type: ignore[assignment]
    output_type: type[TOutput] = field(default=object, init=False, repr=False)  # type: ignore[assignment]

    name: str | None = field(default=None, repr=True)
    strict_validation: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Executor base class with the id."""
        Executor.__init__(self, id=self.id, defer_discovery=True)
        self._discover_handlers()

    @abstractmethod
    async def adapt(self, value: TInput, ctx: WorkflowContext) -> TOutput:
        """Transform the input value to the output type.

        This is the core transformation logic that subclasses must implement.
        The method receives a single input value (already validated if strict_validation
        is enabled) and must return a value of the output type.

        Args:
            value: The input value to transform, guaranteed to match input_type
            ctx: The workflow context providing access to shared state, history, etc.

        Returns:
            The transformed value, which must match output_type

        Raises:
            Any exception is propagated and will terminate the workflow branch.
            For recoverable errors, consider returning a Result type instead.
        """
        ...

    def validate_input(self, value: Any) -> TInput:
        """Validate and potentially coerce the input value.

        Override this method to implement custom input validation or coercion.
        The default implementation performs an isinstance check when
        strict_validation is enabled.

        Args:
            value: The raw input value from the upstream executor

        Returns:
            The validated (and potentially coerced) input value

        Raises:
            TypeError: If the value is not compatible with input_type
        """
        if self.strict_validation:
            # Handle parameterized generics like list[ChatMessage]
            check_type = get_origin(self.input_type) or self.input_type
            if not isinstance(value, check_type):
                type_name = getattr(self.input_type, "__name__", str(self.input_type))
                raise TypeError(
                    f"TypeAdapter {self.name or self.__class__.__name__} expected input of type "
                    f"{type_name}, got {type(value).__name__}"
                )
        return cast(TInput, value)

    def validate_output(self, value: Any) -> TOutput:
        """Validate the output value after transformation.

        Override this method to implement custom output validation.
        The default implementation performs an isinstance check when
        strict_validation is enabled.

        Args:
            value: The output value from the adapt() method

        Returns:
            The validated output value

        Raises:
            TypeError: If the value is not compatible with output_type
        """
        if self.strict_validation:
            # Handle parameterized generics like list[ChatMessage]
            check_type = get_origin(self.output_type) or self.output_type
            if not isinstance(value, check_type):
                type_name = getattr(self.output_type, "__name__", str(self.output_type))
                raise TypeError(
                    f"TypeAdapter {self.name or self.__class__.__name__} produced output of type "
                    f"{type(value).__name__}, expected {type_name}"
                )
        return cast(TOutput, value)

    # Executor interface implementation
    @property
    def input_types(self) -> list[type[Any]]:
        """Return the input types accepted by this adapter.

        This implements the Executor interface, allowing the adapter to be
        used as a node in the workflow graph.
        """
        return [self.input_type]

    @property
    def output_types(self) -> list[type[Any]]:
        """Return the output types produced by this adapter.

        This implements the Executor interface, allowing downstream nodes
        to know what type to expect.
        """
        return [self.output_type]

    @handler
    async def handle_input(self, data: Any, ctx: WorkflowContext[Any, Any]) -> None:
        """Handle input data and send the adapted output.

        This method implements the Executor interface via the handler decorator,
        wrapping the adapt() method with validation.

        Args:
            data: The input data (may be a sequence if multiple inputs)
            ctx: The workflow context
        """
        # Handle sequence inputs (adapters typically expect single values)
        value: Any
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            # Multiple inputs - use first if single, otherwise pass all
            seq: Sequence[Any] = data
            value = seq[0] if len(seq) == 1 else data
        else:
            value = data

        validated_input = self.validate_input(value)
        raw_output = await self.adapt(validated_input, ctx)
        validated_output = self.validate_output(raw_output)

        await ctx.send_message(validated_output)  # type: ignore[arg-type]


@dataclass
class FunctionAdapter(TypeAdapter[TInput, TOutput]):
    """Adapter that wraps a simple function for one-off transformations.

    Use this when you need a quick adapter without creating a full subclass.
    For reusable adapters, prefer creating a TypeAdapter subclass.

    The function can be synchronous or asynchronous:
    - Sync: `fn=lambda x, ctx: x.upper()`
    - Async: `fn=async def(x, ctx): await fetch(x)`

    Example:
        .. code-block:: python

            adapter = FunctionAdapter(
                input_type=str,
                output_type=int,
                fn=lambda s, ctx: int(s),
                name="str_to_int",
            )

    Attributes:
        fn: The transformation function (sync or async)
        input_type: The expected input type
        output_type: The produced output type
    """

    fn: Callable[[TInput, WorkflowContext], TOutput | Awaitable[TOutput]] = field(
        default=None,  # type: ignore[assignment]
        repr=False,
    )

    # Override the inherited class-level defaults with instance fields
    _input_type: type[TInput] = field(default=object, repr=False)  # type: ignore[assignment]
    _output_type: type[TOutput] = field(default=object, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize type fields from instance parameters."""
        if self.fn is None:
            raise ValueError("FunctionAdapter requires a transformation function 'fn'")
        # Set the class-level type attributes from instance fields
        object.__setattr__(self, "input_type", self._input_type)
        object.__setattr__(self, "output_type", self._output_type)
        # Call parent's __post_init__ to initialize Executor
        super().__post_init__()

    async def adapt(self, value: TInput, ctx: WorkflowContext) -> TOutput:
        """Apply the wrapped function to transform the value."""
        result = self.fn(value, ctx)
        if isinstance(result, Awaitable):
            return await result  # type: ignore[return-value]
        return result  # type: ignore[return-value]


# =============================================================================
# Built-in Adapters: String <-> Conversation
# =============================================================================


@dataclass
class TextToConversation(TypeAdapter[str, list[ChatMessage]]):
    """Convert a plain text string to a conversation (list of ChatMessage).

    This is one of the most common adapters needed when composing workflows.
    Many LLM-based workflows expect list[ChatMessage] input, but text processing
    workflows produce plain strings.

    Configuration
    -------------
    role: The role to assign to the created message (default: "user")
    author_name: Optional author name for the message

    Example:
        .. code-block:: python

            adapter = TextToConversation(role=Role.ASSISTANT)
            result = await adapter.adapt("Hello!", ctx)
            assert result == [ChatMessage(role=Role.ASSISTANT, text="Hello!")]
    """

    input_type: type[str] = field(default=str, init=False, repr=False)
    output_type: type[list[ChatMessage]] = field(default=list, init=False, repr=False)  # type: ignore[assignment]

    role: Role | str = field(default_factory=lambda: Role.USER)
    author_name: str | None = field(default=None)

    @property
    def output_types(self) -> list[type[Any]]:
        """Return the parameterized output type list[ChatMessage]."""
        return [list[ChatMessage]]  # type: ignore[list-item]

    async def adapt(self, value: str, ctx: WorkflowContext) -> list[ChatMessage]:
        """Convert the string to a single-message conversation."""
        role = self.role if isinstance(self.role, Role) else Role(self.role)
        return [ChatMessage(role=role, text=value, author_name=self.author_name)]


@dataclass
class ConversationToText(TypeAdapter[list[ChatMessage], str]):
    r"""Convert a conversation (list of ChatMessage) to a plain text string.

    This adapter extracts text from conversation messages, useful when
    a downstream workflow expects plain text input.

    Configuration:
        separator: String to join multiple messages (default: "\n\n")
        include_roles: If True, prefix each message with its role (default: False)
        last_only: If True, only extract the last message (default: False)

    Example:
        .. code-block:: python

            adapter = ConversationToText(last_only=True)
            messages = [
                ChatMessage(role=Role.USER, text="Hello"),
                ChatMessage(role=Role.ASSISTANT, text="Hi there!"),
            ]
            result = await adapter.adapt(messages, ctx)
            assert result == "Hi there!"
    """

    input_type: type[list[ChatMessage]] = field(default=list, init=False, repr=False)  # type: ignore[assignment]
    output_type: type[str] = field(default=str, init=False, repr=False)

    separator: str = field(default="\n\n")
    include_roles: bool = field(default=False)
    last_only: bool = field(default=False)

    @property
    def input_types(self) -> list[type[Any]]:
        """Return the parameterized input type list[ChatMessage]."""
        return [list[ChatMessage]]  # type: ignore[list-item]

    async def adapt(self, value: list[ChatMessage], ctx: WorkflowContext) -> str:
        """Extract text from conversation messages."""
        if not value:
            return ""

        if self.last_only:
            msg = value[-1]
            if self.include_roles:
                return f"{msg.role.value}: {msg.text or ''}"
            return msg.text or ""

        parts: list[str] = []
        for msg in value:
            text = msg.text or ""
            if self.include_roles:
                parts.append(f"{msg.role.value}: {text}")
            else:
                parts.append(text)

        return self.separator.join(parts)


# =============================================================================
# Built-in Adapters: Single <-> List
# =============================================================================


@dataclass
class SingleMessageExtractor(TypeAdapter[list[ChatMessage], ChatMessage]):
    """Extract a single message from a conversation.

    Useful when you need to pass a single ChatMessage to a downstream executor
    that doesn't accept lists.

    Configuration
    -------------
    index: Which message to extract (default: -1, meaning last message)
           Negative indices work like Python list indexing.

    Raises:
    ------
    IndexError: If the index is out of bounds

    Example:
        .. code-block:: python

            adapter = SingleMessageExtractor(index=0)  # Get first message
            messages = [ChatMessage(role=Role.USER, text="First")]
            result = await adapter.adapt(messages, ctx)
            assert result.text == "First"
    """

    input_type: type[list[ChatMessage]] = field(default=list, init=False, repr=False)  # type: ignore[assignment]
    output_type: type[ChatMessage] = field(default=ChatMessage, init=False, repr=False)

    index: int = field(default=-1)

    async def adapt(self, value: list[ChatMessage], ctx: WorkflowContext) -> ChatMessage:
        """Extract the message at the configured index."""
        if not value:
            raise IndexError(
                f"SingleMessageExtractor cannot extract message at index {self.index} from empty conversation"
            )
        try:
            return value[self.index]
        except IndexError as e:
            raise IndexError(
                f"SingleMessageExtractor index {self.index} out of range for conversation with {len(value)} messages"
            ) from e


@dataclass
class MessageWrapper(TypeAdapter[ChatMessage, list[ChatMessage]]):
    """Wrap a single ChatMessage in a list.

    The inverse of SingleMessageExtractor. Use when an upstream produces
    a single message but downstream expects a conversation.

    Example:
        .. code-block:: python

            adapter = MessageWrapper()
            msg = ChatMessage(role=Role.USER, text="Hello")
            result = await adapter.adapt(msg, ctx)
            assert result == [msg]
    """

    input_type: type[ChatMessage] = field(default=ChatMessage, init=False, repr=False)
    output_type: type[list[ChatMessage]] = field(default=list, init=False, repr=False)  # type: ignore[assignment]

    async def adapt(self, value: ChatMessage, ctx: WorkflowContext) -> list[ChatMessage]:
        """Wrap the message in a list."""
        return [value]


@dataclass
class ListToItem(TypeAdapter[list[TInput], TInput]):
    """Extract a single item from a list of any type.

    Generic version of SingleMessageExtractor that works with any list type.

    Configuration
    -------------
    index: Which item to extract (default: -1, meaning last item)

    Type Parameters
    ---------------
    TInput: The type of items in the list

    Example:
        .. code-block:: python

            adapter = ListToItem[str](index=0)
            result = await adapter.adapt(["a", "b", "c"], ctx)
            assert result == "a"
    """

    input_type: type[list[TInput]] = field(default=list, init=False, repr=False)  # type: ignore[assignment]
    output_type: type[TInput] = field(default=object, init=False, repr=False)  # type: ignore[assignment]

    index: int = field(default=-1)

    async def adapt(self, value: list[TInput], ctx: WorkflowContext) -> TInput:
        """Extract the item at the configured index."""
        if not value:
            raise IndexError(f"ListToItem cannot extract item at index {self.index} from empty list")
        try:
            return value[self.index]
        except IndexError as e:
            raise IndexError(f"ListToItem index {self.index} out of range for list with {len(value)} items") from e


@dataclass
class ItemToList(TypeAdapter[TInput, list[TInput]]):
    """Wrap a single item in a list.

    Generic version of MessageWrapper that works with any type.

    Type Parameters
    ---------------
    TInput: The type of the item to wrap

    Example:
        .. code-block:: python

            adapter = ItemToList[str]()
            result = await adapter.adapt("hello", ctx)
            assert result == ["hello"]
    """

    input_type: type[TInput] = field(default=object, init=False, repr=False)  # type: ignore[assignment]
    output_type: type[list[TInput]] = field(default=list, init=False, repr=False)  # type: ignore[assignment]

    async def adapt(self, value: TInput, ctx: WorkflowContext) -> list[TInput]:
        """Wrap the item in a list."""
        return [value]


# =============================================================================
# Built-in Adapters: Identity and Passthrough
# =============================================================================


@dataclass
class IdentityAdapter(TypeAdapter[TInput, TInput]):
    """Pass-through adapter that performs no transformation.

    Useful for:
    - Debugging: Insert into workflow to log values
    - Type narrowing: Validate type without changing value
    - Graph structure: Create explicit waypoints in complex graphs

    The adapter validates that input matches the expected type but
    returns the value unchanged.

    Example:
        .. code-block:: python

            adapter = IdentityAdapter(input_type=str, output_type=str, name="checkpoint")
            result = await adapter.adapt("unchanged", ctx)
            assert result == "unchanged"
    """

    async def adapt(self, value: TInput, ctx: WorkflowContext) -> TInput:
        """Return the value unchanged."""
        return value


# =============================================================================
# Factory Functions for Common Patterns
# =============================================================================


def json_serializer(
    input_type: type[TInput] = object,  # type: ignore[assignment]
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
) -> FunctionAdapter[TInput, str]:
    """Create an adapter that serializes objects to JSON strings.

    This factory creates a FunctionAdapter configured for JSON serialization.
    It handles dataclasses, dicts, and any object with a `to_dict()` method.

    Args:
        input_type: The type of objects to serialize (default: object)
        indent: JSON indentation level (default: None for compact)
        ensure_ascii: If True, escape non-ASCII characters (default: True)

    Returns:
        A FunctionAdapter that serializes to JSON

    Example:
        .. code-block:: python

            adapter = json_serializer(MyDataclass, indent=2)
            json_str = await adapter.adapt(my_obj, ctx)
    """
    import json
    from dataclasses import asdict, is_dataclass

    def serialize(value: TInput, ctx: WorkflowContext) -> str:
        data: Any
        if is_dataclass(value) and not isinstance(value, type):
            data = asdict(value)
        elif hasattr(value, "to_dict"):
            data = value.to_dict()  # type: ignore[union-attr]
        else:
            data = value

        return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

    return FunctionAdapter(
        fn=serialize,
        _input_type=input_type,
        _output_type=str,
        name=f"json_serializer<{input_type.__name__}>",
    )


def json_deserializer(
    output_type: type[TOutput],
    *,
    strict: bool = True,
) -> FunctionAdapter[str, TOutput]:
    """Create an adapter that deserializes JSON strings to objects.

    This factory creates a FunctionAdapter configured for JSON deserialization.
    For dataclasses, it attempts to construct the type from the parsed dict.

    Args:
        output_type: The type to deserialize into
        strict: If True, raise on missing required fields (default: True)

    Returns:
        A FunctionAdapter that deserializes from JSON

    Example:
        .. code-block:: python

            adapter = json_deserializer(MyDataclass)
            obj = await adapter.adapt('{"name": "test"}', ctx)
    """
    import json
    from dataclasses import fields, is_dataclass

    def deserialize(value: str, ctx: WorkflowContext) -> TOutput:
        data = json.loads(value)

        if is_dataclass(output_type):
            # Filter dict to only include valid fields
            valid_fields = {f.name for f in fields(output_type)}
            filtered = {k: v for k, v in data.items() if k in valid_fields}

            if strict:
                required_fields = {f.name for f in fields(output_type) if f.default is f.default_factory}
                missing = required_fields - filtered.keys()
                if missing:
                    raise ValueError(f"Missing required fields for {output_type.__name__}: {missing}")

            return output_type(**filtered)  # type: ignore[return-value]

        if isinstance(data, dict) and hasattr(output_type, "from_dict"):
            return output_type.from_dict(data)  # type: ignore[return-value,union-attr,no-any-return,attr-defined]

        return cast(TOutput, data)

    return FunctionAdapter(
        fn=deserialize,
        _input_type=str,
        _output_type=output_type,
        name=f"json_deserializer<{output_type.__name__}>",
    )


def struct_to_dict_adapter(
    input_type: type[TInput],
) -> FunctionAdapter[TInput, dict[str, Any]]:
    """Create an adapter that converts structured objects to dictionaries.

    Works with dataclasses, objects with `to_dict()`, and objects with `__dict__`.

    Args:
        input_type: The type of structured object to convert

    Returns:
        A FunctionAdapter that converts to dict

    Example:
        .. code-block:: python

            adapter = struct_to_dict_adapter(MyDataclass)
            d = await adapter.adapt(my_obj, ctx)
            assert isinstance(d, dict)
    """
    from dataclasses import asdict, is_dataclass

    def to_dict(value: TInput, ctx: WorkflowContext) -> dict[str, Any]:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if hasattr(value, "to_dict"):
            return value.to_dict()  # type: ignore[union-attr,return-value,no-any-return]
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)  # type: ignore[arg-type]
        raise TypeError(f"Cannot convert {type(value).__name__} to dict")

    return FunctionAdapter(
        fn=to_dict,
        _input_type=input_type,
        _output_type=dict,  # type: ignore[arg-type]
        name=f"struct_to_dict<{input_type.__name__}>",
    )


def dict_to_struct_adapter(
    output_type: type[TOutput],
    *,
    strict: bool = False,
) -> FunctionAdapter[dict[str, Any], TOutput]:
    """Create an adapter that converts dictionaries to structured objects.

    Args:
        output_type: The type to construct from the dict
        strict: If True, raise on extra keys not in the type (default: False)

    Returns:
        A FunctionAdapter that converts from dict

    Example:
        .. code-block:: python

            adapter = dict_to_struct_adapter(MyDataclass)
            obj = await adapter.adapt({"name": "test"}, ctx)
            assert isinstance(obj, MyDataclass)
    """
    from dataclasses import fields, is_dataclass

    def from_dict(value: dict[str, Any], ctx: WorkflowContext) -> TOutput:
        if is_dataclass(output_type):
            valid_fields = {f.name for f in fields(output_type)}
            if strict:
                extra = set(value.keys()) - valid_fields
                if extra:
                    raise ValueError(f"Unexpected fields for {output_type.__name__}: {extra}")
            filtered = {k: v for k, v in value.items() if k in valid_fields}
            return output_type(**filtered)  # type: ignore[return-value]

        if hasattr(output_type, "from_dict"):
            return output_type.from_dict(value)  # type: ignore[return-value,union-attr,no-any-return,attr-defined]

        return output_type(**value)  # type: ignore[return-value]

    return FunctionAdapter(
        fn=from_dict,
        _input_type=dict,  # type: ignore[arg-type]
        _output_type=output_type,
        name=f"dict_to_struct<{output_type.__name__}>",
    )


# =============================================================================
# Adapter Discovery and Registration
# =============================================================================


def find_adapter(
    source_type: type[Any],
    target_type: type[Any],
) -> TypeAdapter[Any, Any] | None:
    """Find a built-in adapter for the given type pair.

    This function searches the built-in adapters for one that can transform
    from source_type to target_type. Returns None if no suitable adapter exists.

    This is useful for automatic adapter insertion when connecting workflows
    with mismatched types.

    Args:
        source_type: The output type of the upstream executor
        target_type: The input type of the downstream executor

    Returns:
        A suitable TypeAdapter instance, or None if no built-in adapter matches

    Example:
        .. code-block:: python

            adapter = find_adapter(str, list[ChatMessage])
            assert isinstance(adapter, TextToConversation)
    """
    from typing import get_args, get_origin

    # str -> list[ChatMessage]
    if source_type is str:
        target_origin = get_origin(target_type)
        target_args = get_args(target_type)
        if target_origin is list and target_args and target_args[0] is ChatMessage:
            return TextToConversation()

    # list[ChatMessage] -> str or ChatMessage
    source_origin = get_origin(source_type)
    source_args = get_args(source_type)
    is_chat_list = source_origin is list and source_args and source_args[0] is ChatMessage
    if is_chat_list and target_type is str:
        return ConversationToText()
    if is_chat_list and target_type is ChatMessage:
        return SingleMessageExtractor()

    # ChatMessage -> list[ChatMessage]
    if source_type is ChatMessage:
        target_origin = get_origin(target_type)
        target_args = get_args(target_type)
        if target_origin is list and target_args and target_args[0] is ChatMessage:
            return MessageWrapper()

    return None
