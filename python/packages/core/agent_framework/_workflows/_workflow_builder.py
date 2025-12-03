# Copyright (c) Microsoft. All rights reserved.

import copy
import logging
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from .._agents import AgentProtocol
from ..observability import OtelAttr, capture_exception, create_workflow_span
from ._agent_executor import AgentExecutor
from ._checkpoint import CheckpointStorage
from ._const import DEFAULT_MAX_ITERATIONS, INTERNAL_SOURCE_ID
from ._edge import (
    Case,
    Default,
    EdgeGroup,
    FanInEdgeGroup,
    FanOutEdgeGroup,
    InternalEdgeGroup,
    SingleEdgeGroup,
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)
from ._executor import Executor
from ._runner_context import InProcRunnerContext
from ._typing_utils import is_type_compatible
from ._validation import validate_workflow_graph
from ._workflow import Workflow

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectionPoint:
    """Describes an executor endpoint with its output types.

    ConnectionPoint represents a named exit point from a workflow fragment,
    carrying full type information for downstream validation. Each point
    corresponds to an executor that produces outputs which can flow to
    subsequent stages.

    Type Information:
        - output_types: Types this executor sends via ctx.send_message()
        - workflow_output_types: Types this executor yields via ctx.yield_output()

    These are derived from the executor's handler signatures and used for
    compile-time validation when connecting fragments.

    Attributes:
        id: The executor ID (potentially prefixed during composition).
        name: Optional semantic name for easier reference in multi-output
              fragments (e.g., "summary", "errors", "analysis").
        output_types: List of types sent to downstream executors.
        workflow_output_types: List of types yielded as workflow outputs.
    """

    id: str
    output_types: list[type[Any]]
    workflow_output_types: list[type[Any]]
    name: str | None = None


class OutputPointsAccessor:
    """Provides both indexed and named access to connection output points.

    This accessor enables two access patterns for multi-output fragments:
        - Indexed: handle.outputs[0], handle.outputs[1]
        - Named: handle.outputs["summary"], handle.outputs["errors"]

    Named access requires that ConnectionPoints have their name attribute set.

    Example:
        .. code-block:: python

            handle = builder.add_workflow(multi_exit_fragment)
            # Access by index (always works)
            builder.connect(handle.outputs[0], next_stage)
            # Access by name (when points are named)
            builder.connect(handle.outputs["summary"], summary_stage)
    """

    def __init__(self, points: list[ConnectionPoint]) -> None:
        self._points = points
        self._by_name: dict[str, ConnectionPoint] = {}
        for p in points:
            if p.name:
                self._by_name[p.name] = p

    def __getitem__(self, key: int | str) -> ConnectionPoint:
        """Access output points by index or name.

        Args:
            key: Integer index or string name of the output point.

        Returns:
            The ConnectionPoint at the specified index or with the specified name.

        Raises:
            IndexError: If integer index is out of range.
            KeyError: If string name is not found.
            TypeError: If key is neither int nor str.
        """
        if isinstance(key, int):
            return self._points[key]
        if isinstance(key, str):
            if key not in self._by_name:
                available = list(self._by_name.keys()) if self._by_name else "none (outputs are unnamed)"
                raise KeyError(f"No output named '{key}'. Available names: {available}")
            return self._by_name[key]
        raise TypeError(f"Output key must be int or str, got {type(key).__name__}")

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[ConnectionPoint]:
        return iter(self._points)

    @property
    def names(self) -> list[str]:
        """Return list of available output names."""
        return list(self._by_name.keys())


class MergeResult:
    """Result of a merge() operation providing access to prefixed executor IDs.

    MergeResult eliminates the need to manually construct prefixed IDs after
    merging a fragment. It maps original executor IDs to their prefixed versions,
    supporting both attribute and dictionary access patterns.

    Access Patterns:
        - Attribute: result.ingest -> "prefix/ingest"
        - Dictionary: result["ingest"] -> "prefix/ingest"
        - Iteration: for original, prefixed in result.items()

    Example:
        .. code-block:: python

            # Merge a fragment and get the ID mapping
            ids = builder.merge(ingest_fragment, prefix="in")

            # Access prefixed IDs via attribute (if valid Python identifier)
            builder.add_edge(ids.ingest, ids.process)

            # Or via dictionary access (works for any ID)
            builder.add_edge(ids["ingest"], ids["process"])

            # List all mappings
            for original, prefixed in ids.items():
                print(f"{original} -> {prefixed}")
    """

    def __init__(self, mapping: dict[str, str], prefix: str) -> None:
        self._mapping = mapping
        self._prefix = prefix

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._mapping:
            available = list(self._mapping.keys())[:10]
            raise AttributeError(
                f"No executor with original id '{name}'. "
                f"Available: {available}{'...' if len(self._mapping) > 10 else ''}"
            )
        return self._mapping[name]

    def __getitem__(self, key: str) -> str:
        if key not in self._mapping:
            available = list(self._mapping.keys())[:10]
            raise KeyError(
                f"No executor with original id '{key}'. "
                f"Available: {available}{'...' if len(self._mapping) > 10 else ''}"
            )
        return self._mapping[key]

    def __contains__(self, key: str) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping.values())

    def __len__(self) -> int:
        return len(self._mapping)

    def items(self) -> Iterator[tuple[str, str]]:
        """Return iterator of (original_id, prefixed_id) pairs."""
        return iter(self._mapping.items())

    def keys(self) -> Iterator[str]:
        """Return original (unprefixed) executor IDs."""
        return iter(self._mapping.keys())

    def values(self) -> Iterator[str]:
        """Return prefixed executor IDs."""
        return iter(self._mapping.values())

    @property
    def prefix(self) -> str:
        """Return the prefix used for this merge."""
        return self._prefix

    def __repr__(self) -> str:
        return f"MergeResult(prefix={self._prefix!r}, ids={list(self._mapping.keys())})"


@dataclass(frozen=True)
class ConnectionHandle:
    """Reference to a merged fragment's entry/exit points with full type metadata.

    ConnectionHandle is the primary interface returned by add_workflow() and merge().
    It provides everything needed to wire the fragment into a larger workflow:

    Entry Point:
        - start: The executor ID that receives input to this fragment
        - start_input_types: Types the entry executor accepts

    Exit Points:
        - outputs: Accessor supporting both indexed and named access to exits
        - output_points: Raw list of ConnectionPoint objects

    Graph Metadata:
        - source_builder: Reference to the (cloned) builder for advanced access

    Design Philosophy:
        ConnectionHandle deliberately exposes only the "surface area" of a fragment -
        its entry and exit points. Internal executor IDs remain encapsulated,
        encouraging composition via the public interface rather than internal wiring.

    Example:
        .. code-block:: python

            # Add a fragment and get its handle
            analysis = builder.add_workflow(concurrent_analysis, prefix="analysis")

            # Wire using the handle's entry point
            builder.connect(data_source, analysis.start)

            # Wire using indexed outputs
            builder.connect(analysis.outputs[0], aggregator)

            # Or use named outputs if the fragment defines them
            builder.connect(analysis.outputs["summary"], report_generator)

    Attributes:
        start_id: Entry executor ID for incoming messages.
        start_input_types: Types accepted by the entry executor.
        output_points: List of exit points with type information.
        source_builder: The WorkflowBuilder backing this fragment (for advanced use).
    """

    start_id: str
    start_input_types: list[type[Any]]
    output_points: list[ConnectionPoint]
    source_builder: "WorkflowBuilder | None" = None

    @property
    def start(self) -> str:
        """Entry point executor ID for connecting upstream sources."""
        return self.start_id

    @property
    def outputs(self) -> OutputPointsAccessor:
        """Exit points supporting both indexed and named access.

        Returns:
            OutputPointsAccessor providing [index] and ["name"] access patterns.
        """
        return OutputPointsAccessor(self.output_points)


@dataclass
class WorkflowConnection:
    """Encapsulates a workflow fragment for composition into another builder.

    WorkflowConnection is an intermediate representation used during composition.
    It wraps a WorkflowBuilder along with metadata about its entry and exit points,
    enabling the composition machinery to correctly wire fragments together.

    When to Use WorkflowConnection Directly:
        Most users should NOT need to interact with WorkflowConnection directly.
        Instead, use add_workflow() which accepts builders, workflows, or connections:

        >>> # Preferred: pass builders directly to add_workflow()
        >>> handle = parent_builder.add_workflow(child_builder)

        WorkflowConnection is useful when:
        1. You need to pre-compute connection metadata for reuse
        2. You're building custom composition utilities
        3. You want explicit control over cloning and prefixing

    Relationship to WorkflowBuilder:
        WorkflowConnection is NOT a replacement for WorkflowBuilder. It's a
        thin wrapper that adds composition metadata. The actual graph structure
        lives in the wrapped builder. Think of it as a "view" of a builder
        prepared for composition.

    Immutability Contract:
        WorkflowConnection.clone() and .with_prefix() always return new instances
        with deep-copied builders. This ensures that:
        1. The original builder/connection is never mutated
        2. Multiple compositions of the same fragment remain independent
        3. Prefix operations are safely isolated

    Attributes:
        builder: The WorkflowBuilder holding the fragment's graph structure.
        entry: Executor ID that serves as the fragment's input entry point.
        exits: List of executor IDs that can connect to downstream stages.
        start_input_types: Types accepted by the entry executor.
        exit_points: Full ConnectionPoint metadata for each exit (types + names).
        output_names: Optional mapping from semantic names to exit executor IDs.

    Example:
        .. code-block:: python

            # Create a reusable connection from a builder
            data_pipeline = (
                WorkflowBuilder(name="pipeline")
                .add_chain([Ingest(id="ingest"), Transform(id="transform"), Load(id="load")])
                .set_start_executor("ingest")
            )
            conn = data_pipeline.as_connection()

            # The connection can be reused with different prefixes
            parent = WorkflowBuilder()
            pipeline_a = parent.add_workflow(conn, prefix="region_a")
            pipeline_b = parent.add_workflow(conn, prefix="region_b")
    """

    builder: "WorkflowBuilder"
    entry: str
    exits: list[str]
    start_input_types: list[type[Any]] | None = None
    exit_points: list[ConnectionPoint] | None = None
    output_names: dict[str, str] | None = None

    def clone(self) -> "WorkflowConnection":
        """Return a deep copy of this connection to avoid shared state."""
        builder_clone = _clone_builder_for_connection(self.builder)
        return WorkflowConnection(
            builder=builder_clone,
            entry=self.entry,
            exits=list(self.exits),
            start_input_types=list(self.start_input_types or []),
            exit_points=[
                ConnectionPoint(p.id, list(p.output_types), list(p.workflow_output_types), name=p.name)
                for p in self.exit_points or []
            ],
            output_names=dict(self.output_names) if self.output_names else None,
        )

    def with_prefix(self, prefix: str | None) -> "WorkflowConnection":
        """Return a prefixed copy to avoid executor ID collisions."""
        if not prefix:
            return self.clone()
        builder_clone = _clone_builder_for_connection(self.builder)
        mapping = _prefix_executor_ids(builder_clone, prefix)
        entry = mapping.get(self.entry, self.entry)
        exits = [mapping.get(e, e) for e in self.exits]
        # Remap output_names to prefixed IDs
        remapped_names: dict[str, str] | None = None
        if self.output_names:
            remapped_names = {name: mapping.get(eid, eid) for name, eid in self.output_names.items()}
        return WorkflowConnection(
            builder=builder_clone,
            entry=entry,
            exits=exits,
            start_input_types=list(self.start_input_types or []),
            exit_points=[
                ConnectionPoint(
                    id=mapping.get(p.id, p.id),
                    output_types=list(p.output_types),
                    workflow_output_types=list(p.workflow_output_types),
                    name=p.name,
                )
                for p in self.exit_points or []
            ],
            output_names=remapped_names,
        )


def _clone_builder_for_connection(builder: "WorkflowBuilder") -> "WorkflowBuilder":
    """Deep copy a builder so connections remain immutable when merged."""
    clone = WorkflowBuilder(
        max_iterations=builder._max_iterations,
        name=builder._name,
        description=builder._description,
    )
    clone._checkpoint_storage = builder._checkpoint_storage
    clone._edge_groups = copy.deepcopy(builder._edge_groups)
    clone._executors = {eid: copy.deepcopy(executor) for eid, executor in builder._executors.items()}
    if builder._start_executor is None:
        clone._start_executor = None
    elif isinstance(builder._start_executor, str):
        clone._start_executor = builder._start_executor
    else:
        clone._start_executor = builder._start_executor.id
    clone._agent_wrappers = {}
    return clone


def _get_executor_input_types(executors: dict[str, Executor], executor_id: str) -> list[type[Any]]:
    """Return input types for a given executor id."""
    exec_obj = executors.get(executor_id)
    if exec_obj is None:
        raise ValueError(f"Unknown executor id '{executor_id}' when deriving connection types.")
    return list(exec_obj.input_types)


def _prefix_executor_ids(builder: "WorkflowBuilder", prefix: str) -> dict[str, str]:
    """Apply a deterministic prefix to executor ids and update edge references."""
    mapping: dict[str, str] = {}
    for executor_id in builder._executors:
        mapping[executor_id] = f"{prefix}/{executor_id}"

    # Update executors and remap keys
    updated_executors: dict[str, Executor] = {}
    for original_id, executor in builder._executors.items():
        prefixed_id = mapping[original_id]
        executor.id = prefixed_id
        updated_executors[prefixed_id] = executor
    builder._executors = updated_executors

    # Update start executor reference
    if builder._start_executor is None:
        pass  # Keep as None
    elif isinstance(builder._start_executor, Executor):
        start_id = builder._start_executor.id
        builder._start_executor = mapping.get(start_id, start_id)
    else:
        builder._start_executor = mapping.get(builder._start_executor, builder._start_executor)

    builder._edge_groups = [_remap_edge_group_ids(group, mapping, prefix) for group in builder._edge_groups]

    return mapping


def _remap_edge_group_ids(group: EdgeGroup, mapping: dict[str, str], prefix: str) -> EdgeGroup:
    """Remap executor ids inside an edge group."""
    remapped = copy.deepcopy(group)
    remapped.id = f"{prefix}/{group.id}"

    for edge in remapped.edges:
        # Adjust internal sources before generic mapping
        for original, new in mapping.items():
            internal_source = INTERNAL_SOURCE_ID(original)
            if edge.source_id == internal_source:
                edge.source_id = INTERNAL_SOURCE_ID(new)
        if edge.source_id in mapping:
            edge.source_id = mapping[edge.source_id]
        if edge.target_id in mapping:
            edge.target_id = mapping[edge.target_id]

    if isinstance(remapped, FanOutEdgeGroup):
        remapped._target_ids = [mapping.get(t, t) for t in remapped._target_ids]  # type: ignore[attr-defined]

    if isinstance(remapped, SwitchCaseEdgeGroup):
        new_targets: list[str] = []
        for idx, target in enumerate(remapped._target_ids):  # type: ignore[attr-defined]
            remapped._target_ids[idx] = mapping.get(target, target)  # type: ignore[attr-defined]
            new_targets.append(remapped._target_ids[idx])  # type: ignore[attr-defined]
        remapped.cases = [  # type: ignore[attr-defined]
            _remap_switch_case(case, mapping)
            for case in remapped.cases  # type: ignore[attr-defined]
        ]
    return remapped


def _remap_switch_case(
    case: SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault,
    mapping: dict[str, str],
) -> SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault:
    """Remap switch-case targets to prefixed ids."""
    case.target_id = mapping.get(case.target_id, case.target_id)
    return case


def _derive_exit_ids(edge_groups: list[EdgeGroup], executors: dict[str, Executor]) -> list[str]:
    """Infer exit executors as those without downstream edges or only targeting terminal nodes."""
    outgoing: dict[str, list[str]] = {}
    for group in edge_groups:
        for edge in group.edges:
            if edge.source_id not in executors:
                continue
            outgoing.setdefault(edge.source_id, []).append(edge.target_id)

    exits: list[str] = []
    for executor_id, _ in executors.items():
        targets = outgoing.get(executor_id, [])
        if not targets:
            exits.append(executor_id)
            continue
        target_are_terminal = True
        for target in targets:
            target_exec = executors.get(target)
            if target_exec is None:
                target_are_terminal = False
                break
            if not target_exec.workflow_output_types:
                target_are_terminal = False
                break
        if target_are_terminal:
            exits.append(executor_id)

    return exits


def _derive_exit_points(edge_groups: list[EdgeGroup], executors: dict[str, Executor]) -> list[ConnectionPoint]:
    """Return connection points (id + output types) for exit executors."""
    exit_ids = _derive_exit_ids(edge_groups, executors)
    points: list[ConnectionPoint] = []
    for executor_id in exit_ids:
        exec_obj = executors.get(executor_id)
        if exec_obj is None:
            continue
        points.append(
            ConnectionPoint(
                id=executor_id,
                output_types=list(exec_obj.output_types),
                workflow_output_types=list(exec_obj.workflow_output_types),
            )
        )
    return points


# Type alias for workflow composition endpoints
Endpoint: TypeAlias = "Executor | AgentProtocol | ConnectionHandle | ConnectionPoint | str"


class WorkflowBuilder:
    """A builder class for constructing workflows.

    This class provides a fluent API for defining workflow graphs by connecting executors
    with edges and configuring execution parameters. Call :meth:`build` to create an
    immutable :class:`Workflow` instance.

    Example:
        .. code-block:: python

            from typing_extensions import Never
            from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


            class UpperCaseExecutor(Executor):
                @handler
                async def process(self, text: str, ctx: WorkflowContext[str]) -> None:
                    await ctx.send_message(text.upper())


            class ReverseExecutor(Executor):
                @handler
                async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
                    await ctx.yield_output(text[::-1])


            # Build a workflow
            workflow = (
                WorkflowBuilder()
                .add_edge(UpperCaseExecutor(id="upper"), ReverseExecutor(id="reverse"))
                .set_start_executor("upper")
                .build()
            )

            # Run the workflow
            events = await workflow.run("hello")
            print(events.get_outputs())  # ['OLLEH']
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        name: str | None = None,
        description: str | None = None,
    ):
        """Initialize the WorkflowBuilder with an empty list of edges and no starting executor.

        Args:
            max_iterations: Maximum number of iterations for workflow convergence. Default is 100.
            name: Optional human-readable name for the workflow.
            description: Optional description of what the workflow does.
        """
        self._edge_groups: list[EdgeGroup] = []
        self._executors: dict[str, Executor] = {}
        self._start_executor: Executor | str | None = None
        self._checkpoint_storage: CheckpointStorage | None = None
        self._max_iterations: int = max_iterations
        self._name: str | None = name
        self._description: str | None = description
        self._fragment_counter: int = 0
        # Maps underlying AgentProtocol object id -> wrapped Executor so we reuse the same wrapper
        # across set_start_executor / add_edge calls. Without this, unnamed agents (which receive
        # random UUID based executor ids) end up wrapped multiple times, giving different ids for
        # the start node vs edge nodes and triggering a GraphConnectivityError during validation.
        self._agent_wrappers: dict[int, Executor] = {}

    # Agents auto-wrapped by builder now always stream incremental updates.

    def as_connection(self, prefix: str | None = None) -> WorkflowConnection:
        """Render this builder as a reusable connection without finalising into a Workflow."""
        if not self._start_executor:
            raise ValueError("Starting executor must be set before calling as_connection().")

        # Validate to ensure we don't emit malformed fragments
        validate_workflow_graph(
            self._edge_groups,
            self._executors,
            self._start_executor,
        )

        clone = _clone_builder_for_connection(self)
        start_exec = clone._start_executor
        if start_exec is None:
            raise ValueError("Starting executor must be set before calling as_connection().")
        entry_id: str = start_exec.id if isinstance(start_exec, Executor) else start_exec
        entry_types = _get_executor_input_types(clone._executors, entry_id)
        exit_points = _derive_exit_points(clone._edge_groups, clone._executors)
        exit_ids = [p.id for p in exit_points]
        connection = WorkflowConnection(
            builder=clone,
            entry=entry_id,
            start_input_types=entry_types,
            exit_points=exit_points,
            exits=exit_ids,
        )
        return connection.with_prefix(prefix) if prefix else connection

    # =========================================================================
    # COMPOSITION APIs - Graph merging and connection
    # =========================================================================

    def merge(
        self,
        other: "WorkflowBuilder | Workflow | WorkflowConnection",
        *,
        prefix: str | None = None,
    ) -> MergeResult:
        """Merge another builder's graph into this one, returning an ID mapping.

        This is the simplest composition primitive. It copies all executors and edges
        from the source into this builder, applying a prefix to avoid ID collisions.
        The returned MergeResult maps original executor IDs to their prefixed versions,
        eliminating the need to manually construct prefixed IDs.

        Unlike add_workflow(), merge() does NOT automatically determine entry/exit
        points - it gives you full control over the resulting graph topology.

        When to Use merge() vs add_workflow():
            - Use merge() when you need low-level control and know the internal
              structure of the builder you're merging
            - Use add_workflow() when you want to treat the fragment as a black box
              with well-defined entry and exit points

        ID Collision Handling:
            If prefix is None, the method attempts to derive a prefix from the
            fragment's name. If no name exists, a collision will raise ValueError.
            Always provide an explicit prefix when merging unnamed builders.

        Args:
            other: A WorkflowBuilder, Workflow, or WorkflowConnection to merge.
            prefix: Prefix applied to all executor IDs from the merged graph.
                    If None, attempts to use the fragment's name property.

        Returns:
            MergeResult mapping original IDs to prefixed IDs. Supports both
            attribute access (result.executor_name) and dictionary access
            (result["executor-name"]).

        Raises:
            ValueError: If executor ID collision occurs without a prefix.

        Example:
            .. code-block:: python

                # Create two separate workflow builders
                data_prep = (
                    WorkflowBuilder(name="prep")
                    .add_chain([Ingest(id="ingest"), Clean(id="clean")])
                    .set_start_executor("ingest")
                )
                analysis = (
                    WorkflowBuilder(name="analysis")
                    .add_edge(Analyze(id="analyze"), Report(id="report"))
                    .set_start_executor("analyze")
                )

                # Merge and wire using the returned ID mapping
                builder = WorkflowBuilder()
                prep = builder.merge(data_prep, prefix="prep")
                analysis = builder.merge(analysis_builder, prefix="analysis")

                # Access prefixed IDs via attribute or dictionary
                builder.add_edge(prep.clean, analysis.analyze)
                # Or: builder.add_edge(prep["clean"], analysis["analyze"])
                builder.set_start_executor(prep.ingest)
        """
        effective_prefix = self._derive_prefix(other, prefix)

        # Capture original IDs before prefixing
        if isinstance(other, WorkflowConnection):
            original_ids = list(other.builder._executors.keys())
        elif isinstance(other, WorkflowBuilder):
            original_ids = list(other._executors.keys())
        elif isinstance(other, Workflow):
            original_ids = list(other.executors.keys())
        else:
            original_ids = []

        connection = self._to_connection(other, prefix=effective_prefix)
        prepared = connection.clone()  # Already prefixed by _to_connection

        # Detect collisions before mutating state
        for executor_id in prepared.builder._executors:
            if executor_id in self._executors:
                raise ValueError(
                    f"Executor id '{executor_id}' already exists in builder. Provide a different prefix when merging."
                )

        # Merge executor map and edge groups
        self._executors.update(prepared.builder._executors)
        self._edge_groups.extend(prepared.builder._edge_groups)

        # Build mapping from original IDs to prefixed IDs
        mapping = {orig: f"{effective_prefix}/{orig}" for orig in original_ids}
        return MergeResult(mapping, effective_prefix)

    def add_workflow(
        self, fragment: "WorkflowBuilder | Workflow | WorkflowConnection", *, prefix: str | None = None
    ) -> ConnectionHandle:
        """Merge a workflow fragment and return a handle for wiring.

        This is the primary composition API. It merges the fragment's graph into
        this builder and returns a ConnectionHandle with type-safe entry and exit
        points. You can then use connect() to wire the fragment to other executors.

        The method accepts any composable type:
            - WorkflowBuilder: Unbuilt builder (recommended for composition)
            - Workflow: Built, immutable workflow (cloned during merge)
            - WorkflowConnection: Pre-computed connection metadata

        No Need for as_connection():
            add_workflow() internally calls as_connection() when needed, so you
            rarely need to call it yourself. Simply pass your builder directly:

            >>> # Preferred - pass builder directly
            >>> handle = parent.add_workflow(child_builder)
            >>>
            >>> # Equivalent but more verbose
            >>> handle = parent.add_workflow(child_builder.as_connection())

        Prefix Derivation:
            If prefix is None, it's derived automatically:
            1. From the fragment's name property if set
            2. From the fragment's class name if a custom subclass
            3. From a counter-based fallback ("fragment-1", "fragment-2", etc.)

        Args:
            fragment: The workflow fragment to merge.
            prefix: Explicit prefix for executor IDs. If None, derived from
                    the fragment's name or class.

        Returns:
            ConnectionHandle with .start (entry point) and .outputs (exit points).

        Example:
            .. code-block:: python

                # Create fragments
                concurrent = ConcurrentBuilder().participants([analyzer_a, analyzer_b])

                # Compose into a larger workflow
                builder = WorkflowBuilder()
                analysis = builder.add_workflow(concurrent, prefix="analysis")

                # Wire using the handle
                builder.connect(data_source, analysis.start)
                builder.connect(analysis.outputs[0], aggregator)
                builder.set_start_executor(data_source)
        """
        effective_prefix = self._derive_prefix(fragment, prefix)
        connection = self._to_connection(fragment, prefix=effective_prefix)
        return self._merge_connection(connection, prefix=None)

    def add_connection(self, connection: WorkflowConnection, *, prefix: str | None = None) -> ConnectionHandle:
        """Merge a WorkflowConnection and return a handle for wiring.

        This is a lower-level API that accepts a pre-computed WorkflowConnection.
        Most users should prefer add_workflow() which accepts any composable type.

        Args:
            connection: A WorkflowConnection with pre-computed entry/exit metadata.
            prefix: Optional additional prefix. Note that if the connection was
                    already created with a prefix, this adds another layer.

        Returns:
            ConnectionHandle for wiring the merged fragment.
        """
        return self._merge_connection(connection, prefix=prefix)

    def connect(self, source: Endpoint, target: Endpoint, /) -> Self:
        """Connect two endpoints with a directed edge.

        This is the composition wiring primitive. It creates an edge from source
        to target, where both can be:
            - Executor instances (added to graph if not present)
            - Agent instances (auto-wrapped in AgentExecutor)
            - ConnectionHandle (uses .start_id as the endpoint)
            - ConnectionPoint (uses .id as the endpoint)
            - String executor IDs (must already exist in graph)

        Type Validation:
            Currently, type compatibility is validated at build() time rather than
            connect() time. Future versions may add eager type checking.

        Args:
            source: The source endpoint (output producer).
            target: The target endpoint (input consumer).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If a string ID is used but not found in the executor map.
            TypeError: If an endpoint type is not supported.

        Example:
            .. code-block:: python

                # Connect executors directly
                builder.connect(producer, consumer)

                # Connect using handles from add_workflow()
                fragment = builder.add_workflow(some_builder)
                builder.connect(data_source, fragment.start)
                builder.connect(fragment.outputs[0], sink)

                # Connect by ID (after merge)
                builder.connect("prep/clean", "analysis/analyze")
        """
        src_id = self._normalize_endpoint(source)
        tgt_id = self._normalize_endpoint(target)
        self._edge_groups.append(SingleEdgeGroup(src_id, tgt_id))  # type: ignore[arg-type]
        return self

    def connect_checked(
        self,
        source: "Endpoint",
        target: "Endpoint",
        /,
        *,
        adapter: "Executor | None" = None,
    ) -> Self:
        """Connect two endpoints with type validation, optionally inserting an adapter.

        This is the type-safe variant of connect(). It validates that the source's
        output types are compatible with the target's input types BEFORE creating
        the edge. If types are incompatible and no adapter is provided, it raises
        a TypeError with guidance on which adapter to use.

        Type Validation:
            The method checks if ANY source output type is compatible with ANY
            target input type. This is permissive by design - the runtime will
            route messages based on actual types.

        Automatic Adapter Suggestions:
            When types are incompatible, the error message suggests built-in
            adapters that could bridge the gap (e.g., TextToConversation for
            str -> list[ChatMessage]).

        Args:
            source: The source endpoint (output producer).
            target: The target endpoint (input consumer).
            adapter: Optional adapter executor to insert between source and target.
                     When provided, the connection becomes: source -> adapter -> target.

        Returns:
            Self for method chaining.

        Raises:
            TypeError: If types are incompatible and no adapter is provided.
            ValueError: If a string ID is not found in the executor map.

        Example:
            .. code-block:: python

                # Direct connection with type checking
                builder.connect_checked(producer, consumer)

                # Insert an adapter for type conversion
                from agent_framework._workflows._type_adapters import TextToConversation

                builder.connect_checked(
                    text_producer,
                    chat_consumer,
                    adapter=TextToConversation(),
                )

                # Error with guidance
                builder.connect_checked(str_producer, chat_consumer)
                # TypeError: Type mismatch: 'str_producer' outputs [str] but 'chat_consumer'
                #            expects [list[ChatMessage]]. Consider inserting a TypeAdapter.
                #            Suggested: TextToConversation()
        """
        src_id = self._normalize_endpoint(source)
        tgt_id = self._normalize_endpoint(target)

        if adapter is not None:
            # With adapter: source -> adapter -> target
            adapter_id = self._normalize_endpoint(adapter)

            # Validate source -> adapter
            self.validate_edge_types(src_id, adapter_id, raise_on_mismatch=True)

            # Validate adapter -> target
            self.validate_edge_types(adapter_id, tgt_id, raise_on_mismatch=True)

            # Create the two edges
            self._edge_groups.append(SingleEdgeGroup(src_id, adapter_id))  # type: ignore[arg-type]
            self._edge_groups.append(SingleEdgeGroup(adapter_id, tgt_id))  # type: ignore[arg-type]
        else:
            # Direct connection: validate and create single edge
            is_compatible, error_msg = self.validate_edge_types(src_id, tgt_id, raise_on_mismatch=False)
            if not is_compatible:
                # Try to suggest an adapter
                suggestion = self._suggest_adapter(src_id, tgt_id)
                if suggestion:
                    error_msg = f"{error_msg}\nSuggested: {suggestion}"
                raise TypeError(error_msg)

            self._edge_groups.append(SingleEdgeGroup(src_id, tgt_id))  # type: ignore[arg-type]

        return self

    def _suggest_adapter(self, source_id: str, target_id: str) -> str | None:
        """Suggest a built-in adapter for type mismatch."""
        from ._type_adapters import find_adapter

        source_exec = self._executors.get(source_id)
        target_exec = self._executors.get(target_id)
        if not source_exec or not target_exec:
            return None

        # Check each output/input pair for a matching adapter
        for src_type in source_exec.output_types:
            for tgt_type in target_exec.input_types:
                adapter = find_adapter(src_type, tgt_type)
                if adapter:
                    return f"{adapter.__class__.__name__}()"
        return None

    def get_executor(self, executor_id: str) -> Executor:
        """Retrieve an executor by ID from this builder.

        This is useful after merge() when you need type-safe access to executors
        from merged fragments for wiring with add_edge().

        Args:
            executor_id: The executor ID (potentially prefixed after merge).

        Returns:
            The Executor instance with the given ID.

        Raises:
            KeyError: If no executor with the given ID exists.

        Example:
            .. code-block:: python

                builder = WorkflowBuilder()
                builder.merge(fragment_a, prefix="a")
                builder.merge(fragment_b, prefix="b")

                # Access merged executors for wiring
                builder.add_edge(builder.get_executor("a/output"), builder.get_executor("b/input"))
        """
        if executor_id not in self._executors:
            available = list(self._executors.keys())
            raise KeyError(
                f"No executor with id '{executor_id}'. "
                f"Available executors: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
        return self._executors[executor_id]

    def get_executors(self) -> dict[str, Executor]:
        """Return a copy of the executor map.

        Returns:
            Dictionary mapping executor IDs to Executor instances.
        """
        return dict(self._executors)

    def validate_edge_types(
        self,
        source_id: str,
        target_id: str,
        *,
        raise_on_mismatch: bool = True,
    ) -> tuple[bool, str | None]:
        """Validate type compatibility between two executors.

        This method checks whether the output types of the source executor
        are compatible with the input types of the target executor using
        the is_type_compatible function from _typing_utils.

        Type Compatibility Rules:
            - Exact match: int -> int (compatible)
            - Subtype: ChildClass -> ParentClass (compatible)
            - Union member: str -> str | int (compatible)
            - List covariance: list[ChildClass] -> list[ParentClass] (compatible)

        Note:
            This performs eager (connect-time) validation. The workflow graph
            validation at build() time performs more comprehensive checks.

        Args:
            source_id: The ID of the source executor.
            target_id: The ID of the target executor.
            raise_on_mismatch: If True, raise TypeError on incompatibility.
                               If False, return (False, error_message) instead.

        Returns:
            A tuple of (is_compatible, error_message). error_message is None
            if compatible.

        Raises:
            TypeError: If raise_on_mismatch is True and types are incompatible.
            KeyError: If either executor ID is not found.

        Example:
            .. code-block:: python

                builder.validate_edge_types("text_producer", "chat_consumer")
                # Returns: (False, "Type mismatch: text_producer outputs [str] ...")

                # With raise_on_mismatch
                builder.validate_edge_types("producer", "consumer", raise_on_mismatch=True)
                # Raises: TypeError: Type mismatch: ...
        """
        if source_id not in self._executors:
            raise KeyError(f"Source executor '{source_id}' not found in builder.")
        if target_id not in self._executors:
            raise KeyError(f"Target executor '{target_id}' not found in builder.")

        source_exec = self._executors[source_id]
        target_exec = self._executors[target_id]

        source_outputs = source_exec.output_types
        target_inputs = target_exec.input_types

        # Check if any source output type is compatible with any target input type
        # This is a permissive check - at least one path must exist
        for src_type in source_outputs:
            for tgt_type in target_inputs:
                if is_type_compatible(src_type, tgt_type):
                    return (True, None)

        # No compatible path found - build error message
        src_type_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in source_outputs]
        tgt_type_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in target_inputs]

        error_msg = (
            f"Type mismatch: '{source_id}' outputs {src_type_names} "
            f"but '{target_id}' expects {tgt_type_names}. "
            f"Consider inserting a TypeAdapter to bridge these types."
        )

        if raise_on_mismatch:
            raise TypeError(error_msg)

        return (False, error_msg)

    @property
    def name(self) -> str | None:
        """The name of this builder, used for prefix derivation."""
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the name of this builder."""
        self._name = value

    @property
    def executor_ids(self) -> list[str]:
        """List of all executor IDs currently in this builder."""
        return list(self._executors.keys())

    def _merge_connection(self, fragment: WorkflowConnection, *, prefix: str | None) -> ConnectionHandle:
        """Merge a connection into this builder, returning a handle to its connection points."""
        prepared = fragment.with_prefix(prefix) if prefix else fragment.clone()

        # Detect collisions before mutating state
        for executor_id in prepared.builder._executors:
            if executor_id in self._executors:
                raise ValueError(
                    f"Executor id '{executor_id}' already exists in builder. "
                    "Provide a prefix when connecting the fragment."
                )

        # Merge executor map and edge groups
        self._executors.update(prepared.builder._executors)
        self._edge_groups.extend(prepared.builder._edge_groups)

        start_executor = prepared.builder._start_executor
        if start_executor is None:
            raise ValueError("Merged fragment must have a start executor set.")
        start_id: str = (
            start_executor.id
            if isinstance(start_executor, Executor)
            else start_executor
        )
        start_types = prepared.start_input_types or _get_executor_input_types(prepared.builder._executors, start_id)
        exit_points = prepared.exit_points or _derive_exit_points(
            prepared.builder._edge_groups, prepared.builder._executors
        )
        return ConnectionHandle(
            start_id=start_id,
            start_input_types=start_types,
            output_points=exit_points,
            source_builder=prepared.builder,
        )

    def _derive_prefix(self, fragment: "WorkflowBuilder | Workflow | WorkflowConnection", explicit: str | None) -> str:
        """Choose a stable prefix from explicit input, fragment name, or a deterministic fallback."""
        if explicit:
            return explicit

        name: str | None = None
        if isinstance(fragment, WorkflowConnection):
            name = fragment.builder._name  # Accessing private name to avoid duplicating state
        elif isinstance(fragment, WorkflowBuilder):
            name = fragment._name
        elif isinstance(fragment, Workflow):
            name = fragment.name

        if name:
            return name

        class_name = type(fragment).__name__
        if class_name not in {"WorkflowBuilder", "Workflow", "WorkflowConnection"}:
            return class_name

        # Fall back to a deterministic suffix when no name exists or is too generic.
        self._fragment_counter += 1
        return f"fragment-{self._fragment_counter}"

    def _to_connection(
        self, fragment: "WorkflowBuilder | Workflow | WorkflowConnection", *, prefix: str
    ) -> WorkflowConnection:
        """Normalize a fragment to a WorkflowConnection, applying a prefix for collision safety."""
        if isinstance(fragment, WorkflowConnection):
            return fragment.with_prefix(prefix)
        if isinstance(fragment, WorkflowBuilder):
            return fragment.as_connection(prefix=prefix)
        if isinstance(fragment, Workflow):
            return fragment.as_connection(prefix=prefix)
        raise TypeError(
            f"add_workflow expects a WorkflowBuilder, Workflow, or WorkflowConnection; got {type(fragment).__name__}."
        )

    def _normalize_endpoint(self, endpoint: Executor | AgentProtocol | ConnectionHandle | ConnectionPoint | str) -> str:
        """Resolve a connect endpoint to an executor id, adding executors when provided."""
        if isinstance(endpoint, ConnectionHandle):
            return endpoint.start_id
        if isinstance(endpoint, ConnectionPoint):
            return endpoint.id
        if isinstance(endpoint, str):
            if endpoint not in self._executors:
                raise ValueError(f"Unknown executor id '{endpoint}' in connect().")
            return endpoint
        executor = self._maybe_wrap_agent(endpoint)  # type: ignore[arg-type]
        executor_id = executor.id
        if executor_id not in self._executors:
            self._add_executor(executor)
        return executor_id

    def _add_executor(self, executor: Executor) -> str:
        """Add an executor to the map and return its ID."""
        existing = self._executors.get(executor.id)
        if existing is not None:
            if existing is executor:
                # Already added
                return executor.id
            # ID conflict
            raise ValueError(f"Duplicate executor ID '{executor.id}' detected in workflow.")

        # New executor
        self._executors[executor.id] = executor
        # Add an internal edge group for each unique executor
        self._edge_groups.append(InternalEdgeGroup(executor.id))  # type: ignore[call-arg]

        return executor.id

    def _maybe_wrap_agent(
        self,
        candidate: Executor | AgentProtocol,
        agent_thread: Any | None = None,
        output_response: bool = False,
        executor_id: str | None = None,
    ) -> Executor:
        """If the provided object implements AgentProtocol, wrap it in an AgentExecutor.

        This allows fluent builder APIs to directly accept agents instead of
        requiring callers to manually instantiate AgentExecutor.

        Args:
            candidate: The executor or agent to wrap.
            agent_thread: The thread to use for running the agent. If None, a new thread will be created.
            output_response: Whether to yield an AgentRunResponse as a workflow output when the agent completes.
            executor_id: A unique identifier for the executor. If None, the agent's name will be used if available.
        """
        try:  # Local import to avoid hard dependency at import time
            from agent_framework import AgentProtocol  # type: ignore
        except Exception:  # pragma: no cover - defensive
            AgentProtocol = object  # type: ignore

        if isinstance(candidate, Executor):  # Already an executor
            return candidate
        if isinstance(candidate, AgentProtocol):  # type: ignore[arg-type]
            # Reuse existing wrapper for the same agent instance if present
            agent_instance_id = id(candidate)
            existing = self._agent_wrappers.get(agent_instance_id)
            if existing is not None:
                return existing
            # Use agent name if available and unique among current executors
            name = getattr(candidate, "name", None)
            proposed_id: str | None = executor_id
            if proposed_id is None and name:
                proposed_id = str(name)
                if proposed_id in self._executors:
                    raise ValueError(
                        f"Duplicate executor ID '{proposed_id}' from agent name. "
                        "Agent names must be unique within a workflow."
                    )
            wrapper = AgentExecutor(
                candidate,
                agent_thread=agent_thread,
                output_response=output_response,
                id=proposed_id,
            )
            self._agent_wrappers[agent_instance_id] = wrapper
            return wrapper
        raise TypeError(
            f"WorkflowBuilder expected an Executor or AgentProtocol instance; got {type(candidate).__name__}."
        )

    def add_agent(
        self,
        agent: AgentProtocol,
        agent_thread: Any | None = None,
        output_response: bool = False,
        id: str | None = None,
    ) -> Self:
        """Add an agent to the workflow by wrapping it in an AgentExecutor.

        This method creates an AgentExecutor that wraps the agent with the given parameters
        and ensures that subsequent uses of the same agent instance in other builder methods
        (like add_edge, set_start_executor, etc.) will reuse the same wrapped executor.

        Note: Agents adapt their behavior based on how the workflow is executed:
        - run_stream(): Agents emit incremental AgentRunUpdateEvent events as tokens are produced
        - run(): Agents emit a single AgentRunEvent containing the complete response

        Args:
            agent: The agent to add to the workflow.
            agent_thread: The thread to use for running the agent. If None, a new thread will be created.
            output_response: Whether to yield an AgentRunResponse as a workflow output when the agent completes.
            id: A unique identifier for the executor. If None, the agent's name will be used if available.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Raises:
            ValueError: If the provided id or agent name conflicts with an existing executor.

        Example:
            .. code-block:: python

                from agent_framework import WorkflowBuilder
                from agent_framework_anthropic import AnthropicAgent

                # Create an agent
                agent = AnthropicAgent(name="writer", model="claude-3-5-sonnet-20241022")

                # Add the agent to a workflow
                workflow = WorkflowBuilder().add_agent(agent, output_response=True).set_start_executor(agent).build()
        """
        executor = self._maybe_wrap_agent(
            agent, agent_thread=agent_thread, output_response=output_response, executor_id=id
        )
        self._add_executor(executor)
        return self

    def add_edge(
        self,
        source: Endpoint,
        target: Endpoint,
        condition: Callable[[Any], bool] | None = None,
    ) -> Self:
        """Add a directed edge between two executors.

        The output types of the source and the input types of the target must be compatible.
        Messages sent by the source executor will be routed to the target executor.

        Supported endpoint types:
            - Executor instances (added to graph if not present)
            - Agent instances (auto-wrapped in AgentExecutor)
            - ConnectionHandle (uses .start_id as the endpoint)
            - ConnectionPoint (uses .id as the endpoint)
            - String executor IDs (must already exist in graph)

        Args:
            source: The source executor or endpoint of the edge.
            target: The target executor or endpoint of the edge.
            condition: An optional condition function that determines whether the edge
                       should be traversed based on the message type.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from typing_extensions import Never
                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


                class ProcessorA(Executor):
                    @handler
                    async def process(self, data: str, ctx: WorkflowContext[int]) -> None:
                        await ctx.send_message(len(data))


                class ProcessorB(Executor):
                    @handler
                    async def process(self, count: int, ctx: WorkflowContext[Never, str]) -> None:
                        await ctx.yield_output(f"Processed {count} characters")


                # Connect executors with an edge
                workflow = (
                    WorkflowBuilder().add_edge(ProcessorA(id="a"), ProcessorB(id="b")).set_start_executor("a").build()
                )


                # With a condition
                def only_large_numbers(msg: int) -> bool:
                    return msg > 100


                workflow = (
                    WorkflowBuilder()
                    .add_edge(ProcessorA(id="a"), ProcessorB(id="b"), condition=only_large_numbers)
                    .set_start_executor("a")
                    .build()
                )

                # Connect by string ID (after merge)
                builder.merge(fragment_a, prefix="a")
                builder.merge(fragment_b, prefix="b")
                builder.add_edge("a/output", "b/input")
        """
        source_id = self._normalize_endpoint(source)
        target_id = self._normalize_endpoint(target)
        self._edge_groups.append(SingleEdgeGroup(source_id, target_id, condition))  # type: ignore[call-arg]
        return self

    def add_fan_out_edges(
        self,
        source: Executor | AgentProtocol,
        targets: Sequence[Executor | AgentProtocol],
    ) -> Self:
        """Add multiple edges to the workflow where messages from the source will be sent to all targets.

        The output types of the source and the input types of the targets must be compatible.
        Messages from the source will be broadcast to all target executors concurrently.

        Args:
            source: The source executor of the edges.
            targets: A list of target executors for the edges.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


                class DataSource(Executor):
                    @handler
                    async def generate(self, count: int, ctx: WorkflowContext[str]) -> None:
                        for i in range(count):
                            await ctx.send_message(f"data_{i}")


                class ValidatorA(Executor):
                    @handler
                    async def validate(self, data: str, ctx: WorkflowContext) -> None:
                        print(f"ValidatorA: {data}")


                class ValidatorB(Executor):
                    @handler
                    async def validate(self, data: str, ctx: WorkflowContext) -> None:
                        print(f"ValidatorB: {data}")


                # Broadcast to multiple validators
                workflow = (
                    WorkflowBuilder()
                    .add_fan_out_edges(DataSource(id="source"), [ValidatorA(id="val_a"), ValidatorB(id="val_b")])
                    .set_start_executor("source")
                    .build()
                )
        """
        source_exec = self._maybe_wrap_agent(source)
        target_execs = [self._maybe_wrap_agent(t) for t in targets]
        source_id = self._add_executor(source_exec)
        target_ids = [self._add_executor(t) for t in target_execs]
        self._edge_groups.append(FanOutEdgeGroup(source_id, target_ids))  # type: ignore[call-arg]

        return self

    def add_switch_case_edge_group(
        self,
        source: Executor | AgentProtocol,
        cases: Sequence[Case | Default],
    ) -> Self:
        """Add an edge group that represents a switch-case statement.

        The output types of the source and the input types of the targets must be compatible.
        Messages from the source executor will be sent to one of the target executors based on
        the provided conditions.

        Think of this as a switch statement where each target executor corresponds to a case.
        Each condition function will be evaluated in order, and the first one that returns True
        will determine which target executor receives the message.

        The last case (the default case) will receive messages that fall through all conditions
        (i.e., no condition matched).

        Args:
            source: The source executor of the edges.
            cases: A list of case objects that determine the target executor for each message.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler, Case, Default
                from dataclasses import dataclass


                @dataclass
                class Result:
                    score: int


                class Evaluator(Executor):
                    @handler
                    async def evaluate(self, text: str, ctx: WorkflowContext[Result]) -> None:
                        await ctx.send_message(Result(score=len(text)))


                class HighScoreHandler(Executor):
                    @handler
                    async def handle(self, result: Result, ctx: WorkflowContext) -> None:
                        print(f"High score: {result.score}")


                class LowScoreHandler(Executor):
                    @handler
                    async def handle(self, result: Result, ctx: WorkflowContext) -> None:
                        print(f"Low score: {result.score}")


                # Route based on score value
                workflow = (
                    WorkflowBuilder()
                    .add_switch_case_edge_group(
                        Evaluator(id="eval"),
                        [
                            Case(condition=lambda r: r.score > 10, target=HighScoreHandler(id="high")),
                            Default(target=LowScoreHandler(id="low")),
                        ],
                    )
                    .set_start_executor("eval")
                    .build()
                )
        """
        source_exec = self._maybe_wrap_agent(source)
        source_id = self._add_executor(source_exec)
        # Convert case data types to internal types that only uses target_id.
        internal_cases: list[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault] = []
        for case in cases:
            # Allow case targets to be agents
            case.target = self._maybe_wrap_agent(case.target)  # type: ignore[attr-defined]
            self._add_executor(case.target)
            if isinstance(case, Default):
                internal_cases.append(SwitchCaseEdgeGroupDefault(target_id=case.target.id))
            else:
                internal_cases.append(SwitchCaseEdgeGroupCase(condition=case.condition, target_id=case.target.id))
        self._edge_groups.append(SwitchCaseEdgeGroup(source_id, internal_cases))  # type: ignore[call-arg]

        return self

    def add_multi_selection_edge_group(
        self,
        source: Executor | AgentProtocol,
        targets: Sequence[Executor | AgentProtocol],
        selection_func: Callable[[Any, list[str]], list[str]],
    ) -> Self:
        """Add an edge group that represents a multi-selection execution model.

        The output types of the source and the input types of the targets must be compatible.
        Messages from the source executor will be sent to multiple target executors based on
        the provided selection function.

        The selection function should take a message and a list of target executor IDs,
        and return a list of executor IDs indicating which target executors should receive the message.

        Args:
            source: The source executor of the edges.
            targets: A list of target executors for the edges.
            selection_func: A function that selects target executors for messages.
                Takes (message, list[executor_id]) and returns list[executor_id].

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
                from dataclasses import dataclass


                @dataclass
                class Task:
                    priority: str
                    data: str


                class TaskDispatcher(Executor):
                    @handler
                    async def dispatch(self, text: str, ctx: WorkflowContext[Task]) -> None:
                        priority = "high" if len(text) > 10 else "low"
                        await ctx.send_message(Task(priority=priority, data=text))


                class WorkerA(Executor):
                    @handler
                    async def process(self, task: Task, ctx: WorkflowContext) -> None:
                        print(f"WorkerA processing: {task.data}")


                class WorkerB(Executor):
                    @handler
                    async def process(self, task: Task, ctx: WorkflowContext) -> None:
                        print(f"WorkerB processing: {task.data}")


                # Select workers based on task priority
                def select_workers(task: Task, executor_ids: list[str]) -> list[str]:
                    if task.priority == "high":
                        return executor_ids  # Send to all workers
                    return [executor_ids[0]]  # Send to first worker only


                workflow = (
                    WorkflowBuilder()
                    .add_multi_selection_edge_group(
                        TaskDispatcher(id="dispatcher"),
                        [WorkerA(id="worker_a"), WorkerB(id="worker_b")],
                        selection_func=select_workers,
                    )
                    .set_start_executor("dispatcher")
                    .build()
                )
        """
        source_exec = self._maybe_wrap_agent(source)
        target_execs = [self._maybe_wrap_agent(t) for t in targets]
        source_id = self._add_executor(source_exec)
        target_ids = [self._add_executor(t) for t in target_execs]
        self._edge_groups.append(FanOutEdgeGroup(source_id, target_ids, selection_func))  # type: ignore[call-arg]

        return self

    def add_fan_in_edges(
        self,
        sources: Sequence[Executor | AgentProtocol],
        target: Executor | AgentProtocol,
    ) -> Self:
        """Add multiple edges from sources to a single target executor.

        The edges will be grouped together for synchronized processing, meaning
        the target executor will only be executed once all source executors have completed.

        The target executor will receive a list of messages aggregated from all source executors.
        Thus the input types of the target executor must be compatible with a list of the output
        types of the source executors.

        Args:
            sources: A list of source executors for the edges.
            target: The target executor for the edges.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from typing_extensions import Never
                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


                class Producer(Executor):
                    @handler
                    async def produce(self, seed: int, ctx: WorkflowContext[str]) -> None:
                        await ctx.send_message(f"result_{seed}")


                class Aggregator(Executor):
                    @handler
                    async def aggregate(self, results: list[str], ctx: WorkflowContext[Never, str]) -> None:
                        combined = ", ".join(results)
                        await ctx.yield_output(f"Combined: {combined}")


                # Collect results from multiple producers
                workflow = (
                    WorkflowBuilder()
                    .add_fan_in_edges([Producer(id="prod_1"), Producer(id="prod_2")], Aggregator(id="agg"))
                    .set_start_executor("prod_1")
                    .build()
                )
        """
        source_execs = [self._maybe_wrap_agent(s) for s in sources]
        target_exec = self._maybe_wrap_agent(target)
        source_ids = [self._add_executor(s) for s in source_execs]
        target_id = self._add_executor(target_exec)
        self._edge_groups.append(FanInEdgeGroup(source_ids, target_id))  # type: ignore[call-arg]

        return self

    def add_chain(self, executors: Sequence[Executor | AgentProtocol]) -> Self:
        """Add a chain of executors to the workflow.

        The output of each executor in the chain will be sent to the next executor in the chain.
        The input types of each executor must be compatible with the output types of the previous executor.

        Circles in the chain are not allowed, meaning the chain cannot have two executors with the same ID.

        Args:
            executors: A list of executors to be added to the chain.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from typing_extensions import Never
                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


                class Step1(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[str]) -> None:
                        await ctx.send_message(text.upper())


                class Step2(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[str]) -> None:
                        await ctx.send_message(text[::-1])


                class Step3(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
                        await ctx.yield_output(f"Final: {text}")


                # Chain executors in sequence
                workflow = (
                    WorkflowBuilder()
                    .add_chain([Step1(id="step1"), Step2(id="step2"), Step3(id="step3")])
                    .set_start_executor("step1")
                    .build()
                )
        """
        # Wrap each candidate first to ensure stable IDs before adding edges
        wrapped: list[Executor] = [self._maybe_wrap_agent(e) for e in executors]
        for i in range(len(wrapped) - 1):
            self.add_edge(wrapped[i], wrapped[i + 1])
        return self

    def set_start_executor(self, executor: Executor | AgentProtocol | str) -> Self:
        """Set the starting executor for the workflow.

        The start executor is the entry point for the workflow. When the workflow is executed,
        the initial message will be sent to this executor.

        Args:
            executor: The starting executor, which can be an Executor instance, AgentProtocol instance,
                or the string ID of an executor previously added to the workflow.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from typing_extensions import Never
                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


                class EntryPoint(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[str]) -> None:
                        await ctx.send_message(text.upper())


                class Processor(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
                        await ctx.yield_output(text)


                # Set by executor instance
                entry = EntryPoint(id="entry")
                workflow = WorkflowBuilder().add_edge(entry, Processor(id="proc")).set_start_executor(entry).build()

                # Set by executor ID string
                workflow = (
                    WorkflowBuilder()
                    .add_edge(EntryPoint(id="entry"), Processor(id="proc"))
                    .set_start_executor("entry")
                    .build()
                )
        """
        if isinstance(executor, str):
            self._start_executor = executor
        else:
            wrapped = self._maybe_wrap_agent(executor)  # type: ignore[arg-type]
            self._start_executor = wrapped
            # Ensure the start executor is present in the executor map so validation succeeds
            # even if no edges are added yet, or before edges wrap the same agent again.
            existing = self._executors.get(wrapped.id)
            if existing is not wrapped:
                self._add_executor(wrapped)
        return self

    def set_max_iterations(self, max_iterations: int) -> Self:
        """Set the maximum number of iterations for the workflow.

        When a workflow contains cycles, this limit prevents infinite loops by capping
        the total number of executor invocations. The default is 100 iterations.

        Args:
            max_iterations: The maximum number of iterations the workflow will run for convergence.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


                class StepA(Executor):
                    @handler
                    async def process(self, count: int, ctx: WorkflowContext[int]) -> None:
                        if count < 10:
                            await ctx.send_message(count + 1)


                class StepB(Executor):
                    @handler
                    async def process(self, count: int, ctx: WorkflowContext[int]) -> None:
                        await ctx.send_message(count)


                # Set a custom iteration limit for workflow with cycles
                workflow = (
                    WorkflowBuilder()
                    .set_max_iterations(500)
                    .add_edge(StepA(id="step_a"), StepB(id="step_b"))
                    .add_edge(StepB(id="step_b"), StepA(id="step_a"))  # Cycle
                    .set_start_executor("step_a")
                    .build()
                )
        """
        self._max_iterations = max_iterations
        return self

    # Removed explicit set_agent_streaming() API; agents always stream updates.

    def with_checkpointing(self, checkpoint_storage: CheckpointStorage) -> Self:
        """Enable checkpointing with the specified storage.

        Checkpointing allows workflows to save their state periodically, enabling
        pause/resume functionality and recovery from failures. The checkpoint storage
        implementation determines where checkpoints are persisted.

        Args:
            checkpoint_storage: The checkpoint storage implementation to use.

        Returns:
            Self: The WorkflowBuilder instance for method chaining.

        Example:
            .. code-block:: python

                from typing_extensions import Never
                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
                from agent_framework import FileCheckpointStorage


                class ProcessorA(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[str]) -> None:
                        await ctx.send_message(text.upper())


                class ProcessorB(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
                        await ctx.yield_output(text)


                # Enable checkpointing with file-based storage
                storage = FileCheckpointStorage("./checkpoints")
                workflow = (
                    WorkflowBuilder()
                    .add_edge(ProcessorA(id="proc_a"), ProcessorB(id="proc_b"))
                    .set_start_executor("proc_a")
                    .with_checkpointing(storage)
                    .build()
                )

                # Run with checkpoint saving
                events = await workflow.run("input")
        """
        self._checkpoint_storage = checkpoint_storage
        return self

    def build(self) -> Workflow:
        """Build and return the constructed workflow.

        This method performs validation before building the workflow to ensure:
        - A starting executor has been set
        - All edges connect valid executors
        - The graph is properly connected
        - Type compatibility between connected executors

        Returns:
            Workflow: An immutable Workflow instance ready for execution.

        Raises:
            ValueError: If starting executor is not set.
            WorkflowValidationError: If workflow validation fails (includes EdgeDuplicationError,
                TypeCompatibilityError, and GraphConnectivityError subclasses).

        Example:
            .. code-block:: python

                from typing_extensions import Never
                from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler


                class MyExecutor(Executor):
                    @handler
                    async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
                        await ctx.yield_output(text.upper())


                # Build and execute a workflow
                workflow = WorkflowBuilder().set_start_executor(MyExecutor(id="executor")).build()

                # The workflow is now immutable and ready to run
                events = await workflow.run("hello")
                print(events.get_outputs())  # ['HELLO']

                # Workflows can be reused multiple times
                events2 = await workflow.run("world")
                print(events2.get_outputs())  # ['WORLD']
        """
        # Create workflow build span that includes validation and workflow creation
        with create_workflow_span(OtelAttr.WORKFLOW_BUILD_SPAN) as span:
            try:
                # Add workflow build started event
                span.add_event(OtelAttr.BUILD_STARTED)

                if not self._start_executor:
                    raise ValueError(
                        "Starting executor must be set using set_start_executor before building the workflow."
                    )

                # Perform validation before creating the workflow
                validate_workflow_graph(
                    self._edge_groups,
                    self._executors,
                    self._start_executor,
                )

                # Add validation completed event
                span.add_event(OtelAttr.BUILD_VALIDATION_COMPLETED)

                context = InProcRunnerContext(self._checkpoint_storage)

                # Create workflow instance after validation
                workflow = Workflow(
                    self._edge_groups,
                    self._executors,
                    self._start_executor,
                    context,
                    self._max_iterations,
                    name=self._name,
                    description=self._description,
                )
                build_attributes: dict[str, Any] = {
                    OtelAttr.WORKFLOW_ID: workflow.id,
                    OtelAttr.WORKFLOW_DEFINITION: workflow.to_json(),
                }
                if workflow.name:
                    build_attributes[OtelAttr.WORKFLOW_NAME] = workflow.name
                if workflow.description:
                    build_attributes[OtelAttr.WORKFLOW_DESCRIPTION] = workflow.description
                span.set_attributes(build_attributes)

                # Add workflow build completed event
                span.add_event(OtelAttr.BUILD_COMPLETED)

                return workflow

            except Exception as exc:
                attributes = {
                    OtelAttr.BUILD_ERROR_MESSAGE: str(exc),
                    OtelAttr.BUILD_ERROR_TYPE: type(exc).__name__,
                }
                span.add_event(OtelAttr.BUILD_ERROR, attributes)  # type: ignore[reportArgumentType, arg-type]
                capture_exception(span, exc)
                raise
