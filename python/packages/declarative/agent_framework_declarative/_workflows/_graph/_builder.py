# Copyright (c) Microsoft. All rights reserved.

"""Builder that transforms declarative YAML into a workflow graph.

This module provides the DeclarativeGraphBuilder which is analogous to
.NET's WorkflowActionVisitor + WorkflowElementWalker. It walks the YAML
action definitions and creates a proper workflow graph with:
- Executor nodes for each action
- Edges for sequential flow
- Async state-aware edge conditions for If/Switch branching
- Loop edges for foreach
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any, cast

from agent_framework._workflows import (
    Workflow,
    WorkflowBuilder,
)
from agent_framework._workflows._shared_state import SharedState

from ._base import (
    DECLARATIVE_STATE_KEY,
    DeclarativeWorkflowState,
    LoopIterationResult,
)
from ._executors_agents import AGENT_ACTION_EXECUTORS, InvokeAzureAgentExecutor
from ._executors_basic import BASIC_ACTION_EXECUTORS
from ._executors_control_flow import (
    CONTROL_FLOW_EXECUTORS,
    ForeachInitExecutor,
    ForeachNextExecutor,
    JoinExecutor,
)
from ._executors_human_input import HUMAN_INPUT_EXECUTORS

# Type alias for async condition functions used in workflow edges
ConditionFunc = Callable[[Any, SharedState], Awaitable[bool]]

# Combined mapping of all action kinds to executor classes
ALL_ACTION_EXECUTORS = {
    **BASIC_ACTION_EXECUTORS,
    **CONTROL_FLOW_EXECUTORS,
    **AGENT_ACTION_EXECUTORS,
    **HUMAN_INPUT_EXECUTORS,
}


def _generate_semantic_id(action_def: dict[str, Any], kind: str) -> str | None:
    """Generate a semantic ID from the action's content.

    Derives a meaningful identifier from the action's primary properties rather
    than using generic index-based naming. Returns None if no semantic name
    can be derived.

    Args:
        action_def: The action definition from YAML
        kind: The action kind (e.g., "SetValue", "SendActivity")

    Returns:
        A semantic ID string, or None if no semantic name can be derived
    """
    # SetValue/SetVariable: use the value being set
    if kind in ("SetValue", "SetVariable", "SetTextVariable"):
        value = action_def.get("value") or action_def.get("text")
        if value and isinstance(value, str) and not value.startswith("="):
            # Simple value like "child", "teenager", "adult"
            slug = _slugify(value)
            if slug:
                return f"set_{slug}"

        # Try to derive from path (e.g., turn.category -> set_category)
        path = action_def.get("path") or action_def.get("variable", {}).get("path")
        if path and isinstance(path, str):
            # Extract last segment: "turn.category" -> "category"
            last_segment = path.split(".")[-1]
            slug = _slugify(last_segment)
            if slug:
                return f"set_{slug}"

    # SendActivity: extract meaningful words from text
    if kind == "SendActivity":
        activity: dict[str, Any] | Any = action_def.get("activity", {})
        text: str | None = None
        if isinstance(activity, dict):
            activity_dict: dict[str, Any] = cast(dict[str, Any], activity)
            text_val = activity_dict.get("text")
            text = str(text_val) if text_val else None
        if text and isinstance(text, str):
            slug = _extract_activity_slug(text)
            if slug:
                return f"send_{slug}"

    # InvokeAzureAgent: use agent name
    if kind == "InvokeAzureAgent":
        agent = action_def.get("agent") or action_def.get("agentName")
        if agent and isinstance(agent, str):
            slug = _slugify(agent)
            if slug:
                return f"invoke_{slug}"

    # AppendValue: use the path
    if kind == "AppendValue":
        path = action_def.get("path")
        if path and isinstance(path, str):
            last_segment = path.split(".")[-1]
            slug = _slugify(last_segment)
            if slug:
                return f"append_{slug}"

    # ResetVariable: use the path
    if kind in ("ResetVariable", "DeleteVariable"):
        path = action_def.get("path") or action_def.get("variable", {}).get("path")
        if path and isinstance(path, str):
            last_segment = path.split(".")[-1]
            slug = _slugify(last_segment)
            if slug:
                prefix = "reset" if kind == "ResetVariable" else "delete"
                return f"{prefix}_{slug}"

    # HumanInput actions: use prompt or variable
    if kind in ("RequestHumanInput", "WaitForHumanInput"):
        prompt = action_def.get("prompt")
        if prompt and isinstance(prompt, str):
            slug = _extract_activity_slug(prompt)
            if slug:
                return f"input_{slug}"
        variable = action_def.get("variable", {}).get("path")
        if variable and isinstance(variable, str):
            slug = _slugify(variable.split(".")[-1])
            if slug:
                return f"input_{slug}"

    return None


def _slugify(text: str, max_words: int = 3) -> str:
    """Convert text to a slug suitable for an ID.

    Args:
        text: The text to slugify
        max_words: Maximum number of words to include

    Returns:
        A lowercase, underscore-separated slug
    """
    # Remove expression prefix if present
    if text.startswith("="):
        text = text[1:]

    # Convert to lowercase and extract words
    text = text.lower()
    # Replace non-alphanumeric with spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)
    # Split and take first N words
    words = text.split()[:max_words]
    # Filter out very short words (articles, etc.) except meaningful ones
    words = [w for w in words if len(w) > 1 or w in ("a", "i")]

    if not words:
        return ""

    return "_".join(words)


def _extract_activity_slug(text: str) -> str:
    """Extract a meaningful slug from activity text.

    Focuses on finding the most distinctive content words.

    Args:
        text: The activity text

    Returns:
        A slug representing the key content
    """
    # Words to skip - common greeting/filler words
    skip_words = {
        "welcome",
        "hello",
        "hi",
        "hey",
        "there",
        "dear",
        "greetings",
        "here",
        "are",
        "some",
        "our",
        "the",
        "for",
        "and",
        "you",
        "your",
        "have",
        "been",
        "with",
        "this",
        "that",
        "these",
        "those",
        "check",
        "out",
        "enjoy",
        "fun",
        "cool",
        "great",
        "things",
        "activities",
    }

    text = text.lower()
    # Remove punctuation
    text = str(re.sub(r"[^a-z0-9\s]+", " ", text))
    words: list[str] = text.split()

    # Find distinctive words (not in skip list, longer than 3 chars)
    meaningful_words: list[str] = []
    for word in words:
        if word not in skip_words and len(word) > 3:
            meaningful_words.append(word)
            if len(meaningful_words) >= 2:
                break

    if not meaningful_words:
        # Fall back to any word longer than 2 chars not in greeting set
        greeting_set = {"welcome", "hello", "hi", "hey", "there", "dear", "greetings"}
        for word in words:
            if len(word) > 2 and word not in greeting_set:
                meaningful_words.append(word)
                if len(meaningful_words) >= 2:
                    break

    if not meaningful_words:
        return ""

    return "_".join(meaningful_words)


def _create_condition_evaluator(condition_expr: str, negate: bool = False) -> ConditionFunc:
    """Create an async condition function that evaluates a declarative expression.

    Args:
        condition_expr: The condition expression (e.g., "=turn.age < 13")
        negate: If True, negate the result (for else branches)

    Returns:
        An async function (data, shared_state) -> bool
    """

    async def evaluate_condition(data: Any, shared_state: SharedState) -> bool:
        # Get DeclarativeWorkflowState from shared state
        try:
            await shared_state.get(DECLARATIVE_STATE_KEY)
        except KeyError:
            # State not initialized - shouldn't happen but handle gracefully
            return not negate  # Default to then branch

        state = DeclarativeWorkflowState(shared_state)
        result = await state.eval(condition_expr)
        bool_result = bool(result)
        return not bool_result if negate else bool_result

    return evaluate_condition


def _create_none_matched_condition(condition_exprs: list[str]) -> ConditionFunc:
    """Create an async condition that returns True when none of the given conditions match.

    Args:
        condition_exprs: List of condition expressions that should all be False

    Returns:
        An async function (data, shared_state) -> bool
    """

    async def evaluate_none_matched(data: Any, shared_state: SharedState) -> bool:
        # Get DeclarativeWorkflowState from shared state
        try:
            await shared_state.get(DECLARATIVE_STATE_KEY)
        except KeyError:
            return True  # Default to else branch if no state

        state = DeclarativeWorkflowState(shared_state)

        # Check that none of the conditions match
        for condition_expr in condition_exprs:
            result = await state.eval(condition_expr)
            if bool(result):
                return False  # A condition matched, so this default branch shouldn't trigger

        return True  # No conditions matched

    return evaluate_none_matched


class DeclarativeGraphBuilder:
    """Builds a Workflow graph from declarative YAML actions.

    This builder transforms declarative action definitions into a proper
    workflow graph with executor nodes and edges. It handles:
    - Sequential actions (simple edges)
    - Conditional branching (If/Switch with condition edges)
    - Loops (Foreach with loop edges)
    - Jumps (Goto with target edges)

    Example usage:
        yaml_def = {
            "actions": [
                {"kind": "SendActivity", "activity": {"text": "Hello"}},
                {"kind": "SetValue", "path": "turn.count", "value": 0},
            ]
        }
        builder = DeclarativeGraphBuilder(yaml_def)
        workflow = builder.build()
    """

    def __init__(
        self,
        yaml_definition: dict[str, Any],
        workflow_id: str | None = None,
        agents: dict[str, Any] | None = None,
        checkpoint_storage: Any | None = None,
    ):
        """Initialize the builder.

        Args:
            yaml_definition: The parsed YAML workflow definition
            workflow_id: Optional ID for the workflow (defaults to name from YAML)
            agents: Registry of agent instances by name (for InvokeAzureAgent actions)
            checkpoint_storage: Optional checkpoint storage for pause/resume support
        """
        self._yaml_def = yaml_definition
        self._workflow_id = workflow_id or yaml_definition.get("name", "declarative_workflow")
        self._executors: dict[str, Any] = {}  # id -> executor
        self._action_index = 0  # Counter for generating unique IDs
        self._agents = agents or {}  # Agent registry for agent executors
        self._checkpoint_storage = checkpoint_storage
        self._pending_gotos: list[tuple[Any, str]] = []  # (goto_executor, target_id)

    def build(self) -> Workflow:
        """Build the workflow graph.

        Returns:
            A Workflow instance with all executors wired together

        Raises:
            ValueError: If no actions are defined (empty workflow)
        """
        builder = WorkflowBuilder(name=self._workflow_id)

        # Enable checkpointing if storage is provided
        if self._checkpoint_storage:
            builder.with_checkpointing(self._checkpoint_storage)

        actions = self._yaml_def.get("actions", [])
        if not actions:
            # Empty workflow - raise an error since we need at least one executor
            raise ValueError("Cannot build workflow with no actions. At least one action is required.")

        # First pass: create all executors
        entry_executor = self._create_executors_for_actions(actions, builder)

        # Set the entry point
        if entry_executor:
            # Check if entry is a control flow structure (If/Switch)
            if getattr(entry_executor, "_is_if_structure", False) or getattr(
                entry_executor, "_is_switch_structure", False
            ):
                # Create an entry passthrough node and wire to the structure's branches
                entry_node = JoinExecutor({"kind": "Entry"}, id="_workflow_entry")
                self._executors[entry_node.id] = entry_node
                builder.set_start_executor(entry_node)
                # Use _add_sequential_edge which knows how to wire to structures
                self._add_sequential_edge(builder, entry_node, entry_executor)
            else:
                builder.set_start_executor(entry_executor)
        else:
            raise ValueError("Failed to create any executors from actions.")

        # Resolve pending gotos (back-edges for loops, forward-edges for jumps)
        self._resolve_pending_gotos(builder)

        return builder.build()

    def _resolve_pending_gotos(self, builder: WorkflowBuilder) -> None:
        """Resolve pending goto edges after all executors are created.

        Creates edges from goto executors to their target executors.
        """
        for goto_executor, target_id in self._pending_gotos:
            target_executor = self._executors.get(target_id)
            if target_executor:
                # Create edge from goto to target
                builder.add_edge(source=goto_executor, target=target_executor)
            # If target not found, the goto effectively becomes a no-op (workflow ends)

    def _create_executors_for_actions(
        self,
        actions: list[dict[str, Any]],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any | None:
        """Create executors for a list of actions and wire them together.

        Args:
            actions: List of action definitions
            builder: The workflow builder
            parent_context: Context from parent (e.g., loop info)

        Returns:
            The first executor in the chain, or None if no actions
        """
        if not actions:
            return None

        first_executor = None
        prev_executor = None
        executors_in_chain: list[Any] = []

        for action_def in actions:
            executor = self._create_executor_for_action(action_def, builder, parent_context)

            if executor is None:
                continue

            executors_in_chain.append(executor)

            if first_executor is None:
                first_executor = executor

            # Wire sequential edge from previous executor
            if prev_executor is not None:
                self._add_sequential_edge(builder, prev_executor, executor)

            prev_executor = executor

        # Store the chain for later reference
        if first_executor is not None:
            first_executor._chain_executors = executors_in_chain  # type: ignore[attr-defined]

        return first_executor

    def _create_executor_for_action(
        self,
        action_def: dict[str, Any],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any | None:
        """Create an executor for a single action.

        Args:
            action_def: The action definition from YAML
            builder: The workflow builder
            parent_context: Context from parent

        Returns:
            The created executor, or None if action type not supported
        """
        kind = action_def.get("kind", "")

        # Handle special control flow actions
        if kind == "If":
            return self._create_if_structure(action_def, builder, parent_context)
        if kind == "Switch" or kind == "ConditionGroup":
            return self._create_switch_structure(action_def, builder, parent_context)
        if kind == "Foreach":
            return self._create_foreach_structure(action_def, builder, parent_context)
        if kind == "Goto" or kind == "GotoAction":
            return self._create_goto_reference(action_def, builder, parent_context)
        if kind == "BreakLoop":
            return self._create_break_executor(action_def, builder, parent_context)
        if kind == "ContinueLoop":
            return self._create_continue_executor(action_def, builder, parent_context)

        # Get the executor class for this action kind
        executor_class = ALL_ACTION_EXECUTORS.get(kind)

        if executor_class is None:
            # Unknown action type - skip with warning
            # In production, might want to log this
            return None

        # Create the executor with ID
        # Priority: explicit ID > semantic ID > fallback index-based ID
        explicit_id = action_def.get("id")
        if explicit_id:
            action_id = explicit_id
        else:
            # Try to generate a semantic ID from action content
            semantic_id = _generate_semantic_id(action_def, kind)
            if semantic_id:
                # Ensure uniqueness by checking if ID already exists
                base_id = semantic_id
                suffix = 0
                while semantic_id in self._executors:
                    suffix += 1
                    semantic_id = f"{base_id}_{suffix}"
                action_id = semantic_id
            else:
                # Fallback to index-based ID
                parent_id = (parent_context or {}).get("parent_id")
                action_id = f"{parent_id}_{kind}_{self._action_index}" if parent_id else f"{kind}_{self._action_index}"
        self._action_index += 1

        # Pass agents to agent-related executors
        executor: Any
        if kind in ("InvokeAzureAgent",):
            executor = InvokeAzureAgentExecutor(action_def, id=action_id, agents=self._agents)
        else:
            executor = executor_class(action_def, id=action_id)
        self._executors[action_id] = executor

        return executor

    def _create_if_structure(
        self,
        action_def: dict[str, Any],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any:
        """Create the graph structure for an If action.

        An If action is implemented without condition evaluator or join executors.
        Conditional edges evaluate expressions against workflow state.
        Branch exits are tracked so they can be wired directly to the successor.

        Args:
            action_def: The If action definition
            builder: The workflow builder
            parent_context: Context from parent

        Returns:
            A structure representing the If with branch entries and exits
        """
        action_id = action_def.get("id") or f"If_{self._action_index}"
        self._action_index += 1

        condition_expr = action_def.get("condition", "true")
        # Normalize boolean conditions from YAML to PowerFx-style strings
        if condition_expr is True:
            condition_expr = "=true"
        elif condition_expr is False:
            condition_expr = "=false"
        elif isinstance(condition_expr, str) and not condition_expr.startswith("="):
            # Bare string conditions should be evaluated as expressions
            condition_expr = f"={condition_expr}"

        # Pass the If's ID as context for child action naming
        branch_context = {
            **(parent_context or {}),
            "parent_id": action_id,
        }

        # Create then branch
        then_actions = action_def.get("then", action_def.get("actions", []))
        then_entry = self._create_executors_for_actions(then_actions, builder, branch_context)

        # Create else branch
        else_actions = action_def.get("else", [])
        else_entry = self._create_executors_for_actions(else_actions, builder, branch_context) if else_actions else None
        else_passthrough = None
        if not else_entry:
            # No else branch - create a passthrough for continuation when condition is false
            else_passthrough = JoinExecutor({"kind": "ElsePassthrough"}, id=f"{action_id}_else_pass")
            self._executors[else_passthrough.id] = else_passthrough

        # Create async conditions
        then_condition = _create_condition_evaluator(condition_expr, negate=False)
        else_condition = _create_condition_evaluator(condition_expr, negate=True)

        # Get branch exit executors for later wiring to successor
        then_exit = self._get_branch_exit(then_entry)
        else_exit = self._get_branch_exit(else_entry) if else_entry else else_passthrough

        # Collect all branch exits (for wiring to successor)
        branch_exits: list[Any] = []
        if then_exit:
            branch_exits.append(then_exit)
        if else_exit:
            branch_exits.append(else_exit)

        # Create an IfStructure to hold all the info needed for wiring
        class IfStructure:
            def __init__(self) -> None:
                self.id = action_id
                self.then_entry = then_entry
                self.else_entry = else_entry
                self.else_passthrough = else_passthrough  # Passthrough when no else actions
                self.then_condition = then_condition
                self.else_condition = else_condition
                self.branch_exits = branch_exits  # All exits that need wiring to successor
                self._is_if_structure = True

        return IfStructure()

    def _create_switch_structure(
        self,
        action_def: dict[str, Any],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any:
        """Create the graph structure for a Switch/ConditionGroup action.

        Like If, Switch is implemented without evaluator or join executors.
        Conditional edges evaluate each condition expression against workflow state.
        Branch exits are tracked for wiring directly to successor.

        Args:
            action_def: The Switch action definition
            builder: The workflow builder
            parent_context: Context from parent

        Returns:
            A SwitchStructure containing branch info for wiring
        """
        action_id = action_def.get("id") or f"Switch_{self._action_index}"
        self._action_index += 1

        conditions = action_def.get("conditions", [])

        # Pass the Switch's ID as context for child action naming
        branch_context = {
            **(parent_context or {}),
            "parent_id": action_id,
        }

        # Collect branches with their conditions and exits
        branches: list[tuple[Any, Any]] = []  # (entry_executor, async_condition)
        branch_exits: list[Any] = []  # All exits that need wiring to successor
        matched_conditions: list[str] = []  # For building the "none matched" condition

        for i, cond_item in enumerate(conditions):
            condition_expr = cond_item.get("condition", "true")
            # Normalize boolean conditions from YAML to PowerFx-style strings
            if condition_expr is True:
                condition_expr = "=true"
            elif condition_expr is False:
                condition_expr = "=false"
            elif isinstance(condition_expr, str) and not condition_expr.startswith("="):
                condition_expr = f"={condition_expr}"
            matched_conditions.append(condition_expr)

            branch_actions = cond_item.get("actions", [])
            # Use branch-specific context
            case_context = {**branch_context, "parent_id": f"{action_id}_case{i}"}
            branch_entry = self._create_executors_for_actions(branch_actions, builder, case_context)

            if branch_entry:
                # Create async condition for this branch
                branch_condition = _create_condition_evaluator(condition_expr, negate=False)
                branches.append((branch_entry, branch_condition))
                # Track exit for later wiring
                branch_exit = self._get_branch_exit(branch_entry)
                if branch_exit:
                    branch_exits.append(branch_exit)

        # Handle else/default branch
        # .NET uses "elseActions", Python fallback to "else" or "default"
        else_actions = action_def.get("elseActions", action_def.get("else", action_def.get("default", [])))
        default_entry = None
        default_passthrough = None
        if else_actions:
            default_context = {**branch_context, "parent_id": f"{action_id}_else"}
            default_entry = self._create_executors_for_actions(else_actions, builder, default_context)
            if default_entry:
                default_exit = self._get_branch_exit(default_entry)
                if default_exit:
                    branch_exits.append(default_exit)
        else:
            # No else actions - create a passthrough for the "no match" case
            # This allows the workflow to continue to the next action when no condition matches
            default_passthrough = JoinExecutor({"kind": "DefaultPassthrough"}, id=f"{action_id}_default")
            self._executors[default_passthrough.id] = default_passthrough
            branch_exits.append(default_passthrough)

        # Create default condition (none of the previous conditions matched)
        default_condition = _create_none_matched_condition(matched_conditions)

        # Create a SwitchStructure to hold all the info needed for wiring
        class SwitchStructure:
            def __init__(self) -> None:
                self.id = action_id
                self.branches = branches
                self.default_entry = default_entry
                self.default_passthrough = default_passthrough  # Passthrough when no else actions
                self.default_condition = default_condition
                self.branch_exits = branch_exits  # All exits that need wiring to successor
                self._is_switch_structure = True

        return SwitchStructure()

    def _create_foreach_structure(
        self,
        action_def: dict[str, Any],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any:
        """Create the graph structure for a Foreach action.

        A Foreach action becomes:
        1. ForeachInit node that initializes the loop
        2. Loop body actions
        3. ForeachNext node that advances to next item
        4. Back-edge from ForeachNext to loop body (when has_next=True)
        5. Exit edge from ForeachNext (when has_next=False)

        Args:
            action_def: The Foreach action definition
            builder: The workflow builder
            parent_context: Context from parent

        Returns:
            The foreach init executor (entry point)
        """
        action_id = action_def.get("id") or f"Foreach_{self._action_index}"
        self._action_index += 1

        # Create foreach init executor
        init_executor = ForeachInitExecutor(action_def, id=f"{action_id}_init")
        self._executors[init_executor.id] = init_executor

        # Create foreach next executor (for advancing to next item)
        next_executor = ForeachNextExecutor(action_def, init_executor.id, id=f"{action_id}_next")
        self._executors[next_executor.id] = next_executor

        # Create join node for loop exit
        join_executor = JoinExecutor({"kind": "Join"}, id=f"{action_id}_exit")
        self._executors[join_executor.id] = join_executor

        # Create loop body
        body_actions = action_def.get("actions", [])
        loop_context = {
            **(parent_context or {}),
            "loop_id": action_id,
            "loop_next_executor": next_executor,
        }
        body_entry = self._create_executors_for_actions(body_actions, builder, loop_context)

        if body_entry:
            # Init -> body (when has_next=True)
            builder.add_edge(
                source=init_executor,
                target=body_entry,
                condition=lambda msg: isinstance(msg, LoopIterationResult) and msg.has_next,
            )

            # Body exit -> Next (get all exits from body and wire to next_executor)
            body_exits = self._get_source_exits(body_entry)
            for body_exit in body_exits:
                builder.add_edge(source=body_exit, target=next_executor)

            # Next -> body (when has_next=True, loop back)
            builder.add_edge(
                source=next_executor,
                target=body_entry,
                condition=lambda msg: isinstance(msg, LoopIterationResult) and msg.has_next,
            )

        # Init -> join (when has_next=False, empty collection)
        builder.add_edge(
            source=init_executor,
            target=join_executor,
            condition=lambda msg: isinstance(msg, LoopIterationResult) and not msg.has_next,
        )

        # Next -> join (when has_next=False, loop complete)
        builder.add_edge(
            source=next_executor,
            target=join_executor,
            condition=lambda msg: isinstance(msg, LoopIterationResult) and not msg.has_next,
        )

        init_executor._exit_executor = join_executor  # type: ignore[attr-defined]
        return init_executor

    def _create_goto_reference(
        self,
        action_def: dict[str, Any],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any | None:
        """Create a GotoAction executor that jumps to the target action.

        GotoAction creates a back-edge (or forward-edge) in the graph to the target action.
        We create a pass-through executor and record the pending edge to be resolved
        after all executors are created.
        """
        from ._executors_control_flow import JoinExecutor

        target_id = action_def.get("target") or action_def.get("actionId")

        if not target_id:
            return None

        # Create a pass-through executor for the goto
        action_id = action_def.get("id") or f"goto_{target_id}_{self._action_index}"
        self._action_index += 1

        # Use JoinExecutor as a simple pass-through node
        goto_executor = JoinExecutor(action_def, id=action_id)
        self._executors[action_id] = goto_executor

        # Record pending goto edge to be resolved after all executors created
        self._pending_gotos.append((goto_executor, target_id))

        return goto_executor

    def _create_break_executor(
        self,
        action_def: dict[str, Any],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any | None:
        """Create a break executor for loop control."""
        from ._executors_control_flow import BreakLoopExecutor

        if parent_context and "loop_next_executor" in parent_context:
            loop_next = parent_context["loop_next_executor"]
            action_id = action_def.get("id") or f"Break_{self._action_index}"
            self._action_index += 1

            executor = BreakLoopExecutor(action_def, loop_next.id, id=action_id)
            self._executors[action_id] = executor

            # Wire break to loop next
            builder.add_edge(source=executor, target=loop_next)

            return executor

        return None

    def _create_continue_executor(
        self,
        action_def: dict[str, Any],
        builder: WorkflowBuilder,
        parent_context: dict[str, Any] | None = None,
    ) -> Any | None:
        """Create a continue executor for loop control."""
        from ._executors_control_flow import ContinueLoopExecutor

        if parent_context and "loop_next_executor" in parent_context:
            loop_next = parent_context["loop_next_executor"]
            action_id = action_def.get("id") or f"Continue_{self._action_index}"
            self._action_index += 1

            executor = ContinueLoopExecutor(action_def, loop_next.id, id=action_id)
            self._executors[action_id] = executor

            # Wire continue to loop next
            builder.add_edge(source=executor, target=loop_next)

            return executor

        return None

    def _add_sequential_edge(
        self,
        builder: WorkflowBuilder,
        source: Any,
        target: Any,
    ) -> None:
        """Add a sequential edge between two executors.

        Handles control flow structures:
        - If source is a structure (If/Switch), wire from all branch exits
        - If target is a structure (If/Switch), wire with conditional edges to branches
        """
        # Get all source exit points
        source_exits = self._get_source_exits(source)

        # Wire each source exit to target
        for source_exit in source_exits:
            self._wire_to_target(builder, source_exit, target)

    def _get_source_exits(self, source: Any) -> list[Any]:
        """Get all exit executors from a source (handles structures with multiple exits)."""
        # Check if source is a structure with branch_exits
        if hasattr(source, "branch_exits"):
            # Collect all exits, recursively flattening nested structures
            all_exits: list[Any] = []
            for exit_item in source.branch_exits:
                if hasattr(exit_item, "branch_exits"):
                    # Nested structure - recurse
                    all_exits.extend(self._collect_all_exits(exit_item))
                else:
                    all_exits.append(exit_item)
            return all_exits if all_exits else []

        # Check if source has a single exit executor
        actual_exit = getattr(source, "_exit_executor", source)
        return [actual_exit]

    def _wire_to_target(
        self,
        builder: WorkflowBuilder,
        source: Any,
        target: Any,
    ) -> None:
        """Wire a single source executor to a target (which may be a structure)."""
        # Check if target is an IfStructure (needs conditional edges)
        if getattr(target, "_is_if_structure", False):
            # Wire from source to then branch with then_condition
            if target.then_entry:
                self._add_conditional_edge(builder, source, target.then_entry, target.then_condition)
            # If no then entry, the condition just doesn't match - no edge needed
            # (the else condition will handle it)

            # Wire from source to else branch with else_condition
            if target.else_entry:
                self._add_conditional_edge(builder, source, target.else_entry, target.else_condition)
            elif target.else_passthrough:
                # No else actions - wire to the passthrough executor for continuation
                self._add_conditional_edge(builder, source, target.else_passthrough, target.else_condition)
            # If neither else_entry nor else_passthrough, unmatched conditions have no destination

        elif getattr(target, "_is_switch_structure", False):
            # Wire from source to switch branches
            for branch_entry, branch_condition in target.branches:
                if branch_entry:
                    self._add_conditional_edge(builder, source, branch_entry, branch_condition)

            # Wire default/else branch
            if target.default_entry:
                self._add_conditional_edge(builder, source, target.default_entry, target.default_condition)
            elif target.default_passthrough:
                # No else actions - wire to the passthrough executor for continuation
                self._add_conditional_edge(builder, source, target.default_passthrough, target.default_condition)
            # If neither default_entry nor default_passthrough, unmatched conditions have no destination

        else:
            # Normal sequential edge to a regular executor
            builder.add_edge(source=source, target=target)

    def _add_conditional_edge(
        self,
        builder: WorkflowBuilder,
        source: Any,
        target: Any,
        condition: Any,
    ) -> None:
        """Add a conditional edge, handling nested structures.

        If target is itself a structure (nested If/Switch), recursively wire
        with combined conditions.
        """
        if getattr(target, "_is_if_structure", False):
            # Nested If - need to combine conditions
            # Source --[condition AND then_condition]--> then_entry
            # Source --[condition AND else_condition]--> else_entry
            combined_then = self._combine_conditions(condition, target.then_condition)
            combined_else = self._combine_conditions(condition, target.else_condition)

            if target.then_entry:
                self._add_conditional_edge(builder, source, target.then_entry, combined_then)

            if target.else_entry:
                self._add_conditional_edge(builder, source, target.else_entry, combined_else)

        elif getattr(target, "_is_switch_structure", False):
            # Nested Switch - combine with each branch condition
            for branch_entry, branch_condition in target.branches:
                if branch_entry:
                    combined = self._combine_conditions(condition, branch_condition)
                    self._add_conditional_edge(builder, source, branch_entry, combined)

            if target.default_entry:
                combined_default = self._combine_conditions(condition, target.default_condition)
                self._add_conditional_edge(builder, source, target.default_entry, combined_default)
        else:
            # Regular executor - just add the edge with condition
            builder.add_edge(source=source, target=target, async_condition=condition)

    def _combine_conditions(
        self, outer_condition: ConditionFunc | None, inner_condition: ConditionFunc | None
    ) -> ConditionFunc | None:
        """Combine two async conditions with AND logic.

        Returns a new async condition that evaluates both conditions.
        """
        if outer_condition is None:
            return inner_condition
        if inner_condition is None:
            return outer_condition

        async def combined_condition(data: Any, shared_state: SharedState) -> bool:
            outer_result = await outer_condition(data, shared_state)
            if not outer_result:
                return False
            return await inner_condition(data, shared_state)

        return combined_condition

    def _get_branch_exit(self, branch_entry: Any) -> Any | None:
        """Get the exit executor of a branch.

        For a linear sequence of actions, returns the last executor.
        For nested structures, returns None (they have their own branch_exits).

        Args:
            branch_entry: The first executor of the branch

        Returns:
            The exit executor, or None if branch is empty or ends with a structure
        """
        if branch_entry is None:
            return None

        # Get the chain of executors in this branch
        chain = getattr(branch_entry, "_chain_executors", [branch_entry])

        if not chain:
            return None

        last_executor = chain[-1]

        # Check if last executor is a structure with branch_exits
        # In that case, we return the structure so its exits can be collected
        if hasattr(last_executor, "branch_exits"):
            return last_executor

        # Regular executor - get its exit point
        return getattr(last_executor, "_exit_executor", last_executor)

    def _collect_all_exits(self, structure: Any) -> list[Any]:
        """Recursively collect all exit executors from a structure."""
        exits: list[Any] = []

        if not hasattr(structure, "branch_exits"):
            # Not a structure - return the executor itself
            actual_exit = getattr(structure, "_exit_executor", structure)
            return [actual_exit]

        for exit_item in structure.branch_exits:
            if hasattr(exit_item, "branch_exits"):
                # Nested structure - recurse
                exits.extend(self._collect_all_exits(exit_item))
            else:
                exits.append(exit_item)

        return exits
