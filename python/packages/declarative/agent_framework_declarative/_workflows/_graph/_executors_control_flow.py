# Copyright (c) Microsoft. All rights reserved.

"""Control flow executors for the graph-based declarative workflow system.

Control flow in the graph-based system is handled differently than the interpreter:
- If/Switch: Condition evaluation happens on edges via async conditions
- Foreach: Loop iteration state managed in SharedState + loop edges
- Goto: Edge to target action (handled by builder)
- Break/Continue: Special signals for loop control

The key insight is that control flow becomes GRAPH STRUCTURE, not executor logic.
Conditions are evaluated on edges, not in separate executor nodes.
"""

from typing import Any

from agent_framework._workflows import (
    WorkflowContext,
    handler,
)

from ._base import (
    ActionComplete,
    ActionTrigger,
    DeclarativeActionExecutor,
    LoopControl,
    LoopIterationResult,
)

# Keys for loop state in SharedState
LOOP_STATE_KEY = "_declarative_loop_state"


class ForeachInitExecutor(DeclarativeActionExecutor):
    """Initializes a foreach loop.

    Sets up the loop state in SharedState and determines if there are items.
    """

    @handler
    async def handle_action(
        self,
        trigger: Any,
        ctx: WorkflowContext[LoopIterationResult],
    ) -> None:
        """Initialize the loop and check for first item."""
        state = await self._ensure_state_initialized(ctx, trigger)

        items_expr = self._action_def.get("itemsSource") or self._action_def.get("items")
        items_raw: Any = await state.eval_if_expression(items_expr) or []

        items: list[Any]
        items = (list(items_raw) if items_raw else []) if not isinstance(items_raw, (list, tuple)) else list(items_raw)  # type: ignore

        loop_id = self.id

        # Store loop state
        state_data = await state.get_state_data()
        loop_states = state_data.setdefault(LOOP_STATE_KEY, {})
        loop_states[loop_id] = {
            "items": items,
            "index": 0,
            "length": len(items),
        }
        await state.set_state_data(state_data)

        # Check if we have items
        if items:
            # Set the iteration variable
            item_var = self._action_def.get("iteratorVariable") or self._action_def.get("item", "turn.item")
            index_var = self._action_def.get("indexVariable") or self._action_def.get("index")

            await state.set(item_var, items[0])
            if index_var:
                await state.set(index_var, 0)

            await ctx.send_message(LoopIterationResult(has_next=True, current_item=items[0], current_index=0))
        else:
            await ctx.send_message(LoopIterationResult(has_next=False))


class ForeachNextExecutor(DeclarativeActionExecutor):
    """Advances to the next item in a foreach loop.

    This executor is triggered after the loop body completes.
    """

    def __init__(
        self,
        action_def: dict[str, Any],
        init_executor_id: str,
        *,
        id: str | None = None,
    ):
        """Initialize with reference to the init executor.

        Args:
            action_def: The Foreach action definition
            init_executor_id: ID of the corresponding ForeachInitExecutor
            id: Optional executor ID
        """
        super().__init__(action_def, id=id)
        self._init_executor_id = init_executor_id

    @handler
    async def handle_action(
        self,
        trigger: Any,
        ctx: WorkflowContext[LoopIterationResult],
    ) -> None:
        """Advance to next item and send result."""
        state = await self._ensure_state_initialized(ctx, trigger)

        loop_id = self._init_executor_id

        # Get loop state
        state_data = await state.get_state_data()
        loop_states = state_data.get(LOOP_STATE_KEY, {})
        loop_state = loop_states.get(loop_id)

        if not loop_state:
            # No loop state - shouldn't happen but handle gracefully
            await ctx.send_message(LoopIterationResult(has_next=False))
            return

        items = loop_state["items"]
        current_index = loop_state["index"] + 1

        if current_index < len(items):
            # Update loop state
            loop_state["index"] = current_index
            await state.set_state_data(state_data)

            # Set the iteration variable
            item_var = self._action_def.get("iteratorVariable") or self._action_def.get("item", "turn.item")
            index_var = self._action_def.get("indexVariable") or self._action_def.get("index")

            await state.set(item_var, items[current_index])
            if index_var:
                await state.set(index_var, current_index)

            await ctx.send_message(
                LoopIterationResult(has_next=True, current_item=items[current_index], current_index=current_index)
            )
        else:
            # Loop complete - clean up
            del loop_states[loop_id]
            await state.set_state_data(state_data)

            await ctx.send_message(LoopIterationResult(has_next=False))

    @handler
    async def handle_loop_control(
        self,
        control: LoopControl,
        ctx: WorkflowContext[LoopIterationResult],
    ) -> None:
        """Handle break/continue signals."""
        state = self._get_state(ctx.shared_state)

        if control.action == "break":
            # Clean up loop state and signal done
            state_data = await state.get_state_data()
            loop_states = state_data.get(LOOP_STATE_KEY, {})
            if self._init_executor_id in loop_states:
                del loop_states[self._init_executor_id]
                await state.set_state_data(state_data)

            await ctx.send_message(LoopIterationResult(has_next=False))

        elif control.action == "continue":
            # Just advance to next iteration
            await self.handle_action(ActionTrigger(), ctx)


class BreakLoopExecutor(DeclarativeActionExecutor):
    """Executor for BreakLoop action.

    Sends a LoopControl signal to break out of the enclosing loop.
    """

    def __init__(
        self,
        action_def: dict[str, Any],
        loop_next_executor_id: str,
        *,
        id: str | None = None,
    ):
        """Initialize with reference to the loop's next executor.

        Args:
            action_def: The action definition
            loop_next_executor_id: ID of the ForeachNextExecutor to signal
            id: Optional executor ID
        """
        super().__init__(action_def, id=id)
        self._loop_next_executor_id = loop_next_executor_id

    @handler
    async def handle_action(
        self,
        trigger: Any,
        ctx: WorkflowContext[LoopControl],
    ) -> None:
        """Send break signal to the loop."""
        await ctx.send_message(LoopControl(action="break"))


class ContinueLoopExecutor(DeclarativeActionExecutor):
    """Executor for ContinueLoop action.

    Sends a LoopControl signal to continue to next iteration.
    """

    def __init__(
        self,
        action_def: dict[str, Any],
        loop_next_executor_id: str,
        *,
        id: str | None = None,
    ):
        """Initialize with reference to the loop's next executor.

        Args:
            action_def: The action definition
            loop_next_executor_id: ID of the ForeachNextExecutor to signal
            id: Optional executor ID
        """
        super().__init__(action_def, id=id)
        self._loop_next_executor_id = loop_next_executor_id

    @handler
    async def handle_action(
        self,
        trigger: Any,
        ctx: WorkflowContext[LoopControl],
    ) -> None:
        """Send continue signal to the loop."""
        await ctx.send_message(LoopControl(action="continue"))


class EndWorkflowExecutor(DeclarativeActionExecutor):
    """Executor for EndWorkflow/EndDialog action.

    This executor simply doesn't send any message, causing the workflow
    to terminate at this point.
    """

    @handler
    async def handle_action(
        self,
        trigger: Any,
        ctx: WorkflowContext[ActionComplete],
    ) -> None:
        """End the workflow by not sending any continuation message."""
        # Don't send ActionComplete - workflow ends here
        pass


class EndConversationExecutor(DeclarativeActionExecutor):
    """Executor for EndConversation action."""

    @handler
    async def handle_action(
        self,
        trigger: Any,
        ctx: WorkflowContext[ActionComplete],
    ) -> None:
        """End the conversation."""
        # For now, just don't continue
        # In a full implementation, this would signal to close the conversation
        pass


# Passthrough executor for joining control flow branches
class JoinExecutor(DeclarativeActionExecutor):
    """Executor that joins multiple branches back together.

    Used after If/Switch to merge control flow back to a single path.
    """

    @handler
    async def handle_action(
        self,
        trigger: dict[str, Any] | str | ActionTrigger | ActionComplete | LoopIterationResult,
        ctx: WorkflowContext[ActionComplete],
    ) -> None:
        """Simply pass through to continue the workflow."""
        await ctx.send_message(ActionComplete())


# Mapping of control flow action kinds to executor classes
# Note: Most control flow is handled by the builder creating graph structure,
# these are the executors that are part of that structure
CONTROL_FLOW_EXECUTORS: dict[str, type[DeclarativeActionExecutor]] = {
    "EndWorkflow": EndWorkflowExecutor,
    "EndDialog": EndWorkflowExecutor,
    "EndConversation": EndConversationExecutor,
}
