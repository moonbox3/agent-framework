            MyExecutor(id="ex3")


# NOTE: The tests below must be module-level functions (no `self`) to avoid pytest
# treating `self` as a fixture/parameter during collection.

def test_handler_future_annotations_workflow_context_generics_resolve() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
            pass

    # Prove postponed-annotations behavior is exercised
    assert isinstance(MyExecutor.example.__annotations__["ctx"], str)
    assert "WorkflowContext" in MyExecutor.example.__annotations__["ctx"]

    ex = MyExecutor(id="ex")
    assert ex.output_types == [TypeA]
    assert ex.workflow_output_types == [TypeB]


def test_handler_future_annotations_with_quoted_forward_refs() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext["TypeA", "TypeB"]) -> None:
            pass

    assert isinstance(MyExecutor.example.__annotations__["ctx"], str)

    ex = MyExecutor(id="ex2")
    assert ex.output_types == [TypeA]
    assert ex.workflow_output_types == [TypeB]


def test_skip_message_annotation_with_unannotated_ctx_does_not_resolve_hints() -> None:
    with pytest.raises(ValueError, match=r"with explicit type parameters must specify 'input' type"):

        class ExplicitModeMissingInput(Executor):
            @handler(output=int)
            async def example(self, input: str, ctx) -> None:
                pass

        ExplicitModeMissingInput(id="bad")


def test_type_hint_resolution_failure_raises_targeted_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[TypeA]) -> None:
            pass

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(executor_module.typing, "get_type_hints", _boom)

def test_handler_future_annotations_workflow_context_generics_resolve() -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[TypeA, TypeB]) -> None:
            pass

    assert isinstance(MyExecutor.example.__annotations__["ctx"], str)
    assert "WorkflowContext" in MyExecutor.example.__annotations__["ctx"]

    ex = MyExecutor(id="ex")
    assert ex.output_types == [TypeA]
    assert ex.workflow_output_types == [TypeB]

def test_type_hint_resolution_failure_raises_targeted_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class MyExecutor(Executor):
        @handler
        async def example(self, input: str, ctx: WorkflowContext[TypeA]) -> None:
            pass

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(executor_module.typing, "get_type_hints", _boom)

    with pytest.raises(ValueError, match=r"annotations could not be resolved"):
        MyExecutor(id="ex3")
