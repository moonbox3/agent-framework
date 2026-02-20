def test_handler_future_annotations_resolves_message_and_workflow_context_generics() -> None:
    class MyOut:
        pass

    class MyWOut:
        pass

    class MyExec(Executor):
        @handler
        async def handle(self, message: int, ctx: WorkflowContext[MyOut, MyWOut]) -> None:
            return None

    ex = MyExec(id="ex")

    assert ex.input_types == [int]
    assert ex.output_types == [MyOut]
    assert ex.workflow_output_types == [MyWOut]


def test_handler_future_annotations_resolves_nested_local_types_in_workflow_context() -> None:
    class MyExec(Executor):
        class NestedOut:
            pass

        class NestedWOut:
            pass

        @handler
        async def handle(self, message: str, ctx: WorkflowContext[NestedOut, NestedWOut]) -> None:
            return None

    ex = MyExec(id="ex")

    assert ex.input_types == [str]
    assert ex.output_types == [MyExec.NestedOut]
    assert ex.workflow_output_types == [MyExec.NestedWOut]


def test_handler_future_annotations_runtime_unavailable_ref_is_actionable() -> None:
    class MyExec(Executor):
        @handler
        async def handle(self, message: int, ctx: WorkflowContext["ZoneInfo"]) -> None:
            return None

    with pytest.raises(ValueError, match=r"annotations that could not be resolved"):
        MyExec(id="ex")
