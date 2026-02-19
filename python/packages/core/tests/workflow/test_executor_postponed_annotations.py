# Copyright (c) Microsoft. All rights reserved.


def test_handler_registration_with_postponed_annotations_exact_sets(self) -> None:
    executor = FutureHandlerExecutor(id="future-handler")

    assert set(executor.input_types) == {str, dict[str, Any]}
    assert set(executor.output_types) == {TypeA, TypeB}
    assert set(executor.workflow_output_types) == {TypeB, TypeC}


def test_fail_closed_when_get_type_hints_fails_and_annotations_unresolvable(self, monkeypatch: Any) -> None:
    import agent_framework._workflows._executor as executor_mod

    def _boom(*_: Any, **__: Any) -> Any:
        raise TypeError("boom")

    monkeypatch.setattr(executor_mod.typing, "get_type_hints", _boom)

    class BadFutureExecutor(Executor):
        @handler
        async def handle_bad(self, message: "MissingType", ctx: WorkflowContext[TypeA, TypeB]) -> None:
            return None

    with pytest.raises(ValueError, match=r"message parameter annotation could not be resolved"):
        BadFutureExecutor(id="bad-future")
