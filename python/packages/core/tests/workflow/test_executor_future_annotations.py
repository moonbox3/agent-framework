N/A - applied via unified_diff

def test_executor_handler_future_annotations_workflow_context_one_arg() -> None:
    ex = _ExecFutureOneArg(id="e1")
    spec = ex._handler_specs[0]
    assert spec["message_type"] is int
    assert spec["output_types"] == [int]
    assert spec["workflow_output_types"] == []

def test_executor_handler_future_annotations_workflow_context_two_args() -> None:
    ex = _ExecFutureTwoArgs(id="e2")
    spec = ex._handler_specs[0]
    assert spec["message_type"] is int
    assert spec["output_types"] == [int]
    assert spec["workflow_output_types"] == [str]

def test_executor_handler_future_annotations_resolves_class_scoped_type() -> None:
    ex = _ExecClassScoped(id="e3")
    spec = ex._handler_specs[0]
    assert spec["output_types"] == [_ExecClassScoped.Inner]

def test_handler_explicit_types_allows_unannotated_ctx_and_sets_ctx_annotation_none() -> None:
    ex = _ExecExplicitTypesNoCtxAnn(id="e4")
    spec = ex._handler_specs[0]
    assert spec["message_type"] is int
    assert spec["output_types"] == [str]
    assert spec["ctx_annotation"] is None

def test_executor_future_annotations_unresolvable_ctx_annotation_raises_clear_error() -> None:
    with pytest.raises(ValueError, match=r"unresolved type annotation"):
        _ExecUnresolvable(id="e5")
