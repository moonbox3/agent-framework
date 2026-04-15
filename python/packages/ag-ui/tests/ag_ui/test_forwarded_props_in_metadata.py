# Copyright (c) Microsoft. All rights reserved.

"""Tests for forwarded_props inclusion in AG-UI session metadata."""

from typing import Any

import pytest

from agent_framework_ag_ui._agent_run import AG_UI_INTERNAL_METADATA_KEYS, _build_safe_metadata


class TestForwardedPropsInSessionMetadata:
    """Verify that forwarded_props is surfaced in session metadata and filtered from LLM metadata."""

    def test_forwarded_props_in_internal_metadata_keys(self):
        """forwarded_props is listed in AG_UI_INTERNAL_METADATA_KEYS to prevent LLM leakage."""
        assert "forwarded_props" in AG_UI_INTERNAL_METADATA_KEYS

    def test_base_metadata_includes_forwarded_props(self):
        """forwarded_props from input_data is added to base_metadata when present."""
        input_data: dict[str, Any] = {
            "thread_id": "t1",
            "run_id": "r1",
            "messages": [],
            "forwarded_props": {
                "custom_flag": True,
                "user_preference": "persist_to_db",
            },
        }

        thread_id = input_data["thread_id"]
        run_id = input_data["run_id"]

        base_metadata: dict[str, Any] = {
            "ag_ui_thread_id": thread_id,
            "ag_ui_run_id": run_id,
        }
        forwarded_props = input_data.get("forwarded_props") or input_data.get("forwardedProps")
        if forwarded_props:
            base_metadata["forwarded_props"] = forwarded_props

        assert "forwarded_props" in base_metadata
        assert base_metadata["forwarded_props"] == {"custom_flag": True, "user_preference": "persist_to_db"}

    def test_base_metadata_excludes_forwarded_props_when_absent(self):
        """forwarded_props is not added to base_metadata when not present in input_data."""
        input_data: dict[str, Any] = {
            "thread_id": "t1",
            "run_id": "r1",
            "messages": [],
        }

        base_metadata: dict[str, Any] = {
            "ag_ui_thread_id": input_data["thread_id"],
            "ag_ui_run_id": input_data["run_id"],
        }
        forwarded_props = input_data.get("forwarded_props") or input_data.get("forwardedProps")
        if forwarded_props:
            base_metadata["forwarded_props"] = forwarded_props

        assert "forwarded_props" not in base_metadata

    def test_camel_case_forwarded_props_also_accepted(self):
        """forwardedProps (camelCase) is accepted as an alternative key."""
        input_data: dict[str, Any] = {
            "thread_id": "t1",
            "run_id": "r1",
            "messages": [],
            "forwardedProps": {"source": "copilotkit"},
        }

        forwarded_props = input_data.get("forwarded_props") or input_data.get("forwardedProps")
        base_metadata: dict[str, Any] = {
            "ag_ui_thread_id": input_data["thread_id"],
            "ag_ui_run_id": input_data["run_id"],
        }
        if forwarded_props:
            base_metadata["forwarded_props"] = forwarded_props

        assert base_metadata["forwarded_props"] == {"source": "copilotkit"}

    def test_forwarded_props_filtered_from_client_metadata(self):
        """forwarded_props is filtered out when building LLM-bound client metadata."""
        session_metadata: dict[str, Any] = {
            "ag_ui_thread_id": "t1",
            "ag_ui_run_id": "r1",
            "forwarded_props": {"custom_flag": True},
        }

        client_metadata = {k: v for k, v in session_metadata.items() if k not in AG_UI_INTERNAL_METADATA_KEYS}

        assert "forwarded_props" not in client_metadata
        assert "ag_ui_thread_id" not in client_metadata

    def test_forwarded_props_safe_metadata_serializes(self):
        """forwarded_props dict is serialized by _build_safe_metadata."""
        metadata: dict[str, Any] = {
            "ag_ui_thread_id": "t1",
            "forwarded_props": {"flag": True, "source": "frontend"},
        }
        result = _build_safe_metadata(metadata)
        assert "forwarded_props" in result
        assert isinstance(result["forwarded_props"], str)
