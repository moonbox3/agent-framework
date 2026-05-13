# Copyright (c) Microsoft. All rights reserved.

"""Participant-oriented workflow output configuration helpers."""

from collections.abc import Sequence

from agent_framework import SupportsAgentRun
from agent_framework._workflows._agent_utils import resolve_agent_id
from agent_framework._workflows._executor import Executor

_ParticipantOutputSpecifier = str | SupportsAgentRun | Executor
_WorkflowExecutorSpecifier = Executor | SupportsAgentRun


def _resolve_participant_output_config(  # pyright: ignore[reportUnusedFunction]
    *,
    participants: Sequence[Executor],
    final_output_from: Sequence[_ParticipantOutputSpecifier] | None,
    intermediate_output_from: Sequence[_ParticipantOutputSpecifier] | None,
    default_final_output_from: Sequence[Executor] = (),
    extra_output_executors: Sequence[Executor] = (),
) -> tuple[list[_WorkflowExecutorSpecifier], list[_WorkflowExecutorSpecifier]]:
    """Resolve public participant output config into workflow executor config."""
    explicit_config = final_output_from is not None or intermediate_output_from is not None
    if explicit_config and not (final_output_from or intermediate_output_from):
        raise ValueError("final_output_from and intermediate_output_from cannot both be empty.")

    participants_by_id = {participant.id: participant for participant in participants}
    known_participants = sorted(participants_by_id)

    intermediate_designated = (
        _resolve_designated_participants(
            intermediate_output_from,
            kind="intermediate",
            participants_by_id=participants_by_id,
            known_participants=known_participants,
        )
        if intermediate_output_from is not None
        else []
    )

    if final_output_from is not None:
        output_designated = _resolve_designated_participants(
            final_output_from,
            kind="output",
            participants_by_id=participants_by_id,
            known_participants=known_participants,
        )
    else:
        # The caller-supplied default applies only to participants not explicitly designated as
        # intermediate. Without this subtraction, builders that pre-populate a default-final list
        # (Handoff defaults to all participants, Sequential defaults to the last) would force
        # an overlap error whenever a user passed `intermediate_output_from=[X]` for an X in
        # the default set, contradicting the public docstring contract.
        intermediate_ids = {participant.id for participant in intermediate_designated}
        output_designated = [
            participant for participant in default_final_output_from if participant.id not in intermediate_ids
        ]

    overlap = sorted(
        {participant.id for participant in output_designated}.intersection(
            participant.id for participant in intermediate_designated
        )
    )
    if overlap:
        raise ValueError(f"Participants cannot be both output and intermediate designated: {overlap}")

    output_executors: list[_WorkflowExecutorSpecifier] = [*extra_output_executors, *output_designated]
    intermediate_executors: list[_WorkflowExecutorSpecifier] = list(intermediate_designated)
    return output_executors, intermediate_executors


def _resolve_designated_participants(
    designations: Sequence[_ParticipantOutputSpecifier],
    *,
    kind: str,
    participants_by_id: dict[str, Executor],
    known_participants: Sequence[str],
) -> list[Executor]:
    resolved: list[Executor] = []
    seen: set[str] = set()
    for designation in designations:
        participant_id = _participant_id(designation)
        if participant_id in seen:
            raise ValueError(f"Duplicate {kind} participant '{participant_id}' in {kind}_participants.")
        seen.add(participant_id)
        try:
            resolved.append(participants_by_id[participant_id])
        except KeyError as exc:
            raise ValueError(
                f"Unknown {kind} participant '{participant_id}'. Known participants: {known_participants}"
            ) from exc
    return resolved


def _participant_id(participant: _ParticipantOutputSpecifier) -> str:
    if isinstance(participant, str):
        return participant
    if isinstance(participant, Executor):
        return participant.id
    return resolve_agent_id(participant)
