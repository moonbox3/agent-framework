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
    output_participants: Sequence[_ParticipantOutputSpecifier] | None,
    intermediate_participants: Sequence[_ParticipantOutputSpecifier] | None,
    default_output_participants: Sequence[Executor] = (),
    extra_output_executors: Sequence[Executor] = (),
) -> tuple[list[_WorkflowExecutorSpecifier], list[_WorkflowExecutorSpecifier]]:
    """Resolve public participant output config into workflow executor config."""
    explicit_config = output_participants is not None or intermediate_participants is not None
    if explicit_config and not (output_participants or intermediate_participants):
        raise ValueError("output_participants and intermediate_participants cannot both be empty.")

    participants_by_id = {participant.id: participant for participant in participants}
    known_participants = sorted(participants_by_id)

    output_designated = (
        _resolve_designated_participants(
            output_participants,
            kind="output",
            participants_by_id=participants_by_id,
            known_participants=known_participants,
        )
        if output_participants is not None
        else list(default_output_participants)
    )
    intermediate_designated = (
        _resolve_designated_participants(
            intermediate_participants,
            kind="intermediate",
            participants_by_id=participants_by_id,
            known_participants=known_participants,
        )
        if intermediate_participants is not None
        else []
    )

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
