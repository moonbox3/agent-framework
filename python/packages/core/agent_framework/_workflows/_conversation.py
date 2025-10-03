# Copyright (c) Microsoft. All rights reserved.

"""Workflow-level conversation orchestration primitives.

This module introduces ConversationSession - the authoritative transcript for a
workflow run - alongside helper classes for coordinating AgentThread bindings,
view materialization, and resumable handles. It is intentionally lightweight so
it can operate entirely in-process while remaining extensible to external
stores.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, MutableMapping, Sequence

from .._threads import AgentThread
from .._types import ChatMessage

if TYPE_CHECKING:
    from ._shared_state import SharedState


@dataclass(slots=True)
class ConversationHandle:
    """Opaque identifier for resuming a workflow conversation."""

    session_id: str
    revision: int | None = None


@dataclass(slots=True)
class ThreadBinding:
    """Tracks linkage between a workflow participant and an AgentThread."""

    participant_id: str
    thread: AgentThread
    last_applied_index: int = 0


@dataclass(slots=True)
class ConversationSession:
    """Authoritative transcript and per-participant metadata for a workflow run."""

    session_id: str
    transcript: list[ChatMessage] = field(default_factory=list)
    thread_bindings: dict[str, ThreadBinding] = field(default_factory=dict)
    participant_profiles: MutableMapping[str, MutableMapping[str, Any]] = field(default_factory=dict)
    attachments: MutableMapping[str, Any] = field(default_factory=dict)
    revision: int = 0


@dataclass(slots=True)
class ConversationView:
    """Materialized view for a participant invocation."""

    messages: list[ChatMessage]
    delta_since_binding: list[ChatMessage]
    thread: AgentThread
    binding: ThreadBinding
    revision: int


class ConversationStore(ABC):
    """Persistence abstraction for ConversationSession objects."""

    @abstractmethod
    async def load(self, handle: ConversationHandle) -> ConversationSession | None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def save(
        self, handle: ConversationHandle, session: ConversationSession
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def delete(self, handle: ConversationHandle) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):
    """Non-persistent store suitable for tests and local workflows."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = asyncio.Lock()

    async def load(self, handle: ConversationHandle) -> ConversationSession | None:
        async with self._lock:
            session = self._sessions.get(handle.session_id)
            if session is None:
                return None
            # Return a shallow copy that can be mutated independently
            return ConversationSession(
                session_id=session.session_id,
                transcript=list(session.transcript),
                thread_bindings=dict(session.thread_bindings),
                participant_profiles=dict(session.participant_profiles),
                attachments=dict(session.attachments),
                revision=session.revision,
            )

    async def save(self, handle: ConversationHandle, session: ConversationSession) -> None:
        async with self._lock:
            self._sessions[handle.session_id] = ConversationSession(
                session_id=session.session_id,
                transcript=list(session.transcript),
                thread_bindings=dict(session.thread_bindings),
                participant_profiles=dict(session.participant_profiles),
                attachments=dict(session.attachments),
                revision=session.revision,
            )

    async def delete(self, handle: ConversationHandle) -> None:
        async with self._lock:
            self._sessions.pop(handle.session_id, None)


_GLOBAL_STORE = InMemoryConversationStore()


class ConversationManager:
    """Workflow-scoped controller for ConversationSession lifecycle."""

    _SHARED_STATE_KEY = "__agent_framework_conversation_manager__"

    def __init__(self, store: ConversationStore | None = None) -> None:
        self._store = store or _GLOBAL_STORE
        self._session: ConversationSession | None = None
        self._handle: ConversationHandle | None = None
        self._lock = asyncio.Lock()

    @property
    def handle(self) -> ConversationHandle:
        if self._handle is None:
            raise RuntimeError("Conversation session has not been initialized.")
        return self._handle

    @property
    def session(self) -> ConversationSession:
        if self._session is None:
            raise RuntimeError("Conversation session has not been initialized.")
        return self._session

    @classmethod
    async def ensure_on_shared_state(
        cls,
        shared_state: "SharedState",
        *,
        store: ConversationStore | None = None,
    ) -> "ConversationManager":
        async with shared_state.hold():
            if await shared_state.has_within_hold(cls._SHARED_STATE_KEY):
                manager = await shared_state.get_within_hold(cls._SHARED_STATE_KEY)
                if isinstance(manager, ConversationManager):
                    return manager
            manager = cls(store=store)
            await shared_state.set_within_hold(cls._SHARED_STATE_KEY, manager)
            return manager

    @classmethod
    async def maybe_get_from_shared_state(
        cls,
        shared_state: "SharedState",
    ) -> "ConversationManager | None":
        async with shared_state.hold():
            if await shared_state.has_within_hold(cls._SHARED_STATE_KEY):
                manager = await shared_state.get_within_hold(cls._SHARED_STATE_KEY)
                if isinstance(manager, ConversationManager):
                    return manager
        return None

    async def ensure_session(
        self,
        *,
        initial_messages: Sequence[ChatMessage] | None = None,
        handle: ConversationHandle | None = None,
    ) -> ConversationSession:
        async with self._lock:
            if self._session is None:
                if handle is not None:
                    loaded = await self._store.load(handle)
                    if loaded is not None:
                        self._session = loaded
                        self._handle = ConversationHandle(handle.session_id, revision=loaded.revision)
                    else:
                        self._session = ConversationSession(session_id=handle.session_id)
                        self._handle = ConversationHandle(handle.session_id, revision=0)
                else:
                    session_id = uuid.uuid4().hex
                    self._session = ConversationSession(session_id=session_id)
                    self._handle = ConversationHandle(session_id=session_id, revision=0)

            if initial_messages is not None:
                await self.replace_transcript(initial_messages)

            return self.session

    async def replace_transcript(self, messages: Sequence[ChatMessage]) -> None:
        """Replace transcript with provided messages, bumping revision if changed."""
        async with self._lock:
            session = self.session
            new_messages = list(messages)
            if session.transcript == new_messages:
                return
            session.transcript = new_messages
            session.revision += 1
            for binding in session.thread_bindings.values():
                binding.last_applied_index = min(binding.last_applied_index, len(session.transcript))
            if self._handle is not None:
                self._handle.revision = session.revision
            await self._store.save(self.handle, session)

    async def append_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Append messages to transcript and advance revision."""
        if not messages:
            return
        async with self._lock:
            session = self.session
            session.transcript.extend(messages)
            session.revision += 1
            for binding in session.thread_bindings.values():
                binding.last_applied_index = len(session.transcript)
            if self._handle is not None:
                self._handle.revision = session.revision
            await self._store.save(self.handle, session)

    async def prepare_invocation(
        self,
        participant_id: str,
        *,
        agent,
        prebound_thread: AgentThread | None = None,
    ) -> ConversationView:
        async with self._lock:
            session = self.session
            binding = session.thread_bindings.get(participant_id)
            if binding is None:
                thread = prebound_thread or agent.get_new_thread()
                binding = ThreadBinding(participant_id=participant_id, thread=thread, last_applied_index=0)
                session.thread_bindings[participant_id] = binding
            else:
                thread = binding.thread

            transcript = list(session.transcript)
            delta: list[ChatMessage] = []
            if binding.last_applied_index < len(transcript):
                delta = transcript[binding.last_applied_index :]
                if delta:
                    await thread.on_new_messages(delta)
                    binding.last_applied_index = len(transcript)

            return ConversationView(
                messages=transcript,
                delta_since_binding=list(delta),
                thread=thread,
                binding=binding,
                revision=session.revision,
            )

    async def commit_agent_response(
        self,
        binding: ThreadBinding,
        response_messages: Sequence[ChatMessage],
    ) -> None:
        """Persist agent response messages and advance binding state."""
        async with self._lock:
            session = self.session
            if response_messages:
                session.transcript.extend(response_messages)
                session.revision += 1
                if self._handle is not None:
                    self._handle.revision = session.revision
                await self._store.save(self.handle, session)
            binding.last_applied_index = len(session.transcript)

    def snapshot_transcript(self) -> list[ChatMessage]:
        return list(self.session.transcript)


@dataclass(slots=True)
class ConversationSnapshot:
    """Serializable wrapper for exposing transcript and handle to callers."""

    messages: list[ChatMessage]
    handle: ConversationHandle
