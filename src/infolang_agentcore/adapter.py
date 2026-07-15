"""Core adapter around the published InfoLang SDK.

This is the only module that touches the ``infolang`` package. The AgentCore
runtime wiring (:mod:`infolang_agentcore.runtime`) builds on it. It uses only
the public SDK surface (``from infolang import InfoLang`` and the documented
memory operations) and never reimplements HTTP nor imports engine internals.

The AgentCore value proposition is *durable* memory: because InfoLang is an
external store keyed by a stable **actor identity**, an agent's memory survives
AgentCore Runtime session teardown/recreation and full reprovisioning. The
ephemeral Runtime ``session_id`` is recorded as provenance, not used as the
durability key.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from infolang import InfoLang, RecallResult, RememberResult

DEFAULT_TOP_K = 5
DEFAULT_SCORE_FLOOR = 0.85
DEFAULT_SOURCE = "agentcore"
DEFAULT_NAMESPACE_TEMPLATE = "{actor_id}"

RememberItem = str | dict[str, Any]


def _join_tags(*tags: str | None) -> str | None:
    parts = [t for t in tags if t]
    return ",".join(parts) if parts else None


class InfoLangMemory:
    """Durable memory for AgentCore agents, backed by the InfoLang SDK.

    Parameters
    ----------
    client:
        A constructed :class:`infolang.InfoLang` client.
    namespace:
        A fixed bank to use for every actor. When set it overrides
        ``namespace_template`` and actors are distinguished by tags/provenance.
    namespace_template:
        Template used to derive a per-actor bank, e.g. ``"{actor_id}"`` (the
        default) or ``"tenant-acme/{actor_id}"``. Applied only when
        ``namespace`` is not set and an ``actor_id`` is available.
    source:
        Default ``source`` label attached to writes.
    top_k:
        Default number of chunks to request on recall/investigate.
    score_floor:
        Confidence floor (matches the SDK's 0.85 weak-match threshold).
    """

    def __init__(
        self,
        client: InfoLang,
        *,
        namespace: str | None = None,
        namespace_template: str = DEFAULT_NAMESPACE_TEMPLATE,
        source: str | None = DEFAULT_SOURCE,
        top_k: int = DEFAULT_TOP_K,
        score_floor: float = DEFAULT_SCORE_FLOOR,
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.namespace_template = namespace_template
        self.source = source
        self.top_k = top_k
        self.score_floor = score_floor

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        namespace: str | None = None,
        namespace_template: str = DEFAULT_NAMESPACE_TEMPLATE,
        workspace: str | None = None,
        source: str | None = DEFAULT_SOURCE,
        top_k: int = DEFAULT_TOP_K,
        score_floor: float = DEFAULT_SCORE_FLOOR,
    ) -> InfoLangMemory:
        """Build the adapter and its managed-cloud client in one call."""

        client = InfoLang.from_api_key(api_key, namespace=namespace, workspace=workspace)
        return cls(
            client,
            namespace=namespace,
            namespace_template=namespace_template,
            source=source,
            top_k=top_k,
            score_floor=score_floor,
        )

    # --- scoping ---------------------------------------------------------

    def namespace_for(self, actor_id: str | None) -> str | None:
        """Resolve the InfoLang bank for a given actor identity.

        * A fixed ``namespace`` always wins (single-bank, tag-scoped model).
        * Otherwise, when an ``actor_id`` is present, the per-actor
          ``namespace_template`` is applied (durable-per-user model).
        * Otherwise the client's own default namespace is used (``None``).
        """

        if self.namespace is not None:
            return self.namespace
        if actor_id:
            return self.namespace_template.format(actor_id=actor_id)
        return None

    def session(
        self,
        *,
        actor_id: str | None = None,
        session_id: str | None = None,
    ) -> MemorySession:
        """Create a :class:`MemorySession` scoped to an actor + runtime session."""

        return MemorySession(self, actor_id=actor_id, session_id=session_id)

    # --- reads -----------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RecallResult:
        return self.client.recall(
            query,
            namespace=namespace,
            top_k=top_k if top_k is not None else self.top_k,
            filters=filters,
        )

    def investigate(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
    ) -> RecallResult:
        return self.client.investigate(
            query,
            namespace_hint=namespace,
            top_k=top_k if top_k is not None else self.top_k,
        )

    # --- writes ----------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        tags: str | None = None,
        source: str | None = None,
    ) -> RememberResult:
        return self.client.remember(
            text,
            namespace=namespace,
            source=source or self.source,
            tags=tags,
        )

    def remember_batch(
        self,
        items: Sequence[RememberItem],
        *,
        namespace: str | None = None,
        source: str | None = None,
    ) -> list[RememberResult]:
        return self.client.remember_batch(
            list(items),
            namespace=namespace,
            source=source or self.source,
        )

    def forget(self, memory_id: str, *, namespace: str | None = None) -> None:
        self.client.forget(memory_id, namespace=namespace)

    # --- rendering -------------------------------------------------------

    def format_chunks(
        self,
        result: RecallResult,
        *,
        include_scores: bool = True,
        header: str = "Relevant memory from InfoLang",
        empty: str = "",
    ) -> str:
        """Render a recall result as an LLM-friendly context block.

        Returns ``empty`` (default: empty string) when there are no chunks so
        callers can cheaply test whether any context was found.
        """

        if not result.chunks:
            return empty
        lines = [f"{header} ({len(result.chunks)} result(s)):"]
        for i, chunk in enumerate(result.chunks, start=1):
            prefix = f"[{i}]"
            if include_scores and chunk.score is not None:
                prefix = f"[{i}] (score {chunk.score:.2f})"
            lines.append(f"{prefix} {chunk.text}")
        if result.weak:
            lines.append(
                "(Weak match: top score is below the "
                f"{self.score_floor:.2f} confidence floor \u2014 treat as a hint.)"
            )
        return "\n".join(lines)


class MemorySession:
    """Memory scoped to one actor identity and one runtime session.

    ``recall``/``remember`` resolve the durable bank from the actor identity so
    the same actor sees the same memory across Runtime session recreation. The
    ephemeral ``session_id`` is attached to writes as provenance.
    """

    def __init__(
        self,
        memory: InfoLangMemory,
        *,
        actor_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.memory = memory
        self.actor_id = actor_id
        self.session_id = session_id
        self.namespace = memory.namespace_for(actor_id)

    def _provenance_tags(self, extra: str | None = None) -> str | None:
        session_tag = f"session:{self.session_id}" if self.session_id else None
        actor_tag = f"actor:{self.actor_id}" if self.actor_id else None
        return _join_tags(actor_tag, session_tag, extra)

    def recall(self, query: str, *, top_k: int | None = None) -> RecallResult:
        return self.memory.recall(query, namespace=self.namespace, top_k=top_k)

    def investigate(self, query: str, *, top_k: int | None = None) -> RecallResult:
        return self.memory.investigate(query, namespace=self.namespace, top_k=top_k)

    def recall_context(self, query: str, *, top_k: int | None = None) -> str:
        """Recall and render context in one call (empty string when nothing)."""

        return self.memory.format_chunks(self.recall(query, top_k=top_k))

    def remember(self, text: str, *, tags: str | None = None) -> RememberResult:
        return self.memory.remember(
            text, namespace=self.namespace, tags=self._provenance_tags(tags)
        )

    def remember_turn(self, prompt: str, answer: str, *, tags: str | None = None) -> RememberResult:
        """Store a user/assistant exchange for future recall."""

        text = f"User: {prompt}\nAssistant: {answer}"
        return self.remember(text, tags=tags)

    def forget(self, memory_id: str) -> None:
        self.memory.forget(memory_id, namespace=self.namespace)


def coerce_memory(
    memory: InfoLangMemory | InfoLang,
    *,
    namespace: str | None = None,
    source: str | None = DEFAULT_SOURCE,
    top_k: int = DEFAULT_TOP_K,
) -> InfoLangMemory:
    """Accept either an :class:`InfoLangMemory` or a raw client and normalise."""

    if isinstance(memory, InfoLangMemory):
        return memory
    return InfoLangMemory(memory, namespace=namespace, source=source, top_k=top_k)
