"""Tests for the InfoLangMemory adapter and MemorySession."""

from __future__ import annotations

import pytest
from infolang import RememberResult

from infolang_agentcore import (
    DEFAULT_SCORE_FLOOR,
    DEFAULT_SOURCE,
    DEFAULT_TOP_K,
    InfoLangMemory,
    MemorySession,
    coerce_memory,
)

from .conftest import FakeInfoLang, make_chunk, make_recall

# --- namespace resolution ------------------------------------------------


def test_namespace_for_per_actor_default(memory: InfoLangMemory) -> None:
    assert memory.namespace_for("alice") == "alice"


def test_namespace_for_template() -> None:
    mem = InfoLangMemory(
        FakeInfoLang(),  # type: ignore[arg-type]
        namespace_template="tenant-acme/{actor_id}",
    )
    assert mem.namespace_for("bob") == "tenant-acme/bob"


def test_namespace_for_fixed_wins() -> None:
    mem = InfoLangMemory(FakeInfoLang(), namespace="shared")  # type: ignore[arg-type]
    assert mem.namespace_for("alice") == "shared"


def test_namespace_for_none_when_no_actor(memory: InfoLangMemory) -> None:
    assert memory.namespace_for(None) is None


# --- low-level ops ------------------------------------------------------


def test_recall_uses_top_k_default(memory: InfoLangMemory) -> None:
    memory.recall("q", namespace="ns")
    kwargs = memory.client.kwargs_for("recall")  # type: ignore[attr-defined]
    assert kwargs["namespace"] == "ns"
    assert kwargs["top_k"] == DEFAULT_TOP_K


def test_recall_top_k_override(memory: InfoLangMemory) -> None:
    memory.recall("q", namespace="ns", top_k=2)
    assert memory.client.kwargs_for("recall")["top_k"] == 2  # type: ignore[attr-defined]


def test_investigate(memory: InfoLangMemory) -> None:
    memory.investigate("q", namespace="ns", top_k=8)
    kwargs = memory.client.kwargs_for("investigate")  # type: ignore[attr-defined]
    assert kwargs["namespace_hint"] == "ns"
    assert kwargs["top_k"] == 8


def test_remember_default_source(memory: InfoLangMemory) -> None:
    memory.remember("fact", namespace="ns", tags="a,b")
    kwargs = memory.client.kwargs_for("remember")  # type: ignore[attr-defined]
    assert kwargs["source"] == "unit-test"
    assert kwargs["tags"] == "a,b"
    assert kwargs["namespace"] == "ns"


def test_remember_source_override(memory: InfoLangMemory) -> None:
    memory.remember("fact", source="over")
    assert memory.client.kwargs_for("remember")["source"] == "over"  # type: ignore[attr-defined]


def test_remember_batch(memory: InfoLangMemory) -> None:
    results = memory.remember_batch(["a", "b"], namespace="ns")
    assert len(results) == 2
    assert memory.client.kwargs_for("remember_batch")["namespace"] == "ns"  # type: ignore[attr-defined]


def test_forget(memory: InfoLangMemory) -> None:
    memory.forget("m3", namespace="ns")
    assert memory.client.kwargs_for("forget")["namespace"] == "ns"  # type: ignore[attr-defined]


def test_defaults() -> None:
    mem = InfoLangMemory(FakeInfoLang())  # type: ignore[arg-type]
    assert mem.source == DEFAULT_SOURCE
    assert mem.top_k == DEFAULT_TOP_K
    assert mem.score_floor == DEFAULT_SCORE_FLOOR


# --- formatting ----------------------------------------------------------


def test_format_chunks_empty_default(memory: InfoLangMemory) -> None:
    assert memory.format_chunks(make_recall()) == ""


def test_format_chunks_with_scores(memory: InfoLangMemory) -> None:
    out = memory.format_chunks(make_recall(make_chunk(text="hi", score=0.9)))
    assert "[1] (score 0.90) hi" in out


def test_format_chunks_no_scores(memory: InfoLangMemory) -> None:
    out = memory.format_chunks(
        make_recall(make_chunk(text="hi", score=0.9)), include_scores=False
    )
    assert "[1] hi" in out
    assert "score" not in out


def test_format_chunks_missing_score(memory: InfoLangMemory) -> None:
    out = memory.format_chunks(make_recall(make_chunk(text="hi", score=None)))
    assert "[1] hi" in out


def test_format_chunks_weak(memory: InfoLangMemory) -> None:
    assert "Weak match" in memory.format_chunks(make_recall(make_chunk(score=0.4)))


def test_format_chunks_strong(memory: InfoLangMemory) -> None:
    assert "Weak match" not in memory.format_chunks(make_recall(make_chunk(score=0.99)))


# --- from_api_key + coerce ----------------------------------------------


def test_from_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_api_key(api_key: str, **kwargs: object) -> FakeInfoLang:
        captured["api_key"] = api_key
        captured.update(kwargs)
        return FakeInfoLang()

    monkeypatch.setattr(
        "infolang_agentcore.adapter.InfoLang.from_api_key", fake_from_api_key
    )
    mem = InfoLangMemory.from_api_key(
        "il_live_x", namespace="ns", workspace="ws", top_k=3
    )
    assert captured == {"api_key": "il_live_x", "namespace": "ns", "workspace": "ws"}
    assert mem.namespace == "ns"
    assert mem.top_k == 3


def test_coerce_passthrough(memory: InfoLangMemory) -> None:
    assert coerce_memory(memory) is memory


def test_coerce_wraps_client() -> None:
    mem = coerce_memory(FakeInfoLang(), namespace="ns")  # type: ignore[arg-type]
    assert isinstance(mem, InfoLangMemory)
    assert mem.namespace == "ns"


# --- MemorySession -------------------------------------------------------


def test_session_resolves_namespace(memory: InfoLangMemory) -> None:
    session = memory.session(actor_id="alice", session_id="s1")
    assert session.namespace == "alice"
    assert isinstance(session, MemorySession)


def test_session_recall_uses_namespace(memory: InfoLangMemory) -> None:
    session = memory.session(actor_id="alice", session_id="s1")
    session.recall("q")
    assert memory.client.kwargs_for("recall")["namespace"] == "alice"  # type: ignore[attr-defined]


def test_session_investigate(memory: InfoLangMemory) -> None:
    memory.session(actor_id="alice").investigate("q")
    assert memory.client.kwargs_for("investigate")["namespace_hint"] == "alice"  # type: ignore[attr-defined]


def test_session_recall_context(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall(make_chunk(text="berlin"))  # type: ignore[attr-defined]
    out = memory.session(actor_id="alice").recall_context("where?")
    assert "berlin" in out


def test_session_remember_tags_provenance(memory: InfoLangMemory) -> None:
    session = memory.session(actor_id="alice", session_id="s1")
    session.remember("a fact")
    tags = memory.client.kwargs_for("remember")["tags"]  # type: ignore[attr-defined]
    assert "actor:alice" in tags
    assert "session:s1" in tags


def test_session_remember_extra_tags(memory: InfoLangMemory) -> None:
    session = memory.session(actor_id="alice", session_id="s1")
    session.remember("a fact", tags="topic:billing")
    tags = memory.client.kwargs_for("remember")["tags"]  # type: ignore[attr-defined]
    assert "topic:billing" in tags


def test_session_remember_no_ids_no_tags() -> None:
    mem = InfoLangMemory(FakeInfoLang())  # type: ignore[arg-type]
    mem.session().remember("a fact")
    assert mem.client.kwargs_for("remember")["tags"] is None  # type: ignore[attr-defined]


def test_session_remember_turn(memory: InfoLangMemory) -> None:
    session = memory.session(actor_id="alice", session_id="s1")
    result = session.remember_turn("what is x?", "x is y")
    assert isinstance(result, RememberResult)
    text = memory.client.args_for("remember")[0]  # type: ignore[attr-defined]
    assert "User: what is x?" in text
    assert "Assistant: x is y" in text


def test_session_forget(memory: InfoLangMemory) -> None:
    session = memory.session(actor_id="alice")
    session.forget("m9")
    assert memory.client.kwargs_for("forget")["namespace"] == "alice"  # type: ignore[attr-defined]
