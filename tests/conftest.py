"""Shared fixtures: an in-memory fake InfoLang client and a fake AgentCore context.

Every test runs fully offline against these fakes; there are no real AWS or
InfoLang calls. Tests needing real backends are marked ``@pytest.mark.live`` and
skipped unless ``--run-live`` is passed.
"""

from __future__ import annotations

from typing import Any

import pytest
from infolang import Chunk, RecallResult, RememberResult

from infolang_agentcore import InfoLangMemory


def make_chunk(
    id: str = "m1",
    text: str = "a stored fact",
    score: float | None = 0.92,
    tags: str | None = None,
) -> Chunk:
    return Chunk(id=id, text=text, score=score, tags=tags)


def make_recall(*chunks: Chunk, namespace: str | None = "default") -> RecallResult:
    return RecallResult(chunks=list(chunks), namespace=namespace)


class FakeInfoLang:
    """Records every call and returns canned results."""

    def __init__(
        self,
        *,
        recall_result: RecallResult | None = None,
        remember_result: RememberResult | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
    ) -> None:
        self.namespace = namespace
        self.workspace = workspace
        self.recall_result = recall_result or make_recall(make_chunk())
        self.remember_result = remember_result or RememberResult(id="m1")
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def kwargs_for(self, name: str) -> dict[str, Any]:
        for call_name, _args, kwargs in self.calls:
            if call_name == name:
                return kwargs
        raise AssertionError(f"{name} was not called; calls={[c[0] for c in self.calls]}")

    def args_for(self, name: str) -> tuple[Any, ...]:
        for call_name, args, _kwargs in self.calls:
            if call_name == name:
                return args
        raise AssertionError(f"{name} was not called; calls={[c[0] for c in self.calls]}")

    def recall(self, query: str, **kwargs: Any) -> RecallResult:
        self._record("recall", query, **kwargs)
        return self.recall_result

    def investigate(self, query: str, **kwargs: Any) -> RecallResult:
        self._record("investigate", query, **kwargs)
        return self.recall_result

    def remember(self, text: str, **kwargs: Any) -> RememberResult:
        self._record("remember", text, **kwargs)
        return self.remember_result

    def remember_batch(self, items: list[Any], **kwargs: Any) -> list[RememberResult]:
        self._record("remember_batch", items, **kwargs)
        return [self.remember_result for _ in items]

    def forget(self, memory_id: str, **kwargs: Any) -> None:
        self._record("forget", memory_id, **kwargs)


class FakeContext:
    """Stand-in for the AgentCore ``RequestContext`` (only ``session_id`` used)."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id


@pytest.fixture
def fake_client() -> FakeInfoLang:
    return FakeInfoLang()


@pytest.fixture
def memory(fake_client: FakeInfoLang) -> InfoLangMemory:
    # Per-actor bank derivation (namespace not fixed) is the default.
    return InfoLangMemory(fake_client, source="unit-test")  # type: ignore[arg-type]


# --- live marker plumbing ------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.live against real AWS/InfoLang",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live test; pass --run-live to enable")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
