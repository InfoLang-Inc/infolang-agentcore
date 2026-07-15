"""Tests for AgentCore runtime extraction helpers and the memory_entrypoint decorator."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from infolang_agentcore import (
    InfoLangMemory,
    extract_actor_id,
    extract_answer,
    extract_prompt,
    extract_session_id,
    memory_entrypoint,
    session_for,
)

from .conftest import FakeContext, FakeInfoLang, make_chunk, make_recall

# --- extract_session_id --------------------------------------------------


def test_session_id_from_context() -> None:
    assert extract_session_id(FakeContext("sess-1")) == "sess-1"


def test_session_id_none_context_no_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "infolang_agentcore.runtime._session_id_from_runtime", lambda: None
    )
    assert extract_session_id(None) is None


def test_session_id_falls_back_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "infolang_agentcore.runtime._session_id_from_runtime", lambda: "runtime-sess"
    )
    assert extract_session_id(FakeContext(None)) == "runtime-sess"


def test_session_id_from_runtime_reads_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import bedrock_agentcore

    monkeypatch.setattr(
        bedrock_agentcore.BedrockAgentCoreContext,
        "get_session_id",
        staticmethod(lambda: "sdk-sess"),
    )
    assert extract_session_id(None) == "sdk-sess"


def test_session_id_from_runtime_handles_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import bedrock_agentcore

    def boom() -> str:
        raise RuntimeError("no runtime context")

    monkeypatch.setattr(
        bedrock_agentcore.BedrockAgentCoreContext, "get_session_id", staticmethod(boom)
    )
    assert extract_session_id(None) is None


# --- extract_actor_id ----------------------------------------------------


def test_actor_id_from_payload() -> None:
    assert extract_actor_id({"actor_id": "alice"}, FakeContext("s1")) == "alice"


def test_actor_id_custom_key() -> None:
    assert extract_actor_id({"user": "bob"}, actor_key="user") == "bob"


def test_actor_id_falls_back_to_session() -> None:
    assert extract_actor_id({}, FakeContext("s1")) == "s1"


def test_actor_id_none() -> None:
    assert extract_actor_id({}, FakeContext(None)) is None


def test_actor_id_non_dict_payload() -> None:
    assert extract_actor_id("just a string", FakeContext("s1")) == "s1"


# --- extract_prompt ------------------------------------------------------


@pytest.mark.parametrize("key", ["prompt", "inputText", "input", "query", "text", "message"])
def test_prompt_from_keys(key: str) -> None:
    assert extract_prompt({key: "hello"}) == "hello"


def test_prompt_from_string() -> None:
    assert extract_prompt("hi there") == "hi there"


def test_prompt_empty_string_is_none() -> None:
    assert extract_prompt("") is None


def test_prompt_missing() -> None:
    assert extract_prompt({"unrelated": "x"}) is None


def test_prompt_non_str_value() -> None:
    assert extract_prompt({"prompt": 123}) is None


def test_prompt_non_str_non_dict_payload() -> None:
    assert extract_prompt(None) is None
    assert extract_prompt(42) is None


def test_prompt_priority() -> None:
    assert extract_prompt({"prompt": "a", "query": "b"}) == "a"


# --- extract_answer ------------------------------------------------------


@pytest.mark.parametrize("key", ["result", "output", "response", "completion", "answer"])
def test_answer_from_keys(key: str) -> None:
    assert extract_answer({key: "the answer"}) == "the answer"


def test_answer_from_string() -> None:
    assert extract_answer("done") == "done"


def test_answer_missing() -> None:
    assert extract_answer({"nope": 1}) is None


def test_answer_empty_string_is_none() -> None:
    assert extract_answer("") is None


def test_answer_non_str_non_dict() -> None:
    assert extract_answer(None) is None
    assert extract_answer(123) is None


# --- session_for ---------------------------------------------------------


def test_session_for(memory: InfoLangMemory) -> None:
    session = session_for(memory, {"actor_id": "alice"}, FakeContext("s1"))
    assert session.actor_id == "alice"
    assert session.session_id == "s1"
    assert session.namespace == "alice"


# --- memory_entrypoint decorator ----------------------------------------


def _handler_calls() -> tuple[Any, list[Any]]:
    seen: list[Any] = []

    def handler(payload: Any, context: Any) -> dict[str, str]:
        seen.append(payload)
        return {"result": "the answer"}

    return handler, seen


def test_decorator_preserves_signature(memory: InfoLangMemory) -> None:
    handler, _ = _handler_calls()
    wrapped = memory_entrypoint(memory)(handler)
    params = list(inspect.signature(wrapped).parameters)
    assert params == ["payload", "context"]


def test_decorator_copies_name(memory: InfoLangMemory) -> None:
    def my_handler(payload: Any, context: Any) -> str:
        return "x"

    wrapped = memory_entrypoint(memory)(my_handler)
    assert wrapped.__name__ == "my_handler"


def test_decorator_injects_context(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall(make_chunk(text="berlin"))  # type: ignore[attr-defined]
    handler, seen = _handler_calls()
    wrapped = memory_entrypoint(memory)(handler)
    wrapped({"prompt": "where?", "actor_id": "alice"}, FakeContext("s1"))
    assert "berlin" in seen[0]["infolang_context"]


def test_decorator_recall_uses_actor_namespace(memory: InfoLangMemory) -> None:
    handler, _ = _handler_calls()
    memory_entrypoint(memory)(handler)({"prompt": "q", "actor_id": "alice"}, FakeContext("s1"))
    assert memory.client.kwargs_for("recall")["namespace"] == "alice"  # type: ignore[attr-defined]


def test_decorator_remembers_exchange(memory: InfoLangMemory) -> None:
    handler, _ = _handler_calls()
    memory_entrypoint(memory)(handler)({"prompt": "what?", "actor_id": "alice"}, FakeContext("s1"))
    text = memory.client.args_for("remember")[0]  # type: ignore[attr-defined]
    assert "User: what?" in text
    assert "Assistant: the answer" in text


def test_decorator_no_prompt_skips_all(memory: InfoLangMemory) -> None:
    handler, seen = _handler_calls()
    memory_entrypoint(memory)(handler)({"actor_id": "alice"}, FakeContext("s1"))
    assert memory.client.calls == []  # type: ignore[attr-defined]
    assert seen[0] == {"actor_id": "alice"}


def test_decorator_no_answer_skips_remember(memory: InfoLangMemory) -> None:
    def handler(payload: Any, context: Any) -> dict[str, Any]:
        return {"unrelated": 1}

    memory_entrypoint(memory)(handler)({"prompt": "q", "actor_id": "a"}, FakeContext("s1"))
    assert "remember" not in [c[0] for c in memory.client.calls]  # type: ignore[attr-defined]


def test_decorator_recall_disabled(memory: InfoLangMemory) -> None:
    handler, seen = _handler_calls()
    memory_entrypoint(memory, recall=False)(handler)(
        {"prompt": "q", "actor_id": "a"}, FakeContext("s1")
    )
    assert "recall" not in [c[0] for c in memory.client.calls]  # type: ignore[attr-defined]
    assert "infolang_context" not in seen[0]


def test_decorator_remember_disabled(memory: InfoLangMemory) -> None:
    handler, _ = _handler_calls()
    memory_entrypoint(memory, remember=False)(handler)(
        {"prompt": "q", "actor_id": "a"}, FakeContext("s1")
    )
    assert "remember" not in [c[0] for c in memory.client.calls]  # type: ignore[attr-defined]


def test_decorator_no_context_injection_for_string_payload(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall(make_chunk(text="berlin"))  # type: ignore[attr-defined]
    received: list[Any] = []

    def handler(payload: Any, context: Any) -> str:
        received.append(payload)
        return "answer"

    # String payload: prompt is the string; no dict to inject into.
    memory_entrypoint(memory)(handler)("just a prompt", FakeContext("s1"))
    assert received[0] == "just a prompt"


def test_decorator_top_k_override(memory: InfoLangMemory) -> None:
    handler, _ = _handler_calls()
    memory_entrypoint(memory, top_k=2)(handler)(
        {"prompt": "q", "actor_id": "a"}, FakeContext("s1")
    )
    assert memory.client.kwargs_for("recall")["top_k"] == 2  # type: ignore[attr-defined]


def test_decorator_custom_inject_key(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall(make_chunk(text="berlin"))  # type: ignore[attr-defined]
    handler, seen = _handler_calls()
    memory_entrypoint(memory, inject_key="mem")(handler)(
        {"prompt": "q", "actor_id": "a"}, FakeContext("s1")
    )
    assert "mem" in seen[0]


def test_decorator_no_injection_when_no_chunks(memory: InfoLangMemory) -> None:
    memory.client.recall_result = make_recall()  # type: ignore[attr-defined]
    handler, seen = _handler_calls()
    memory_entrypoint(memory)(handler)({"prompt": "q", "actor_id": "a"}, FakeContext("s1"))
    assert "infolang_context" not in seen[0]


def test_decorator_returns_handler_result(memory: InfoLangMemory) -> None:
    handler, _ = _handler_calls()
    out = memory_entrypoint(memory)(handler)({"prompt": "q", "actor_id": "a"}, FakeContext("s1"))
    assert out == {"result": "the answer"}


def test_decorator_accepts_raw_client() -> None:
    client = FakeInfoLang()
    handler, _ = _handler_calls()
    memory_entrypoint(client)(handler)(  # type: ignore[arg-type]
        {"prompt": "q", "actor_id": "alice"}, FakeContext("s1")
    )
    assert client.kwargs_for("recall")["namespace"] == "alice"


# --- error handling ------------------------------------------------------


class BoomClient(FakeInfoLang):
    def recall(self, query: str, **kwargs: Any) -> Any:
        raise RuntimeError("boom recall")

    def remember(self, text: str, **kwargs: Any) -> Any:
        raise RuntimeError("boom remember")


def test_recall_error_swallowed() -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]
    handler, seen = _handler_calls()
    out = memory_entrypoint(mem)(handler)({"prompt": "q", "actor_id": "a"}, FakeContext("s1"))
    assert out == {"result": "the answer"}
    assert "infolang_context" not in seen[0]


def test_remember_error_swallowed() -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]

    def handler(payload: Any, context: Any) -> str:
        return "the answer"

    # recall raises too, but is swallowed; ensure remember error is also swallowed.
    out = memory_entrypoint(mem, recall=False)(handler)(
        {"prompt": "q", "actor_id": "a"}, FakeContext("s1")
    )
    assert out == "the answer"


def test_recall_error_raise_opt_in() -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]
    handler, _ = _handler_calls()
    with pytest.raises(RuntimeError, match="boom recall"):
        memory_entrypoint(mem, raise_on_error=True)(handler)(
            {"prompt": "q", "actor_id": "a"}, FakeContext("s1")
        )


def test_remember_error_raise_opt_in() -> None:
    mem = InfoLangMemory(BoomClient())  # type: ignore[arg-type]

    def handler(payload: Any, context: Any) -> str:
        return "the answer"

    with pytest.raises(RuntimeError, match="boom remember"):
        memory_entrypoint(mem, recall=False, raise_on_error=True)(handler)(
            {"prompt": "q", "actor_id": "a"}, FakeContext("s1")
        )
