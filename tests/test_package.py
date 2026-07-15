"""Package-level smoke tests, incl. AgentCore entrypoint compatibility."""

from __future__ import annotations

import inspect

import infolang_agentcore
from infolang_agentcore import InfoLangMemory, memory_entrypoint

from .conftest import FakeInfoLang


def test_version_is_string() -> None:
    assert isinstance(infolang_agentcore.__version__, str)
    assert infolang_agentcore.__version__.count(".") >= 2


def test_public_exports() -> None:
    for name in (
        "InfoLangMemory",
        "MemorySession",
        "coerce_memory",
        "memory_entrypoint",
        "session_for",
        "extract_actor_id",
        "extract_session_id",
        "extract_prompt",
        "extract_answer",
    ):
        assert name in infolang_agentcore.__all__
        assert hasattr(infolang_agentcore, name)


def test_wrapped_handler_matches_agentcore_takes_context() -> None:
    """The wrapper must look like a ``(payload, context)`` handler.

    Mirrors ``BedrockAgentCoreApp._takes_context``: 2nd param must be 'context'.
    """

    mem = InfoLangMemory(FakeInfoLang())  # type: ignore[arg-type]

    def handler(payload: object, context: object) -> str:
        return "x"

    wrapped = memory_entrypoint(mem)(handler)
    params = list(inspect.signature(wrapped).parameters.keys())
    assert len(params) >= 2 and params[1] == "context"


def test_registers_with_bedrock_agentcore_app() -> None:
    from bedrock_agentcore import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()
    mem = InfoLangMemory(FakeInfoLang())  # type: ignore[arg-type]

    @app.entrypoint
    @memory_entrypoint(mem)
    def handler(payload: object, context: object) -> dict[str, str]:
        return {"result": "ok"}

    assert app.handlers["main"] is handler
