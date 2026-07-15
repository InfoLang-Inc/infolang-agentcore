"""AgentCore Runtime wiring: recall on session start, remember on session end.

Amazon Bedrock AgentCore Runtime hosts arbitrary agent code and invokes the
function you register with ``@app.entrypoint`` as ``handler(payload, context)``,
where ``context.session_id`` identifies the (ephemeral) runtime session. The
*sanctioned* extension point is this entrypoint — not Bedrock Agents action-group
Lambdas, which belong to the older managed-agents product.

:func:`memory_entrypoint` decorates such a handler so that, transparently:

1. the incoming prompt is used to **recall** durable memory (keyed by actor
   identity) and the rendered context is injected into the payload; then
2. after the wrapped handler returns, the prompt/answer pair is **remembered**.

Because memory lives in InfoLang (external, identity-keyed), it survives Runtime
session teardown/recreate and account reprovisioning — the WP17 story.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from infolang import InfoLang

from .adapter import InfoLangMemory, MemorySession, coerce_memory

_log = logging.getLogger("infolang_agentcore")

Handler = Callable[[Any, Any], Any]

DEFAULT_PROMPT_KEYS: tuple[str, ...] = (
    "prompt",
    "inputText",
    "input",
    "query",
    "text",
    "message",
)
DEFAULT_ANSWER_KEYS: tuple[str, ...] = (
    "result",
    "output",
    "response",
    "completion",
    "answer",
    "text",
    "message",
)
DEFAULT_INJECT_KEY = "infolang_context"
DEFAULT_ACTOR_KEY = "actor_id"


def extract_session_id(context: Any) -> str | None:
    """Read the AgentCore runtime session id from the request context.

    Falls back to the SDK's process-global
    ``BedrockAgentCoreContext.get_session_id()`` when no context is supplied.
    """

    session_id = getattr(context, "session_id", None)
    if isinstance(session_id, str) and session_id:
        return session_id
    return _session_id_from_runtime()


def _session_id_from_runtime() -> str | None:
    try:
        from bedrock_agentcore import BedrockAgentCoreContext

        value = BedrockAgentCoreContext.get_session_id()
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def extract_actor_id(
    payload: Any,
    context: Any = None,
    *,
    actor_key: str = DEFAULT_ACTOR_KEY,
) -> str | None:
    """Resolve the durable actor identity.

    Precedence: an explicit ``actor_id`` in the payload (the stable per-user
    key an app should pass) → the runtime ``session_id`` (per-session fallback).
    """

    if isinstance(payload, dict):
        value = payload.get(actor_key)
        if isinstance(value, str) and value:
            return value
    return extract_session_id(context)


def extract_prompt(payload: Any, *, keys: tuple[str, ...] = DEFAULT_PROMPT_KEYS) -> str | None:
    """Pull the user prompt out of an AgentCore payload."""

    if isinstance(payload, str):
        return payload or None
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def extract_answer(result: Any, *, keys: tuple[str, ...] = DEFAULT_ANSWER_KEYS) -> str | None:
    """Pull the assistant answer out of a handler's return value."""

    if isinstance(result, str):
        return result or None
    if isinstance(result, dict):
        for key in keys:
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def session_for(
    memory: InfoLangMemory,
    payload: Any,
    context: Any = None,
    *,
    actor_key: str = DEFAULT_ACTOR_KEY,
) -> MemorySession:
    """Build a :class:`MemorySession` from an AgentCore payload + context."""

    return memory.session(
        actor_id=extract_actor_id(payload, context, actor_key=actor_key),
        session_id=extract_session_id(context),
    )


def memory_entrypoint(
    memory: InfoLangMemory | InfoLang,
    *,
    recall: bool = True,
    remember: bool = True,
    actor_key: str = DEFAULT_ACTOR_KEY,
    prompt_keys: tuple[str, ...] = DEFAULT_PROMPT_KEYS,
    answer_keys: tuple[str, ...] = DEFAULT_ANSWER_KEYS,
    inject_key: str = DEFAULT_INJECT_KEY,
    top_k: int | None = None,
    raise_on_error: bool = False,
) -> Callable[[Handler], Handler]:
    """Wrap an AgentCore ``(payload, context)`` handler with InfoLang memory.

    Usage::

        app = BedrockAgentCoreApp()
        memory = InfoLangMemory.from_api_key("il_live_...")

        @app.entrypoint
        @memory_entrypoint(memory)
        def handler(payload, context):
            prompt = payload["prompt"]
            context_block = payload.get("infolang_context", "")
            return {"result": my_agent(prompt, context_block)}

    Memory failures are best-effort by default (logged, not raised); set
    ``raise_on_error=True`` to surface them.
    """

    mem = coerce_memory(memory)

    def decorator(func: Handler) -> Handler:
        def wrapper(payload: Any, context: Any) -> Any:
            session = session_for(mem, payload, context, actor_key=actor_key)
            prompt = extract_prompt(payload, keys=prompt_keys)

            if recall and prompt:
                block = _safe(
                    lambda: session.recall_context(prompt, top_k=top_k),
                    "recall",
                    raise_on_error,
                )
                if block and isinstance(payload, dict):
                    payload = {**payload, inject_key: block}

            result = func(payload, context)

            if remember and prompt:
                answer = extract_answer(result, keys=answer_keys)
                if answer:
                    _safe(
                        lambda: session.remember_turn(prompt, answer),
                        "remember",
                        raise_on_error,
                    )
            return result

        wrapper.__name__ = getattr(func, "__name__", "wrapper")
        wrapper.__doc__ = func.__doc__
        wrapper.__qualname__ = getattr(func, "__qualname__", wrapper.__name__)
        return wrapper

    return decorator


def _safe(fn: Callable[[], Any], op: str, raise_on_error: bool) -> Any:
    try:
        return fn()
    except Exception:
        if raise_on_error:
            raise
        _log.warning("InfoLang %s failed; continuing without memory.", op, exc_info=True)
        return None
