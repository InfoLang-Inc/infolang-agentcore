"""InfoLang durable memory for Amazon Bedrock AgentCore Runtime.

Wire identity-keyed memory into an AgentCore Runtime entrypoint so it survives
ephemeral session teardown/recreate::

    from bedrock_agentcore import BedrockAgentCoreApp
    from infolang_agentcore import InfoLangMemory, memory_entrypoint

    app = BedrockAgentCoreApp()
    memory = InfoLangMemory.from_api_key("il_live_...")

    @app.entrypoint
    @memory_entrypoint(memory)
    def handler(payload, context):
        prompt = payload["prompt"]
        recalled = payload.get("infolang_context", "")
        return {"result": run_agent(prompt, recalled)}

    app.run()
"""

from __future__ import annotations

from ._version import __version__
from .adapter import (
    DEFAULT_NAMESPACE_TEMPLATE,
    DEFAULT_SCORE_FLOOR,
    DEFAULT_SOURCE,
    DEFAULT_TOP_K,
    InfoLangMemory,
    MemorySession,
    coerce_memory,
)
from .runtime import (
    extract_actor_id,
    extract_answer,
    extract_prompt,
    extract_session_id,
    memory_entrypoint,
    session_for,
)

__all__ = [
    "__version__",
    "InfoLangMemory",
    "MemorySession",
    "coerce_memory",
    "memory_entrypoint",
    "session_for",
    "extract_actor_id",
    "extract_session_id",
    "extract_prompt",
    "extract_answer",
    "DEFAULT_TOP_K",
    "DEFAULT_SCORE_FLOOR",
    "DEFAULT_SOURCE",
    "DEFAULT_NAMESPACE_TEMPLATE",
]
