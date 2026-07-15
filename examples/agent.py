"""A minimal Amazon Bedrock AgentCore Runtime agent with durable InfoLang memory.

This is the sanctioned AgentCore extension surface: a ``BedrockAgentCoreApp``
with an ``@app.entrypoint`` handler (framework-agnostic; NOT a Bedrock Agents
action-group Lambda). ``memory_entrypoint`` transparently recalls durable memory
on the way in and remembers the exchange on the way out.

Local run (talks to InfoLang; uses an echo "model")::

    export INFOLANG_API_KEY=il_live_...
    python examples/agent.py            # serves on http://localhost:8080

Invoke it::

    curl -s localhost:8080/invocations \
      -H 'content-type: application/json' \
      -d '{"prompt": "What is our SLA?", "actor_id": "customer-42"}'

Deploy to AgentCore Runtime with the starter toolkit (`agentcore configure`
then `agentcore launch`) or with the Terraform in ``deploy/``.
"""

from __future__ import annotations

import os
from typing import Any

from bedrock_agentcore import BedrockAgentCoreApp

from infolang_agentcore import InfoLangMemory, memory_entrypoint

app = BedrockAgentCoreApp()

# One managed InfoLang client. A per-actor bank ("{actor_id}") keeps each
# customer's memory durable across AgentCore Runtime session recreation.
memory = InfoLangMemory.from_api_key(
    os.environ["INFOLANG_API_KEY"],
    namespace_template="{actor_id}",
    source="agentcore-example",
)


def generate_reply(prompt: str, recalled_context: str) -> str:
    """Stand-in for a real model call.

    Replace this with a Bedrock ``converse`` call or a Strands ``Agent`` that
    reads ``recalled_context`` as grounding. Kept dependency-free here so the
    example runs with only ``infolang-agentcore``.
    """

    if recalled_context:
        return f"(grounded in memory)\n{recalled_context}\n\nAnswering: {prompt}"
    return f"Answering: {prompt}"


@app.entrypoint
@memory_entrypoint(memory)
def handler(payload: dict[str, Any], context: Any) -> dict[str, str]:
    prompt = payload.get("prompt", "")
    recalled = payload.get("infolang_context", "")
    return {"result": generate_reply(prompt, recalled)}


if __name__ == "__main__":
    app.run()
