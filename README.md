# infolang-agentcore

Durable [InfoLang](https://infolang.ai) memory for
[Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) Runtime
agents.

AgentCore Runtime sessions are ephemeral — when a session is torn down and
recreated (or the account is reprovisioned), in-process state is gone. This
package backs an agent's memory with InfoLang, an external store keyed by a
stable **actor identity**, so memory *survives* that lifecycle. It wires recall
on the way into an invocation and remember on the way out.

Built on the published InfoLang Python SDK (`infolang`); it never talks HTTP
directly or touches runtime internals.

## Which AgentCore extension surface?

AgentCore Runtime is framework-agnostic and hosts your code behind an
`@app.entrypoint` handler (`bedrock-agentcore` SDK). That entrypoint — **not**
Bedrock Agents "action group" Lambdas (a different, older product) — is the
sanctioned extension point, and it is what this package hooks into. Memory here
is an external service call, so there is no AWS memory resource to provision and
nothing to mock in tests.

## Install

```bash
pip install infolang-agentcore
```

Pulls in `infolang` (the public SDK) and `bedrock-agentcore`.

## Quickstart

```python
from bedrock_agentcore import BedrockAgentCoreApp
from infolang_agentcore import InfoLangMemory, memory_entrypoint

app = BedrockAgentCoreApp()

# A per-actor bank keeps each user's memory durable across session recreation.
memory = InfoLangMemory.from_api_key("il_live_...", namespace_template="{actor_id}")

@app.entrypoint
@memory_entrypoint(memory)
def handler(payload, context):
    prompt = payload["prompt"]
    recalled = payload.get("infolang_context", "")   # injected by the decorator
    answer = run_your_agent(prompt, grounding=recalled)
    return {"result": answer}                          # remembered by the decorator

app.run()
```

`memory_entrypoint` derives the actor identity from `payload["actor_id"]`
(falling back to the runtime `session_id`), recalls relevant memory, injects it
as `payload["infolang_context"]`, calls your handler, then stores the
prompt/answer pair. Memory failures are best-effort by default (logged, not
raised); pass `raise_on_error=True` to opt out.

### Explicit control

Skip the decorator and drive it yourself:

```python
from infolang_agentcore import InfoLangMemory, session_for

memory = InfoLangMemory.from_api_key("il_live_...")

@app.entrypoint
def handler(payload, context):
    session = session_for(memory, payload, context)   # keyed by actor + session
    grounding = session.recall_context(payload["prompt"])
    answer = run_your_agent(payload["prompt"], grounding=grounding)
    session.remember_turn(payload["prompt"], answer)
    return {"result": answer}
```

## Scoping

InfoLang scopes memory by `workspace` (tenant) and `namespace` (bank):

| Model | How |
|-------|-----|
| Durable per user (default) | `namespace_template="{actor_id}"` — each actor gets its own bank |
| Single shared bank | `namespace="support"` — actors distinguished by provenance tags (`actor:...`, `session:...`) |
| Multi-tenant | set `workspace="acme"` on the client |

A managed API key honours `namespace` on both reads and writes.

## API

| Symbol | Purpose |
|--------|---------|
| `InfoLangMemory(client, *, namespace, namespace_template, source, top_k, score_floor)` | Adapter over an `infolang.InfoLang` client |
| `InfoLangMemory.from_api_key(key, *, namespace, namespace_template, workspace, ...)` | Build adapter + managed client |
| `InfoLangMemory.session(*, actor_id, session_id)` / `session_for(memory, payload, context)` | A `MemorySession` scoped to an actor + runtime session |
| `memory_entrypoint(memory, *, recall, remember, actor_key, prompt_keys, answer_keys, inject_key, top_k, raise_on_error)` | Decorator for an `@app.entrypoint` `(payload, context)` handler |

## Deploy

A runnable example (`BedrockAgentCoreApp` + Dockerfile + Terraform for the WP17
per-customer-account model) lives in [`examples/`](examples/). See
[`examples/deploy/README.md`](examples/deploy/README.md), including a check that
recall works across a session-id change.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

Tests run fully offline against a fake InfoLang client and a fake AgentCore
context. Live tests (`-m live --run-live`) require `INFOLANG_API_KEY` and prove
recall across a simulated session recreation.

## Verified against

- `bedrock-agentcore` 1.18.0
- `boto3` 1.43.48
- `infolang` 0.2.0

## License

Apache-2.0
