"""Opt-in live tests against real InfoLang.

Run with::

    INFOLANG_API_KEY=il_live_... pytest --run-live -m live

Demonstrates the core AgentCore promise: memory written under one runtime
session id is recalled under a *different* session id, keyed by a stable actor
identity (i.e. it survives session teardown/recreate).
"""

from __future__ import annotations

import os
import uuid

import pytest
from infolang import InfoLang

from infolang_agentcore import InfoLangMemory

pytestmark = pytest.mark.live


def _memory() -> InfoLangMemory:
    api_key = os.environ.get("INFOLANG_API_KEY")
    if not api_key:
        pytest.skip("INFOLANG_API_KEY not set")
    workspace = os.environ.get("INFOLANG_WORKSPACE")
    client = InfoLang.from_api_key(api_key, workspace=workspace)
    return InfoLangMemory(client, namespace_template="wp39-agentcore-live/{actor_id}")


def test_live_memory_survives_session_recreation() -> None:
    memory = _memory()
    actor = f"actor-{uuid.uuid4().hex[:8]}"
    token = uuid.uuid4().hex

    # Session #1 writes.
    first = memory.session(actor_id=actor, session_id="session-1")
    first.remember_turn("what is the canary token?", f"the token is {token}")

    # Session #2 (as if the runtime was torn down and recreated) recalls.
    second = memory.session(actor_id=actor, session_id="session-2")
    result = second.recall("what is the canary token?", top_k=5)
    assert any(token in chunk.text for chunk in result.chunks)
