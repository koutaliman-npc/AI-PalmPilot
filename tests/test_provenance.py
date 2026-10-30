"""
Provenance & namespace-default regression tests.

Locks in two behaviours introduced to stop automatic captures claiming to be
operator-stated and to stop every uncategorised memory landing in a junk
"default" / "agent-memory" bucket:

  1. ``source`` defaults to ``"inferred"`` across every capture path; only an
     explicit ``source="user"`` stamps operator attribution.
  2. An omitted ``namespace`` is derived by a deterministic keyword classifier
     into a small canonical set instead of the literal ``"default"``.
"""

import os
import shutil
import tempfile

import pytest

from agent_palm import PalmPilot
from agent_palm.extract import classify_namespace, DEFAULT_NAMESPACE


@pytest.fixture
def tmp_root():
    root = tempfile.mkdtemp(prefix="mempalm_prov_")
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def palm(tmp_root):
    p = PalmPilot(agent_id="prov_agent", root=tmp_root, use_hash_embed=True)
    yield p
    p.close()


def _row(palm, uri):
    conn = palm.store.connect()
    return conn.execute(
        "SELECT source, namespace FROM memories WHERE uri = ?", (uri,)
    ).fetchone()


# ── classifier ──────────────────────────────────────────────────────────

def test_classify_namespace_infra():
    assert classify_namespace("Deployed nginx to the staging server") == "infra"
    assert classify_namespace("nmap scan found port 22 open on the host") == "infra"


def test_classify_namespace_home():
    assert classify_namespace("The boiler in the kitchen needs a service") == "home"


def test_classify_namespace_people():
    assert classify_namespace("Alice's birthday is in March, she is a colleague") == "people"


def test_classify_namespace_projects():
    assert classify_namespace("Next milestone is the search feature before the release") == "projects"


def test_classify_namespace_estate_events():
    assert classify_namespace("The estate is hosting a wedding next weekend") == "estate-events"


def test_classify_namespace_fallback():
    assert classify_namespace("a vague free-floating thought") == "agent-memory"
    assert classify_namespace("") == "agent-memory"
    assert DEFAULT_NAMESPACE == "agent-memory"


def test_classify_namespace_word_boundary():
    # "report"/"important" contain "port", but "port" must only match as a token.
    assert classify_namespace("this report is important") == "agent-memory"


# ── source default: inferred, unless explicitly user ─────────────────────

def test_remember_default_source_is_inferred(palm):
    uri = palm.remember(content="Automatic capture of a deployment state", namespace="ns")
    assert _row(palm, uri)[0] == "inferred"


def test_remember_explicit_source_user(palm):
    uri = palm.remember(
        content="Operator explicitly stated this preference",
        namespace="ns",
        source="user",
    )
    assert _row(palm, uri)[0] == "user"


# ── namespace default: classified, not "default" ─────────────────────────

def test_remember_default_namespace_is_classified(palm):
    uri = palm.remember(content="Deployed docker containers to the cluster")
    source, namespace = _row(palm, uri)
    assert namespace == "infra"
    assert namespace != "default"
    assert source == "inferred"


def test_hermes_provider_defaults(tmp_root):
    from agent_palm.hermes_provider import VaultMemoryProvider

    provider = VaultMemoryProvider({
        "agent_id": "provider_agent",
        "root": tmp_root,
        "hash_embed": "true",
    })
    uri = provider.store("Scanned the network and found a new server online")
    source, namespace = provider._store.connect().execute(
        "SELECT source, namespace FROM memories WHERE uri = ?", (uri,)
    ).fetchone()
    assert source == "inferred"
    assert namespace == "infra"


def test_hermes_provider_explicit_user_source(tmp_root):
    from agent_palm.hermes_provider import VaultMemoryProvider

    provider = VaultMemoryProvider({
        "agent_id": "provider_agent",
        "root": tmp_root,
        "hash_embed": "true",
    })
    uri = provider.store(
        "The operator dictated this",
        metadata={"namespace": "people", "source": "user"},
    )
    source, namespace = provider._store.connect().execute(
        "SELECT source, namespace FROM memories WHERE uri = ?", (uri,)
    ).fetchone()
    assert source == "user"
    assert namespace == "people"


def test_mcp_tool_defaults(tmp_root, monkeypatch):
    from agent_palm.mcp_tool import MCPPalmServer

    monkeypatch.setenv("MEMPALM_PATH", tmp_root)
    server = MCPPalmServer(use_hash_embed=True)
    result = server.tool_remember(content="Backed up the postgres database")
    uri = result["uri"]
    source, namespace = server.personal_store.connect().execute(
        "SELECT source, namespace FROM memories WHERE uri = ?", (uri,)
    ).fetchone()
    assert source == "inferred"
    assert namespace == "infra"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
