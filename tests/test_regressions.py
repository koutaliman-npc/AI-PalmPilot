"""
Regression tests for the bugs found in the v1.2.0 review (PR #1 follow-up).

Every test here corresponds to a defect that was reproduced against
origin/main or against the PR branch before being fixed.
"""

import os
import sys
import tempfile
import gc
import shutil
import time

import pytest

from agent_palm import PalmPilot
from agent_palm.store import PalmStore
from agent_palm.extract import heuristic_extract


@pytest.fixture
def tmp_root():
    """Isolated MEMPALM root per test (Windows-safe cleanup)."""
    root = tempfile.mkdtemp(prefix="mempalm_reg_")
    yield root
    for _ in range(3):
        try:
            shutil.rmtree(root, ignore_errors=False)
            break
        except PermissionError:
            gc.collect()
            time.sleep(0.1)


@pytest.fixture
def palm(tmp_root):
    p = PalmPilot(agent_id="reg_agent", root=tmp_root, use_hash_embed=True)
    yield p
    p.close()


# ─────────────────────────────────────────────────────────────────────
# 1. Duplicate remember() must not crash with a UNIQUE constraint error
# ─────────────────────────────────────────────────────────────────────
def test_duplicate_remember_does_not_crash(palm):
    """sqlite-vec vec0 tables reject INSERT OR REPLACE - the old store_vector
    used it, so re-remembering identical content raised
    'UNIQUE constraint failed on memories_vec primary key'."""
    first = palm.remember(content="Exactly the same content twice", namespace="ns")
    second = palm.remember(content="Exactly the same content twice", namespace="ns")
    assert first == second
    # Only one memory row, and exactly one vector row for it.
    conn = palm.store.connect()
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0] == 1


def test_store_vector_is_idempotent(palm):
    """store_vector must be a true upsert (DELETE + INSERT), not INSERT OR REPLACE."""
    uri = palm.remember(content="Vector upsert probe", namespace="ns")
    node_id = palm.store.get_node_id(uri)
    emb = palm.embedder.embed_single("probe")
    palm.store.store_vector(node_id, emb)   # second write, same id
    palm.store.store_vector(node_id, emb)   # third write
    conn = palm.store.connect()
    assert conn.execute("SELECT COUNT(*) FROM memories_vec WHERE id = ?", (node_id,)).fetchone()[0] == 1


# ─────────────────────────────────────────────────────────────────────
# 2. recall() must report the token counts for the requested level
# ─────────────────────────────────────────────────────────────────────
def test_recall_token_counts_increase_by_level(palm):
    """The old recall() read row[12]/[13]/[14] (tok_l1/tok_l2/created_at), so
    l0 reported l1's tokens, l1 reported l2's, and l2 reported the row's
    created_at epoch as its token count."""
    content = "Token counting verification content. " * 20
    uri = palm.remember(content=content, namespace="ns")
    l0 = palm.recall(uri, level="l0")["total_tokens"]
    l1 = palm.recall(uri, level="l1")["total_tokens"]
    l2 = palm.recall(uri, level="l2")["total_tokens"]
    assert 0 < l0 < l1 < l2
    # created_at epoch would be ~1.7e9 - a clear signature of the off-by-one.
    assert l2 < 1_000_000


# ─────────────────────────────────────────────────────────────────────
# 3. annotate() must not reference a non-existent column
# ─────────────────────────────────────────────────────────────────────
def test_annotate_works(palm):
    """The annotations table has no `author` column; the old SELECT raised
    'no such column: author'."""
    uri = palm.remember(content="Content that will be annotated", namespace="ns")
    result = palm.annotate(uri, verdict="confirmed", confidence=0.9, review_text="verified")
    assert result["memory_id"] > 0
    assert "new_trust" in result


# ─────────────────────────────────────────────────────────────────────
# 4. heuristic_extract() must never produce an empty L0
# ─────────────────────────────────────────────────────────────────────
def test_heuristic_extract_l0_never_empty():
    """Content with no sentence boundaries used to yield an empty L0."""
    l0, l1 = heuristic_extract("a" * 500)
    assert len(l0) > 0
    assert len(l0) <= 280


# ─────────────────────────────────────────────────────────────────────
# 5. Re-remembering must not wipe review-derived trust
# ─────────────────────────────────────────────────────────────────────
def test_dedup_preserves_trust(palm):
    """The dedup UPDATE set trust = 0.5, silently discarding trust earned
    from annotations."""
    uri = palm.remember(content="Trust must survive a re-write", namespace="ns")
    palm.annotate(uri, verdict="confirmed", confidence=1.0, review_text="human verified")
    conn = palm.store.connect()
    before = conn.execute("SELECT trust FROM memories WHERE uri = ?", (uri,)).fetchone()[0]

    palm.remember(content="Trust must survive a re-write", namespace="ns")
    after = conn.execute("SELECT trust FROM memories WHERE uri = ?", (uri,)).fetchone()[0]

    assert after == before
    assert after > 0.5, "review-derived trust was reset to the 0.5 prior"


def test_dedup_preserves_author(palm):
    """The dedup UPDATE hardcoded author = 'agent', discarding the real author."""
    uri = palm.remember(content="Authorship must survive a re-write", namespace="ns")
    conn = palm.store.connect()
    before = conn.execute("SELECT author FROM memories WHERE uri = ?", (uri,)).fetchone()[0]
    palm.remember(content="Authorship must survive a re-write", namespace="ns")
    after = conn.execute("SELECT author FROM memories WHERE uri = ?", (uri,)).fetchone()[0]
    assert after == before


# ─────────────────────────────────────────────────────────────────────
# 6. A failed embed must not leave an orphaned, un-vectorized memory
# ─────────────────────────────────────────────────────────────────────
def test_failed_embed_leaves_no_orphan_row(tmp_root):
    """Embedding happens BEFORE the row is committed, so a failing embedder
    leaves no half-written memory behind (which would be invisible to
    semantic search and unrepairable by a retry)."""
    palm = PalmPilot(agent_id="orphan_agent", root=tmp_root, use_hash_embed=True)

    class BoomEmbedder:
        def embed_single(self, text):
            raise RuntimeError("simulated embedder failure")

        def embed_batch(self, texts):
            raise RuntimeError("simulated embedder failure")

    palm.embedder = BoomEmbedder()
    with pytest.raises(RuntimeError):
        palm.remember(content="This must not be persisted", namespace="ns")

    conn = palm.store.connect()
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0] == 0
    palm.close()


def test_every_memory_has_a_vector(palm):
    """Invariant: no memory row may exist without a matching vector row."""
    for i in range(5):
        palm.remember(content=f"Vector invariant memory number {i}", namespace="ns")
    conn = palm.store.connect()
    missing = conn.execute(
        "SELECT m.id, m.uri FROM memories m "
        "LEFT JOIN memories_vec v ON m.id = v.id WHERE v.id IS NULL"
    ).fetchall()
    assert missing == [], f"memories without vectors: {missing}"


# ─────────────────────────────────────────────────────────────────────
# 7. The MCP tool path must handle duplicates (Eps's PR only fixed
#    PalmPilot.remember, not the MCP server - the main agent-facing path)
# ─────────────────────────────────────────────────────────────────────
def test_mcp_duplicate_remember_does_not_crash(tmp_root, monkeypatch):
    from agent_palm.mcp_tool import MCPPalmServer

    monkeypatch.setenv("MEMPALM_PATH", tmp_root)
    server = MCPPalmServer(use_hash_embed=True)

    first = server.tool_remember(content="MCP duplicate probe content", namespace="ns")
    second = server.tool_remember(content="MCP duplicate probe content", namespace="ns")

    assert first["uri"] == second["uri"]
    assert second["is_duplicate"] is True


def test_mcp_server_honours_root_argument(tmp_root):
    """MCPPalmServer ignored its root= argument and always used
    $MEMPALM_PATH / ~/.mempalm."""
    from agent_palm.mcp_tool import MCPPalmServer

    explicit = os.path.join(tmp_root, "explicit_root")
    server = MCPPalmServer(root=explicit, use_hash_embed=True)
    assert os.path.abspath(server.root) == os.path.abspath(explicit)


# ─────────────────────────────────────────────────────────────────────
# 8. The Hermes MemoryProvider path must handle duplicates too
# ─────────────────────────────────────────────────────────────────────
def test_hermes_provider_duplicate_store_does_not_crash(tmp_root):
    from agent_palm.hermes_provider import VaultMemoryProvider

    provider = VaultMemoryProvider({
        "agent_id": "provider_agent",
        "root": tmp_root,
        "hash_embed": "true",
    })
    first = provider.store("Provider duplicate probe content", metadata={"namespace": "ns"})
    second = provider.store("Provider duplicate probe content", metadata={"namespace": "ns"})
    assert first == second


# ─────────────────────────────────────────────────────────────────────
# 9. Shared-pool writes from two agents must not silently merge
# ─────────────────────────────────────────────────────────────────────
def test_shared_pool_dedup_is_by_pool_and_namespace(tmp_root):
    """Different namespaces in the shared pool must stay separate rows."""
    palm = PalmPilot(agent_id="shared_agent", root=tmp_root, use_hash_embed=True)
    a = palm.remember(content="Same words, different namespace", namespace="ns_a", pool="shared")
    b = palm.remember(content="Same words, different namespace", namespace="ns_b", pool="shared")
    assert a != b
    palm.close()


# ─────────────────────────────────────────────────────────────────────
# 10. A cross-agent review must count FULL weight; only a self-review
#     counts half. The annotations table has no `author` column, so the
#     self-review check must compare the reviewer against the MEMORY's
#     author. Aliasing `reviewer AS author` made every annotation look
#     like a self-review and halved every trust contribution.
# ─────────────────────────────────────────────────────────────────────
def test_cross_agent_review_counts_full_weight(tmp_root):
    from agent_palm.review import recompute_trust, PRIOR_TRUST

    palm = PalmPilot(agent_id="alice", root=tmp_root, use_hash_embed=True)
    conn = palm.store.connect()

    uri = palm.remember(content="A memory reviewed by a second agent", namespace="ns")
    node_id, mem_author, source = conn.execute(
        "SELECT id, author, source FROM memories WHERE uri = ?", (uri,)
    ).fetchone()

    # Insert an annotation from a DIFFERENT reviewer
    conn.execute(
        """INSERT INTO annotations (memory_id, reviewer, verdict, confidence,
                                    review_text, review_hash, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (node_id, "bob", "confirmed", 1.0, "bob confirms", "cross-hash", 1000.0),
    )
    conn.commit()

    anns = conn.execute(
        """SELECT reviewer, verdict, confidence, review_text, evidence_uri, created_at
           FROM annotations WHERE memory_id = ?""",
        (node_id,),
    ).fetchall()
    ann_list = [
        {"reviewer": a[0], "verdict": a[1], "confidence": a[2],
         "review_text": a[3], "evidence_uri": a[4], "created_at": a[5],
         "author": mem_author}
        for a in anns
    ]

    trust = recompute_trust(palm.store, node_id, source, ann_list)
    prior = PRIOR_TRUST.get(source, 0.5)
    full_weight = (prior + 1.0 * 1.0 * 1.0) / (1.0 + 1.0)
    half_weight = (prior + 1.0 * 0.5 * 1.0) / (1.0 + 0.5)

    assert abs(trust - full_weight) < 1e-9, (
        f"cross-agent review was penalised as self-review: "
        f"got {trust}, expected {full_weight} (not {half_weight})"
    )
    palm.close()


def test_self_review_counts_half_weight(tmp_root):
    """The author reviewing their own memory must still be halved."""
    from agent_palm.review import recompute_trust, PRIOR_TRUST

    palm = PalmPilot(agent_id="alice", root=tmp_root, use_hash_embed=True)
    conn = palm.store.connect()

    uri = palm.remember(content="A memory the author reviews themselves", namespace="ns")
    node_id, mem_author, source = conn.execute(
        "SELECT id, author, source FROM memories WHERE uri = ?", (uri,)
    ).fetchone()
    conn.execute(
        """INSERT INTO annotations (memory_id, reviewer, verdict, confidence,
                                    review_text, review_hash, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (node_id, mem_author, "confirmed", 1.0, "self review", "self-hash", 1000.0),
    )
    conn.commit()

    anns = conn.execute(
        """SELECT reviewer, verdict, confidence, review_text, evidence_uri, created_at
           FROM annotations WHERE memory_id = ?""",
        (node_id,),
    ).fetchall()
    ann_list = [
        {"reviewer": a[0], "verdict": a[1], "confidence": a[2],
         "review_text": a[3], "evidence_uri": a[4], "created_at": a[5],
         "author": mem_author}
        for a in anns
    ]

    trust = recompute_trust(palm.store, node_id, source, ann_list)
    prior = PRIOR_TRUST.get(source, 0.5)
    half_weight = (prior + 1.0 * 0.5 * 1.0) / (1.0 + 0.5)
    assert abs(trust - half_weight) < 1e-9, (
        f"self-review was not halved: got {trust}, expected {half_weight}"
    )
    palm.close()


def test_annotate_uses_memory_author_for_self_review_detection(tmp_root):
    """End-to-end: annotating as a different reviewer must raise trust MORE
    than annotating as the memory's author."""
    from agent_palm.review import recompute_trust

    def trust_after(reviewer):
        p = PalmPilot(agent_id="author_agent", root=tempfile.mkdtemp(prefix="mempalm_trust_"), use_hash_embed=True)
        conn = p.store.connect()
        uri = p.remember(content=f"Trust comparison {reviewer}", namespace="ns")
        node_id, mem_author, source = conn.execute(
            "SELECT id, author, source FROM memories WHERE uri = ?", (uri,)
        ).fetchone()
        conn.execute(
            """INSERT INTO annotations (memory_id, reviewer, verdict, confidence,
                                        review_text, review_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (node_id, reviewer, "confirmed", 1.0, "r", f"h-{reviewer}", 1000.0),
        )
        conn.commit()
        anns = conn.execute(
            """SELECT reviewer, verdict, confidence, review_text, evidence_uri, created_at
               FROM annotations WHERE memory_id = ?""",
            (node_id,),
        ).fetchall()
        ann_list = [
            {"reviewer": a[0], "verdict": a[1], "confidence": a[2],
             "review_text": a[3], "evidence_uri": a[4], "created_at": a[5],
             "author": mem_author}
            for a in anns
        ]
        t = recompute_trust(p.store, node_id, source, ann_list)
        p.close()
        return t

    self_trust = trust_after("author_agent")
    cross_trust = trust_after("different_agent")
    assert cross_trust > self_trust, (
        f"cross-agent review ({cross_trust}) did not outrank self-review ({self_trust})"
    )


# ─────────────────────────────────────────────────────────────────────
# 11. A memory row that lost its vector must be REPAIRED on re-write
# ─────────────────────────────────────────────────────────────────────
def test_missing_vector_is_repaired_on_rewrite(tmp_root):
    """A row written without a vector (e.g. by an older version) stays
    invisible to semantic search forever unless a rewrite repairs it."""
    palm = PalmPilot(agent_id="repair_agent", root=tmp_root, use_hash_embed=True)
    uri = palm.remember(content="Memory that will lose its vector", namespace="ns")
    conn = palm.store.connect()
    node_id = conn.execute("SELECT id FROM memories WHERE uri = ?", (uri,)).fetchone()[0]

    # Simulate the orphan: drop the vector row
    conn.execute("DELETE FROM memories_vec WHERE id = ?", (node_id,))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0] == 0

    palm.remember(content="Memory that will lose its vector", namespace="ns")
    assert conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0] == 1, \
        "the missing vector was not repaired by the rewrite"
    palm.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
