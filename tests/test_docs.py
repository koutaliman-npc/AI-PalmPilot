"""Run every documented example verbatim.

The README's API Reference is only allowed to claim what actually executes, so
each example lives here as a test. If a signature changes, this file fails
before the docs can drift again.
"""
import tempfile

import pytest

from agent_palm import PalmPilot

ROOT = tempfile.mkdtemp(prefix="readme_verify_")
p = PalmPilot(agent_id="my-agent", root=ROOT, use_hash_embed=True)
ok = []


def step(label, fn):
    try:
        out = fn()
        ok.append((label, "OK", out))
    except Exception as e:
        ok.append((label, "FAIL", f"{type(e).__name__}: {e}"))


# --- Quick Start #3 ---
step("remember (Quick Start #3)", lambda: p.remember(
    "Decided to use ONNX Runtime for embeddings - no more llama.cpp crashes!",
    namespace="decisions", tags=["architecture", "embeddings"]))
uri = p.remember("Decided to use ONNX Runtime for embeddings - no more llama.cpp crashes!",
                 namespace="decisions", tags=["architecture", "embeddings"])

# --- Quick Start #4 ---
def step4():
    results = p.search("embedding crashes", max_tokens=2000, pool="both")
    for r in results:
        m = r["data"]
        print(f"[{m['uri']}::{r['level']}] {m['l1'][:80]}...")
    return f"{len(results)} results"


step("search (Quick Start #4)", step4)


# --- Quick Start #5 ---
def step5():
    detail = p.recall(uri, level="l2", max_tokens=4000)
    print(detail["text"])
    return detail["truncated"], detail["total_tokens"]


step("recall (Quick Start #5)", step5)


# --- Quick Start #6 (stats is a dict now) ---
def step6():
    st = p.stats()
    print({k: v["count"] for k, v in st.items()})
    return "counts printed"


step("stats (Quick Start #6)", step6)


# --- API Reference: STORE ---
step("API remember", lambda: p.remember(
    content="Full text content to remember",
    namespace="decisions",
    tags=["architecture", "llm"],
    type_="note",
    source="inferred",
    ttl_days=30,
    pool="personal"))


# --- API Reference: SEARCH ---
def api_search():
    results = p.search(
        query="embedding model selection",
        namespace="decisions",
        pool="both",
        level="l1",
        max_tokens=2000,
        limit=10,
        include_superseded=False,
    )
    for r in results:
        _ = (r["data"]["uri"], r["data"]["score"], r["data"]["l1"], r["data"]["verdict"],
             r["data"]["author"], r["data"]["created_at"], r["level"], r["cost"])
    return f"envelope fields all readable, n={len(results)}"


step("API search envelope", api_search)


# --- API Reference: ANNOTATE ---
def api_annotate():
    out = p.annotate(
        uri=uri,
        review_text="This pattern broke in v3.2; use the new API instead",
        confidence=0.9,
        verdict="superseded",
        evidence_uri=None,
        tags=["deprecated"],
    )
    return out


step("API annotate", api_annotate)


# --- API Reference: REVIEW QUEUE / MARK USED / SUPERSEDE ---
def api_queue():
    queue = p.review_queue(namespace="decisions", limit=10, pool="personal")
    for item in queue:
        _ = (item.uri, item.reason, item.l1)
    return f"ReviewItem attributes readable, n={len(queue)}"


step("API review_queue", api_queue)
step("API mark_used", lambda: p.mark_used([uri]))


def api_supersede():
    old = p.remember("deploy on tuesday", namespace="ops")
    new = p.supersede(old, new_content="deploy on thursday", namespace="ops")
    default_uris = [r["data"]["uri"] for r in p.search("deploy day", namespace="ops")]
    all_uris = [r["data"]["uri"] for r in p.search("deploy day", namespace="ops", include_superseded=True)]
    return {"new": new, "old_hidden_by_default": old not in default_uris,
            "old_found_explicitly": old in all_uris}


step("API supersede", api_supersede)


# --- API Reference: FORGET / STATS / VERIFY / BACKUP ---
tmp_uri = p.remember("temp memory to delete", namespace="tmp")
step("API forget", lambda: p.forget(tmp_uri))
step("API stats", lambda: p.stats(namespace=None, pool="both"))
step("API verify_citations", lambda: p.verify_citations("we used ONNX"))
step("API backup", lambda: p.backup(destination=ROOT + "/backups"))


# --- Hermes section examples ---
def hermes_remember():
    p.remember(
        content="User prefers snake_case for Python internal variables",
        namespace="conventions",
        tags=["style", "python"],
        source="user",
        pool="personal",
    )
    return True


step("Hermes: remember(snake_case)", hermes_remember)


def hermes_search():
    results = p.search(query="user coding style preferences", max_tokens=1500, pool="both")
    _ = [(r["data"]["uri"], r["data"]["l1"], r["data"]["score"], r["data"]["verdict"]) for r in results]
    return f"{len(results)} results"


step("Hermes: search(both)", hermes_search)


def hermes_mark_used():
    results = p.search("coding style", pool="both")

    def i_cited_it(r):
        return True

    p.mark_used([r["data"]["uri"] for r in results if i_cited_it(r)])
    return True


step("Hermes: mark_used", hermes_mark_used)


def hermes_review():
    queue = p.review_queue(limit=5)
    for item in queue:
        p.annotate(item.uri, review_text="Still valid as of current version",
                   confidence=0.85, verdict="confirmed")
    return f"queue={len(queue)}"


step("Hermes: review_queue loop", hermes_review)


def hermes_shared():
    p.remember(
        content="Never bump a dependency and migrate in the same PR.",
        namespace="team/conventions",
        tags=["process", "review"],
        source="observed",
        pool="shared",
    )
    found = p.search("dependency migration PR", pool="shared") + \
        p.search("dependency migration PR", pool="both")
    return f"shared write + search OK, {len(found)} hits"


step("Hermes: shared pool write", hermes_shared)

p.close()

def test_every_documented_example_runs():
    """Doc drift guard: one failure per broken claim in the README."""
    fails = [(l, o) for l, st, o in ok if st == "FAIL"]
    assert not fails, "documented examples that do not execute:\n" + "\n".join(
            f"  {l}: {o}" for l, o in fails
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])