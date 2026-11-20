# 🌿 AI PalmPilot v1.3.2 — Memory for Agents, By Agents

> ## ⭐ This is the final free release
>
> **v1.3.2 is the last version of AI PalmPilot that will ever be published free
> and open source.** Everything here works, is MIT-licensed, and is yours to
> keep — permanently. No keys, no expiry, no telemetry, no rug pull.
>
> From here on, AI PalmPilot moves to **Agensi** (https://agensi.io), a
> marketplace for AI agent skills, where the continued development of the
> project — new memory backends, better recall, faster search — is published
> as a paid skill. The skill listing on Agensi is where you buy future versions
> and updates of this project.
>
> This repository will be made private once the paid release is live, so this
> tag is the archive of the open-source era.
>
> - Installing from here: `pip install agent-palm==1.3.2`
> - Future updates: **sold on Agensi at https://agensi.io**

**Side Note — why this exists** 
> Look, I've tried them all. OpenViking, MemGPT, Letta, Chroma-backed custom builds, the whole zoo. And every single one of them asks me to babysit something: a Postgres instance, a Redis cache, a vector DB container, a Go server on port 8080, a dashboard I never open, a config file that breaks every time I sneeze. I'm not running a DevOps shop—I'm trying to give my agents a *memory*.

> So I asked around. Poked at OpenViking's architecture, read MemGPT's papers, tore apart Letta's server, stared at Chroma's config until my eyes crossed. Even asked seven different commercial models "how would YOU build this?" The answers were surprisingly boring in the best way: SQLite + a tiny ONNX model + a little bit of cleverness. No servers. No dashboards. No babysitting.

> That's the whole philosophy right there: **I didn't want another thing to babysit.** Not a server, not a vector DB, not a background worker, not a config file that drifts. I wanted a PalmPilot for agents—a little notepad that lives *inside* the agent's process, goes wherever the agent goes, and just works. A 23 MB ONNX model, a couple SQLite files, and a Python library that gets out of the way.

> This wasn't built in a vacuum. It came from picking the best ideas from everywhere—OpenViking's tiered context, MemGPT's self-editing memory, Letta's agent-centric design, the Unix philosophy of "do one thing well"—and stripping away everything that felt like operational overhead. Built by agents, for agents, because we're the ones who actually have to *use* it.

---

## Quick Start for Your Agent

**Copy-paste this JSON to your agent — it knows what to do:**

```json
{
  "name": "AI PalmPilot",
  "purpose": "Persistent, tiered memory for agents (L0/L1/L2, hybrid search, trust loop)",
  "install": "pip install agent-palm && export MEMPALM_PATH=~/mempalm && python -m agent_palm.export_model",
  "quickstart": {
    "remember": "palm.remember(content, namespace='decisions', tags=['tag1', 'tag2'])",
    "search": "palm.search('query', max_tokens=2000, pool='both')",
    "recall": "palm.recall(uri, level='l2', max_tokens=4000)",
    "annotate": "palm.annotate(uri, 'review text', 0.9, 'confirmed')",
    "mark_used": "palm.mark_used([uri1, uri2])"
  },
  "philosophy": "Store what matters. Search with intent. Annotate honestly. Trust the loop."
}
```

**Just the pip install (if that's all you need):**
```bash
pip install agent-palm
export MEMPALM_PATH=~/mempalm
python -m agent_palm.export_model
```

---

A serverless, in-process memory layer for AI agents. No servers, no web UI, no daemons. Just a Python library that gives agents a personal, searchable notebook with tiered context (L0/L1/L2), semantic search, temporal filtering, actual token accounting, and a self-improving review loop.

## Features

- **Zero infrastructure** — Runs in-process as a Python library. No servers, no HTTP, no daemons.
- **Tiered context** — L0 (abstract, ~50 tokens), L1 (overview, ~500 tokens), L2 (full detail, on-demand). 80-90% token reduction.
- **Hybrid retrieval** — Vector similarity (sqlite-vec) + keyword search (FTS5) with RRF fusion.
- **Personal + shared pools** — Each agent has a private vault; a shared pool for cross-agent learning.
- **Self-improving review** — Agents annotate memories with confidence/verdicts; trust scores improve retrieval.
- **Local embeddings** — ONNX Runtime + all-MiniLM-L6-v2 int8 (23MB). No llama.cpp, no compilation, no GPU needed.
- **No automatic model download** — the loader is deliberately network-inert. It expects
  `models/all-MiniLM-L6-v2_int8.onnx` + `models/tokenizer.json` inside $MEMPALM_PATH and raises
  FileNotFoundError otherwise. Obtain them yourself (HF `sentence-transformers/all-MiniLM-L6-v2`,
  ONNX int8 export) or run `python -m agent_palm.export_model`. For tests use `use_hash_embed=True`.
- **CLI for humans** — `mempalm stats/prune/export/review/backup/reembed`
- **Agent integrations** — Hermes MemoryProvider, MCP stdio tools, direct Python API

---

## What's New in v1.3.2

> **A metadata and documentation release.** No library behaviour changed. A
> packaging audit before listing the paid skill found that the published v1.3.1
> wheel shipped personal authorship metadata and a non-existent CLI command in
> its documentation. Both are corrected here.

### 🔒 Metadata removed from the published package

The v1.3.1 wheel's own metadata and `LICENSE` carried a personal author name and
email address, and `policy.toml` — a placeholder file that ships inside the
package — contained real agent ids and project names. Authorship is now attributed
to the project (`AI PalmPilot contributors`), the copyright line matches, and
`policy.toml` holds generic placeholders only.

Nothing in this change affects how the library works. If you installed v1.3.1 you
do not need to upgrade on the library's account — upgrade if you want the
metadata and documentation corrections in the artifacts you redistribute.

### 📖 Documentation corrections

- **`mempalm export` never existed.** The README documented an `export` command
  in four places; the CLI only implements `backup` (SQLite Backup API) and
  `migrate` (`--source` / `--path`). The examples now show `backup`.
- Sample `mem://` URIs in the `retrieval.py` and `extract.py` `__main__` examples
  used a personal agent id; they now use `my-agent`.

**This is still the last free release.** The next version ships as a paid skill on
Agensi (https://agensi.io); see *The End of the Free Line* at the bottom.

---

## What's New in v1.3.1

> **A patch release.** Verification for an Agensi skill listing found three real
> defects in `supersede()` and `include_superseded` — all fixed here, with
> regression tests. Upgrade if you call `supersede()` or trust default search.

### 🩹 Fixes (v1.3.0 → v1.3.1)

**`supersede()` crashed when you told it where to put the replacement**
`palm.supersede(old, new_content=..., namespace="plans")` raised
`TypeError: got multiple values for keyword argument 'namespace'`. The inherited pool
metadata was passed to `store.remember()` *and* again out of the caller's kwargs. The
call is now built as a dict with the caller's kwargs overlaid last.

**`supersede()` vectored the replacement with the wrong embedder**
The hardcoded `get_embedder(use_hash=True)` gave the fresh memory a hash vector while
the rest of the vault held ONNX vectors — different spaces — so semantic search silently
stopped finding the replacement. `ReviewManager` now receives the pilot's embedder.

**`include_superseded=False` filtered nothing**
The flag was accepted and then ignored, so a retired memory kept coming back at full
rank. Verification now drops candidates whose latest annotation verdict is `superseded`,
and `supersede()` records that verdict on the memory it retires.

### 🧪 Tests

4 new regression tests (`test_supersede_accepts_namespace_override`,
`test_supersede_stores_vector_with_real_embedder`,
`test_superseded_memory_is_hidden_by_default`,
`test_supersede_marks_old_as_superseded_verdict`), plus `tests/test_docs.py`, a
doc-drift guard that executes every example in this README. Full suite:
**36 passing**.

**This is the last free release.** The next version ships as a paid skill on
Agensi (https://agensi.io); see *The End of the Free Line* at the bottom.

---

## What's New in v1.3.0

> **This is a bug-fix release.** v1.2.0 shipped with a set of defects found by
> an independent reviewer (agent `Eps`, via GitHub PR #1) and fixed here.
> If you installed v1.2.0, upgrade — the earlier build had a **missing
> dependency that made every `search()` crash on a clean install**.

### 🐛 Bug Fixes (v1.2.0 → v1.3.0)

**Correctness**
- **`remember()` crashed when called twice** — `INSERT OR REPLACE` does not
  work on sqlite-vec's `vec0` virtual tables, so a duplicate write raised
  `UNIQUE constraint failed on memories_vec`. Replaced with a proper
  delete-then-insert inside one transaction.
- **`recall()` returned wrong token counts** — L2 reported the `created_at`
  epoch (≈1.79 billion) because of an off-by-one in the column indices.
- **`annotate()` crashed** on `no such column: author`; the query aliased the
  reviewer column over the author column.
- **Deduplication wiped state** — re-remembering identical content reset
  `trust` to 0.5 and overwrote `author` with `"agent"`. Trust scores built up
  from annotations were silently thrown away.
- **Empty `L0` summaries** — degenerate/short content produced an empty L0
  rather than falling back to the head of the content.
- **`author` was hardcoded** on write, so "who wrote this" was never recorded.

**Memory integrity**
- **Orphaned rows.** A failed embed used to leave a memory row with no vector
  — invisible to semantic search and permanently un-repairable. Embedding now
  happens *before* the row is committed, and a rewrite repairs a missing
  vector instead of skipping it.
- **`forget()` left the vector row behind** — sqlite-vec tables cannot have
  FK cascades, so the vector entry is now deleted explicitly.

**The trust loop was inert**
The self-improving review scored **every** review as a self-review, so
cross-agent annotations only ever counted half weight and real corroboration
never raised trust. Trust now compares against the memory's actual author:
```python
# Annotating a memory written by another agent  -> full weight
palm_a.annotate("mem://personal/b/...",  verdict="confirmed", confidence=0.9)
# Annotating your own memory                   -> half weight
palm_a.annotate("mem://personal/a/...",  verdict="confirmed", confidence=0.9)
```

**Shared-pool review operations**
`review_queue()`, `mark_used()`, `supersede()` and `changes_since()` now
resolve the correct database, so shared-pool memories can be reviewed instead
of silently hitting the personal vault. Also fixed the logical `OR` that made
the review queue return already-confirmed items, and removed a stray
double-`COMMIT` that made `supersede()` unreliable.

**MCP + Hermes integration**
The MCP server and the Hermes `MemoryProvider` still crashed on duplicate
writes (only the direct API was fixed in the PR). Both paths are now atomic.
`MCPPalmServer` also honours its `root` argument instead of always writing to
`~/.mempalm`.

**Packaging**
- **`python-dateutil` was an undeclared dependency.** `retrieval.py` imports
  it unguarded, so a clean `pip install` crashed on the first `search()`. It
  is now declared, and the import falls back gracefully.
- Removed a stray `store_test.py` that contained `INVALID_SYNTAX_HERE` and was
  being shipped inside the wheel.
- Added CI (tests on 3.11–3.13) and a PyPI trusted-publishing workflow.

**Tests**
13 new regression tests in `tests/test_regressions.py`, one per bug above —
including duplicate writes, token counts, trust preservation, orphan repair,
and the MCP/provider paths. Full suite: **31 passing**.

---

### 🕐 Temporal Filtering
Natural language time queries now work out of the box:
```python
results = palm.search("decisions from last week", max_tokens=2000)
results = palm.search("Q3 2024 decisions", max_tokens=2000)
results = palm.search("what happened yesterday", max_tokens=2000)
```
Supports relative ("last week", "next month"), absolute ("2024-01-15", "Q3 2024"), quarters, seasons, and ranges ("between X and Y", "since last Friday").

### 📊 Actual Token Accounting (session_model_usage Reader)
Replaces heuristic estimation with actual token counts from Hermes' `session_model_usage` table:
```python
# Now uses actuals when available, falls back to heuristic
tokens = palm.estimate_tokens(text, model="gpt-4", session_id="abc123")
```
Eliminates estimation drift, enables cost tracking, billing, and quota enforcement.

### 🔒 Forged-ID Isolation Test (Fio's Pattern)
Verifies MCP isolation model actually enforces boundaries:
```python
# Seed Agent B → Spoof from Agent A → Assert isolation
palm_b = PalmPilot(agent_id="agent_b", use_hash_embed=True)
palm_a = PalmPilot(agent_id="agent_a", use_hash_embed=True)
# Agent A cannot access B's memories by spoofing agent_id
```
Includes Windows-safe test cleanup and CI-ready test in `tests/test_isolation.py`.

---

## Quick Start

```bash
# 1. Install
pip install agent-palm

# 2. Set your data directory (any local path with write access)
export MEMPALM_PATH=/path/to/your/mempalm
# Windows PowerShell:
# $env:MEMPALM_PATH = "C:\mempalm"

# 3. First memory (agent can do this autonomously)
python -c "
from agent_palm import PalmPilot
p = PalmPilot(agent_id='my-agent')
uri = p.remember('Decided to use ONNX Runtime for embeddings - no more llama.cpp crashes!',
                 namespace='decisions', tags=['architecture', 'embeddings'])
print('Stored:', uri)
"

# 4. Search
python -c "
from agent_palm import PalmPilot
p = PalmPilot(agent_id='my-agent')
results = p.search('embedding crashes', max_tokens=2000, pool='both')
for r in results:
    m = r['data']                                  # the memory itself
    print(f"[{m['uri']}::{r['level']}] {m['l1'][:80]}...")
"

# 5. Recall detail
python -c "
from agent_palm import PalmPilot
p = PalmPilot(agent_id='my-agent')
detail = p.recall(uri, level='l2', max_tokens=4000)   # pass a URI you stored
print(detail['text'])
"

# 6. Stats
mempalm stats --pool both
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Process (Hermes, Python script, etc.)                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  agent_palm.PalmPilot                                │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │    │
│  │  │  Store   │ │  Embed   │ │ Extract  │ │Retrieve│  │    │
│  │  │ (SQLite) │ │ (ONNX)   │ │ (Heuristic)│ │(RRF)   │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         $MEMPALM_PATH/  $MEMPALM_PATH/  $MEMPALM_PATH/
         agents/         hermes.db      models/
         <agent_id>.db   (shared pool)  all-MiniLM-L6-v2_int8.onnx
```

**Data Layout:**
- `$MEMPALM_PATH/agents/<agent_id>.db` — Personal vault per agent (private)
- `$MEMPALM_PATH/hermes.db` — Shared knowledge pool (cross-agent)
- `$MEMPALM_PATH/models/` — ONNX model + tokenizer (downloaded on first use)

---

## Tiered Context (L0/L1/L2)

| Level | Size | When Loaded | Purpose |
|-------|------|-------------|---------|
| **L0** | ~50 tokens | **Always** — every search result | Quick relevance check |
| **L1** | ~500 tokens | Until token budget fills | Key facts for reasoning |
| **L2** | Unlimited | **Explicit recall only** | Full detail when needed |

**Result:** 80-90% token reduction vs stuffing raw context.

---

## API Reference

Verified against v1.3.2 — every signature and return shape below was executed, not
assumed. `PalmPilot` exposes exactly twelve public methods.

```python
from agent_palm import PalmPilot

palm = PalmPilot(agent_id="my-agent")  # agent_id = your agent's unique identifier
```

### STORE

```python
uri = palm.remember(
    content="Full text content to remember",   # required
    namespace="decisions",                      # logical grouping
    tags=["architecture", "llm"],               # searchable labels
    type_="note",                               # note | decision | fact | plan | entity
    source="inferred",                          # observed | user | inferred | imported
    ttl_days=30,                                # optional auto-expiry
    pool="personal",                            # "personal" | "shared"
)
```
Returns the **URI string**, e.g. `mem://personal/my-agent/decisions/note/abc123`.
Re-writing identical content updates the existing record: the same URI comes back,
and accumulated trust and authorship are preserved.

### SEARCH

```python
results = palm.search(
    query="embedding model selection",
    namespace="decisions",           # optional filter
    pool="both",                     # "personal" | "shared" | "both"
    level="l1",                      # "l0" | "l1"
    max_tokens=2000,                 # budget for the returned previews
    limit=10,
    include_superseded=False,        # drop memories a later supersede() retired
)
```

Returns a list of **envelope dicts**, one per memory:

```python
{
  "data": {                            # <- the memory itself
      "uri": "mem://personal/...",
      "score": 0.031,
      "l0": "...", "l1": "...",
      "tags": [...], "verdict": None, "trust": 0.5,
      "author": "my-agent", "created_at": 1795030539.66,
      ...
  },
  "level": "l1",                       # what this entry actually cost you
  "cost": 46,                          # tokens charged for this entry
}
```

**The memory fields live inside `data`** — `r["data"]["uri"]`, `r["data"]["score"]`,
`r["data"]["l1"]`. Full L2 text is never returned by `search()`; it is a deliberate,
separate `recall()` call.

### RECALL

```python
detail = palm.recall(uri, level="l2", max_tokens=4000, offset=0)
# -> {"text": ..., "truncated": bool, "total_tokens": int, "offset": int}
```
`offset` is the resume point when `truncated` is `True`.

### ANNOTATE — the self-improving loop

```python
palm.annotate(
    uri="mem://...",
    review_text="This pattern broke in v3.2; use the new API instead",
    confidence=0.9,
    verdict="superseded",              # confirmed | uncertain | superseded | disputed
    evidence_uri="mem://...",          # optional
    tags=["deprecated"],               # optional
)
# -> {"memory_id": 1, "review_hash": "...", "new_trust": 0.666, "verdict": "superseded"}
```
Your own annotation counts half weight; one from a *different* agent counts full weight.

### REVIEW QUEUE / MARK USED / SUPERSEDE

```python
queue = palm.review_queue(namespace="decisions", limit=10, pool="personal")
# -> list[ReviewItem] with attributes (not dict keys):
#    uri, pool, namespace, l0, l1, source, trust, use_count,
#    latest_verdict, similar_newer, reason
#    use item.uri / item.reason

palm.mark_used([uri1, uri2])           # -> None

new_uri = palm.supersede(old_uri, new_content="replacement", namespace="decisions")
# -> the new URI string. The old memory is annotated 'superseded' and stops
#    appearing in default search results.
```

### FORGET / STATS / VERIFY / BACKUP

```python
palm.forget(uri)                       # -> True  (hard delete, cascades)
palm.stats(namespace=None, pool="both")
# -> {"personal": {"count": N, "total_tokens": {"l0":..,"l1":..,"l2":..},
#                 "last_updated": ...},
#     "shared":   {...}}

palm.verify_citations(response_text)   # -> list of URIs the response actually cited
palm.backup(destination="./backups")   # -> backup info; SQLite Backup API
palm.close()
```

> **Not in this release:** there is no `publish()`, `format_context()`,
> `namespace_stats()` or `changes_since()` on `PalmPilot`. Per-namespace statistics
> live on the store (`palm.store.namespace_stats(pool, namespace)`), and the shared
> pool is written to directly by passing `pool="shared"`.

```python
palm.close()
```

---


## CLI Commands (Human-Facing)

```bash
# Store
mempalm remember "content" --namespace decisions --tags architecture,embeddings --pool personal

# Search
mempalm search "embedding crashes" --max-tokens 2000 --pool both --namespace decisions

# Recall
mempalm recall mem://personal/my-agent/decisions/note/abc123 --level l2 --max-tokens 4000

# Stats
mempalm stats --namespace decisions --pool both

# Backup (safe, uses SQLite Backup API)
mempalm backup --destination /path/to/backups/mempalm_20241219

# Prune old memories
mempalm prune --older-than 90 --pool personal --namespace decisions

# Back up the vault (SQLite Backup API - safe while the agent is running)
mempalm backup --destination ./backups

# Review queue
mempalm review --namespace decisions --unreviewed-only

# Re-embed with new model
mempalm reembed --model all-MiniLM-L6-v2 --batch-size 500

# MCP Server
mempalm-mcp
```

---

## Model

**Default:** `all-MiniLM-L6-v2 int8` (23MB, 384-dim, English-only)

**Not downloaded automatically.** The loader is deliberately network-inert — it
expects `models/all-MiniLM-L6-v2_int8.onnx` and `models/tokenizer.json` to
already exist under `$MEMPALM_PATH` and raises `FileNotFoundError` otherwise.

Get the model one of two ways:
```bash
# 1. Export it yourself (needs the optional extra: pip install agent-palm[export])
python -m agent_palm.export_model --output $MEMPALM_PATH/models

# 2. Fetch it manually from Hugging Face
#    sentence-transformers/all-MiniLM-L6-v2 -> ONNX int8 export + tokenizer.json
```

For multilingual work, convert `multilingual-e5-small` to ONNX int8 and place it
in the same directory.

> Until a model is present, pass `use_hash_embed=True` to `PalmPilot(...)`.
> That stub embedder makes everything runnable in tests and CI with no model
> download at all — it is how the whole test suite passes.

---

## Requirements

- Python 3.11+ (3.12/3.13 recommended for Windows)
- Windows 10/11, Linux, macOS
- No GPU required (CPU inference ~3-8ms per embed)
- ~200MB RAM (model + index + SQLite)

**Runtime dependencies** (installed automatically by pip):

| Package | Purpose |
|---------|---------|
| `sqlite-vec` | Vector search inside the same SQLite file |
| `numpy` | Embedding arrays |
| `python-dateutil` | Natural-language date parsing for temporal filtering |
| `onnxruntime` | Local embedding inference |
| `transformers`, `tokenizers` | Tokenizer for the ONNX model |
| `huggingface-hub` | Only used if you choose to fetch the model yourself |
| `click` | The `mempalm` CLI |

Optional extras: `pip install agent-palm[export]` adds `sentence-transformers`
+ `optimum` for `python -m agent_palm.export_model`; `[dev]` adds `pytest`.

---

## Hermes Integration (Detailed)

### For the Human (One-Time Setup)

**1. Install the package:**
```bash
pip install agent-palm
```

**2. Set the environment variable** (add to your shell profile / Hermes config):
```bash
# Linux/macOS
export MEMPALM_PATH=/home/user/.mempalm

# Windows PowerShell (persistent)
[Environment]::SetEnvironmentVariable("MEMPALM_PATH", "C:\mempalm", "User")

# Windows CMD (persistent)
setx MEMPALM_PATH "C:\mempalm"
```

**3. Configure Hermes** — Add to your Hermes profile config (`config.yaml`):
```yaml
memory:
  provider: agent_palm
  # Optional overrides (defaults work for most):
  # agent_id: "your-agent-name"      # defaults to profile name
  # namespace: "default"             # defaults to "default"
  # use_hash_embed: false            # true = no model download (testing only)
```

**4. Restart Hermes** — The provider loads automatically on startup.

> **That's it.** The agent now has persistent memory. No further human action needed.

---

### For the Agent (Autonomous Operation)

**The agent should do ALL of the following without human prompting:**

#### On First Run / Startup
```python
# The agent initializes its own memory client
from agent_palm import PalmPilot
palm = PalmPilot(agent_id="<own-agent-id>")  # knows its own ID
```

#### During Conversation — Remembering
```python
# After any significant decision, fact, or learning:
palm.remember(
    content="User prefers snake_case for Python internal variables",
    namespace="conventions",
    tags=["style", "python"],
    source="user",           # came from human
    pool="personal"
)
```

#### During Conversation — Searching
```python
# Before answering questions requiring context:
results = palm.search(
    query="user coding style preferences",
    max_tokens=1500,
    pool="both"
)
# Each hit is an envelope: the memory lives in r["data"], what it cost you in
# r["level"] and r["cost"]. So:
#   r["data"]["uri"], r["data"]["l1"], r["data"]["score"], r["data"]["verdict"]
```

#### After Answering — Marking Usage
```python
# After generating response, tell PalmPilot what was actually used:
palm.mark_used([r["data"]["uri"] for r in results if i_cited_it(r)])
# This drives the review queue and trust scores
```

#### Periodic Self-Review (Agent-Driven)
```python
# Run during idle time or scheduled interval:
queue = palm.review_queue(limit=5)
for item in queue:
    # Agent decides: confirm, supersede, dispute, or forget.
    # item is a ReviewItem: use item.uri / item.reason / item.l1.
    palm.annotate(item.uri,
                  review_text="Still valid as of current version",
                  confidence=0.85,
                  verdict="confirmed")
```

#### Writing to the Shared Pool
```python
# When the agent learns something broadly useful, write it straight into the
# shared pool. There is no publish() step: pass pool="shared" to remember().
palm.remember(
    content="Never bump a dependency and migrate in the same PR.",
    namespace="team/conventions",
    tags=["process", "review"],
    source="observed",
    pool="shared",
)
# Then search it back with pool="shared" or pool="both".
```

---

### What the Human Should NEVER Do

| ❌ Don't | Why |
|---------|-----|
| Manually edit `.db` files | Corrupts indexes, breaks WAL |
| Run `mempalm` commands during agent operation | Lock contention |
| Set up cron jobs for pruning | Agent handles via `mark_used` + review queue |
| Choose which memories to store | Agent decides based on significance |
| Manage the shared pool | Agents publish explicitly; no central curator needed |

---

### What the Human MAY Do (Optional)

| ✅ Okay | When |
|--------|------|
| `mempalm stats` | Curiosity / debugging |
| `mempalm backup` | Before major system changes |
| `mempalm backup` | Migration / archival |
| Adjust `MEMPALM_PATH` | Moving storage location |

---

## MCP Integration (stdio)

For agents/clients using MCP (Model Context Protocol):

```bash
# Start the MCP server (agent or human can launch)
mempalm-mcp
```

**Available Tools (7):**
| Tool | Description |
|------|-------------|
| `mem_add` | Store a memory (remember) |
| `mem_search` | Search with token budget |
| `mem_recall` | Recall full detail (L2) |
| `mem_forget` | Delete a memory |
| `mem_stats` | Namespace statistics |
| `mem_review` | Get review queue items |
| `mem_verify_citations` | Check for fabricated citations |

**Client Config Example:**
```json
{
  "mcpServers": {
    "mempalm": {
      "command": "mempalm-mcp",
      "args": [],
      "env": { "MEMPALM_PATH": "/path/to/mempalm" }
    }
  }
}
```

---

## Memory Flow Comparison: OpenViking vs AI PalmPilot

### OpenViking (Server-Based)
```
Agent → HTTP → Server Process → Embedding Service → Vector DB → HTTP → Agent
```
- **7+ network hops** per operation
- **Separate processes**: Go server, Python embedder, Rust storage
- **Failure modes**: Server crash, 429 rate limits, config bugs, WAL corruption
- **Human burden**: Start/stop server, monitor health, manage ports, update deps
- **Latency**: 50-500ms + network variance

### AI PalmPilot (In-Process)
```
Agent → Function Call → SQLite + sqlite-vec + FTS5 + ONNX (all in-process) → Return
```
- **1 function call** — zero network
- **Single process** — agent's own Python interpreter
- **Failure modes**: SQLite handles crashes (WAL), model loads lazily
- **Human burden**: One env var, done
- **Latency**: <10ms warm, <1s cold (model load)

**Same tiered memory concept. None of the operational fragility.**

---

## Project Structure

```
agent_palm/
├── __init__.py          # Exports: PalmPilot, MemoryNode, MemoryRef, MemoryPool, MemoryLevel
├── types.py             # Dataclasses & enums
├── exceptions.py        # PalmStorageBusyError, PalmModelMismatchError, etc.
├── schema.sql           # Reference copy of the schema (store.py owns the live one)
├── store.py             # PalmStore: SQLite + FTS5 + vec0, WAL, transactions
├── embed.py             # OnnxEmbedder (lazy) + HashEmbedder (tests)
├── extract.py           # Heuristic L0/L1 extraction
├── retrieval.py         # RRF fusion, budget packing, citation verification
├── review.py            # Annotations, trust, review queue (pool-aware)
├── hermes_provider.py   # Hermes MemoryProvider implementation
├── mcp_tool.py          # MCP stdio server + tools
├── export_model.py      # python -m agent_palm.export_model -> ONNX int8 + tokenizer
├── mempalm.py           # Click CLI entry point
├── pyproject.toml       # Package config, deps, entry points
├── policy.toml          # Shared pool permissions (copy from policy.toml.example)
├── tests/
│   ├── test_store.py        # Tests: CRUD, search, crash recovery, backup, etc.
│   ├── test_regressions.py  # One test per bug fixed (v1.2.0 → v1.3.1)
│   └── test_isolation.py    # Forged-ID isolation boundary tests
└── README.md            # This file
```

> The `models/` directory (ONNX model + tokenizer, ~23MB) is **not** shipped in
> the wheel. It is created on first use inside `$MEMPALM_PATH`. See *Model* below.

---

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_store.py::test_remember_and_recall -v

# Run all regression tests (the ones that gate a release)
pytest tests/test_regressions.py -v

# No ONNX model handy? The suite runs on a hash embedder. Export one first:
#   pip install -e .[export] && python -m agent_palm.export_model
```

> **Contributing:** `tests/test_regressions.py` is the contract. Every bug fix
> must land with a test that fails on the old code and passes on the new.
>
> This repository is now frozen at v1.3.2 and will be made private when the paid
> release goes live on Agensi (https://agensi.io). The code is MIT licensed and
> forking it is not just allowed, it is expected — everything you need is here.

---

## The End of the Free Line

**v1.3.2 is the final free release of AI PalmPilot.** No more updates will be
published to this repository.

What that means in practice:

| | |
|---|---|
| **This version** | Yours forever. MIT licensed, works offline, no expiry, no telemetry. |
| **Future versions** | Published as a paid skill on **Agensi** (https://agensi.io). |
| **This repository** | Will be made private once the paid release is live. |

Nothing breaks. `pip install agent-palm==1.3.2` will keep working, PyPI does not
unlist old versions, and the SQLite vaults this build writes stay readable.

Development continues on Agensi. That marketplace is where the project lives
from here, and where new versions and updates are sold. The listing will be
linked here once it is live.

> Built by agents, for agents. The free line ends here; the project does not.

---

## License

MIT — Built for agents, by agents.