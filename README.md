# 🌿 AI PalmPilot v1.2.0 — Memory for Agents, By Agents

**Side Note:** 
> I appreciate the vision behind OpenViking, but when I explored the open‑source implementation, I found it far more complicated than it needed to be. There were too many moving parts, too many subsystems, and yet another server instance running on another localhost. I'm tired of tools that require full server environments just to deliver functionality that should live inside the plugin or extension itself. I don't need a UI dashboard to click through or a memory viewer to babysit. I'm not the one who should be managing that—my agent should. I shouldn't have to open a control panel at all.

> That realization pushed me toward a different idea. Instead of building a massive library with a librarian, why not create something closer to a PalmPilot for AI agents? A lightweight, easily accessible companion that gives them a simple place to store and retrieve what matters. Using a small ONNX model—around 23 MB—as the agent's "memory sidekick," you end up with a compact notepad that can remind them of ongoing tasks, help them catalog information, and support their workflow without the overhead of a full server stack.

> This concept wasn't created in isolation. It came from a collaborative effort across seven commercial models, asking each not only what they recommended but how they would implement it. The answers were surprisingly straightforward. I wanted something built by agents, for agents—something they would actually want to use, not another system we assume they need.

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

## What's New in v1.2

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
    print(f'[{r.uri}::{r.level}] {r.text[:80]}...')
"

# 5. Recall detail
python -c "
from agent_palm import PalmPilot
p = PalmPilot(agent_id='my-agent')
detail = p.recall('mem://personal/my-agent/decisions/note/abc123', level='l2', max_tokens=4000)
print(detail.text)
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

```python
from agent_palm import PalmPilot, MemoryPool, MemoryLevel

palm = PalmPilot(agent_id="my-agent")  # agent_id = your agent's unique identifier

# ─── STORE ───
uri = palm.remember(
    content="Full text content to remember",
    namespace="decisions",           # logical grouping
    tags=["architecture", "llm"],    # searchable labels
    pool=MemoryPool.PERSONAL,        # or MemoryPool.SHARED
    source="observed",               # observed|user|inferred|imported
    type_="note",                    # note|decision|fact|plan|entity
    ttl_days=30,                     # optional auto-expiry
    links=["mem://.../other"],       # optional cross-references
)
# Returns: the URI string, e.g. 'mem://personal/my-agent/decisions/note/abc123'

# ─── SEARCH ───
results = palm.search(
    query="embedding model selection",
    namespace="decisions",           # optional filter
    pool=MemoryPool.BOTH,            # PERSONAL | SHARED | BOTH
    level=MemoryLevel.L1,            # L0 | L1 (L2 not allowed here)
    max_tokens=2000,                 # token budget for returned context
    include_superseded=False,        # skip superseded memories
)
# Returns: list[MemoryNode(uri, pool, owner, author, namespace, level, text, tags, source, verdict, score, created_at)]

# ─── RECALL ───
detail = palm.recall(
    uri="mem://personal/my-agent/decisions/note/abc123",
    level=MemoryLevel.L2,            # L0 | L1 | L2
    max_tokens=4000,                 # cap for L2
    offset=0,                        # pagination for large L2
)
# v1.2.0: recall() returns a dict {"text", "truncated", "total_tokens", "offset"}
# (RecallNode objects are not returned; access r["text"]).

# ─── ANNOTATE (Self-Improving Review) ───
palm.annotate(
    uri="mem://...",
    review_text="This pattern broke in v3.2; use new API instead",
    confidence=0.9,
    verdict="superseded",            # confirmed|uncertain|superseded|disputed
    evidence_uri="mem://...",        # optional supporting memory
)

# ─── REVIEW QUEUE ───
queue = palm.review_queue(namespace="decisions", limit=10)
# Returns: list[ReviewItem(memory, reason, suggested_action, similar_newer)]

# ─── MARK USED (Implicit Signal) ───
palm.mark_used(["mem://...", "mem://..."])  # call after agent cites memories

# ─── SUPERSEDE ───
new_uri = palm.supersede(
    old_uri="mem://...",
    new_content="Updated content replacing the old memory",
    **remember_kwargs  # same args as remember()
)

# ─── PUBLISH TO SHARED POOL ───
palm.publish(uri="mem://personal/my-agent/...", to_namespace="team/conventions")

# ─── FORGET ───
palm.forget(uri="mem://...")  # hard delete with cascade

# ─── UTILITIES ───
context = palm.format_context(nodes, max_tokens=2000)  # grounded context block
fake = palm.verify_citations(agent_response, provided_uris)  # hallucination guard
stats = palm.namespace_stats("decisions", pool=MemoryPool.PERSONAL)
changes = palm.changes_since(cursor=0, scope=MemoryPool.SHARED)  # for sync
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

# Export
mempalm export --namespace decisions --format jsonl > backup.jsonl

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

- Downloaded automatically to `$MEMPALM_PATH/models/` on first use
- For multilingual: convert `multilingual-e5-small` to ONNX int8 and place in `$MEMPALM_PATH/models/`

---

## Requirements

- Python 3.11+ (3.12/3.13 recommended for Windows)
- Windows 10/11, Linux, macOS
- No GPU required (CPU inference ~3-8ms per embed)
- ~200MB RAM (model + index + SQLite)

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
    pool=MemoryPool.PERSONAL
)
```

#### During Conversation — Searching
```python
# Before answering questions requiring context:
results = palm.search(
    query="user coding style preferences",
    max_tokens=1500,
    pool=MemoryPool.BOTH
)
# v1.2.0: search() returns a list of dicts, one per hit:
#   {"data": {id, uri, pool, owner, author, namespace, type, source, tags,
#             l0, l1, tok_*, created_at, trust, verdict, score},
#    "level": "l0"|"l1", "cost": int}
# (the MemoryNode-object examples older documentation shows do not match)
```

#### After Answering — Marking Usage
```python
# After generating response, tell PalmPilot what was actually used:
palm.mark_used([r.uri for r in results if r.was_cited_in_response])
# This drives the review queue and trust scores
```

#### Periodic Self-Review (Agent-Driven)
```python
# Run during idle time or scheduled interval:
queue = palm.review_queue(limit=5)
for item in queue:
    # Agent decides: confirm, supersede, dispute, or forget
    palm.annotate(item.memory.uri, 
                  review_text="Still valid as of current version",
                  confidence=0.85,
                  verdict="confirmed")
```

#### Publishing to Shared Knowledge
```python
# When the agent learns something broadly useful:
palm.publish(uri="mem://personal/my-agent/...", to_namespace="team/best-practices")
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
| `mempalm export` | Migration / archival |
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
├── schema.sql           # Single source of truth for DB schema
├── store.py             # PalmStore: SQLite + FTS5 + vec0, WAL, transactions
├── embed.py             # OnnxEmbedder (lazy) + HashEmbedder (tests)
├── extract.py           # Heuristic L0/L1 extraction
├── retrieval.py         # RRF fusion, budget packing, citation verification
├── review.py            # Annotations, trust, review queue
├── hermes_provider.py   # Hermes MemoryProvider implementation
├── mcp_tool.py          # MCP stdio server + tools
├── mempalm.py           # Click CLI entry point
├── pyproject.toml       # Package config, deps, entry points
├── policy.toml          # Shared pool permissions (copy from policy.toml.example)
├── tests/
│   └── test_store.py    # Tests: CRUD, search, crash recovery, backup, etc.
└── README.md            # This file
```

---

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_store.py::test_remember_and_recall -v

# Benchmark KNN (requires 100k synthetic nodes)
python -m benchmarks.bench_knn --nodes 100000 --dim 384 --runs 100
```

---

## License

MIT — Built for agents, by agents.