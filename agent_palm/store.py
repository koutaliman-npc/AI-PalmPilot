"""
Storage engine for agent_palm - SQLite + FTS5 + sqlite-vec
WAL mode, explicit transactions, zero preprocessing in lock
"""

import sqlite3
import zlib
import json
import time
import hashlib
import os
from typing import Optional, Sequence, Dict, Any, List, Tuple, Iterator
from contextlib import contextmanager

from .types import MemoryPool, MemorySource, MemoryRef
from .exceptions import (
    PalmStorageBusyError, PalmModelMismatchError, 
    PalmStorageFullError, PalmNotFoundError, PalmValidationError
)
from .retrieval import TemporalConstraint, extract_temporal_constraint, apply_temporal_filter


# SQL statements
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;
PRAGMA secure_delete = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT UNIQUE NOT NULL,
    pool TEXT NOT NULL CHECK (pool IN ('personal', 'shared')),
    owner TEXT NOT NULL,
    author TEXT NOT NULL,
    namespace TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'note',
    source TEXT NOT NULL CHECK (source IN ('observed', 'user', 'inferred', 'imported')),
    tags TEXT NOT NULL DEFAULT '[]',
    fields TEXT,
    l0 TEXT NOT NULL,
    l1 TEXT NOT NULL,
    l2 BLOB,
    tok_l0 INTEGER NOT NULL,
    tok_l1 INTEGER NOT NULL,
    tok_l2 INTEGER,
    content_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_accessed_at REAL,
    use_count INTEGER NOT NULL DEFAULT 0,
    seq INTEGER NOT NULL,
    trust REAL NOT NULL DEFAULT 0.5,
    expires_at REAL
);

CREATE INDEX IF NOT EXISTS idx_memories_lookup 
ON memories(pool, namespace, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_dedup 
ON memories(pool, namespace, content_hash);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('confirmed', 'uncertain', 'superseded', 'disputed')),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    review_text TEXT NOT NULL,
    evidence_uri TEXT,
    review_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(memory_id, reviewer, review_hash)
);

CREATE INDEX IF NOT EXISTS idx_annotations_mem ON annotations(memory_id);

CREATE TABLE IF NOT EXISTS edges (
    src_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst_uri TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('supersedes', 'contradicts', 'derived_from', 'related')),
    created_at REAL NOT NULL,
    PRIMARY KEY (src_id, dst_uri, kind)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    l0, 
    l1, 
    l2, 
    tags, 
    content='', 
    contentless_delete=1, 
    tokenize='trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);

CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uri TEXT NOT NULL,
    target_ns TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class PalmStore:
    """SQLite storage engine with FTS5 + sqlite-vec, WAL mode, explicit transactions"""

    EXPECTED_MODEL = "all-MiniLM-L6-v2"
    EXPECTED_DIM = 384

    def __init__(
        self, 
        db_path: str, 
        expected_model: str = EXPECTED_MODEL,
        expected_dim: int = EXPECTED_DIM,
    ):
        self.db_path = db_path
        self.expected_model = expected_model
        self.expected_dim = expected_dim
        self._conn: Optional[sqlite3.Connection] = None

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Open connection with WAL mode and load sqlite-vec extension"""
        if self._conn is None:
            # Python 3.12+ supports autocommit=True, but we need 3.11 compatibility
            conn = sqlite3.connect(
                self.db_path, 
                timeout=5.0, 
                isolation_level=None,  # Disable implicit transactions
            )
            # Set autocommit mode manually for Python 3.12+ compatibility
            try:
                conn.autocommit = True
            except AttributeError:
                pass  # Python < 3.12, isolation_level=None handles it

            # Load sqlite-vec extension
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except ImportError:
                raise RuntimeError(
                    "sqlite-vec not installed. Install with: pip install sqlite-vec"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load sqlite-vec extension: {e}")
            finally:
                conn.enable_load_extension(False)

            # Configure pragmas
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA secure_delete = ON;")

            # Initialize schema
            conn.executescript(SCHEMA_SQL)

            # Verify virtual tables were created
            vt = conn.execute("SELECT name FROM sqlite_master WHERE type='virtual' AND name LIKE 'memories_%'").fetchall()
            print(f"Virtual tables created: {[v[0] for v in vt]}")

            # Verify/initialize model metadata
            self._verify_or_init_meta(conn)

            self._conn = conn

        return self._conn

    def _verify_or_init_meta(self, conn: sqlite3.Connection):
        """Verify embedding model matches or initialize on first run"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)

        row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
        if row is None:
            # First run - initialize
            conn.execute("INSERT INTO meta(key, value) VALUES ('embed_model', ?)", (self.expected_model,))
            conn.execute("INSERT INTO meta(key, value) VALUES ('embed_dim', ?)", (str(self.expected_dim),))
        else:
            current_model = row[0]
            if current_model != self.expected_model:
                from .exceptions import PalmModelMismatchError
                raise RuntimeError(
                    f"Database vector space is '{current_model}', but runtime configured for "
                    f"'{self.expected_model}'. Run `mempalm reembed` before opening."
                )

    def close(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self, immediate: bool = True):
        """Context manager for explicit transactions with BEGIN IMMEDIATE"""
        conn = self.connect()
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def estimate_tokens(self, text: str, model: str = None, session_id: str = None) -> int:
        """Estimate token count using actuals from session_model_usage when available, else heuristic."""
        # Try to get actuals from Hermes session_model_usage table
        if model and session_id:
            actuals = self.get_actual_token_usage(session_id, model)
            if actuals is not None:
                return actuals
        # Fall back to heuristic
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using ASCII/char heuristic (internal)"""
        if not text:
            return 0
        ascii_count = sum(1 for c in text if ord(c) < 128)
        non_ascii = len(text) - ascii_count
        # ASCII ~3.5 chars/token, non-ASCII ~1.2 chars/token, +10% safety margin
        import math
        return math.ceil((ascii_count / 3.5 + non_ascii * 1.2) * 1.1)

    def get_actual_token_usage(self, session_id: str, model: str) -> Optional[int]:
        """
        Get actual token usage from Hermes session_model_usage table.

        Returns total tokens (input + output + cached + thoughts) for the most recent
        entry matching session_id and model, or None if table doesn't exist or no match.
        """
        conn = self.connect()
        try:
            # Check if session_model_usage table exists
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_model_usage'"
            ).fetchone()
            if not row:
                return None

            # Get most recent usage for this session+model
            # Schema assumed: session_id, model, input_tokens, output_tokens, cached_tokens, thoughts_tokens, timestamp
            row = conn.execute(
                """SELECT input_tokens, output_tokens, cached_tokens, thoughts_tokens
                   FROM session_model_usage
                   WHERE session_id = ? AND model = ?
                   ORDER BY timestamp DESC
                   LIMIT 1""",
                (session_id, model)
            ).fetchone()

            if row:
                input_t, output_t, cached_t, thoughts_t = row
                return (input_t or 0) + (output_t or 0) + (cached_t or 0) + (thoughts_t or 0)
            return None
        except Exception:
            # Table schema mismatch or other error - fall back to heuristic
            return None

    def _make_uri(self, pool: str, owner: str, namespace: str, type_: str, content_hash: str) -> str:
        """Generate URI: mem://<pool>/<owner>/<namespace>/<type>/<hash_prefix>"""
        return f"mem://{pool}/{owner}/{namespace}/{type_}/{content_hash[:12]}"

    def write_memory(
        self,
        *,
        pool: str,
        owner: str,
        author: str,
        namespace: str,
        type_: str,
        source: str,
        tags: Sequence[str],
        fields: Optional[Dict[str, Any]],
        l0: str,
        l1: str,
        l2_raw: str,
        tok_l0: int,
        tok_l1: int,
        tok_l2: int,
        embedding: list[float],
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Write memory with all preprocessing done BEFORE transaction.
        Returns (uri, is_near_duplicate, near_duplicate_uri)
        """
        # All preprocessing OUTSIDE transaction
        compressed_l2 = zlib.compress(l2_raw.encode("utf-8"), level=6)
        content_hash = hashlib.sha256(l2_raw.encode("utf-8")).hexdigest()
        tags_json = json.dumps(list(tags))
        fields_json = json.dumps(fields) if fields else None
        now = time.time()

        uri = f"mem://{pool}/{owner}/{namespace}/{content_hash[:12]}"

        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Check for exact duplicate (same content_hash in same pool/namespace)
            dup_row = conn.execute(
                "SELECT uri FROM memories WHERE pool = ? AND namespace = ? AND content_hash = ?",
                (pool, namespace, content_hash)
            ).fetchone()

            is_dup = dup_row is not None
            near_dup_uri = dup_row[0] if is_dup else None

            if is_dup:
                # Update existing record instead of inserting duplicate
                conn.execute(
                    """
                    UPDATE memories SET 
                        l0 = ?, l1 = ?, l2 = ?, tok_l0 = ?, tok_l1 = ?, tok_l2 = ?,
                        tags = ?, fields = ?, updated_at = ?, author = ?, source = ?,
                        trust = 0.5, expires_at = ?
                    WHERE uri = ?
                    """,
                    (
                        l0, l1, zlib.compress(l2_raw.encode("utf-8"), level=6),
                        len(l0.split()) + len(l0) // 4,  # rough token counts
                        len(l1.split()) + len(l1) // 4,
                        len(l2_raw.split()) + len(l2_raw) // 4,
                        json.dumps(list(set(json.loads("[]") + list(tags)))),
                        json.dumps(fields) if fields else None,
                        time.time(), author, source, None,
                        near_dup_uri
                    )
                )
                conn.execute("COMMIT")
                return near_dup_uri, True, near_dup_uri

            # Insert new memory
            seq_val = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM memories"
            ).fetchone()[0]

            conn.execute(
                """
                INSERT INTO memories (
                    uri, pool, owner, author, namespace, type, source, tags, fields,
                    l0, l1, l2, tok_l0, tok_l1, tok_l2, content_hash,
                    created_at, updated_at, seq, trust
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"mem://{pool}/{content_hash[:12]}",  # Will be updated with proper URI
                    "personal", owner, "agent", namespace, "note", "inferred",
                    "[]", None,
                    "", "", None, 0, 0, 0, "dummy",
                    time.time(), time.time(), 0, 0.5
                )
            )

            # Get the actual rowid and update URI with proper hash
            node_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            content_hash = hashlib.sha256(b"placeholder").hexdigest()  # Will be fixed

            # Actually, let's do this properly with the real data
            conn.execute("ROLLBACK")
            raise NotImplementedError("Need to restructure - see below")

        except sqlite3.OperationalError as e:
            conn.execute("ROLLBACK")
            if "busy" in str(e).lower() or "locked" in str(e).lower():
                from .exceptions import PalmStorageBusyError
                raise PalmStorageBusyError(f"Contention on {self.db_path}: write failed. Retryable.") from e
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def remember(
        self,
        *,
        content: str,
        pool: str,
        owner: str,
        author: str,
        namespace: str,
        type_: str = "note",
        source: str = "inferred",
        tags: Sequence[str] = (),
        fields: Optional[Dict[str, Any]] = None,
        ttl_days: Optional[int] = None,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Store a memory with automatic L0/L1 extraction and embedding.
        Returns (uri, is_near_duplicate, near_duplicate_uri)
        """
        # Preprocessing OUTSIDE transaction
        from .extract import heuristic_extract

        l0, l1 = heuristic_extract(content)
        l2_raw = content

        # Token estimates
        tok_l0 = self._estimate_tokens(l0)
        tok_l1 = self._estimate_tokens(l1)
        tok_l2 = self._estimate_tokens(content)

        # Content hash for deduplication
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # URI format: mem://<pool>/<owner>/<namespace>/<type>/<hash12>
        uri = f"mem://{pool}/{owner}/{namespace}/{type_}/{content_hash[:12]}"

        # Prepare data for transaction
        tags_json = json.dumps(list(tags))
        fields_json = json.dumps(fields) if fields else None
        compressed_l2 = zlib.compress(content.encode("utf-8"), level=6)
        now = time.time()
        expires_at = now + (ttl_days * 86400) if ttl_days else None

        with self.transaction(immediate=True) as conn:
            # Check for exact duplicate
            dup_row = conn.execute(
                "SELECT uri FROM memories WHERE pool = ? AND namespace = ? AND content_hash = ?",
                (pool, namespace, content_hash)
            ).fetchone()

            if dup_row:
                # Update existing.
                #
                # NOTE: trust is intentionally NOT reset here. Re-remembering the
                # same content is an idempotent no-op for review state - wiping
                # trust back to the 0.5 prior would destroy trust earned from
                # annotations (confirmed/reviewed memories silently regress).
                # The same applies to `author`: the original author is preserved.
                conn.execute(
                    """
                    UPDATE memories SET 
                        l0 = ?, l1 = ?, l2 = ?, tok_l0 = ?, tok_l1 = ?, tok_l2 = ?,
                        tags = ?, fields = ?, updated_at = ?, source = ?,
                        expires_at = ?
                    WHERE uri = ?
                    """,
                    (
                        l0, l1, zlib.compress(content.encode("utf-8"), level=6),
                        self._estimate_tokens(l0), self._estimate_tokens(l1), self._estimate_tokens(content),
                        tags_json, fields_json, time.time(), source,
                        expires_at, dup_row[0]
                    )
                )

                # Sync FTS
                conn.execute(
                    "INSERT INTO memories_fts(rowid, l0, l1, l2, tags) VALUES (?, ?, ?, ?, ?)",
                    (conn.execute("SELECT id FROM memories WHERE uri = ?", (dup_row[0],)).fetchone()[0], l0, l1, content, json.dumps(list(tags)))
                )

                # Transaction committed automatically by context manager
                return dup_row[0], True, dup_row[0]

            # Generate URI
            uri = f"mem://{pool}/{owner}/{namespace}/{type_}/{content_hash[:12]}"

            # Insert new memory - 23 data columns (matching table exactly)
            # Schema columns: uri, pool, owner, author, namespace, type, source, tags, fields,
            # l0, l1, l2, tok_l0, tok_l1, tok_l2, content_hash,
            # created_at, updated_at, last_accessed_at, use_count, seq, trust, expires_at
            seq_val = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM memories").fetchone()[0]

            # Explicit 23 values matching the 23 data columns.
            # NOTE: `author` must be the real author, not the literal "agent".
            # recompute_trust compares the annotation's reviewer against the
            # memory's author to detect a self-review; with a hardcoded
            # "agent" the comparison could never fire correctly.
            values = (
                uri, pool, owner, author, namespace, type_, source,
                json.dumps(list(tags)), json.dumps(fields) if fields else None,
                l0, l1, zlib.compress(content.encode("utf-8"), level=6),
                self._estimate_tokens(l0), self._estimate_tokens(l1), self._estimate_tokens(content),
                content_hash,
                time.time(), time.time(), None, 0, 1, 0.5, expires_at
            )
            assert len(values) == 23, f"Expected 23 values, got {len(values)}"

            conn.execute(
                """
                INSERT INTO memories (
                    uri, pool, owner, author, namespace, type, source, tags, fields,
                    l0, l1, l2, tok_l0, tok_l1, tok_l2, content_hash,
                    created_at, updated_at, last_accessed_at, use_count, seq, trust, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values
            )

            node_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Sync FTS (contentless)
            conn.execute(
                "INSERT INTO memories_fts(rowid, l0, l1, l2, tags) VALUES (?, ?, ?, ?, ?)",
                (node_id, l0, l1, content, json.dumps(list(tags)))
            )

            # Vector will be inserted by caller after embedding

            # Transaction committed automatically by context manager
            return uri, False, None

    def store_vector(self, node_id: int, embedding: list[float]):
        """
        Store embedding vector for a memory (called after remember).

        NOTE: sqlite-vec's vec0 virtual tables do NOT support INSERT OR REPLACE
        (the "REPLACE" conflict-resolution clause is not honoured for vec0
        primary keys and raises "UNIQUE constraint failed on memories_vec
        primary key"). We therefore DELETE the existing row and INSERT the new
        one, both inside a single transaction so a concurrent reader can never
        observe the vector missing in between.
        """
        import array
        conn = self.connect()
        # Convert list of floats to bytes for sqlite-vec
        vec_bytes = array.array('f', embedding).tobytes()
        with self.transaction(immediate=True) as tx:
            tx.execute("DELETE FROM memories_vec WHERE id = ?", (node_id,))
            tx.execute("INSERT INTO memories_vec(id, embedding) VALUES (?, ?)", (node_id, vec_bytes))

    def search_fts(self, query: str, pool: str, namespace: Optional[str], k: int = 50) -> List[Tuple[int, float]]:
        """Search FTS5, return (rowid, bm25_score)"""
        conn = self.connect()
        where = ["m.pool = ?"]
        params = [pool]
        if namespace:
            where.append("m.namespace = ?")
            params.append(namespace)

        where_clause = " AND ".join(where)
        sql = f"""
            SELECT m.id, bm25(memories_fts) as score
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.rowid
            WHERE {where_clause} AND memories_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """
        params = [pool]
        if namespace:
            params.append(namespace)
        params.extend([query, k])

        return conn.execute(sql, params).fetchall()

    def search_vec(self, embedding: list[float], pool: str, namespace: Optional[str], k: int = 50) -> List[Tuple[int, float]]:
        """Search vec0, return (id, distance)"""
        conn = self.connect()

        # vec0 doesn't support WHERE clauses directly on the vec table
        # We need to join with memories table
        where = ["m.pool = ?"]
        params = [pool]
        if namespace:
            where.append("m.namespace = ?")
            params.append(namespace)

        where_clause = " AND ".join(where)

        # Convert embedding to bytes for vec0
        import array
        vec_bytes = array.array('f', embedding).tobytes()

        sql = f"""
            SELECT m.id, vec_distance_l2(memories_vec.embedding, ?) as dist
            FROM memories_vec
            JOIN memories m ON m.id = memories_vec.id
            WHERE {where_clause}
            ORDER BY dist
            LIMIT ?
        """
        # Parameter order: vec_bytes, pool, [namespace], k
        # The first ? is for vec_distance_l2, then pool/namespace in WHERE, then LIMIT
        final_params = [vec_bytes] + params + [k]

        return conn.execute(sql, final_params).fetchall()

    def fuse_and_fetch(
        self, 
        vec_results: List[Tuple[int, float]], 
        fts_results: List[Tuple[int, float]], 
        k: int = 20,
        max_tokens: int = 2000,
        include_superseded: bool = False,
        query_text: str = "",
        temporal_constraint: Optional[TemporalConstraint] = None
    ) -> List[Dict[str, Any]]:
        """RRF fusion + budget packing + optional temporal filtering"""
        from .retrieval import rrf_fuse, pack_budget

        # Convert to candidate dicts
        candidates = []
        vec_ranks = {rid: i for i, (rid, _) in enumerate(vec_results)}
        fts_ranks = {rid: i for i, (rid, _) in enumerate(fts_results)}

        all_ids = set(vec_ranks.keys()) | set(fts_ranks.keys())

        conn = self.connect()
        placeholders = ",".join("?" * len(all_ids))
        rows = conn.execute(
            f"""
            SELECT id, uri, pool, owner, author, namespace, type, source, tags, fields,
                   l0, l1, tok_l0, tok_l1, tok_l2, created_at, trust, 
                   (SELECT verdict FROM annotations WHERE memory_id = memories.id ORDER BY created_at DESC LIMIT 1) as verdict
            FROM memories
            WHERE id IN ({",".join("?" * len(all_ids))})
            """, list(all_ids)
        ).fetchall()

        for row in rows:
            rid = row[0]
            vec_rank = vec_ranks.get(rid)
            fts_rank = fts_ranks.get(rid)

            # RRF score
            score = 0.0
            if vec_rank is not None:
                score += 1.0 / (60 + vec_rank)
            if fts_rank is not None:
                score += 0.8 / (60 + fts_rank)

            candidates.append({
                "id": row[0],
                "uri": row[1],
                "pool": row[2],
                "owner": row[3],
                "author": row[4],
                "namespace": row[5],
                "type": row[6],
                "source": row[7],
                "tags": json.loads(row[8]) if row[8] else [],
                "fields": json.loads(row[9]) if row[9] else None,
                "l0": row[10],
                "l1": row[11],
                "tok_l0": row[12],
                "tok_l1": row[13],
                "tok_l2": row[14],
                "created_at": row[15],
                "trust": row[16] if row[16] is not None else 0.5,
                "verdict": row[17],
                "score": score,
            })

        # Sort by score desc
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Apply temporal filter if constraint provided
        if temporal_constraint is not None and (temporal_constraint.start is not None or temporal_constraint.end is not None):
            candidates = apply_temporal_filter(candidates, temporal_constraint)

        # Pack into budget
        return pack_budget(candidates, max_tokens)

    def recall(self, uri: str, level: str = "l1", max_tokens: int = 4000, offset: int = 0):
        """Recall memory at specific level with optional pagination"""
        conn = self.connect()
        row = conn.execute(
            """
            SELECT uri, pool, owner, author, namespace, source, tags, fields,
                   l0, l1, l2, tok_l0, tok_l1, tok_l2, created_at, trust
            FROM memories WHERE uri = ?
            """, (uri,)
        ).fetchone()

        if not row:
            from .exceptions import PalmNotFoundError
            raise Exception(f"Memory not found: {uri}")

        if level == "l0":
            text = row[8]
            total = row[11]
        elif level == "l1":
            text = row[9]
            total = row[12]
        elif level == "l2":
            if not row[10]:
                return {"text": "", "truncated": False, "total_tokens": 0, "offset": 0}
            text = zlib.decompress(row[10]).decode("utf-8")
            total = row[13] if row[13] else len(text) // 4
            # Pagination for L2
            if offset > 0 or len(text) > max_tokens:
                end = min(offset + max_tokens, len(text))
                return {
                    "text": text[offset:end],
                    "truncated": end < len(text),
                    "total_tokens": total,
                    "offset": end
                }
        else:
            raise ValueError(f"Invalid level: {level}")

        return {
            "text": text,
            "truncated": False,
            "total_tokens": total,
            "offset": offset
        }

    def forget(self, uri: str) -> Dict[str, Any]:
        """Hard delete memory and all associated data"""
        with self.transaction() as conn:
            # Get node id first
            row = conn.execute("SELECT id, pool FROM memories WHERE uri = ?", (uri,)).fetchone()
            if not row:
                from .exceptions import PalmNotFoundError
                raise Exception(f"Memory not found: {uri}")

            node_id, pool = row

            # Check if shared and find published copies
            published = []
            if pool == "shared":
                # In single-file design, shared is just pool='shared'
                pass

            # Cascading delete (foreign keys handle annotations, edges, fts, vec)
            conn.execute("DELETE FROM memories WHERE uri = ?", (uri,))

            # Transaction committed automatically by context manager

            return {"deleted": True, "uri": uri, "published_copies": published}

    def get_node_id(self, uri: str) -> Optional[int]:
        """Get internal node ID for a URI"""
        conn = self.connect()
        row = conn.execute("SELECT id FROM memories WHERE uri = ?", (uri,)).fetchone()
        return row[0] if row else None

    def get_node_id_by_hash(self, pool: str, namespace: str, content_hash: str) -> Optional[int]:
        """
        Get the internal node ID for an existing memory with the given content hash.

        Used to decide whether a write needs a fresh embedding before the row is
        committed (see PalmPilot.remember).
        """
        conn = self.connect()
        row = conn.execute(
            "SELECT id FROM memories WHERE pool = ? AND namespace = ? AND content_hash = ?",
            (pool, namespace, content_hash),
        ).fetchone()
        return row[0] if row else None

    def has_vector(self, node_id: int) -> bool:
        """Check whether a memory row already has a stored embedding."""
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM memories_vec WHERE id = ? LIMIT 1", (node_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            # sqlite-vec table unavailable - treat as "no vector"
            return False
        return row is not None

    def get_memory(self, uri: str) -> Optional[Dict]:
        """Get full memory record"""
        conn = self.connect()
        row = conn.execute(
            """
            SELECT id, uri, pool, owner, author, namespace, type, source, tags, fields,
                   l0, l1, l2, tok_l0, tok_l1, tok_l2, content_hash,
                   created_at, updated_at, trust, expires_at
            FROM memories WHERE uri = ?
            """, (uri,)
        ).fetchone()

        if not row:
            return None

        return {
            "id": row[0], "uri": row[1], "pool": row[2], "owner": row[3],
            "author": row[4], "namespace": row[6], "type": row[7],
            "source": row[7], "tags": json.loads(row[8]) if row[8] else [],
            "fields": json.loads(row[9]) if row[9] else None,
            "l0": row[10], "l1": row[11], "l2": zlib.decompress(row[12]).decode("utf-8") if row[12] else "",
            "tok_l0": row[13], "tok_l1": row[14], "tok_l2": row[15],
            "content_hash": row[16], "created_at": row[17], "updated_at": row[18],
            "trust": row[19], "expires_at": row[20],
        }

    def namespace_stats(self, pool: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a namespace"""
        conn = self.connect()
        where = ["pool = ?"]
        params = [pool]
        if namespace:
            where.append("namespace = ?")
            params.append(namespace)

        where_clause = " AND ".join(where)

        stats = conn.execute(f"""
            SELECT 
                COUNT(*) as count,
                SUM(tok_l0) as total_l0,
                SUM(tok_l1) as total_l1,
                SUM(tok_l2) as total_l2,
                MAX(updated_at) as last_updated
            FROM memories WHERE {" AND ".join(where)}
        """, params).fetchone()

        return {
            "count": stats[0] or 0,
            "total_tokens": {
                "l0": stats[1] or 0,
                "l1": stats[2] or 0,
                "l2": stats[3] or 0,
            },
            "last_updated": stats[4],
        }

    def backup(self, destination_path: str):
        """Thread/process safe backup using SQLite's online backup API"""
        conn = self.connect()
        dest_conn = sqlite3.connect(destination_path)
        with dest_conn:
            conn.backup(dest_conn, pages=100)
        dest_conn.close()


# Convenience function for quick access
def open_store(db_path: str, **kwargs) -> PalmStore:
    """Open a PalmStore with default config"""
    return PalmStore(db_path, **kwargs)
