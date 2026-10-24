"""
Self-improving review loop for agent_palm
Annotations, trust scores, review queue, supersession
"""

import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Sequence, Tuple
from dataclasses import dataclass

from .store import PalmStore
from .types import ReviewVerdict, MemorySource


@dataclass
class Annotation:
    """Review annotation on a memory"""
    memory_id: int
    reviewer: str
    verdict: str  # confirmed, uncertain, superseded, disputed
    confidence: float
    review_text: str
    evidence_uri: Optional[str]
    review_hash: str
    created_at: float


@dataclass
class ReviewItem:
    """Item in review queue"""
    uri: str
    pool: str
    namespace: str
    l0: str
    l1: str
    source: str
    trust: float
    use_count: int
    latest_verdict: Optional[str]
    similar_newer: List[Dict[str, Any]]  # 3 most similar newer memories
    reason: str  # Why this needs review


PRIOR_TRUST = {
    "observed": 0.85,
    "user": 0.9,
    "inferred": 0.55,
    "imported": 0.6,
}

VERDICT_WEIGHT = {
    "confirmed": 1.0,
    "uncertain": 0.0,
    "superseded": -1.0,
    "disputed": -0.5,
}


def recompute_trust(
    store, 
    node_id: int, 
    source: MemorySource,
    annotations: List[Dict[str, Any]]
) -> float:
    """
    Recompute trust score based on annotations and source prior.
    Self-review counts half; cross-agent confirmation counts fully.
    Evidence-backed annotations count double.
    """
    num = PRIOR_TRUST.get(source, 0.5)
    den = 1.0
    
    # One vote per agent (latest wins)
    latest_per_agent = {}
    for a in annotations:
        agent = a["reviewer"]
        if agent not in latest_per_agent or a["created_at"] > latest_per_agent[agent]["created_at"]:
            latest_per_agent[agent] = a
    
    for agent, ann in latest_per_agent.items():
        # Self-review counts half
        w = 0.5 if agent == ann.get("author") else 1.0
        
        # Evidence-backed counts double
        if ann.get("evidence_uri"):
            w *= 2.0
        
        verdict_weight = VERDICT_WEIGHT.get(ann["verdict"], 0.0)
        num += ann["confidence"] * w * verdict_weight
        den += w
    
    return num / den if den > 0 else 0.5


def latest_per_agent(annotations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Get latest annotation per agent"""
    latest = {}
    for a in annotations:
        agent = a["reviewer"]
        if agent not in latest or a["created_at"] > latest[agent]["created_at"]:
            latest[agent] = a
    return latest


class ReviewManager:
    """Manages review loop: annotations, trust updates, review queue"""
    
    def __init__(self, store, shared_store=None):
        """
        Args:
            store: the personal store
            shared_store: the shared store (optional). When supplied, URIs in
                the shared pool are resolved against it, so shared memories
                can be annotated/reviewed instead of raising "Memory not found".
        """
        self.store = store
        self.shared_store = shared_store
    
    def _store_for_uri(self, uri: str):
        """Pick the store that owns a given memory URI."""
        if self.shared_store is not None and isinstance(uri, str) and uri.startswith("mem://shared/"):
            return self.shared_store
        return self.store
    
    def _store_for_pool(self, pool: str):
        """Pick the store backing a given pool name."""
        if pool == "shared" and self.shared_store is not None:
            return self.shared_store
        return self.store
    
    def annotate(
        self,
        uri: str,
        reviewer: str,
        verdict: str,
        confidence: float,
        review_text: str,
        evidence_uri: Optional[str] = None,
        tags: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Add review annotation and update trust"""
        if verdict not in ("confirmed", "uncertain", "superseded", "disputed"):
            raise ValueError(f"Invalid verdict: {verdict}")
        
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be 0.0-1.0")
        
        # Resolve the owning store from the URI so shared-pool memories can
        # be annotated (they live in the shared DB, not the personal one).
        store = self._store_for_uri(uri)
        
        conn = store.connect()
        with store.transaction() as conn:
            # Get memory
            row = conn.execute(
                "SELECT id, author, source FROM memories WHERE uri = ?", (uri,)
            ).fetchone()
            if not row:
                from .exceptions import PalmNotFoundError
                raise Exception(f"Memory not found: {uri}")
            
            node_id, author, source = row
            
            # Create annotation
            review_hash = hashlib.sha256(
                f"{uri}{reviewer}{verdict}{confidence}{review_text}{time.time()}".encode()
            ).hexdigest()[:16]
            
            now = time.time()
            
            # Insert annotation (ON CONFLICT DO NOTHING via UNIQUE constraint)
            conn.execute(
                """
                INSERT INTO annotations (
                    memory_id, reviewer, verdict, confidence, review_text,
                    evidence_uri, review_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, reviewer, review_hash) DO NOTHING
                """,
                (node_id, reviewer, verdict, confidence, review_text, evidence_uri, review_hash, now)
            )
            
            # Get all annotations for this memory.
            #
            # NOTE: the `annotations` table has no `author` column - the
            # "author" used by recompute_trust to detect a SELF-review is the
            # memory's author (the agent that wrote the memory), not the
            # reviewer's own name. Selecting `reviewer AS author` here would
            # make `reviewer == author` true for every annotation, silently
            # halving the weight of every cross-agent confirmation and making
            # the self-improving trust loop unable to distinguish a second
            # pair of eyes from the author rubber-stamping their own memory.
            annotations = conn.execute(
                """
                SELECT reviewer, verdict, confidence, review_text, evidence_uri, created_at
                FROM annotations WHERE memory_id = ?
                ORDER BY created_at
                """, (node_id,)
            ).fetchall()
            
            # Recompute trust
            annotations_list = [
                {
                    "reviewer": a[0], "verdict": a[1], "confidence": a[2],
                    "review_text": a[3], "evidence_uri": a[4], "created_at": a[5],
                    "author": author,
                }
                for a in annotations
            ]
            
            new_trust = recompute_trust(store, row[0], source, annotations_list)
            
            # Update trust and verdict in memories table
            latest_verdict = max(annotations, key=lambda x: x[2])[1] if annotations else None
            
            conn.execute(
                "UPDATE memories SET trust = ?, updated_at = ? WHERE uri = ?",
                (new_trust, time.time(), uri)
            )
            
            return {
                "memory_id": row[0],
                "review_hash": review_hash,
                "new_trust": new_trust,
                "verdict": verdict,
            }
    
    def supersede(
            self,
            old_uri: str,
            new_content: str,
            **remember_kwargs
        ) -> str:
            """Create new memory that supersedes old one"""
            # Create new memory
            from .extract import heuristic_extract
            from .embed import get_embedder
        
            embedder = get_embedder(use_hash=True)  # Use hash for now
            l0, l1 = heuristic_extract(new_content)
            embedding = embedder.embed_single(l0 + " " + l1)
        
            # The superseding memory lives in the same pool as the one it
            # supersedes, so resolve the store from the old URI.
            store = self._store_for_uri(old_uri)
        
            # Extract pool metadata from the old memory so the new one lands
            # in the same pool with the same owner/author/namespace.
            conn = store.connect()
            old_meta = conn.execute(
                "SELECT pool, owner, author, namespace FROM memories WHERE uri = ?",
                (old_uri,)
            ).fetchone()
            if not old_meta:
                from .exceptions import PalmNotFoundError
                raise Exception(f"Memory not found: {old_uri}")
            old_pool, old_owner, old_author, old_namespace = old_meta
        
            # Remember new content - inherit pool metadata from the superseded memory
            new_uri, is_dup, _ = store.remember(
                content=new_content,
                pool=old_pool,
                owner=old_owner,
                author=old_author,
                namespace=old_namespace,
                **remember_kwargs
            )
        
            # Store the vector for the new memory (skipped on the dedup path -
            # the existing embedding is still valid because the content hash
            # matched).
            if not is_dup:
                new_node_id = store.get_node_id(new_uri)
                if new_node_id:
                    store.store_vector(new_node_id, embedding)
        
            # Link old -> new
            conn = store.connect()
            with store.transaction() as conn:
                old_id = conn.execute("SELECT id FROM memories WHERE uri = ?", (old_uri,)).fetchone()
                new_id = conn.execute("SELECT id FROM memories WHERE uri = ?", (new_uri,)).fetchone()
            
                if old_id and new_id:
                    conn.execute(
                        "INSERT INTO edges (src_id, dst_uri, kind, created_at) VALUES (?, ?, 'supersedes', ?)",
                        (old_id[0], new_uri, time.time())
                    )
                # Transaction committed automatically by the context manager
                # (an explicit COMMIT here would double-commit and raise).
        
            return new_uri
    
    def review_queue(
        self, 
        namespace: Optional[str] = None, 
        limit: int = 10,
        pool: str = "personal"
    ) -> List[ReviewItem]:
        """Get memories needing review, ranked by urgency"""
        # Shared-pool memories live in the shared DB.
        store = self._store_for_pool(pool)
        conn = store.connect()
        
        # Base conditions (ANDed)
        where = ["m.pool = ?", "m.use_count >= 3"]
        params = [pool]
        if namespace:
            where.append("m.namespace = ?")
            params.append(namespace)
        
        # Exclude already confirmed/superseded
        not_confirmed = """
            NOT EXISTS (
                SELECT 1 FROM annotations a 
                WHERE a.memory_id = m.id 
                AND a.verdict IN ('confirmed', 'superseded')
                AND a.created_at > m.updated_at
            )
        """
        
        # Also include memories that contradict another memory
        contradicts = """
            EXISTS (
                SELECT 1 FROM edges e
                WHERE e.src_id = m.id AND e.kind = 'contradicts'
            )
        """
        
        # (base conditions) AND (not recently confirmed OR contradicts)
        where_clause = " AND ".join(where) + f" AND ({not_confirmed} OR {contradicts})"
        
        sql = f"""
            SELECT m.id, m.uri, m.pool, m.owner, m.author, m.namespace, m.type,
                   m.source, m.l0, m.l1, m.trust, m.use_count, m.created_at,
                   a.verdict as latest_verdict
            FROM memories m
            LEFT JOIN (
                SELECT memory_id, verdict, MAX(created_at) as latest
                FROM annotations
                GROUP BY memory_id
            ) a ON a.memory_id = m.id
            WHERE {where_clause}
            ORDER BY 
                CASE WHEN m.source = 'inferred' AND a.verdict IS NULL THEN 0 ELSE 1 END,
                m.use_count DESC,
                m.trust ASC,
                m.created_at ASC
            LIMIT ?
        """
        
        rows = store.connect().execute(sql, params + [limit]).fetchall()
        
        # Column map for the SELECT above:
        #  0 id, 1 uri, 2 pool, 3 owner, 4 author, 5 namespace, 6 type,
        #  7 source, 8 l0, 9 l1, 10 trust, 11 use_count, 12 created_at,
        # 13 latest_verdict
        items = []
        for row in rows:
            # Get 3 most similar newer memories
            similar = self._find_similar_newer(row[0], pool, limit=3)
            
            # Determine reason
            if row[13] is None and row[7] == "inferred":
                reason = "Unreviewed inferred memory with high use_count"
            elif "contradicts" in str(row):
                reason = "Contradicts another memory"
            elif row[13] == "uncertain":
                reason = "Previously marked uncertain"
            else:
                reason = "Review needed"
            
            items.append(ReviewItem(
                uri=row[1],
                pool=row[2],
                namespace=row[5],
                l0=row[8],
                l1=row[9],
                source=row[7],
                trust=row[10],
                use_count=row[11],
                latest_verdict=row[13],
                similar_newer=similar,
                reason=reason,
            ))
        
        return items
    
    def _find_similar_newer(
        self, node_id: int, pool: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find 3 most similar newer memories via vector search"""
        # This would use vector search - simplified for now
        return []
    
    def mark_used(self, uris: Sequence[str]):
        """Increment use_count for cited memories"""
        # URIs may span both pools - group them so each store is updated
        # with its own transaction.
        by_store = {}
        for uri in uris:
            by_store.setdefault(self._store_for_uri(uri), []).append(uri)
        
        for store, store_uris in by_store.items():
            conn = store.connect()
            with store.transaction() as conn:
                for uri in store_uris:
                    conn.execute(
                        "UPDATE memories SET use_count = use_count + 1, last_accessed_at = ? WHERE uri = ?",
                        (time.time(), uri)
                    )
                # Transaction committed automatically by the context manager
                # (an explicit COMMIT here would double-commit and raise
                # "cannot commit - no transaction is active").
    
    def changes_since(
        self, cursor: int, scope: str = "shared"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get memories changed since cursor (for pull-based sync)"""
        if scope == "shared":
            store = self._store_for_pool("shared")
            where = "pool = 'shared' AND seq > ?"
        elif scope == "personal":
            store = self._store_for_pool("personal")
            where = "pool = 'personal' AND seq > ?"
        else:
            store = self.store
            where = "seq > ?"
        
        conn = store.connect()
        
        rows = conn.execute(
            f"""
            SELECT uri, pool, owner, author, namespace, type, source,
                   l0, l1, tok_l0, tok_l1, tok_l2, created_at, updated_at, seq
            FROM memories
            WHERE {where}
            ORDER BY seq
            LIMIT 1000
            """, (cursor,)
        ).fetchall()
        
        new_cursor = rows[-1][-1] if rows else cursor
        
        # Column map for the SELECT above:
        #  0 uri, 1 pool, 2 owner, 3 author, 4 namespace, 5 type, 6 source,
        #  7 l0, 8 l1, 9 tok_l0, 10 tok_l1, 11 tok_l2,
        # 12 created_at, 13 updated_at, 14 seq
        changes = [
            {
                "uri": r[0], "pool": r[1], "owner": r[2], "author": r[3],
                "namespace": r[4], "type": r[5], "source": r[6],
                "l0": r[7], "l1": r[8], "tok_l0": r[9], "tok_l1": r[10],
                "tok_l2": r[11], "created_at": r[12], "updated_at": r[13],
                "seq": r[14],
            }
            for r in rows
        ]
        
        return changes, new_cursor


def build_review_item(store, node_id: int, reason: str) -> ReviewItem:
    """Build a ReviewItem from a node"""
    conn = store.connect()
    row = conn.execute(
        "SELECT uri, pool, owner, author, namespace, source, l0, l1, trust, use_count, created_at FROM memories WHERE id = ?",
        (node_id,)
    ).fetchone()
    
    if not row:
        return None
    
    return ReviewItem(
        uri=row[0],
        pool=row[1],
        namespace=row[4],
        l0=row[6],
        l1=row[7],
        source=row[5],
        trust=row[8],
        use_count=row[9],
        latest_verdict=None,
        similar_newer=[],
        reason=reason,
    )