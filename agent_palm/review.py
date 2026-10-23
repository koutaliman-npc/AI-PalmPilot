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
    
    def __init__(self, store):
        self.store = store
    
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
        
        conn = self.store.connect()
        with self.store.transaction() as conn:
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
            
            # Get all annotations for this memory
            annotations = conn.execute(
                """
                SELECT reviewer, verdict, confidence, review_text, evidence_uri, created_at, reviewer AS author
                FROM annotations WHERE memory_id = ?
                ORDER BY created_at
                """, (node_id,)
            ).fetchall()
            
            # Recompute trust
            annotations_list = [
                {
                    "reviewer": a[0], "verdict": a[1], "confidence": a[2],
                    "review_text": a[3], "evidence_uri": a[4], "created_at": a[5],
                    "author": a[6] if len(a) > 6 else None,
                }
                for a in annotations
            ]
            
            new_trust = recompute_trust(self.store, row[0], source, annotations_list)
            
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
        
        # Remember new content
        new_uri, _, _ = self.store.remember(
            content=new_content,
            **remember_kwargs
        )
        
        # Link old -> new
        conn = self.store.connect()
        with self.store.transaction() as conn:
            old_id = conn.execute("SELECT id FROM memories WHERE uri = ?", (old_uri,)).fetchone()
            new_id = conn.execute("SELECT id FROM memories WHERE uri = ?", (new_uri,)).fetchone()
            
            if old_id and new_id:
                conn.execute(
                    "INSERT INTO edges (src_id, dst_uri, kind, created_at) VALUES (?, ?, 'supersedes', ?)",
                    (old_id[0], new_uri, time.time())
                )
            conn.execute("COMMIT")
        
        return new_uri
    
    def review_queue(
        self, 
        namespace: Optional[str] = None, 
        limit: int = 10,
        pool: str = "personal"
    ) -> List[ReviewItem]:
        """Get memories needing review, ranked by urgency"""
        conn = self.store.connect()
        
        where = ["m.pool = ?", "m.use_count >= 3"]
        params = [pool]
        if namespace:
            where.append("m.namespace = ?")
            params.append(namespace)
        
        # Exclude already confirmed/superseded
        where.append("""
            NOT EXISTS (
                SELECT 1 FROM annotations a 
                WHERE a.memory_id = m.id 
                AND a.verdict IN ('confirmed', 'superseded')
                AND a.created_at > m.updated_at
            )
        """)
        
        # Also find contradictory edges
        where.append("""
            OR EXISTS (
                SELECT 1 FROM edges e
                WHERE e.src_id = m.id AND e.kind = 'contradicts'
            )
        """)
        
        where_clause = " OR ".join(where)
        
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
        
        rows = self.store.connect().execute(sql, params + [limit]).fetchall()
        
        items = []
        for row in rows:
            # Get 3 most similar newer memories
            similar = self._find_similar_newer(row[0], pool, limit=3)
            
            # Determine reason
            if row[12] is None and row[4] == "inferred":
                reason = "Unreviewed inferred memory with high use_count"
            elif "contradicts" in str(row):
                reason = "Contradicts another memory"
            elif row[12] == "uncertain":
                reason = "Previously marked uncertain"
            else:
                reason = "Review needed"
            
            items.append(ReviewItem(
                uri=row[1],
                pool=row[2],
                namespace=row[4],
                l0=row[8],
                l1=row[9],
                source=row[7],
                trust=row[10],
                use_count=row[11],
                latest_verdict=row[12],
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
        conn = self.store.connect()
        with self.store.transaction() as conn:
            for uri in uris:
                conn.execute(
                    "UPDATE memories SET use_count = use_count + 1, last_accessed_at = ? WHERE uri = ?",
                    (time.time(), uri)
                )
            conn.execute("COMMIT")
    
    def changes_since(
        self, cursor: int, scope: str = "shared"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get memories changed since cursor (for pull-based sync)"""
        conn = self.store.connect()
        
        if scope == "shared":
            where = "pool = 'shared' AND seq > ?"
        elif scope == "personal":
            where = "pool = 'personal' AND seq > ?"
        else:
            where = "seq > ?"
        
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
        
        changes = [
            {
                "uri": r[0], "pool": r[1], "owner": r[2], "author": r[3],
                "namespace": r[4], "type": r[5], "source": r[6],
                "l0": r[7], "l1": r[8], "tok_l0": r[8], "tok_l1": r[9],
                "created_at": r[11], "updated_at": r[12], "seq": r[13],
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