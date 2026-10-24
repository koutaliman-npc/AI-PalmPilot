"""
agent_palm - AI PalmPilot
Agent-native memory layer (serverless, in-process)
"""

from .types import (
    MemoryPool,
    MemorySource,
    ReviewVerdict,
    MemoryLevel,
    MemoryRef,
    MemoryNode,
    RecallNode,
)
from .store import PalmStore, PalmStorageBusyError, PalmModelMismatchError
from .embed import OnnxEmbedder, HashEmbedder
from .extract import heuristic_extract
from .retrieval import pack_budget, verify_citations, format_context, build_grounded_context, extract_temporal_constraint, TemporalConstraint, apply_temporal_filter

__all__ = [
    "MemoryPool",
    "MemorySource", 
    "ReviewVerdict",
    "MemoryLevel",
    "MemoryRef",
    "MemoryNode",
    "RecallNode",
    "PalmStore",
    "PalmStorageBusyError",
    "PalmModelMismatchError",
    "OnnxEmbedder",
    "HashEmbedder",
    "heuristic_extract",
    "pack_budget",
    "verify_citations",
    "format_context",
    "build_grounded_context",
    "PalmPilot",
]

__version__ = "1.2.0"


class PalmPilot:
    """
    High-level agent-facing API for agent_palm.
    
    Usage:
        palm = PalmPilot(agent_id="sebastian")
        
        # Store a memory
        uri = palm.remember(
            "Decided to use ONNX Runtime for embeddings - no more llama.cpp crashes!",
            namespace="decisions",
            tags=["architecture", "embeddings"],
        )
        
        # Search with token budget
        results = palm.search(
            "embedding crashes", 
            max_tokens=2000,
            pool="both",
        )
        
        # Explicit full-detail recall
        detail = palm.recall(uri, level="l2", max_tokens=4000)
        
        # Self-improving review
        palm.annotate(uri, "This decision was superseded by newer benchmarks", 
                      confidence=0.9, verdict="superseded")
    """
    
    def __init__(
        self, 
        agent_id: str = None,
        root: str = None,
        use_hash_embed: bool = False,
    ):
        import os
        
        self.agent_id = agent_id or os.environ.get("USER", "agent")
        self.root = root or os.environ.get("MEMPALM_PATH", os.path.expanduser("~/.mempalm"))
        self.use_hash_embed = use_hash_embed
        
        # Initialize stores
        self._init_stores()
        self._init_embedder()
        
        # Review manager - given both stores so shared-pool memories can be
        # annotated, superseded, queued for review and marked as used.
        from .review import ReviewManager
        self._review = ReviewManager(self.store, shared_store=self.shared_store)
    
    def _init_stores(self):
        """Initialize personal and shared stores"""
        import os
        from .store import PalmStore
        
        os.makedirs(os.path.join(self.root, "agents"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "models"), exist_ok=True)
        
        # Personal store
        personal_path = os.path.join(self.root, "agents", f"{self.agent_id}.db")
        self.store = PalmStore(db_path=personal_path)
        
        # Shared store - always create
        shared_path = os.path.join(self.root, "hermes.db")
        self.shared_store = PalmStore(db_path=shared_path)
    
    def _init_embedder(self):
        """Initialize embedder"""
        from .embed import get_embedder
        self.embedder = get_embedder(use_hash=self.use_hash_embed)
    
    def remember(
        self,
        content: str,
        namespace: str = "default",
        tags: list = None,
        type_: str = "note",
        source: str = "inferred",
        ttl_days: int = None,
        pool: str = "personal",
    ) -> str:
        """
        Store a memory with automatic L0/L1 extraction and embedding.
        
        Returns:
            URI of the stored memory
        """
        tags = tags or []
        
        # Auto-extract L0/L1
        l0, l1 = heuristic_extract(content)
        
        # Select store
        store = self.shared_store if pool == "shared" else self.store
        owner = "hermes" if pool == "shared" else self.agent_id
        
        # ── Embed BEFORE the row is committed ──────────────────────────────
        #
        # The row must not exist without its vector: an un-vectorized memory
        # is invisible to semantic search (FTS still finds it), so a partially
        # written row is a silent data-quality bug. If the embedder raises
        # (e.g. the ONNX model is not installed yet), nothing is written and
        # the caller gets the original error.
        #
        # The duplicate case is detected up-front so duplicates never pay the
        # embed cost. When an existing row already carries a vector we reuse
        # it: the content hash matched, so the content (and therefore the
        # embedding) is unchanged.
        #
        # If the row exists but its vector is MISSING (e.g. it was written by
        # an older version that could fail between the row insert and the
        # vector insert), we embed and re-store the vector so the memory is
        # repaired rather than left permanently un-searchable.
        import hashlib
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing_id = store.get_node_id_by_hash(pool, namespace, content_hash)
        existing_has_vector = existing_id is not None and store.has_vector(existing_id)
        
        embedding = None
        if not existing_has_vector:
            embedding = self.embedder.embed_single(l0 + " " + l1)
        
        # Store
        uri, is_dup, near_dup = store.remember(
            content=content,
            pool=pool,
            owner=owner,
            author=self.agent_id,
            namespace=namespace,
            type_=type_,
            source=source,
            tags=tags,
            fields=None,
            ttl_days=ttl_days,
        )
        
        # Store the vector whenever we computed one:
        #   * new row            -> always needs its vector
        #   * dedup, no vector  -> repairing a missing vector
        #   * dedup, has vector -> skipped above (embedding is None)
        if embedding is not None:
            node_id = store.get_node_id(uri)
            if node_id:
                store.store_vector(node_id, embedding)
        
        return uri
    
    def search(
            self,
            query: str,
            namespace: str = None,
            pool: str = "both",
            level: str = "l1",
            max_tokens: int = 2000,
            limit: int = 10,
            include_superseded: bool = False,
        ) -> list:
            """
            Search memories with hybrid retrieval and temporal filtering.

            Args:
                query: Search query
                namespace: Namespace to search (None = all)
                pool: "personal", "shared", or "both"
                level: "l0" or "l1" (L2 requires explicit recall)
                max_tokens: Token budget for results
                limit: Maximum results
                include_superseded: Include superseded memories

            Returns:
                List of MemoryNode-like dicts
            """
            # Embed query
            query_embedding = self.embedder.embed_single(query)
        
            # Extract temporal constraint from query
            temporal_constraint = extract_temporal_constraint(query)
        
            all_results = []
            pools = ["personal", "shared"] if pool == "both" else [pool]
        
            for p in pools:
                if p == "shared":
                    if not self.shared_store:
                        continue
                    store = self.shared_store
                else:
                    store = self.store
            
                try:
                    # Vector search
                    vec_results = store.search_vec(query_embedding, pool=p, namespace=namespace, k=50)
                
                    # FTS search
                    fts_results = store.search_fts(query, pool=p, namespace=namespace, k=50)
                
                    # Fuse and fetch with temporal filtering
                    results = store.fuse_and_fetch(
                        vec_results, fts_results,
                        k=limit, max_tokens=max_tokens,
                        include_superseded=include_superseded,
                        query_text=query,
                        temporal_constraint=temporal_constraint
                    )
                
                    all_results.extend(results)
                
                except Exception as e:
                    # Log but continue
                    pass
        
            # Sort by score
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
            return all_results[:limit]
    
    def recall(
        self, 
        uri: str, 
        level: str = "l1", 
        max_tokens: int = 4000, 
        offset: int = 0
    ) -> dict:
        """
        Recall full memory content at specified level.
        
        Args:
            uri: Memory URI
            level: "l0", "l1", or "l2"
            max_tokens: Max tokens for L2 pagination
            offset: Pagination offset for L2
            
        Returns:
            Dict with text, truncated, total_tokens, offset
        """
        # Try personal first
        try:
            result = self.store.recall(uri, level=level, max_tokens=max_tokens, offset=offset)
            if result["text"]:
                return result
        except Exception:
            pass
        
        # Try shared
        if self.shared_store:
            try:
                result = self.shared_store.recall(uri, level=level, max_tokens=max_tokens, offset=offset)
                if result["text"]:
                    return result
            except Exception:
                pass
        
        return {"error": f"Memory not found: {uri}", "text": "", "truncated": False, "total_tokens": 0, "offset": 0}
    
    def forget(self, uri: str) -> bool:
        """Permanently delete a memory"""
        try:
            self.store.forget(uri)
            return True
        except Exception:
            if self.shared_store:
                try:
                    self.shared_store.forget(uri)
                    return True
                except Exception:
                    pass
        return False
    
    def annotate(
        self,
        uri: str,
        review_text: str,
        confidence: float,
        verdict: str,
        evidence_uri: str = None,
        tags: list = None,
    ) -> dict:
        """
        Add review annotation (self-improving review loop).
        
        Args:
            uri: Memory URI to annotate
            review_text: Review content
            confidence: 0.0-1.0
            verdict: "confirmed", "uncertain", "superseded", "disputed"
            evidence_uri: Optional URI of supporting evidence
            tags: Optional tags
            
        Returns:
            Dict with new trust score and annotation info
        """
        return self._review.annotate(
            uri=uri,
            reviewer=self.agent_id,
            verdict=verdict,
            confidence=confidence,
            review_text=review_text,
            evidence_uri=evidence_uri,
            tags=tags or [],
        )
    
    def supersede(
        self,
        old_uri: str,
        new_content: str,
        **remember_kwargs
    ) -> str:
        """Create new memory that supersedes old one"""
        return self._review.supersede(old_uri, new_content, **remember_kwargs)
    
    def review_queue(
        self, 
        namespace: str = None, 
        limit: int = 10,
        pool: str = "personal"
    ) -> list:
        """Get memories needing review"""
        return self._review.review_queue(namespace, limit, pool)
    
    def mark_used(self, uris: list):
        """Mark memories as used (for implicit review signal)"""
        self._review.mark_used(uris)
    
    def verify_citations(self, response: str) -> list:
        """Verify citations in model response"""
        from agent_palm.retrieval import verify_citations
        # Get recently cited URIs from recent searches
        # This would need to track recently retrieved URIs
        return verify_citations(response, [])
    
    def stats(self, namespace: str = None, pool: str = "both") -> dict:
        """Get namespace statistics"""
        results = {}
        
        if pool in ["personal", "both"]:
            results["personal"] = self.store.namespace_stats("personal", namespace)
        
        if pool in ["shared", "both"] and self.shared_store:
            results["shared"] = self.shared_store.namespace_stats("shared", namespace)
        
        return results
    
    def backup(self, destination: str = None, pool: str = "both") -> dict:
        """Backup database using SQLite Backup API"""
        import os
        import time
        
        if not destination:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            destination = os.path.join(self.root, "backups", f"mempalm_backup_{timestamp}")
        
        results = {}
        
        if pool in ["personal", "both"]:
            dest = f"{destination}_{self.agent_id}.db"
            self.store.backup(dest)
            results["personal"] = dest
        
        if pool in ["shared", "both"] and self.shared_store:
            dest = f"{destination}_hermes.db"
            self.shared_store.backup(dest)
            results["shared"] = dest
        
        return results
    
    def close(self):
        """Close all connections"""
        self.store.close()
        if self.shared_store:
            self.shared_store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()