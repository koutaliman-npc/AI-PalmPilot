"""
Hermes MemoryProvider adapter for agent_palm
Implements the MemoryProvider interface for Hermes Agent
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Sequence

from agent_palm.store import PalmStore, open_store
from agent_palm.embed import get_embedder
from agent_palm.extract import heuristic_extract
from agent_palm.retrieval import build_grounded_context, verify_citations
from agent_palm.types import MemoryPool, MemorySource


class VaultMemoryProvider:
    """
    Hermes MemoryProvider implementation using agent_palm.
    
    Configuration via environment variables:
    - MEMPALM_PATH: Root directory for databases (default: ~/.mempalm)
    - MEMPALM_AGENT_ID: Agent identifier (default: current user)
    - MEMPALM_NAMESPACE: Default namespace (default: "default")
    - MEMPALM_HASH_EMBED: Use hash embedder for testing (default: false)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.root = self.config.get("path") or os.environ.get("MEMPALM_PATH", os.path.expanduser("~/.mempalm"))
        self.agent_id = self.config.get("agent_id") or os.environ.get("MEMPALM_AGENT_ID") or os.environ.get("USER", "agent")
        self.default_namespace = self.config.get("namespace") or os.environ.get("MEMPALM_NAMESPACE", "default")
        self.use_hash_embed = self.config.get("hash_embed", os.environ.get("MEMPALM_HASH_EMBED", "false").lower() == "true")
        
        # Initialize store and embedder
        self._init_store()
        self._init_embedder()
        
        # Track cited URIs for citation verification
        self._last_cited_uris: List[str] = []
    
    def _init_store(self):
        """Initialize PalmStore"""
        db_path = os.path.join(self.root, "agents", f"{self.agent_id}.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._store = PalmStore(db_path=db_path)
        
        # Also open shared pool if available
        self._shared_store = None
        shared_path = os.path.join(self.root, "hermes.db")
        if os.path.exists(shared_path):
            self._shared_store = PalmStore(db_path=shared_path)
    
    def _init_embedder(self):
        """Initialize embedder"""
        from agent_palm.embed import get_embedder
        self.embedder = get_embedder(use_hash=self.use_hash_embed)
    
    # --- MemoryProvider Interface ---
    
    def store(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store a memory with automatic extraction and embedding.
        
        Args:
            text: Content to remember
            metadata: Optional dict with:
                - namespace: Namespace (default: config default)
                - tags: List of tags
                - type: Memory type (default: "note")
                - source: "observed" | "user" | "inferred" | "imported" (default: "inferred")
                - ttl_days: Time to live in days
        
        Returns:
            URI of stored memory
        """
        metadata = metadata or {}
        
        namespace = metadata.get("namespace", self.default_namespace)
        tags = metadata.get("tags", [])
        type_ = metadata.get("type", "note")
        source = metadata.get("source", "inferred")
        ttl_days = metadata.get("ttl_days")
        
        # Auto-extract L0/L1
        l0, l1 = heuristic_extract(text)
        
        # Embed BEFORE the row is committed: a row without a vector is
        # invisible to semantic search, so a failed embed must not leave a
        # half-written memory behind. Duplicates are detected up-front so
        # they never pay the embed cost.
        import hashlib
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing_id = self._store.get_node_id_by_hash("personal", namespace, content_hash)
        existing_has_vector = existing_id is not None and self._store.has_vector(existing_id)
        
        embedding = None
        if not existing_has_vector:
            embedding = self.embedder.embed_single(l0 + " " + l1)
        
        # Store in personal pool
        uri, is_dup, near_dup = self._store.remember(
            content=text,
            pool="personal",
            owner=self.agent_id,
            author=self.agent_id,
            namespace=namespace,
            type_=type_,
            source=source,
            tags=metadata.get("tags", []),
            fields=metadata.get("fields"),
            ttl_days=metadata.get("ttl_days"),
        )
        
        # Store the vector whenever we computed one (new row, or repairing a
        # row whose vector went missing).
        if embedding is not None:
            node_id = self._store.get_node_id(uri)
            if node_id:
                self._store.store_vector(node_id, embedding)
        
        return uri
    
    def search(self, query: str, limit: int = 10, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search memories with hybrid retrieval.
        
        Args:
            query: Search query
            limit: Maximum number of results
            metadata: Optional filters:
                - namespace: Namespace to search
                - pool: "personal" | "shared" | "both" (default: "both")
                - max_tokens: Token budget (default: 2000)
        
        Returns:
            List of memory dicts with uri, text, score, metadata
        """
        metadata = metadata or {}
        namespace = metadata.get("namespace")
        pool = metadata.get("pool", "both")
        max_tokens = metadata.get("max_tokens", 2000)
        limit = metadata.get("limit", limit)
        
        # Embed query
        query_embedding = self.embedder.embed_single(query)
        
        all_results = []
        pools = ["personal", "shared"] if pool == "both" else [pool]
        
        for p in ["personal", "shared"] if pool == "both" else [pool]:
            if p == "shared" and self._shared_store:
                store = self._shared_store
            elif p == "personal":
                store = self._store
            else:
                continue
            
            try:
                # Vector search
                vec_results = store.search_vec(
                    self.embedder.embed_single(query),
                    pool=p, namespace=namespace, k=50
                )
                
                # FTS search
                fts_results = store.search_fts(query, pool=p, namespace=namespace, k=50)
                
                # Fuse and fetch
                results = store.fuse_and_fetch(
                    vec_results, fts_results,
                    k=limit, max_tokens=max_tokens
                )
                

                # pack_budget wraps results in {"data": ..., "level": ..., "cost": ...}
                # Unwrap to get the actual memory dicts
                results = [r["data"] for r in results if "data" in r]
                all_results.extend(results)
                
            except Exception as e:
                # Log but continue
                pass
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Track cited URIs for verification
        self._last_cited_uris = [r["uri"] for r in all_results[:limit]]
        
        return all_results[:limit]
    
    def recall(self, uri: str, level: str = "l1", max_tokens: int = 4000) -> Optional[Dict[str, Any]]:
        """Recall full memory content"""
        # Try personal first
        try:
            result = self._store.recall(uri, level=level, max_tokens=max_tokens)
            if result["text"]:
                return result
        except Exception:
            pass
        
        # Try shared
        if self._shared_store:
            try:
                result = self._shared_store.recall(uri, level=level, max_tokens=max_tokens)
                if result["text"]:
                    return result
            except Exception:
                pass
        
        return None
    
    def forget(self, uri: str) -> bool:
        """Delete a memory"""
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
    
    def verify_citations(self, response: str) -> List[str]:
        """Verify citations in model response"""
        return verify_citations(response, self._last_cited_uris)
    
    def build_context(self, query: str, max_tokens: int = 2000) -> str:
        """Build grounded context for agent prompt"""
        results = self.search(query, max_tokens=max_tokens)
        return build_grounded_context(results, max_tokens)
    
    def get_stats(self, namespace: str = None) -> Dict[str, Any]:
        """Get namespace statistics"""
        personal = self._store.namespace_stats("personal", namespace)
        shared = self._shared_store.namespace_stats("shared", namespace) if self._shared_store else {"count": 0}
        
        return {
            "personal": personal,
            "shared": shared,
            "total": personal["count"] + shared["count"],
        }


# Factory function for Hermes
def create_memory_provider(config: Dict[str, Any] = None) -> VaultMemoryProvider:
    """Factory function for Hermes MemoryProvider interface"""
    return VaultMemoryProvider(config)


# Hermes expects this interface
class MemoryProvider:
    """Hermes MemoryProvider interface compatibility"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.provider = VaultMemoryProvider(config)
    
    def store(self, text: str, metadata: Dict[str, Any] = None) -> str:
        return self.provider.store(text, metadata)
    
    def search(self, query: str, limit: int = 10, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        return self.provider.search(query, limit, metadata)
    
    def recall(self, uri: str, level: str = "l1", max_tokens: int = 4000) -> Optional[Dict[str, Any]]:
        return self.provider.recall(uri, level, max_tokens)
    
    def forget(self, uri: str) -> bool:
        return self.provider.forget(uri)
    
    def verify_citations(self, response: str) -> List[str]:
        return self.provider.verify_citations(response)
    
    def build_context(self, query: str, max_tokens: int = 2000) -> str:
        return self.provider.build_context(query, max_tokens)
    
    def get_stats(self, namespace: str = None) -> Dict[str, Any]:
        return self.provider.get_stats(namespace)


# For Hermes config.yaml:
# memory:
#   provider: agent_palm.hermes_provider.MemoryProvider
#   agent_palm:
#     path: D:/mempalm
#     agent_id: sebastian
#     namespace: default