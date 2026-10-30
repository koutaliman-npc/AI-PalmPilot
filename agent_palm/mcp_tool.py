"""
MCP stdio tools for agent_palm
Exposes memory operations as MCP tools via stdio transport
"""

import sys
import json
import os
from typing import List, Dict, Any, Optional, Sequence

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_palm.store import PalmStore, open_store
from agent_palm.embed import HashEmbedder, OnnxEmbedder, get_embedder
from agent_palm.extract import heuristic_extract, classify_namespace
from agent_palm.retrieval import build_grounded_context, verify_citations
from agent_palm.types import MemoryPool


class MCPPalmServer:
    """MCP stdio server for agent_palm"""
    
    def __init__(self, root: str = None, use_hash_embed: bool = False):
        # Use passed parameter if explicitly True, otherwise fall back to env var
        # This allows explicit True to override env var False
        if use_hash_embed is True:
            self.use_hash_embed = True
        else:
            self.use_hash_embed = os.environ.get("MEMPALM_HASH_EMBED", "false").lower() == "true"
        # Honour an explicit root= argument; fall back to $MEMPALM_PATH, then ~/.mempalm
        self.root = root or os.environ.get("MEMPALM_PATH", os.path.expanduser("~/.mempalm"))
        
        # Initialize stores
        os.makedirs(os.path.join(self.root, "agents"), exist_ok=True)
        self.personal_store = PalmStore(
            db_path=os.path.join(self.root, "agents", f"{os.environ.get('USER', 'agent')}.db")
        )
        
        shared_path = os.path.join(self.root, "hermes.db")
        if os.path.exists(shared_path):
            self.shared_store = PalmStore(db_path=shared_path)
        else:
            self.shared_store = None
        
        # Embedder
        if self.use_hash_embed:
            self.embedder = HashEmbedder()
        else:
            self.embedder = OnnxEmbedder()
    
    def _get_store(self, pool: str):
        if pool == "shared" and self.shared_store:
            return self.shared_store
        return self.personal_store
    
    def tool_remember(self, content: str, namespace: str = None, pool: str = "personal", 
                      tags: list = None, type_: str = "note", source: str = "inferred",
                      ttl_days: int = None) -> dict:
        """Store a memory"""
        store = self._get_store(pool)
        if namespace is None:
            namespace = classify_namespace(content)
        
        # Auto-extract L0/L1
        l0, l1 = heuristic_extract(content)
        
        # Embed BEFORE the row is committed: a row without a vector is
        # invisible to semantic search, so a failed embed must not leave a
        # half-written memory behind. Duplicates are detected up-front so
        # they never pay the embed cost.
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
            owner="mcp_user",
            author="mcp_user",
            namespace=namespace,
            type_=type_,
            source=source,
            tags=tags or [],
            ttl_days=ttl_days,
        )
        
        # Store the vector whenever we computed one (new row, or repairing a
        # row whose vector went missing).
        if embedding is not None:
            node_id = store.get_node_id(uri)
            if node_id:
                store.store_vector(node_id, embedding)
        
        return {
            "uri": uri,
            "is_duplicate": is_dup,
            "near_duplicate_of": near_dup,
        }
    
    def tool_search(self, query: str, namespace: str = None, pool: str = "both",
                    level: str = "l1", max_tokens: int = 2000, limit: int = 10,
                    include_superseded: bool = False) -> dict:
        """Search memories with hybrid retrieval"""
        from agent_palm.embed import get_embedder
        
        embedder = HashEmbedder() if self.use_hash_embed else OnnxEmbedder()
        query_embedding = embedder.embed_single(query)
        
        pools = ["personal", "shared"] if pool == "both" else [pool]
        all_results = []
        
        for p in pools:
            if p == "shared":
                store = self.shared_store
                if not store:
                    continue
            else:
                store = self.personal_store
            
            try:
                vec_results = store.search_vec(query_embedding, pool=p, namespace=namespace, k=50)
                fts_results = store.search_fts(query, pool=p, namespace=namespace, k=50)
                
                results = store.fuse_and_fetch(
                    vec_results, fts_results,
                    k=limit, max_tokens=max_tokens,
                    include_superseded=include_superseded
                )
                
                all_results.extend(results)
            except Exception as e:
                pass
        
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return {
            "results": all_results[:limit],
            "count": len(all_results),
        }
    
    def tool_recall(self, uri: str, level: str = "l1", max_tokens: int = 4000, offset: int = 0) -> dict:
        """Recall full memory content"""
        # Try personal first
        try:
            result = self.personal_store.recall(uri, level=level, max_tokens=max_tokens, offset=offset)
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
        
        return {"error": f"Memory not found: {uri}"}
    
    def tool_forget(self, uri: str) -> dict:
        """Delete a memory"""
        try:
            self.personal_store.forget(uri)
            return {"success": True, "uri": uri}
        except Exception:
            if self.shared_store:
                try:
                    self.shared_store.forget(uri)
                    return {"success": True, "uri": uri}
                except Exception:
                    pass
        return {"success": False, "error": f"Memory not found: {uri}"}
    
    def tool_stats(self, namespace: str = None, pool: str = "both") -> dict:
        """Get namespace statistics"""
        results = {}
        
        if pool in ["personal", "both"]:
            results["personal"] = self.personal_store.namespace_stats("personal", namespace)
        
        if pool in ["shared", "both"] and self.shared_store:
            results["shared"] = self.shared_store.namespace_stats("shared", namespace)
        
        return results
    
    def tool_verify_citations(self, response: str, provided_uris: list = None) -> dict:
        """Verify citations in model response"""
        if provided_uris is None:
            provided_uris = []
        
        invalid = verify_citations(response, provided_uris)
        coverage = len(set(provided_uris) & set(extract_cited_uris(response))) / max(len(extract_cited_uris(response)), 1)
        
        return {
            "invalid_citations": invalid,
            "coverage": coverage,
            "total_cited": len(extract_cited_uris(response)),
        }
    
    def tool_grounded_context(self, query: str, max_tokens: int = 2000, pool: str = "both") -> dict:
        """Build grounded context for agent prompt"""
        result = self.tool_search(query, pool=pool, max_tokens=max_tokens)
        context = build_grounded_context(result["results"], max_tokens)
        return {
            "context": context,
            "cited_uris": [r["uri"] for r in result["results"]],
        }


# MCP stdio protocol handling
def handle_request(request: dict) -> dict:
    """Handle MCP JSON-RPC request"""
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    # Use hash embedder for testing (no model needed)
    server = MCPPalmServer(use_hash_embed=True)
    
    try:
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "memory_remember",
                            "description": "Store a memory with automatic L0/L1 extraction and embedding",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string", "description": "Content to remember"},
                                    "namespace": {"type": "string", "description": "Namespace (auto-classified when omitted)"},
                                    "pool": {"type": "string", "enum": ["personal", "shared"], "default": "personal"},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                    "type": {"type": "string", "default": "note"},
                                    "source": {"type": "string", "enum": ["observed", "user", "inferred", "imported"], "default": "inferred"},
                                    "ttl_days": {"type": "integer"},
                                },
                                "required": ["content"],
                            },
                        },
                        {
                            "name": "memory_search",
                            "description": "Search memories with hybrid vector + keyword retrieval",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "namespace": {"type": "string"},
                                    "pool": {"type": "string", "enum": ["personal", "shared", "both"], "default": "both"},
                                    "level": {"type": "string", "enum": ["l0", "l1"], "default": "l1"},
                                    "max_tokens": {"type": "integer", "default": 2000},
                                    "limit": {"type": "integer", "default": 10},
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "memory_recall",
                            "description": "Recall full memory content at specified tier",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "uri": {"type": "string"},
                                    "level": {"type": "string", "enum": ["l0", "l1", "l2"], "default": "l1"},
                                    "max_tokens": {"type": "integer", "default": 4000},
                                    "offset": {"type": "integer", "default": 0},
                                },
                                "required": ["uri"],
                            },
                        },
                        {
                            "name": "memory_forget",
                            "description": "Permanently delete a memory",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "uri": {"type": "string"},
                                },
                                "required": ["uri"],
                            },
                        },
                        {
                            "name": "memory_stats",
                            "description": "Get namespace statistics",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "namespace": {"type": "string"},
                                    "pool": {"type": "string", "enum": ["personal", "shared", "both"], "default": "both"},
                                },
                            },
                        },
                        {
                            "name": "memory_verify_citations",
                            "description": "Verify citations in model response against retrieved memories",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "response": {"type": "string"},
                                    "provided_uris": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["response"],
                            },
                        },
                        {
                            "name": "memory_grounded_context",
                            "description": "Build grounded context block for agent prompt",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "max_tokens": {"type": "integer", "default": 2000},
                                    "pool": {"type": "string", "enum": ["personal", "shared", "both"], "default": "both"},
                                },
                                "required": ["query"],
                            },
                        },
                    ]
                },
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            server = MCPPalmServer()
            
            try:
                if tool_name == "memory_remember":
                    result = server.tool_remember(**arguments)
                elif tool_name == "memory_search":
                    result = server.tool_search(**arguments)
                elif tool_name == "memory_recall":
                    result = server.tool_recall(**arguments)
                elif tool_name == "memory_forget":
                    result = server.tool_forget(**arguments)
                elif tool_name == "memory_stats":
                    result = server.tool_stats(**arguments)
                elif tool_name == "memory_verify_citations":
                    result = server.tool_verify_citations(**arguments)
                elif tool_name == "memory_grounded_context":
                    result = server.tool_grounded_context(**arguments)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                    }
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(e)}
                }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            }
    
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
        }


def main():
    """MCP stdio server entry point"""
    print("AI PalmPilot MCP Server starting...", file=sys.stderr)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()