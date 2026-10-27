#!/usr/bin/env python
"""
mempalm - AI PalmPilot CLI
Agent-native memory layer management
"""

import os
import sys
import json
import time
import click
from pathlib import Path
from typing import Optional, List

# Add project root to path for local development
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_palm.store import PalmStore, open_store
from agent_palm.embed import OnnxEmbedder, HashEmbedder, get_embedder
from agent_palm.extract import heuristic_extract
from agent_palm.retrieval import build_grounded_context, verify_citations
from agent_palm.types import MemoryPool
from agent_palm.exceptions import PalmModelMismatchError, PalmStorageBusyError


# Default paths
DEFAULT_ROOT = os.environ.get("MEMPALM_PATH", os.path.expanduser("~/.mempalm"))
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIM = 384


def get_store(pool: str = "personal", owner: str = None, use_hash_embed: bool = False) -> tuple:
    """Get PalmStore and embedder instances"""
    if owner is None:
        import getpass
        owner = getpass.getuser()
    
    db_path = os.path.join(DEFAULT_ROOT, "agents", f"{owner}.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    store = PalmStore(db_path)
    embedder = HashEmbedder() if use_hash_embed else OnnxEmbedder()
    
    return store, embedder, owner


def get_shared_store(use_hash_embed: bool = False) -> tuple:
    """Get shared pool store (hermes.db)"""
    db_path = os.path.join(DEFAULT_ROOT, "hermes.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    store = PalmStore(db_path)
    embedder = HashEmbedder() if use_hash_embed else OnnxEmbedder()
    
    return store, embedder, "hermes"


@click.group()
@click.version_option(version="1.3.0")
@click.option("--root", default=DEFAULT_ROOT, envvar="MEMPALM_PATH", help="Root directory for mempalm data")
@click.option("--hash-embed", is_flag=True, help="Use hash embedder (for testing, no model needed)")
@click.pass_context
def cli(ctx, root: str, hash_embed: bool):
    """mempalm - AI PalmPilot: Agent-native memory layer"""
    ctx.ensure_object(dict)
    ctx.obj["root"] = root
    ctx.obj["hash_embed"] = hash_embed
    ctx.obj["use_hash"] = hash_embed
    
    # Ensure directories exist
    os.makedirs(os.path.join(root, "agents"), exist_ok=True)
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    os.makedirs(os.path.join(root, "backups"), exist_ok=True)


@cli.command()
@click.argument("content")
@click.option("--namespace", "-n", default="default", help="Namespace for the memory")
@click.option("--pool", type=click.Choice(["personal", "shared"]), default="personal", help="Memory pool")
@click.option("--tags", "-t", multiple=True, help="Tags for the memory")
@click.option("--type", default="note", help="Memory type")
@click.option("--source", type=click.Choice(["observed", "user", "inferred", "imported"]), default="inferred")
@click.option("--ttl", type=int, help="Time to live in days")
@click.pass_context
def remember(ctx, content: str, namespace: str, pool: str, tags: tuple, type: str, source: str, ttl: int):
    """Store a memory with automatic L0/L1 extraction"""
    use_hash = ctx.obj["use_hash"]
    root = ctx.obj["root"]
    
    if pool == "shared":
        store, embedder, owner = get_shared_store(use_hash)
        owner = "hermes"
    else:
        store, embedder, owner = get_store(pool="personal", use_hash_embed=use_hash)
    
    try:
        from agent_palm.extract import heuristic_extract
        
        # Extract L0/L1
        l0, l1 = heuristic_extract(content)
        
        # Embed L0 + L1 for vector search
        embedding = embedder.embed_single(l0 + " " + l1)
        
        # Store
        uri, is_dup, near_dup = store.remember(
            content=content,
            pool=pool,
            owner=owner,
            author=owner,
            namespace=namespace,
            type_=type,
            source=source,
            tags=list(tags),
            fields=None,
            ttl_days=ttl,
        )
        
        # Store vector (need node_id - get it from store)
        node_id = store.get_node_id(uri)
        if node_id:
            store.store_vector(node_id, embedding)
        
        if is_dup:
            click.echo(f"⚠ Updated existing memory: {uri}")
            if near_dup:
                click.echo(f"   (near duplicate of: {near_dup})")
        else:
            click.echo(f"✓ Remembered: {uri}")
            click.echo(f"  L0: {store.estimate_tokens(l0)} tokens")
            click.echo(f"  L1: {store.estimate_tokens(l1)} tokens")
            click.echo(f"  L2: {store.estimate_tokens(content)} tokens")
        
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--namespace", "-n", help="Namespace to search (default: all)")
@click.option("--pool", type=click.Choice(["personal", "shared", "both"]), default="both")
@click.option("--level", type=click.Choice(["l0", "l1"]), default="l1")
@click.option("--max-tokens", "-m", default=2000, help="Token budget for results")
@click.option("--include-superseded", is_flag=True, help="Include superseded memories")
@click.pass_context
def search(ctx, query: str, namespace: str, pool: str, level: str, max_tokens: int, include_superseded: bool):
    """Search memories with hybrid retrieval"""
    use_hash = ctx.obj["use_hash"]
    root = ctx.obj["root"]
    
    pools = ["personal", "shared"] if pool == "both" else [pool]
    
    # Get embedder
    embedder = HashEmbedder() if use_hash else OnnxEmbedder()
    query_embedding = embedder.embed_single(query)
    
    all_results = []
    
    for p in pools:
        if p == "shared":
            store, _, owner = get_shared_store(use_hash)
            owner = "hermes"
        else:
            store, _, owner = get_store(pool="personal", use_hash_embed=use_hash)
        
        try:
            # Vector search
            vec_results = store.search_vec(query_embedding, pool=p, namespace=namespace, k=50)
            
            # FTS search
            fts_results = store.search_fts(query, pool=p, namespace=namespace, k=50)
            
            # Fuse and fetch
            results = store.fuse_and_fetch(
                vec_results, fts_results,
                k=20, max_tokens=max_tokens,
                include_superseded=include_superseded
            )
            
            all_results.extend(results)
            
        except Exception as e:
            click.echo(f"⚠ Error searching {p} pool: {e}", err=True)
    
    if not all_results:
        click.echo("No results found.")
        return
    
    # Sort by score
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Format output
    click.echo(f"\nFound {len(all_results)} results for: '{query}'\n")
    
    for i, node in enumerate(all_results, 1):
        data = node.get('data', node)  # Handle both formats
        tags_str = f" [{', '.join(data['tags'])}]" if data.get('tags') else ""
        verdict_str = f" ({data['latest_verdict']})" if data.get('latest_verdict') else ""
        
        click.echo(f"{i}. {data['uri']}{tags_str}{verdict_str}")
        click.echo(f"    Score: {data.get('score', node.get('score', 0)):.3f} | Level: {data.get('level', node.get('level', 'l1'))} | Trust: {data.get('trust', node.get('trust', 0.5)):.2f}")
        text = data.get('l1', data.get('text', ''))
        click.echo(f"    {text[:120]}{'...' if len(text) > 120 else ''}")
        click.echo()


@cli.command()
@click.argument("uri")
@click.option("--level", type=click.Choice(["l0", "l1", "l2"]), default="l1")
@click.option("--max-tokens", default=4000)
@click.option("--offset", default=0)
@click.pass_context
def recall(ctx, uri: str, level: str, max_tokens: int, offset: int):
    """Recall full memory content at specified level"""
    use_hash = ctx.obj["use_hash"]
    root = ctx.obj["root"]
    
    # Try both pools
    for pool in ["personal", "shared"]:
        if pool == "shared":
            store, _, _ = get_shared_store(use_hash)
        else:
            store, _, _ = get_store(pool="personal", use_hash_embed=use_hash)
        
        try:
            result = store.recall(uri, level=level, max_tokens=max_tokens, offset=offset)
            
            if result["text"]:
                click.echo(f"URI: {uri}")
                click.echo(f"Level: {level}")
                click.echo(f"Tokens: {result['total_tokens']} | Offset: {offset}")
                if result["truncated"]:
                    click.echo(f"⚠ Truncated (next offset: {result['offset']})")
                click.echo("-" * 40)
                click.echo(result["text"])
            else:
                click.echo(f"Memory not found or empty: {uri}")
            return
            
        except Exception:
            continue
    
    click.echo(f"Memory not found in any pool: {uri}", err=True)
    sys.exit(1)


@cli.command()
@click.argument("uri")
@click.pass_context
def forget(ctx, uri: str):
    """Permanently delete a memory"""
    use_hash = ctx.obj["use_hash"]
    
    for pool in ["personal", "shared"]:
        if pool == "shared":
            store, _, _ = get_shared_store(use_hash)
        else:
            store, _, _ = get_store(pool="personal", use_hash_embed=use_hash)
        
        try:
            result = store.forget(uri)
            click.echo(f"✓ Forgotten: {uri}")
            if result.get("published_copies"):
                click.echo(f"  Published copies also removed: {len(result['published_copies'])}")
            return
        except Exception:
            continue
    
    click.echo(f"Memory not found: {uri}", err=True)
    sys.exit(1)


@cli.command()
@click.option("--namespace", "-n", help="Namespace to stats")
@click.option("--pool", type=click.Choice(["personal", "shared", "both"]), default="both")
@click.pass_context
def stats(ctx, namespace: str, pool: str):
    """Show namespace statistics"""
    use_hash = ctx.obj["use_hash"]
    root = ctx.obj["root"]
    
    pools = ["personal", "shared"] if pool == "both" else [pool]
    
    for p in pools:
        if p == "shared":
            store, _, owner = get_shared_store(ctx.obj["use_hash"])
            owner = "hermes"
        else:
            store, _, owner = get_store(pool="personal", use_hash_embed=ctx.obj["use_hash"])
        
        stats = store.namespace_stats(p, namespace)
        
        ns_label = namespace or "all"
        click.echo(f"\n📊 {p.upper()} pool - {owner} - namespace: {ns_label}")
        click.echo(f"   Memories: {stats['count']}")
        click.echo(f"   L0 tokens: {stats['total_tokens']['l0']:,}")
        click.echo(f"   L1 tokens: {stats['total_tokens']['l1']:,}")
        click.echo(f"   L2 tokens: {stats['total_tokens']['l2']:,}")
        
        # DB size
        if p == "shared":
            db_path = os.path.join(DEFAULT_ROOT, "hermes.db")
        else:
            db_path = os.path.join(DEFAULT_ROOT, "agents", f"{owner}.db")
        
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            click.echo(f"   DB size: {size_mb:.1f} MB")


@cli.command()
@click.option("--namespace", "-n", help="Namespace to prune")
@click.option("--pool", type=click.Choice(["personal", "shared", "both"]), default="both")
@click.option("--older-than", default=90, help="Prune memories older than N days")
@click.option("--dry-run", is_flag=True, help="Show what would be pruned without deleting")
@click.pass_context
def prune(ctx, namespace: str, pool: str, older_than: int, dry_run: bool):
    """Prune old memories"""
    use_hash = ctx.obj["use_hash"]
    root = ctx.obj["root"]
    
    cutoff = time.time() - (older_than * 86400)
    
    pools = ["personal", "shared"] if pool == "both" else [pool]
    
    for p in pools:
        if p == "shared":
            store, _, owner = get_shared_store(use_hash)
        else:
            store, _, owner = get_store(pool="personal", use_hash_embed=use_hash)
        
        # This would need a prune implementation in store
        click.echo(f"Prune not yet implemented for {p} pool")


@cli.command()
@click.option("--destination", "-d", help="Backup destination path")
@click.option("--pool", type=click.Choice(["personal", "shared", "both"]), default="both")
@click.pass_context
def backup(ctx, destination: str, pool: str):
    """Backup database using SQLite Backup API"""
    use_hash = ctx.obj["use_hash"]
    root = ctx.obj["root"]
    
    if not destination:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        destination = os.path.join(DEFAULT_ROOT, "backups", f"mempalm_backup_{timestamp}")
    
    pools = ["personal", "shared"] if pool == "both" else [pool]
    
    for p in pools:
        if p == "shared":
            store, _, _ = get_shared_store(use_hash)
            dest = f"{destination}_hermes.db"
        else:
            store, _, owner = get_store(pool="personal", use_hash_embed=use_hash)
            dest = f"{destination}_{owner}.db"
        
        try:
            store.backup(dest)
            click.echo(f"✓ Backed up {p} pool to {dest}")
        except Exception as e:
            click.echo(f"✗ Backup failed for {p}: {e}", err=True)


@cli.command()
@click.option("--model", default="all-MiniLM-L6-v2", help="Model name for new vectors")
@click.pass_context
def reembed(ctx, model: str):
    """Re-embed all vectors with new model"""
    click.echo("Re-embedding not yet implemented")
    click.echo(f"Would re-embed all memories with model: {model}")


@cli.command()
@click.option("--namespace", "-n", help="Namespace to review")
@click.option("--unreviewed-only", is_flag=True, help="Show only unreviewed memories")
@click.pass_context
def review(ctx, namespace: str, unreviewed_only: bool):
    """Show memories needing review"""
    click.echo("Review queue not yet implemented")


@cli.command()
@click.option("--source", type=click.Choice(["openviking", "markdown", "jsonl"]), default="markdown")
@click.option("--path", help="Path to import from")
@click.option("--namespace", "-n", default="imported")
@click.option("--pool", type=click.Choice(["personal", "shared"]), default="personal")
@click.pass_context
def migrate(ctx, source: str, path: str, namespace: str, pool: str):
    """Migrate from OpenViking or other sources"""
    click.echo(f"Migration from {source} not yet implemented")


if __name__ == "__main__":
    cli()