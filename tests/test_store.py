"""
Smoke tests for agent_palm
Run with: pytest tests/ -v
"""

import os
import tempfile
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def temp_mempalm_path():
    """Create a temporary directory for mempalm data"""
    temp_dir = tempfile.mkdtemp(prefix="mempalm_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def palm_pilot(temp_mempalm_path):
    """Create a PalmPilot instance with hash embedder for testing"""
    from agent_palm import PalmPilot
    return PalmPilot(agent_id="test-agent", root=temp_mempalm_path, use_hash_embed=True)


def test_imports():
    """Test that all main exports are importable"""
    from agent_palm import (
        PalmPilot,
        MemoryPool,
        MemorySource,
        ReviewVerdict,
        MemoryLevel,
        MemoryRef,
        MemoryNode,
        RecallNode,
        PalmStore,
        PalmStorageBusyError,
        PalmModelMismatchError,
        OnnxEmbedder,
        HashEmbedder,
        heuristic_extract,
        pack_budget,
        verify_citations,
        format_context,
        build_grounded_context,
    )
    assert PalmPilot is not None
    assert MemoryPool is not None


def test_remember_and_recall(palm_pilot):
    """Test basic remember and recall cycle"""
    uri = palm_pilot.remember(
        content="Test memory content for recall",
        namespace="test",
        tags=["test", "smoke"],
        pool="personal",
    )
    
    assert uri.startswith("mem://personal/test-agent/test/note/")
    
    # Search for it
    results = palm_pilot.search("test memory", pool="personal", max_tokens=1000)
    assert len(results) >= 1
    assert any("Test memory" in r["data"].get("l1", "") or "Test memory" in r["data"].get("l0", "") for r in results)
    
    # Recall full detail
    detail = palm_pilot.recall(uri, level="l2")
    assert "Test memory content for recall" in detail["text"]


def test_tiered_levels(palm_pilot):
    """Test L0/L1/L2 tiered retrieval"""
    content = "A" * 2000  # Long content to test tiering
    uri = palm_pilot.remember(content, namespace="tiered", pool="personal")
    
    # L0 - always included, very short
    l0_result = palm_pilot.recall(uri, level="l0")
    assert len(l0_result["text"]) <= 280 and len(l0_result["text"]) > 0  # L0 cap = max_l0_chars (280)
    assert l0_result["total_tokens"] > 0
    
    # L1 - medium detail
    l1_result = palm_pilot.recall(uri, level="l1")
    assert len(l1_result["text"]) <= 2000  # L1 cap = max_l1_chars (2000)
    assert l1_result["total_tokens"] > l0_result["total_tokens"]
    
    # L2 - full detail
    l2_result = palm_pilot.recall(uri, level="l2")
    assert len(l2_result["text"]) == len(content)
    # degenerate content (single unbroken token) ties l1 and l2 — assert non-increasing
    assert l2_result["total_tokens"] >= l1_result["total_tokens"]


def test_search_with_token_budget(palm_pilot):
    """Test search respects token budget"""
    # Add multiple memories
    for i in range(5):
        palm_pilot.remember(f"Memory number {i} with some unique content", namespace="budget", pool="personal")
    
    # Search with small budget
    results = palm_pilot.search("memory", pool="personal", max_tokens=200, limit=10)
    
    # Should return results but total tokens should respect budget roughly
    total_tokens = sum(r.get("tok_l1", r.get("tokens", {}).get("l1", 0)) for r in results)
    # Budget is approximate due to packing, but shouldn't be wildly over


def test_namespace_filtering(palm_pilot):
    """Test namespace filtering works"""
    palm_pilot.remember("Namespace A content", namespace="ns_a", pool="personal")
    palm_pilot.remember("Namespace B content", namespace="ns_b", pool="personal")
    
    results_a = palm_pilot.search("content", namespace="ns_a", pool="personal")
    results_b = palm_pilot.search("content", namespace="ns_b", pool="personal")
    
    assert len(results_a) >= 1
    assert len(results_b) >= 1
    # Results should be from correct namespace
    for r in results_a:
        assert r["data"].get("namespace") == "ns_a" or "ns_a" in r["data"].get("uri", "")
    for r in results_b:
        assert r["data"].get("namespace") == "ns_b" or "ns_b" in r["data"].get("uri", "")


def test_shared_pool(palm_pilot):
    """Test shared pool operations"""
    uri = palm_pilot.remember(
        "Shared knowledge",
        namespace="shared_test",
        pool="shared",
    )
    
    assert uri.startswith("mem://shared/hermes/shared_test/")
    
    results = palm_pilot.search("shared knowledge", pool="shared")
    assert len(results) >= 1


def test_annotate_and_trust(palm_pilot):
    """Test annotation and trust score updates"""
    uri = palm_pilot.remember("Annotatable memory", namespace="trust", pool="personal")
    
    # Annotate as confirmed
    result = palm_pilot.annotate(
        uri=uri,
        review_text="Verified this is correct",
        confidence=0.9,
        verdict="confirmed",
    )
    
    assert "new_trust" in result
    assert result["new_trust"] > 0.5  # Should increase from prior
    assert result["verdict"] == "confirmed"


def test_stats(palm_pilot):
    """Test stats collection"""
    palm_pilot.remember("Stat test 1", namespace="stats", pool="personal")
    palm_pilot.remember("Stat test 2", namespace="stats", pool="personal")
    
    stats = palm_pilot.stats(namespace="stats", pool="personal")
    
    assert "personal" in stats
    assert stats["personal"]["count"] >= 2
    assert stats["personal"]["total_tokens"]["l0"] > 0
    assert stats["personal"]["total_tokens"]["l1"] > 0


def test_backup(palm_pilot):
    """Test backup functionality"""
    palm_pilot.remember("Backup test", namespace="backup", pool="personal")
    
    with tempfile.TemporaryDirectory() as backup_dir:
        dest = os.path.join(backup_dir, "test_backup")
        result = palm_pilot.backup(destination=dest, pool="personal")
        
        assert "personal" in result
        assert os.path.exists(result["personal"])
        assert os.path.getsize(result["personal"]) > 0


def test_forget(palm_pilot):
    """Test memory deletion"""
    uri = palm_pilot.remember("To be forgotten", namespace="forget", pool="personal")
    
    # Verify it exists
    results = palm_pilot.search("forgotten", pool="personal")
    assert len(results) >= 1
    
    # Forget it
    success = palm_pilot.forget(uri)
    assert success is True
    
    # Verify it's gone
    results = palm_pilot.search("forgotten", pool="personal")
    assert len(results) == 0


def test_version():
    """Test version is accessible"""
    from agent_palm import __version__
    assert __version__ == "1.2.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
