"""
Retrieval utilities for agent_palm
RRF fusion, budget packing, citation verification, grounded context formatting,
and temporal filtering for time-aware recall.
"""

import re
import math
import time
from typing import List, Dict, Any, Sequence, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


def rrf_fuse(
    vec_results: List[Tuple[int, float]], 
    fts_results: List[Tuple[int, float]], 
    k: int = 60,
    vec_weight: float = 1.0,
    fts_weight: float = 0.8
) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion (RRF) for combining vector and FTS results.
    Returns dict of node_id -> fused score.
    """
    scores = {}
    
    # Vector results
    for rank, (node_id, _) in enumerate(vec_results):
        scores[node_id] = scores.get(node_id, 0.0) + vec_weight / (k + rank)
    
    # FTS results
    for rank, (node_id, _) in enumerate(fts_results):
        scores[node_id] = scores.get(node_id, 0.0) + fts_weight / (k + rank)
    
    return scores


@dataclass
class TemporalConstraint:
    """Parsed temporal constraint from a query"""
    start: Optional[float] = None  # Unix timestamp
    end: Optional[float] = None    # Unix timestamp
    raw_text: str = ""
    confidence: float = 0.0
    
    def overlaps(self, event_start: Optional[float], event_end: Optional[float], mentioned_at: Optional[float] = None) -> bool:
        """Check if a memory's temporal range overlaps this constraint"""
        if self.start is None and self.end is None:
            return True
            
        # Check the memory's time range
        mem_start = event_start
        mem_end = event_end
        
        # Fallback to mentioned_at if no explicit range
        if mem_start is None and mem_end is None and mentioned_at is not None:
            mem_start = mentioned_at
            mem_end = mentioned_at
            
        if mem_start is None and mem_end is None:
            return True  # No temporal info on memory, don't filter out
            
        # Check overlap
        if self.start is not None and mem_end is not None and mem_end < self.start:
            return False
        if self.end is not None and mem_start is not None and mem_start > self.end:
            return False
        return True


# Temporal extraction patterns
TEMPORAL_PATTERNS = [
    # Relative patterns
    (r'\b(last|past|previous)\s+(week|month|year|quarter|day|hour)\b', 'relative_backward'),
    (r'\b(next|coming|upcoming)\s+(week|month|year|quarter|day|hour)\b', 'relative_forward'),
    (r'\bthis\s+(week|month|year|quarter|day)\b', 'relative_this'),
    (r'\b(yesterday|today|tomorrow)\b', 'relative_day'),
    
    # Absolute date patterns
    (r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b', 'iso_date'),
    (r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b', 'us_date'),
    (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', 'month_name_date'),
    
    # Quarter patterns
    (r'\bQ[1-4]\s*\d{4}\b', 'quarter'),
    (r'\b(Q[1-4])\s+(?:FY)?\s*(\d{4})\b', 'quarter_fy'),
    
    # Year patterns
    (r'\b(19|20)\d{2}\b', 'year'),
    
    # Seasonal
    (r'\b(spring|summer|fall|autumn|winter)\s+\d{4}\b', 'season'),
    
    # Time ranges
    (r'\bbetween\s+(.+?)\s+and\s+(.+?)\b', 'between'),
    (r'\bfrom\s+(.+?)\s+to\s+(.+?)\b', 'from_to'),
    (r'\b(since|after)\s+(.+?)\b', 'since_after'),
    (r'\b(before|until)\s+(.+?)\b', 'before_until'),
]


def extract_temporal_constraint(
    query: str, 
    reference_time: Optional[float] = None,
    llm_extract: bool = False
) -> TemporalConstraint:
    """
    Extract temporal constraint from natural language query.
    
    Args:
        query: The search query text
        reference_time: Reference timestamp (defaults to now)
        llm_extract: If True, would use LLM for complex extraction (not implemented yet)
    
    Returns:
        TemporalConstraint with start/end timestamps
    """
    if reference_time is None:
        reference_time = time.time()
    
    ref_dt = datetime.fromtimestamp(reference_time)
    query_lower = query.lower()
    
    # Try heuristic patterns first
    constraint = _extract_heuristic(query_lower, ref_dt)
    
    if constraint.start is not None or constraint.end is not None:
        constraint.raw_text = query
        constraint.confidence = 0.7  # Heuristic confidence
        return constraint
    
    # If no heuristic match, return empty constraint (no filtering)
    return TemporalConstraint(raw_text=query, confidence=0.0)


def _extract_heuristic(query: str, ref_dt: datetime) -> TemporalConstraint:
    """Heuristic temporal extraction using regex patterns"""
    # python-dateutil is a declared dependency, but import it defensively so a
    # partial/incomplete environment degrades this ONE pattern (month-name
    # dates like "January 15, 2024") instead of crashing every search.
    try:
        import dateutil.parser
        _have_dateutil = True
    except ImportError:
        _have_dateutil = False
    
    # Relative: "last week", "past month", etc.
    for pattern, ptype in TEMPORAL_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            continue
            
        groups = match.groups()
        
        try:
            if ptype == 'relative_backward':
                # "last week", "past month"
                unit = groups[1].lower()
                return _relative_backward(ref_dt, unit)
                
            elif ptype == 'relative_forward':
                # "next month"
                unit = groups[1].lower()
                return _relative_forward(ref_dt, unit)
                
            elif ptype == 'relative_this':
                # "this week"
                unit = groups[0].lower()
                return _relative_this(ref_dt, unit)
                
            elif ptype == 'relative_day':
                # "yesterday", "today", "tomorrow"
                day = groups[0].lower()
                return _relative_day(ref_dt, day)
                
            elif ptype == 'iso_date':
                # YYYY-MM-DD
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                dt = datetime(year, month, day)
                return TemporalConstraint(start=dt.timestamp(), end=(dt + timedelta(days=1)).timestamp())
                
            elif ptype == 'us_date':
                # MM/DD/YYYY or MM/DD/YY
                month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                if year < 100:
                    year += 2000 if year < 50 else 1900
                dt = datetime(year, month, day)
                return TemporalConstraint(start=dt.timestamp(), end=(dt + timedelta(days=1)).timestamp())
                
            elif ptype == 'month_name_date':
                # "January 15, 2024"
                dt = dateutil.parser.parse(match.group(0))
                return TemporalConstraint(start=dt.timestamp(), end=(dt + timedelta(days=1)).timestamp())
                
            elif ptype == 'quarter':
                # "Q1 2024"
                q = int(match.group(0)[1])
                year = int(match.group(0)[-4:])
                return _quarter_to_range(year, q)
                
            elif ptype == 'quarter_fy':
                q = int(groups[0][1])
                year = int(groups[1])
                return _quarter_to_range(year, q)
                
            elif ptype == 'year':
                year = int(match.group(0))
                dt_start = datetime(year, 1, 1)
                dt_end = datetime(year + 1, 1, 1)
                return TemporalConstraint(start=dt_start.timestamp(), end=dt_end.timestamp())
                
            elif ptype == 'season':
                season, year = match.group(0).split()
                year = int(year)
                return _season_to_range(season.lower(), year)
                
            elif ptype in ('between', 'from_to'):
                # "between X and Y" or "from X to Y"
                start_str, end_str = groups[0], groups[1]
                start_dt = _parse_flexible_date(start_str.strip(), ref_dt)
                end_dt = _parse_flexible_date(end_str.strip(), ref_dt)
                if start_dt and end_dt:
                    return TemporalConstraint(
                        start=min(start_dt.timestamp(), end_dt.timestamp()),
                        end=max(start_dt.timestamp(), end_dt.timestamp())
                    )
                    
            elif ptype in ('since_after', 'before_until'):
                # "since X" or "before Y"
                date_str = groups[1].strip()
                dt = _parse_flexible_date(date_str, ref_dt)
                if dt:
                    if ptype == 'since_after':
                        return TemporalConstraint(start=dt.timestamp(), end=None)
                    else:
                        return TemporalConstraint(start=None, end=dt.timestamp())
                        
        except (ValueError, IndexError):
            continue
    
    return TemporalConstraint()


def _relative_backward(ref_dt: datetime, unit: str) -> TemporalConstraint:
    """Handle 'last week', 'past month', etc."""
    unit = unit.lower()
    if unit in ('week', 'weeks'):
        start = ref_dt - timedelta(weeks=1)
        return TemporalConstraint(start=start.timestamp(), end=ref_dt.timestamp())
    elif unit in ('month', 'months'):
        start = ref_dt.replace(day=1) - timedelta(days=1)  # First day of previous month
        start = start.replace(day=1)
        end = ref_dt.replace(day=1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('year', 'years'):
        start = datetime(ref_dt.year - 1, 1, 1)
        end = datetime(ref_dt.year, 1, 1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('quarter', 'quarters'):
        q = (ref_dt.month - 1) // 3 + 1
        if q == 1:
            year = ref_dt.year - 1
            q = 4
        else:
            year = ref_dt.year
            q -= 1
        return _quarter_to_range(year, q)
    elif unit in ('day', 'days'):
        start = ref_dt - timedelta(days=1)
        return TemporalConstraint(start=start.timestamp(), end=ref_dt.timestamp())
    elif unit in ('hour', 'hours'):
        start = ref_dt - timedelta(hours=1)
        return TemporalConstraint(start=start.timestamp(), end=ref_dt.timestamp())
    return TemporalConstraint()


def _relative_forward(ref_dt: datetime, unit: str) -> TemporalConstraint:
    """Handle 'next month', etc."""
    unit = unit.lower()
    if unit in ('week', 'weeks'):
        start = ref_dt
        end = ref_dt + timedelta(weeks=1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('month', 'months'):
        # Next month
        if ref_dt.month == 12:
            start = datetime(ref_dt.year + 1, 1, 1)
            end = datetime(ref_dt.year + 1, 2, 1)
        else:
            start = datetime(ref_dt.year, ref_dt.month + 1, 1)
            if start.month == 12:
                end = datetime(start.year + 1, 1, 1)
            else:
                end = datetime(start.year, start.month + 1, 1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('year', 'years'):
        start = datetime(ref_dt.year + 1, 1, 1)
        end = datetime(ref_dt.year + 2, 1, 1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('quarter', 'quarters'):
        q = (ref_dt.month - 1) // 3 + 1
        if q == 4:
            year = ref_dt.year + 1
            q = 1
        else:
            year = ref_dt.year
            q += 1
        return _quarter_to_range(year, q)
    return TemporalConstraint()


def _relative_this(ref_dt: datetime, unit: str) -> TemporalConstraint:
    """Handle 'this week', 'this month', etc."""
    unit = unit.lower()
    if unit in ('week', 'weeks'):
        # Monday of this week
        start = ref_dt - timedelta(days=ref_dt.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(weeks=1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('month', 'months'):
        start = ref_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = datetime(start.year + 1, 1, 1)
        else:
            end = datetime(start.year, start.month + 1, 1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('year', 'years'):
        start = datetime(ref_dt.year, 1, 1)
        end = datetime(ref_dt.year + 1, 1, 1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif unit in ('day', 'days'):
        start = ref_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    return TemporalConstraint()


def _relative_day(ref_dt: datetime, day: str) -> TemporalConstraint:
    """Handle 'yesterday', 'today', 'tomorrow'"""
    day = day.lower()
    if day == 'yesterday':
        start = (ref_dt - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif day == 'today':
        start = ref_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    elif day == 'tomorrow':
        start = (ref_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return TemporalConstraint(start=start.timestamp(), end=end.timestamp())
    return TemporalConstraint()


def _quarter_to_range(year: int, quarter: int) -> TemporalConstraint:
    """Convert quarter to date range"""
    if quarter < 1 or quarter > 4:
        return TemporalConstraint()
    month = (quarter - 1) * 3 + 1
    start = datetime(year, month, 1)
    if quarter == 4:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 3, 1)
    return TemporalConstraint(start=start.timestamp(), end=end.timestamp())


def _season_to_range(season: str, year: int) -> TemporalConstraint:
    """Convert season to date range"""
    seasons = {
        'spring': (3, 6),
        'summer': (6, 9),
        'fall': (9, 12),
        'autumn': (9, 12),
        'winter': (12, 3),
    }
    if season not in seasons:
        return TemporalConstraint()
    start_m, end_m = seasons[season]
    start = datetime(year, start_m, 1)
    if end_m == 3:
        end = datetime(year + 1, 3, 1)
    else:
        end = datetime(year, end_m, 1)
    return TemporalConstraint(start=start.timestamp(), end=end.timestamp())


def _parse_flexible_date(date_str: str, ref_dt: datetime) -> Optional[datetime]:
    """Parse flexible date strings like 'last monday', 'next week', 'Jan 15'"""
    try:
        import dateutil.parser
        # Try dateutil first
        return dateutil.parser.parse(date_str, default=ref_dt)
    except Exception:
        pass
    
    # Try relative patterns
    date_lower = date_str.lower().strip()
    
    # "last monday", "next friday"
    match = re.search(r'\b(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', date_lower)
    if match:
        direction, weekday = match.groups()
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        target = weekdays.index(weekday)
        current = ref_dt.weekday()
        if direction == 'last':
            delta = (current - target) % 7
            if delta == 0:
                delta = 7
            return ref_dt - timedelta(days=delta)
        else:  # next
            delta = (target - current) % 7
            if delta == 0:
                delta = 7
            return ref_dt + timedelta(days=delta)
    
    return None


def apply_temporal_filter(
    candidates: List[Dict[str, Any]], 
    constraint: TemporalConstraint
) -> List[Dict[str, Any]]:
    """
    Filter candidates by temporal constraint.
    
    Expects candidates to have created_at, and optionally event_date/occurred_start/occurred_end
    (these would be added to the schema for full temporal support).
    """
    if constraint.start is None and constraint.end is None:
        return candidates
    
    filtered = []
    for c in candidates:
        created_at = c.get('created_at')
        # For now, use created_at as the temporal anchor
        # In future, memory schema could include explicit event_date/occurred_start/occurred_end
        if created_at is not None:
            if constraint.overlaps(created_at, created_at):
                filtered.append(c)
        else:
            # No temporal info - include by default (don't filter out)
            filtered.append(c)
    
    return filtered


def compute_recency_boost(
    created_at: float, 
    memory_type: str = "note",
    trust: float = 0.5
) -> float:
    """
    Time-decay relevance boost.
    Half-life by type, scaled by trust.
    """
    age_days = (time.time() - created_at) / 86400
    
    half_lives = {
        "decision": 180,
        "fact": 365,
        "status": 7,
        "plan": 30,
        "note": 30,
        "observation": 90,
    }
    
    half_life = half_lives.get(memory_type, 30)
    
    # Exponential decay with floor scaled by trust
    decay = 0.5 ** (age_days / half_life)
    floor = 0.1 + (trust * 0.3)  # 0.1 to 0.4
    
    return max(decay, floor)


def pack_budget(
    candidates: List[Dict[str, Any]], 
    max_tokens: int,
    format_overhead_tokens: int = 35,
    l0_budget_ratio: float = 0.40
) -> List[Dict[str, Any]]:
    """
    Progressive budget packing for tiered context.
    
    Phase 1: Fit L0 items until 40% of budget
    Phase 2: Upgrade to L1 in priority order until budget exhausted
    Phase 3: Never include L2 (explicit recall only)
    
    Returns list of MemoryNode-like dicts with level field.
    """
    if not candidates or max_tokens <= 0:
        return []
    
    total_budget = max_tokens
    l0_ceiling = int(total_budget * l0_budget_ratio)
    
    selected = []
    accumulated_tokens = 0
    
    # Sort by score descending
    candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    
    # Phase 1: Fit L0 items
    for item in candidates:
        l0_cost = item.get("tok_l0", 0) + format_overhead_tokens
        if accumulated_tokens + l0_cost <= l0_ceiling:
            accumulated_tokens += l0_cost
            selected.append({
                "data": item,
                "level": "l0",
                "cost": l0_cost,
            })
        else:
            break
    
    # Phase 2: Upgrade to L1 in priority order
    remaining_budget = total_budget - accumulated_tokens
    
    for entry in selected:
        item = entry["data"]
        upgrade_cost = item.get("tok_l1", 0) - item.get("tok_l0", 0)
        if remaining_budget >= upgrade_cost:
            remaining_budget -= upgrade_cost
            entry["level"] = "l1"
            entry["cost"] = item.get("tok_l1", 0) + format_overhead_tokens
        else:
            # Keep at L0, stop upgrading
            break
    
    return selected


def build_memory_nodes(
    packed: List[Dict[str, Any]],
    format_overhead_tokens: int = 35
) -> List[Dict[str, Any]]:
    """Convert packed entries to MemoryNode dicts"""
    nodes = []
    for entry in packed:
        data = entry["data"]
        level = entry["level"]
        
        nodes.append({
            "uri": data["uri"],
            "pool": data["pool"],
            "owner": data["owner"],
            "author": data["author"],
            "namespace": data["namespace"],
            "level": level,
            "text": data["l1"] if level == "l1" else data["l0"],
            "tags": data.get("tags", []),
            "source": data.get("source", "inferred"),
            "latest_verdict": data.get("verdict"),
            "score": entry["data"].get("score", 0.0),
            "created_at": data.get("created_at", 0),
            "fields": data.get("fields"),
            "trust": data.get("trust", 0.5),
        })
    
    return nodes


def format_context(nodes: List[Dict[str, Any]], max_tokens: int = 4000) -> str:
    """
    Format nodes into grounded context block for agent consumption.
    Each memory wrapped with URI, level, and metadata for citation.
    """
    if not nodes:
        return ""
    
    lines = []
    total_tokens = 0
    overhead_per = 45  # URI header + metadata wrapper
    
    for node in nodes:
        # Estimate tokens for this entry
        text = node.get("text", "")
        token_est = len(text) // 4 + overhead_per
        
        if total_tokens + token_est > max_tokens:
            break
        
        uri = node.get("uri", "")
        level = node.get("level", "l0")
        source = node.get("source", "inferred")
        author = node.get("author", "unknown")
        trust = node.get("trust", 0.5)
        
        header = f'<memory uri="{uri}" level="{level}" source="{source}" author="{author}" trust="{trust:.2f}">'
        footer = '</memory>'
        
        lines.append(f"{header}\n{text}\n{footer}")
        total_tokens += token_est
    
    return "\n".join(lines)


def verify_citations(model_response: str, provided_uris: Sequence[str]) -> List[str]:
    """
    Hallucination guard: detect fabricated citations.
    Returns list of URIs cited in response that were NOT in provided context.
    """
    if not model_response or not provided_uris:
        return []
    
    # Pattern to match mem:// URIs
    uri_pattern = re.compile(r'mem://[^\s\]\)\>]+')
    cited = set(uri_pattern.findall(model_response))
    allowed = set(provided_uris)
    
    return list(cited - allowed)


def extract_cited_uris(model_response: str) -> List[str]:
    """Extract all mem:// URIs from model response"""
    return re.findall(r'mem://[^\s\]\)\>]+', model_response)


def citation_coverage(model_response: str, provided_uris: Sequence[str]) -> Dict[str, Any]:
    """
    Analyze citation quality in model response.
    Returns dict with coverage stats.
    """
    if not model_response:
        return {"total_citations": 0, "valid_citations": 0, "invalid_citations": 0, "coverage": 0.0}
    
    cited = extract_cited_uris(model_response)
    allowed = set(provided_uris)
    
    valid = [u for u in cited if u in allowed]
    invalid = [u for u in cited if u not in allowed]
    
    coverage = len(valid) / len(cited) if cited else 1.0
    
    return {
        "total_citations": len(cited),
        "valid_citations": len(valid),
        "invalid_citations": len(invalid),
        "coverage": coverage,
        "invalid_uris": invalid,
    }


def rrf_score(rank: Optional[int], k: int = 60, weight: float = 1.0) -> float:
    """Single RRF score for a rank"""
    if rank is None:
        return 0.0
    return weight / (k + rank)


def format_memory_for_context(node: Dict[str, Any], level: str = "l1") -> str:
    """Format a single memory node for context injection"""
    uri = node.get("uri", "")
    level_str = node.get("level", level)
    source = node.get("source", "inferred")
    author = node.get("author", "unknown")
    trust = node.get("trust", 0.5)
    text = node.get("text", "")
    
    return f'<memory uri="{uri}" level="{level_str}" source="{source}" author="{author}" trust="{trust:.2f}">\n{text}\n</memory>'


def build_grounded_context(nodes: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
    """
    Build grounded context block with token budgeting.
    Returns formatted string ready for agent context.
    """
    if not nodes:
        return ""
    
    # Use pack_budget for tiered selection
    candidates = []
    for node in nodes:
        candidates.append({
            "uri": node.get("uri", ""),
            "pool": node.get("pool", "personal"),
            "owner": node.get("owner", ""),
            "author": node.get("author", ""),
            "namespace": node.get("namespace", ""),
            "l0": node.get("l0", ""),
            "l1": node.get("l1", ""),
            "tok_l0": len(node.get("l0", "")) // 4,
            "tok_l1": len(node.get("l1", "")) // 4,
            "score": node.get("score", 0.0),
            "source": node.get("source", "inferred"),
            "verdict": node.get("verdict"),
            "tags": node.get("tags", []),
            "fields": node.get("fields"),
            "created_at": node.get("created_at", 0),
            "trust": node.get("trust", 0.5),
        })
    
    packed = pack_budget(candidates, max_tokens)
    return format_context(build_memory_nodes(packed), max_tokens)


if __name__ == "__main__":
    # Quick test
    test_nodes = [
        {
            "uri": "mem://personal/sebastian/test/note/abc123",
            "pool": "personal",
            "owner": "sebastian",
            "author": "sebastian",
            "namespace": "test",
            "level": "l1",
            "text": "Decided to use ONNX Runtime for embeddings instead of llama.cpp.",
            "tags": ["decision", "embeddings"],
            "source": "inferred",
            "score": 0.85,
            "created_at": 1700000000,
            "trust": 0.8,
        },
        {
            "uri": "mem://shared/hermes/decision/xyz789",
            "pool": "shared",
            "owner": "hermes",
            "author": "ling",
            "namespace": "decisions",
            "level": "l0",
            "text": "Team agreed to use Python 3.11+ for all new services.",
            "tags": ["policy", "python"],
            "source": "observed",
            "score": 0.72,
            "created_at": 1699000000,
            "trust": 0.9,
        },
    ]
    
    context = build_grounded_context(test_nodes, max_tokens=500)
    print("Context:")
    print(context)
    print("---")
    
    # Test temporal extraction
    print("\nTemporal extraction tests:")
    test_queries = [
        "What happened last week?",
        "Decisions made in Q3 2024",
        "Notes from yesterday",
        "Planning for next month",
        "Meeting on 2024-01-15",
        "What did we decide about embeddings?",  # No temporal constraint
    ]
    
    for q in test_queries:
        constraint = extract_temporal_constraint(q)
        print(f"  '{q}' -> start={constraint.start}, end={constraint.end}, conf={constraint.confidence}")
    
    # Test temporal filter
    print("\nTemporal filter test:")
    constraint = extract_temporal_constraint("last week")
    filtered = apply_temporal_filter(test_nodes, constraint)
    print(f"  Original: {len(test_nodes)}, Filtered: {len(filtered)}")
    
    # Test citation verification
    response = 'We decided to use ONNX [mem://personal/sebastian/test/note/abc123] and the team agreed on Python 3.11 [mem://shared/hermes/decision/xyz789].'
    provided = ["mem://personal/sebastian/test/note/abc123", "mem://shared/hermes/decision/xyz789"]
    
    invalid = verify_citations(response, provided)
    print(f"\nInvalid citations: {invalid}")
    
    coverage = citation_coverage(response, provided)
    print(f"Coverage: {coverage}")