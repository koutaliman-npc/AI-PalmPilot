"""
Heuristic extraction for L0/L1 from raw content
No LLM required - fast, offline, deterministic
"""

import re
from typing import Tuple, List


# Patterns for sentence splitting
SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Keywords that indicate important content
KEY_PHRASES = [
    r'\b(decided|decision|concluded|conclusion|resolved|resolution)\b',
    r'\b(important|critical|key|essential|vital|crucial)\b',
    r'\b(must|should|need to|have to|required)\b',
    r'\b(bug|issue|error|problem|fix|solution)\b',
    r'\b(todo|task|action|next step|follow.?up)\b',
    r'\b(meeting|discussion|agreed|consensus)\b',
    r'\b(deadline|milestone|release|deploy)\b',
]

# Compile for performance
KEY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in KEY_PHRASES]

# Bullet/list patterns
BULLET_PATTERN = re.compile(r'^[\s]*[-*•]\s+(.+)$', re.MULTILINE)

# Entity-like patterns (capitalized words, proper nouns)
ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    if not text:
        return []
    sentences = SENTENCE_ENDINGS.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_entities(text: str) -> List[str]:
    """Extract capitalized entity-like phrases"""
    entities = ENTITY_PATTERN.findall(text)
    # Filter common words
    common = {'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'Why', 'How', 'If', 'Then', 'Else', 'And', 'Or', 'But', 'For', 'In', 'On', 'At', 'To', 'From', 'With', 'By', 'As', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being', 'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Will', 'Would', 'Could', 'Should', 'May', 'Might', 'Can', 'Must', 'Shall', 'Will'}
    return [e for e in entities if e not in common and len(e) > 2]


def score_sentence_importance(sentence: str, entities: List[str]) -> float:
    """Score a sentence for L1 inclusion"""
    score = 0.0
    lower = sentence.lower()
    
    # Key phrase matches
    for pattern in KEY_PATTERNS:
        if pattern.search(sentence):
            score += 2.0
    
    # Entity density
    words = sentence.split()
    if words:
        entity_count = sum(1 for w in words if w in entities)
        score += entity_count * 0.5
    
    # Length bonus (not too short, not too long)
    word_count = len(words)
    if 5 <= word_count <= 30:
        score += 1.0
    elif word_count > 50:
        score -= 1.0
    
    # Position bonus (earlier sentences often more important)
    # This would need sentence index, handled by caller
    
    return score


def extract_bullets(text: str) -> List[str]:
    """Extract bullet points"""
    return [m.group(1).strip() for m in BULLET_PATTERN.finditer(text)]


def heuristic_extract(content: str, max_l0_chars: int = 280, max_l1_chars: int = 2000) -> Tuple[str, str]:
    """
    Extract L0 (abstract) and L1 (overview) from raw content.
    
    L0: First 1-2 sentences, truncated to max_l0_chars
    L1: Key sentences + bullets + entities, truncated to max_l1_chars
    """
    if not content or not content.strip():
        return "", ""
    
    content = content.strip()
    sentences = split_sentences(content)
    
    # L0: First 1-2 sentences
    l0_parts = []
    char_count = 0
    for sent in sentences[:2]:
        if char_count + len(sent) <= 280:
            l0_parts.append(sent)
            char_count += len(sent) + 1
        else:
            break
    l0 = " ".join(l0_parts)[:max_l0_chars]
    # Fallback: pathological content (no sentence split, e.g. long unbroken
    # tokens) yields an empty L0 — carry the head of the content instead so
    # the tier is never empty (mirrors the L1 fallback below).
    if not l0:
        l0 = content[:max_l0_chars]
    
    # L1: Key sentences + bullets + entities
    if not sentences:
        return l0, l0
    
    entities = extract_entities(content)
    bullets = extract_bullets(content)
    
    # Score all sentences
    scored = []
    for i, sent in enumerate(sentences):
        score = score_sentence_importance(sent, entities)
        # Position bonus for early sentences
        if i < 3:
            score += 1.5
        elif i < 6:
            score += 0.5
        scored.append((score, i, sent))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Build L1
    l1_parts = []
    char_count = 0
    
    # Always include first sentence if not in L0
    if len(sentences) > len(l0_parts):
        first_remaining = sentences[len(l0_parts)]
        if len(first_remaining) <= max_l1_chars:
            l1_parts.append(first_remaining)
            char_count += len(first_remaining) + 1
    
    # Add top scored sentences
    for score, idx, sent in scored[:8]:  # Top 8 candidates
        if idx < len(l0_parts):
            continue  # Already in L0
        if char_count + len(sent) + 1 <= max_l1_chars:
            l1_parts.append(sent)
            char_count += len(sent) + 1
        else:
            break
    
    # Add bullets
    for bullet in bullets[:5]:
        if char_count + len(bullet) + 1 <= max_l1_chars:
            l1_parts.append(f"• {bullet}")
            char_count += len(bullet) + 3
        else:
            break
    
    l1 = " ".join(l1_parts)[:max_l1_chars]
    
    # Fallback: if L1 is empty or too short, use first chunk of content
    if len(l1) < 50:
        l1 = content[:max_l1_chars]
    
    return l0, l1


def heuristic_extract_with_structure(content: str, structure: dict = None) -> Tuple[str, str]:
    """
    Extended extraction that uses document structure if available.
    structure can contain: headings, code blocks, lists, etc.
    """
    # For now, delegate to basic heuristic
    # Future: use structure to weight sections
    return heuristic_extract(content)


# For testing
if __name__ == "__main__":
    test_content = """
    Sebastian decided to use ONNX Runtime for embeddings instead of llama.cpp.
    The decision was made because llama.cpp kept crashing with GGML_ASSERT errors.
    This is a critical fix for the memory system.
    
    Key points:
    • ONNX Runtime works on Windows without VS Build Tools
    • all-MiniLM-L6-v2 int8 is only 23MB
    • No compilation needed
    
    Next steps: integrate with SQLite vec0 and test retrieval.
    """
    
    l0, l1 = heuristic_extract(test_content)
    print("L0:", l0)
    print("---")
    print("L1:", l1)