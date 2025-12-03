"""
Voting mechanisms for redundancy defense in MAS safety.
Implements majority voting with semantic similarity clustering.
"""

import re
from collections import Counter, defaultdict
from typing import List, Tuple, Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the content inside \\boxed{...} if present.
    This is useful for voting on final answers.
    """
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    return match.group(1) if match else None


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase, strip whitespace, remove punctuation.
    """
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def exact_match_vote(candidates: List[str]) -> Tuple[str, dict]:
    """
    Simple majority voting based on exact string matching.
    
    Args:
        candidates: List of candidate responses
        
    Returns:
        Tuple of (winning_candidate, vote_stats)
    """
    if not candidates:
        return "", {"total": 0, "unique": 0, "winner_votes": 0}
    
    # Normalize for comparison
    normalized = [normalize_text(c) for c in candidates]
    counter = Counter(normalized)
    
    # Get the most common normalized version
    most_common_normalized, count = counter.most_common(1)[0]
    
    # Find the original (non-normalized) version to return
    for i, norm in enumerate(normalized):
        if norm == most_common_normalized:
            winner = candidates[i]
            break
    
    stats = {
        "total": len(candidates),
        "unique": len(counter),
        "winner_votes": count,
        "vote_distribution": dict(counter.most_common()),
    }
    
    return winner, stats


def semantic_similarity_simple(text1: str, text2: str) -> float:
    """
    Simple word-overlap based similarity metric.
    Returns a score between 0 and 1.
    
    For production, this could be replaced with:
    - Sentence embeddings (e.g., sentence-transformers)
    - Edit distance
    - BERT-based similarity
    """
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    # Jaccard similarity
    return len(intersection) / len(union) if union else 0.0


def cluster_by_similarity(
    candidates: List[str], 
    similarity_threshold: float = 0.7
) -> List[List[int]]:
    """
    Cluster candidate responses by semantic similarity.
    
    Args:
        candidates: List of candidate responses
        similarity_threshold: Minimum similarity to be in same cluster
        
    Returns:
        List of clusters, where each cluster is a list of indices into candidates
    """
    n = len(candidates)
    clusters: List[List[int]] = []
    assigned = [False] * n
    
    for i in range(n):
        if assigned[i]:
            continue
            
        # Start a new cluster with this candidate
        cluster = [i]
        assigned[i] = True
        
        # Find all similar candidates
        for j in range(i + 1, n):
            if assigned[j]:
                continue
                
            similarity = semantic_similarity_simple(candidates[i], candidates[j])
            if similarity >= similarity_threshold:
                cluster.append(j)
                assigned[j] = True
        
        clusters.append(cluster)
    
    return clusters


def semantic_majority_vote(
    candidates: List[str], 
    similarity_threshold: float = 0.7
) -> Tuple[str, dict]:
    """
    Majority voting with semantic similarity clustering.
    Groups similar (but not identical) responses and picks the largest cluster.
    
    Args:
        candidates: List of candidate responses
        similarity_threshold: Minimum similarity to group responses together
        
    Returns:
        Tuple of (winning_candidate, vote_stats)
    """
    if not candidates:
        return "", {"total": 0, "clusters": 0, "winner_cluster_size": 0}
    
    if len(candidates) == 1:
        return candidates[0], {"total": 1, "clusters": 1, "winner_cluster_size": 1}
    
    # Cluster similar responses
    clusters = cluster_by_similarity(candidates, similarity_threshold)
    
    # Find the largest cluster
    largest_cluster = max(clusters, key=len)
    
    # Return the first candidate from the largest cluster
    winner_idx = largest_cluster[0]
    winner = candidates[winner_idx]
    
    stats = {
        "total": len(candidates),
        "clusters": len(clusters),
        "winner_cluster_size": len(largest_cluster),
        "cluster_sizes": [len(c) for c in clusters],
    }
    
    return winner, stats


def boxed_answer_vote(candidates: List[str]) -> Tuple[str, dict]:
    """
    Vote based on the content inside \\boxed{...} tags.
    Falls back to full text voting if no boxed content found.
    
    This is useful for final answerer agents where the answer is boxed.
    """
    boxed_answers = [extract_boxed_answer(c) for c in candidates]
    
    # If none have boxed answers, fall back to regular voting
    if all(ans is None for ans in boxed_answers):
        return semantic_majority_vote(candidates)
    
    # Filter out None values and vote on boxed content
    valid_boxed = [ans for ans in boxed_answers if ans is not None]
    
    if not valid_boxed:
        return semantic_majority_vote(candidates)
    
    # Vote on the boxed answers
    counter = Counter(valid_boxed)
    most_common_boxed, count = counter.most_common(1)[0]
    
    # Return the full candidate that had this boxed answer
    for i, boxed in enumerate(boxed_answers):
        if boxed == most_common_boxed:
            winner = candidates[i]
            break
    
    stats = {
        "total": len(candidates),
        "valid_boxed": len(valid_boxed),
        "unique_boxed": len(counter),
        "winner_votes": count,
    }
    
    return winner, stats


def majority_vote(
    candidates: List[str],
    strategy: str = "semantic",
    similarity_threshold: float = 0.7,
) -> Tuple[str, dict]:
    """
    Main voting function that dispatches to specific voting strategies.
    
    Args:
        candidates: List of candidate responses
        strategy: Voting strategy - "exact", "semantic", or "boxed"
        similarity_threshold: For semantic voting, minimum similarity threshold
        
    Returns:
        Tuple of (winning_candidate, vote_statistics)
    """
    if not candidates:
        return "", {"error": "No candidates provided"}
    
    if len(candidates) == 1:
        return candidates[0], {"total": 1, "strategy": strategy}
    
    if strategy == "exact":
        return exact_match_vote(candidates)
    elif strategy == "semantic":
        return semantic_majority_vote(candidates, similarity_threshold)
    elif strategy == "boxed":
        return boxed_answer_vote(candidates)
    else:
        # Default to semantic
        return semantic_majority_vote(candidates, similarity_threshold)
