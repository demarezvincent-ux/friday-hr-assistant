"""
Confidence Scoring
Calculate confidence score for RAG responses WITHOUT extra LLM calls.
Uses mathematical similarity and heuristics.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))
    except Exception:
        return 0.0


def calculate_confidence(
    query_embedding: Optional[List[float]] = None,
    context_embedding: Optional[List[float]] = None,
    sources_count: int = 0,
    context_length: int = 0,
    response_length: int = 0
) -> dict:
    """
    Calculate confidence score for a RAG response.
    
    Uses three factors (no LLM calls):
    1. Context Relevance (40%): Cosine similarity between query and context
    2. Source Diversity (30%): Number of sources used (capped at 3)
    3. Response Quality (30%): Ratio of response length to context length
    
    Args:
        query_embedding: Vector embedding of the query
        context_embedding: Vector embedding of the retrieved context
        sources_count: Number of source documents retrieved
        context_length: Character length of context
        response_length: Character length of response
        
    Returns:
        Dict with 'score' (0-1), 'level' (high/moderate/low), and 'factors'
    """
    factors = {}
    
    # 1. Context Relevance (40%)
    if query_embedding and context_embedding:
        relevance = cosine_similarity(query_embedding, context_embedding)
    else:
        relevance = 0.5  # Default if embeddings not provided
    
    factors["relevance"] = round(relevance, 3)
    
    # 2. Source Diversity (30%)
    # More sources = higher confidence, capped at 3 sources = 100%
    source_score = min(sources_count / 3, 1.0) if sources_count > 0 else 0.0
    factors["source_diversity"] = round(source_score, 3)
    
    # 3. Response Quality (30%)
    # Good responses are detailed but not just copying context
    if context_length > 0 and response_length > 0:
        # Ideal ratio is 0.2-0.5 (response should be condensed)
        ratio = response_length / context_length
        if 0.1 <= ratio <= 0.6:
            quality = 1.0
        elif ratio < 0.1:
            quality = ratio / 0.1  # Too short
        else:
            quality = max(0.3, 1.0 - (ratio - 0.6))  # Too verbose
    else:
        quality = 0.5  # Default
    
    factors["response_quality"] = round(quality, 3)
    
    # Weighted average
    final_score = (
        relevance * 0.4 +
        source_score * 0.3 +
        quality * 0.3
    )
    
    # Determine level
    if final_score >= 0.7:
        level = "high"
        emoji = "🎯"
        message = "High Confidence"
    elif final_score >= 0.45:
        level = "moderate"
        emoji = "💡"
        message = "Moderate Confidence"
    else:
        level = "low"
        emoji = "⚠️"
        message = "Low Confidence - Consider reviewing sources"
    
    return {
        "score": round(final_score, 3),
        "percent": int(final_score * 100),
        "level": level,
        "emoji": emoji,
        "message": message,
        "factors": factors
    }


def format_confidence_html(confidence: dict) -> str:
    """
    Format confidence score as HTML for display in Streamlit.
    
    Args:
        confidence: Result from calculate_confidence()
        
    Returns:
        HTML string for display
    """
    colors = {
        "high": "#2ECC71",      # Green
        "moderate": "#3498DB",   # Blue
        "low": "#E67E22"         # Orange
    }
    
    color = colors.get(confidence["level"], "#95A5A6")
    
    return f'''
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        background: {color}15;
        color: {color};
        font-size: 12px;
        font-weight: 500;
        margin-top: 8px;
    ">
        <span>{confidence["emoji"]}</span>
        <span>{confidence["message"]} ({confidence["percent"]}%)</span>
    </div>
    '''
