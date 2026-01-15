"""
Result Reranker Service
FlashRank-based semantic reranking for retrieved documents.
Gracefully degrades to similarity-based ordering when FlashRank unavailable.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

# Track if FlashRank is available
FLASHRANK_AVAILABLE = False
_ranker_instance = None

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
    logger.info("FlashRank: available")
except ImportError:
    logger.warning("FlashRank: not available, using similarity-based ordering")


@dataclass
class RankedDocument:
    """A document with its reranking score."""
    id: int
    content: str
    metadata: dict
    original_similarity: float
    rerank_score: float


def get_reranker():
    """
    Get or create the FlashRank reranker (singleton pattern).
    
    Returns:
        Ranker instance or None if FlashRank is not available.
    """
    global _ranker_instance
    
    if not FLASHRANK_AVAILABLE:
        return None
    
    if _ranker_instance is not None:
        return _ranker_instance
    
    try:
        from flashrank import Ranker
        _ranker_instance = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        logger.info("Reranker: initialized successfully")
        return _ranker_instance
    except Exception as e:
        logger.error(f"Reranker: initialization failed ({e})")
        return None


# Try to initialize on import (for Streamlit caching alternative)
try:
    import streamlit as st
    
    @st.cache_resource
    def _get_cached_reranker():
        return get_reranker()
    
    def get_reranker_cached():
        return _get_cached_reranker()
except ImportError:
    def get_reranker_cached():
        return get_reranker()


class ResultReranker:
    """
    Semantic reranker using FlashRank.
    Re-orders retrieved documents by relevance to the query.
    Falls back to similarity-based ordering when FlashRank unavailable.
    """

    def __init__(self):
        """Initialize with cached ranker instance."""
        self.ranker = get_reranker_cached()

    def rerank_docs(
        self,
        query: str,
        docs: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """
        Rerank documents by semantic relevance to the query.

        Args:
            query: The search query (preferably the corrected natural query)
            docs: List of documents from hybrid search 
                  (each with 'id', 'content', 'metadata', 'similarity')
            top_k: Number of top results to return

        Returns:
            List of top_k documents, reordered by relevance.
            Falls back to similarity-based order on failure.
        """
        if not docs:
            return []

        # If FlashRank not available, use similarity-based ordering
        if not self.ranker or not FLASHRANK_AVAILABLE:
            logger.info("Reranker: using similarity-based ordering (FlashRank unavailable)")
            return self._fallback_rerank(docs, top_k)

        try:
            from flashrank import RerankRequest

            # Prepare passages for FlashRank
            passages = [
                {"id": i, "text": doc.get("content", "")}
                for i, doc in enumerate(docs)
            ]

            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)
            
            # Perform reranking
            results = self.ranker.rerank(rerank_request)

            # Map back to original documents with new scores
            reranked_docs = []
            for result in results[:top_k]:
                original_idx = result["id"]
                if 0 <= original_idx < len(docs):
                    doc = docs[original_idx].copy()
                    doc["rerank_score"] = result.get("score", 0.0)
                    reranked_docs.append(doc)

            logger.info(f"Reranker: success - reranked {len(docs)} -> top {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.warning(f"Reranker: error ({e}), using fallback")
            return self._fallback_rerank(docs, top_k)

    def _fallback_rerank(self, docs: List[dict], top_k: int) -> List[dict]:
        """
        Fallback reranking using existing similarity scores.
        Simply sorts by similarity and returns top_k.
        """
        sorted_docs = sorted(
            docs, 
            key=lambda x: x.get("similarity", 0.0), 
            reverse=True
        )
        return sorted_docs[:top_k]

    def is_available(self) -> bool:
        """Check if FlashRank reranker is ready to use."""
        return self.ranker is not None and FLASHRANK_AVAILABLE

