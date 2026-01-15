"""
RAG Controller / Main Orchestrator
Coordinates Intelligence Engine, Hybrid Search, and Reranker.
"""

import asyncio
import logging
from typing import List, Tuple

from supabase import Client

from .search_service import SearchIntelligence, SearchParams
from .ranker_service import ResultReranker

logger = logging.getLogger(__name__)


def get_context_with_strategy(
    raw_query: str,
    company_id: str,
    supabase: Client,
    groq_api_key: str,
    get_embeddings_fn,
    match_count: int = 40,
    top_k: int = 5
) -> Tuple[str, List[str]]:
    """
    Main RAG orchestrator with Layered Intelligence.

    Flow:
    1. Intelligence Engine: Analyze query -> corrected_query + fts_string
    2. Embedding: Generate vector from CORRECTED query (not raw)
    3. Hybrid Search: Vector + FTS with high recall (match_count=40)
    4. Reranker: FlashRank re-orders candidates by semantic relevance
    5. Return: Top K results as context string + source filenames

    Args:
        raw_query: Original user input (may have typos, no spaces, etc.)
        company_id: Filter for company-specific documents
        supabase: Supabase client instance
        groq_api_key: API key for Groq LLM
        get_embeddings_fn: Function to generate embeddings (existing from main.py)
        match_count: Number of candidates to fetch before reranking (default 40)
        top_k: Number of final results to return (default 5)

    Returns:
        Tuple of (context_string, list_of_source_filenames)
    """
    # =========================================================================
    # STEP 1: Intelligence Engine - Query Analysis
    # =========================================================================
    try:
        intelligence = SearchIntelligence(groq_api_key, timeout=2.0)
        # Wrap async call for Streamlit sync context
        search_params = asyncio.run(intelligence.analyze_query(raw_query))
        
        corrected_query = search_params.corrected_natural_query
        fts_string = search_params.fts_search_string
        
        logger.info(f"Orchestrator: corrected='{corrected_query[:50]}...', fts='{fts_string[:50]}...'")
        
    except Exception as e:
        logger.warning(f"Orchestrator: Intelligence Engine failed ({e}), using raw query")
        corrected_query = raw_query
        fts_string = raw_query.strip().lower()

    # =========================================================================
    # STEP 2: Generate Embeddings on CORRECTED Query
    # =========================================================================
    try:
        # CRITICAL: Use corrected query, not raw query
        vectors = get_embeddings_fn([corrected_query])
        if not vectors or not vectors[0]:
            logger.warning("Orchestrator: embedding generation failed")
            return "", []
        query_embedding = vectors[0]
    except Exception as e:
        logger.error(f"Orchestrator: embedding error ({e})")
        return "", []

    # =========================================================================
    # STEP 3: Hybrid Search (Vector + FTS) with High Recall
    # =========================================================================
    try:
        rpc_params = {
            "query_embedding": query_embedding,
            "text_search_query": fts_string,
            "match_threshold": 0.15,
            "match_count": match_count,
            "company_id_filter": company_id
        }
        
        # Try new Elite function first, fallback to existing
        try:
            result = supabase.rpc("match_documents_hybrid_elite", rpc_params).execute()
        except Exception:
            # Fallback to existing function if elite not deployed yet
            logger.info("Orchestrator: Elite RPC not available, using legacy")
            legacy_params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.15,
                "match_count": match_count,
                "filter_company_id": company_id,
                "query_text": fts_string
            }
            result = supabase.rpc("match_documents_hybrid", legacy_params).execute()
        
        candidates = result.data if result.data else []
        logger.info(f"Orchestrator: retrieved {len(candidates)} candidates")
        
    except Exception as e:
        logger.error(f"Orchestrator: database search failed ({e})")
        return "", []

    if not candidates:
        return "", []

    # =========================================================================
    # STEP 4: Rerank Candidates
    # =========================================================================
    try:
        reranker = ResultReranker()
        reranked_docs = reranker.rerank_docs(
            query=corrected_query,  # Use corrected query for reranking
            docs=candidates,
            top_k=top_k
        )
    except Exception as e:
        logger.warning(f"Orchestrator: reranking failed ({e}), using raw order")
        reranked_docs = candidates[:top_k]

    # =========================================================================
    # STEP 5: Build Context String and Extract Sources
    # =========================================================================
    context_parts = []
    sources = []
    
    for doc in reranked_docs:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        
        context_parts.append(f"-- SOURCE: {filename} --\n{content}")
        
        if filename not in sources:
            sources.append(filename)

    context_str = "\n\n".join(context_parts)
    
    logger.info(f"Orchestrator: returning {len(reranked_docs)} docs from {len(sources)} sources")
    
    return context_str, sources
