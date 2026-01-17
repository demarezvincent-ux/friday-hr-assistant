"""
RAG Controller / Main Orchestrator
Coordinates Intelligence Engine, Query Router, Hybrid Search, Web Search, and Reranker.
"""

import asyncio
import logging
from typing import List, Tuple, Optional

from supabase import Client

from .search_service import SearchIntelligence, SearchParams
from .ranker_service import ResultReranker, rerank_with_huggingface
from .query_router import QueryRouter, QueryIntent
from .web_search import cached_web_search, format_web_results_as_context

logger = logging.getLogger(__name__)


def get_context_with_strategy(
    raw_query: str,
    company_id: str,
    supabase: Client,
    groq_api_key: str,
    get_embeddings_fn,
    hf_api_key: Optional[str] = None,
    match_count: int = 40,
    top_k: int = 5
) -> Tuple[str, List[str]]:
    """
    Main RAG orchestrator with Layered Intelligence.

    Flow:
    1. Query Router: Classify intent -> DATABASE, WEB, or CHITCHAT
    2. Intelligence Engine: Analyze query -> corrected_query + fts_string
    3. Embedding: Generate vector from CORRECTED query
    4. Hybrid Search: Vector + FTS with high recall (or Web Search for WEB intent)
    5. Reranker: Re-order candidates by semantic relevance
    6. Return: Top K results as context string + source filenames

    Args:
        raw_query: Original user input (may have typos, no spaces, etc.)
        company_id: Filter for company-specific documents
        supabase: Supabase client instance
        groq_api_key: API key for Groq LLM
        get_embeddings_fn: Function to generate embeddings
        hf_api_key: HuggingFace API key for cross-encoder reranking (optional)
        match_count: Number of candidates to fetch before reranking (default 40)
        top_k: Number of final results to return (default 5)

    Returns:
        Tuple of (context_string, list_of_source_filenames)
    """
    # =========================================================================
    # STEP 0: Query Routing - Classify Intent
    # =========================================================================
    router = QueryRouter()
    intent, confidence = router.classify(raw_query)
    logger.info(f"Orchestrator: intent={intent.value} (conf={confidence:.2f})")
    
    # Handle CHITCHAT intent - no context needed
    if intent == QueryIntent.CHITCHAT:
        logger.info("Orchestrator: CHITCHAT detected, returning empty context")
        return "", []
    
    # Handle WEB intent - search the web
    if intent == QueryIntent.WEB:
        logger.info("Orchestrator: WEB search triggered")
        web_results = cached_web_search(raw_query, max_results=3)
        if web_results:
            context = format_web_results_as_context(web_results)
            sources = [f"Web: {r['title'][:30]}..." for r in web_results]
            return context, sources
        # Fallback to database if web search fails
        logger.info("Orchestrator: Web search returned nothing, falling back to DB")

    # =========================================================================
    # STEP 1: Intelligence Engine - Query Analysis
    # =========================================================================
    try:
        intelligence = SearchIntelligence(groq_api_key, timeout=2.0)
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
        # Try HuggingFace API reranker first (if key provided)
        if hf_api_key:
            reranked_docs = rerank_with_huggingface(
                query=corrected_query,
                docs=candidates,
                hf_api_key=hf_api_key,
                top_k=top_k
            )
        else:
            # Fall back to FlashRank or similarity-based
            reranker = ResultReranker()
            reranked_docs = reranker.rerank_docs(
                query=corrected_query,
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
    
    for i, doc in enumerate(reranked_docs, 1):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        
        # Add [Doc{i}] prefix for citation tracking
        context_parts.append(f"[Doc{i}] -- SOURCE: {filename} --\n{content}")
        
        if filename not in sources:
            sources.append(filename)

    context_str = "\n\n".join(context_parts)
    
    logger.info(f"Orchestrator: returning {len(reranked_docs)} docs from {len(sources)} sources")
    
    return context_str, sources

