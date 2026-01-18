"""
RAG Controller / Main Orchestrator
Coordinates Intelligence Engine, Hybrid Search, and Reranker.
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


def diversify_by_source(candidates: List[dict], max_per_source: int = 5) -> List[dict]:
    """
    Limit the number of chunks per source document to ensure result diversity.
    
    This prevents a single document's chunks from dominating all results.
    Keeps the top-scoring chunks from each source.
    
    Args:
        candidates: List of document chunks with metadata containing 'filename'
        max_per_source: Maximum chunks to keep per source file
    
    Returns:
        Diversified list of candidates
    """
    if not candidates:
        return []
    
    # Group candidates by source filename
    source_groups = {}
    for doc in candidates:
        metadata = doc.get("metadata", {})
        filename = metadata.get("filename", "unknown")
        if filename not in source_groups:
            source_groups[filename] = []
        source_groups[filename].append(doc)
    
    # DEBUG: Log all unique sources found
    source_counts = {fn: len(docs) for fn, docs in source_groups.items()}
    logger.info(f"Diversifier DEBUG: Found {len(source_groups)} unique sources in {len(candidates)} candidates: {source_counts}")
    
    # Sort each group by similarity score (keep best per source)
    diversified = []
    for filename, docs in source_groups.items():
        # Sort by similarity descending
        sorted_docs = sorted(docs, key=lambda x: x.get("similarity", 0), reverse=True)
        # Keep only top N from this source
        diversified.extend(sorted_docs[:max_per_source])
    
    # Re-sort the diversified list by similarity for fair reranking
    diversified.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    
    logger.info(f"Diversifier: {len(candidates)} candidates -> {len(diversified)} from {len(source_groups)} sources (max {max_per_source}/source)")
    
    return diversified



async def get_context_with_strategy(
    raw_query: str,
    company_id: str,
    supabase: Client,
    groq_api_key: str,
    get_embeddings_fn,
    hf_api_key: Optional[str] = None,
    match_count: int = 200,
    top_k: int = 10
) -> Tuple[str, List[str]]:
    """
    Main RAG orchestrator with Layered Intelligence.
    """
    # =========================================================================
    # STEP 0: Query Routing - Classify Intent
    # =========================================================================
    router = QueryRouter()
    intent, confidence = router.classify(raw_query)
    logger.info(f"Orchestrator: intent={intent.value} (conf={confidence:.2f})")
    
    if intent == QueryIntent.CHITCHAT:
        return "", []
    
    if intent == QueryIntent.WEB:
        web_results = cached_web_search(raw_query, max_results=3)
        if web_results:
            context = format_web_results_as_context(web_results)
            sources = [f"Web: {r['title'][:30]}..." for r in web_results]
            return context, sources

    # =========================================================================
    # STEP 1: Intelligence Engine - Query Analysis
    # =========================================================================
    try:
        intelligence = SearchIntelligence(groq_api_key, timeout=2.0)
        # Use await since we are in an async function
        search_params = await intelligence.analyze_query(raw_query)
        
        corrected_query = search_params.corrected_natural_query
        fts_string = search_params.fts_search_string
        
        logger.info(f"Orchestrator: corrected='{corrected_query[:50]}...', fts='{fts_string}'")
        
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
            "match_threshold": 0.05,  # Very low for maximum recall
            "match_count": 300,  # Increased to catch more candidates
            "company_id_filter": company_id
        }
        
        # Try new Elite function first, fallback to existing
        try:
            result = supabase.rpc("match_documents_hybrid_elite", rpc_params).execute()
        except Exception:
            # Fallback to legacy function with different parameter names
            fallback_params = {
                "query_embedding": query_embedding,
                "query_text": fts_string,
                "match_threshold": 0.05,  # Very low for maximum recall
                "match_count": 300,  # Increased to catch more candidates
                "filter_company_id": company_id
            }
            result = supabase.rpc("match_documents_hybrid", fallback_params).execute()
        
        candidates = result.data if result.data else []
        logger.info(f"Orchestrator: retrieved {len(candidates)} candidates")
        
    except Exception as e:
        logger.error(f"Orchestrator: database search failed ({e})")
        return "", []

    if not candidates:
        return "", []

    # =========================================================================
    # STEP 4: Diversify and Rerank Candidates
    # =========================================================================
    # CRITICAL FIX: Diversify by source BEFORE reranking to prevent single-source dominance
    diversified_candidates = diversify_by_source(candidates, max_per_source=5)
    
    try:
        if hf_api_key:
            reranked_docs = rerank_with_huggingface(
                query=corrected_query,
                docs=diversified_candidates,  # Use diversified candidates
                hf_api_key=hf_api_key,
                top_k=min(top_k * 2, len(diversified_candidates))  # Get more for diversity
            )
        else:
            reranker = ResultReranker()
            reranked_docs = reranker.rerank_docs(
                query=corrected_query,
                docs=diversified_candidates,  # Use diversified candidates
                top_k=min(top_k * 2, len(diversified_candidates))  # Get more for diversity
            )
    except Exception as e:
        logger.warning(f"Orchestrator: reranking failed ({e}), using raw order")
        reranked_docs = diversified_candidates[:top_k]  # Use diversified candidates
    
    # CRITICAL: Post-rerank diversification - ensure ALL sources are represented
    # Keep up to 2 chunks per source to catch variations (like password in 2nd chunk)
    final_docs = []
    source_count = {}
    MAX_PER_SOURCE = 2
    
    # First pass: take top 2 docs from each unique source
    for doc in reranked_docs:
        filename = doc.get("metadata", {}).get("filename", "Unknown")
        if source_count.get(filename, 0) < MAX_PER_SOURCE:
            final_docs.append(doc)
            source_count[filename] = source_count.get(filename, 0) + 1
            if len(final_docs) >= top_k:
                break
    
    # Second pass: fill remaining slots if we still have room
    for doc in reranked_docs:
        if len(final_docs) >= top_k:
            break
        if doc not in final_docs:
            final_docs.append(doc)
    
    reranked_docs = final_docs



    # =========================================================================
    # STEP 5: Build Context String and Extract Sources
    # =========================================================================
    context_parts = []
    sources = []
    
    for doc in reranked_docs:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        
        # Simplified header as requested
        context_parts.append(f"-- SOURCE: {filename} --\n{content}")
        
        if filename not in sources:
            sources.append(filename)

    context_str = "\n\n".join(context_parts)
    
    logger.info(f"Orchestrator: returning {len(reranked_docs)} docs from {len(sources)} sources")
    
    return context_str, sources
