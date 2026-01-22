"""
RAG Controller / Main Orchestrator
Coordinates Intelligence Engine, Hybrid Search, Reranker, and Semantic Cache.
"""

import asyncio
import logging
from typing import List, Tuple, Optional

from supabase import Client

from services.search_service import SearchIntelligence, SearchParams
from services.ranker_service import ResultReranker, rerank_with_huggingface
from services.query_router import QueryRouter, QueryIntent
from services.web_search import cached_web_search, format_web_results_as_context
from services.agentic.cache import SemanticCache, is_cache_available

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
    top_k: int = 10,
    use_cache: bool = True
) -> Tuple[str, List[str]]:
    """
    Main RAG orchestrator with Layered Intelligence and Semantic Caching.
    
    If caching is enabled and available, checks for similar queries first.
    This can reduce API calls by ~40% for repeated/similar questions.
    """
    
    # =========================================================================
    # STEP 0a: Check Semantic Cache (if available)
    # =========================================================================
    cache = None
    query_embedding = None
    
    if use_cache:
        try:
            if is_cache_available(supabase):
                cache = SemanticCache(supabase)
                
                # Get embedding for cache lookup (reuse later)
                vectors = get_embeddings_fn([raw_query])
                if vectors and vectors[0]:
                    query_embedding = vectors[0]
                    
                    # Check cache
                    cached = cache.get_cached_response(query_embedding, company_id)
                    if cached:
                        logger.info("Orchestrator: CACHE HIT - returning cached response")
                        return cached
        except Exception as e:
            logger.debug(f"Cache check failed (non-critical): {e}")
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
    
    # =========================================================================
    # STEP 6: Form Discovery & Proactive Linking (Option B)
    # =========================================================================
    try:
        forms = await get_relevant_forms(corrected_query, context_str, company_id, supabase)
        if forms:
            form_section = "\n\n=== RECOMMENDED FORMS (Secure Downloads) ===\n"
            for f in forms:
                form_section += f"- [{f['filename']}]({f['url']}) (Expires in 1h)\n"
            context_str += form_section
            logger.info(f"Orchestrator: Added {len(forms)} form links to context")
    except Exception as e:
        logger.warning(f"Form discovery failed (non-critical): {e}")
    
    logger.info(f"Orchestrator: returning {len(reranked_docs)} docs from {len(sources)} sources")
    
    return context_str, sources


async def get_relevant_forms(query: str, context: str, company_id: str, supabase: Client) -> List[dict]:
    """
    Discovery logic for Option B:
    1. Check if 'form', 'template', 'application' keywords exist in Context or Query.
    2. If yes, search DB for files matching query keywords AND 'form'/'template' etc.
    3. Generate 1-hour signed URLs.
    """
    # 1. Intent Detection - expanded with Dutch/French HR terms
    triggers = [
        "form", "template", "aanvraag", "formulier", "document", "sheet", 
        "application", "overdracht", "transfer", "verlof", "vakantie",
        "request", "demande", "congé", "download", "invullen", "fill"
    ]
    text_to_check = (query + " " + context).lower()
    
    if not any(t in text_to_check for t in triggers):
        return []

    # 2. Extract specific topics from query (naive but fast)
    # remove common stop words to find the "topic"
    stops = {
        "what", "is", "the", "how", "to", "can", "i", "do", "for", "a", "an", 
        "of", "in", "on", "my", "form", "template", "waar", "hoe", "een", "het", "de"
    }
    query_words = [w for w in query.lower().split() if w.isalnum() and w not in stops]
    
    if not query_words:
        return []

    found_forms = []
    
    try:
        # 3. Simple Search in Documents Table
        # We look for files that are relevant to the query topic
        # Relaxed: No longer requires 'form' in filename - just relevance
        
        # Get all active filenames for this company (usually small list < 1000)
        # For larger scale, we'd use a DB-side text search function
        res = supabase.table("documents").select("filename").eq("company_id", company_id).eq("is_active", True).execute()
        all_files = [d['filename'] for d in res.data]
        
        candidates = []
        for fname in all_files:
            fname_lower = fname.lower()
            # Match if filename relates to the query topic (at least one word match)
            is_relevant = any(w in fname_lower for w in query_words)
            if is_relevant:
                candidates.append(fname)
        
        # Limit to 3 most relevant
        candidates = candidates[:3]
        
        # 4. Generate Signed URLs
        for fname in candidates:
            try:
                # 3600 seconds = 1 hour
                # Sanitize company_id for storage path (must match upload path)
                import re
                safe_company_id = re.sub(r'[^\w\-]', '_', company_id)
                path = f"{safe_company_id}/{fname}"
                url_res = supabase.storage.from_("documents").create_signed_url(path, 3600, options={'download': True})
                
                # Check if response is a string (URL) or dict (depending on SDK version)
                # supabase-py v2 usually returns dict with 'signedURL' or just str?
                # Let's handle safe extraction
                signed_url = url_res
                if isinstance(url_res, dict):
                     signed_url = url_res.get("signedURL") or url_res.get("signed_url")

                if signed_url:
                    found_forms.append({"filename": fname, "url": signed_url})
            except Exception as e:
                logger.warning(f"Failed to sign URL for {fname}: {e}")
                
    except Exception as e:
        logger.error(f"Form search validation failed: {e}")
        
    return found_forms
