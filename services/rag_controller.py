"""
RAG Controller / Main Orchestrator
Coordinates Intelligence Engine, Hybrid Search, Reranker, and Semantic Cache.
"""

import asyncio
import logging
import re
from typing import List, Tuple, Optional

from supabase import Client

from services.search_service import SearchIntelligence, SearchParams
from services.ranker_service import ResultReranker, rerank_with_huggingface
from services.query_router import QueryRouter, QueryIntent
from services.web_search import cached_web_search, format_web_results_as_context
from services.agentic.cache import SemanticCache, is_cache_available

logger = logging.getLogger(__name__)


def sanitize_fts_query(query: str) -> str:
    """
    Sanitize search string for Postgres to_tsquery.
    Removes special characters that cause syntax errors (like ' in CAO's).
    """
    # Remove apostrophes, quotes, and tsquery special chars
    sanitized = re.sub(r"['\"\(\)\[\]:&!|<>*]", " ", query)
    # Collapse whitespace and filter short terms
    terms = [t.strip() for t in sanitized.split() if len(t.strip()) > 2]
    # Create pipe-delimited query (max 10 terms)
    return " | ".join(terms[:10]) if terms else ""


async def search_legal_knowledge(
    query: str,
    query_embedding: List[float],
    supabase: Client,
    fts_string: str = "",
    source_filter: Optional[str] = None,
    match_count: int = 5
) -> Tuple[str, List[str]]:
    """
    Search the legal_knowledge table for Belgian labor law.
    Returns context string and list of sources.
    """
    try:
        # Sanitize FTS string to prevent tsquery syntax errors (e.g., CAO's)
        safe_fts = sanitize_fts_query(fts_string or query)
        
        rpc_params = {
            "query_embedding": query_embedding,
            "text_search_query": safe_fts,
            "match_threshold": 0.15,
            "match_count": match_count,
            "source_filter": source_filter
        }
        
        result = supabase.rpc("match_legal_documents", rpc_params).execute()
        
        if not result.data:
            logger.info("Legal search: No matches found")
            return "", []
        
        context_parts = []
        sources = []
        
        for doc in result.data:
            source = doc.get("metadata", {}).get("source", "Legal")
            
            # Article-aware header for Belgian law articles
            if source == "BELGIAN_LAW":
                law_name = doc.get("metadata", {}).get("law_name", "")
                law_date = doc.get("metadata", {}).get("law_date", "")
                article_number = doc.get("metadata", {}).get("article_number", "")
                title = doc.get("metadata", {}).get("title", "")
                chapter = doc.get("metadata", {}).get("chapter", "")
                
                header = f"--- [CITE THIS: {article_number} van de {law_name} ({law_date})] ---"
                if title or chapter:
                    hierarchy_parts = [p for p in [title, chapter] if p]
                    header += f"\n{' > '.join(hierarchy_parts)}"
                
                # Use full content for law articles (not truncated)
                content = doc.get("content", "")
                source_label = f"{article_number} {law_name}"
            else:
                # Existing logic for other legal sources
                topic = doc.get("metadata", {}).get("topic", "")
                effective_date = doc.get("metadata", {}).get("effective_date", "")
                
                header = f"-- LEGAL SOURCE: {source}"
                if topic:
                    header += f" ({topic})"
                if effective_date:
                    header += f" [Effective: {effective_date}]"
                header += " --"
                
                # Prefer summary if available, otherwise use content snippet
                content = doc.get("summary") or doc.get("content", "")[:1500]
                source_label = f"{source} - {topic}" if topic else source
            
            context_parts.append(f"{header}\n{content}")
            
            if source_label not in sources:
                sources.append(source_label)
        
        logger.info(f"Legal search: Found {len(result.data)} docs from {len(sources)} sources")
        return "\n\n".join(context_parts), sources
        
    except Exception as e:
        logger.warning(f"Legal knowledge search failed: {e}")
        return "", []


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
) -> Tuple[str, dict]:
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
        return "", {"legal_sources": [], "company_sources": []}
    
    if intent == QueryIntent.WEB:
        web_results = cached_web_search(raw_query, max_results=3)
        if web_results:
            context = format_web_results_as_context(web_results)
            sources = [f"Web: {r['title'][:30]}..." for r in web_results]
            return context, {"legal_sources": [], "company_sources": sources}
    
    # =========================================================================
    # STEP 0c: For LEGAL or DATABASE intent, we'll search BOTH legal_knowledge AND company docs
    # Rationale: Most HR questions have Belgian labor law implications, even if not explicitly stated
    # =========================================================================
    is_legal_query = (intent == QueryIntent.LEGAL or intent == QueryIntent.DATABASE)

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
            return "", {"legal_sources": [], "company_sources": []}
        query_embedding = vectors[0]
    except Exception as e:
        logger.error(f"Orchestrator: embedding error ({e})")
        return "", {"legal_sources": [], "company_sources": []}

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
        return "", {"legal_sources": [], "company_sources": []}

    if not candidates:
        return "", {"legal_sources": [], "company_sources": []}

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
    # STEP 5: Build Context String and Extract Sources (Company Docs)
    # =========================================================================
    context_parts = []
    company_sources = []
    legal_sources = []
    
    for doc in reranked_docs:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        
        # Simplified header as requested
        context_parts.append(f"-- SOURCE: {filename} --\n{content}")
        
        if filename not in company_sources:
            company_sources.append(filename)

    context_str = "\n\n".join(context_parts)
    
    # =========================================================================
    # STEP 5b: Legal Knowledge Search (for LEGAL intent queries)
    # =========================================================================
    if is_legal_query:
        try:
            legal_context, legal_src = await search_legal_knowledge(
                query=corrected_query,
                query_embedding=query_embedding,
                supabase=supabase,
                fts_string=fts_string,
                match_count=5
            )
            if legal_context:
                # Prepend legal context (prioritize law over company policy)
                context_str = f"=== BELGIAN LABOR LAW ===\n{legal_context}\n\n=== COMPANY POLICY ===\n{context_str}"
                legal_sources = legal_src
                logger.info(f"Orchestrator: Added legal context from {len(legal_sources)} sources")
        except Exception as e:
            logger.warning(f"Legal knowledge search failed (non-critical): {e}")
    
    # =========================================================================
    # STEP 6: Form Discovery & Proactive Linking (Option B)
    # =========================================================================
    try:
        forms = await get_relevant_forms(raw_query, corrected_query, context_str, company_id, supabase)
        if forms:
            form_section = "\n\n=== RECOMMENDED FORMS (Secure Downloads) ===\n"
            for f in forms:
                form_section += f"- [{f['filename']}]({f['url']}) (Expires in 1h)\n"
            context_str += form_section
            logger.info(f"Orchestrator: Added {len(forms)} form links to context")
    except Exception as e:
        logger.warning(f"Form discovery failed (non-critical): {e}")
    
    all_count = len(legal_sources) + len(company_sources)
    logger.info(f"Orchestrator: returning {len(reranked_docs)} docs from {all_count} sources (legal={len(legal_sources)}, company={len(company_sources)})")
    
    return context_str, {"legal_sources": legal_sources, "company_sources": company_sources}


async def get_relevant_forms(raw_query: str, corrected_query: str, context: str, company_id: str, supabase: Client) -> List[dict]:
    """
    Discovery logic for Option B:
    1. Check if 'form', 'template', 'application' keywords exist in Context or Query.
    2. If yes, search DB for files matching query keywords AND 'form'/'template' etc.
    3. Generate 1-hour signed URLs.
    
    Uses BOTH raw_query (original language) and corrected_query (English) to match filenames.
    """
    import re
    
    # 1. Intent Detection - expanded with Dutch/French HR terms
    triggers = [
        "form", "template", "aanvraag", "formulier", "document", "sheet", 
        "application", "overdracht", "transfer", "verlof", "vakantie",
        "request", "demande", "congé", "download", "invullen", "fill"
    ]
    text_to_check = (raw_query + " " + corrected_query + " " + context).lower()
    
    triggered_words = [t for t in triggers if t in text_to_check]
    logger.info(f"Form Discovery: triggers found: {triggered_words}")
    
    if not triggered_words:
        logger.info("Form Discovery: No triggers found, skipping")
        return []

    # 2. Extract specific topics from BOTH queries (original language + translated)
    # This allows matching Dutch filenames from Dutch queries AND English from English
    stops = {
        # English - expanded
        "what", "is", "the", "how", "to", "can", "i", "do", "for", "a", "an", 
        "of", "in", "on", "my", "form", "template", "you", "me", "we", "they",
        "with", "this", "that", "are", "was", "were", "been", "have", "has",
        "will", "would", "could", "should", "may", "might", "get", "give",
        "please", "provide", "want", "need", "there", "here",
        # Dutch
        "waar", "hoe", "een", "het", "de", "ik", "mijn", "je", "jouw", "dit", "dat",
        "naar", "voor", "met", "bij", "om", "als", "maar", "dan", "wel", "niet",
        "wil", "kan", "kun", "mag", "moet", "zal", "zou", "geven",
        # French
        "le", "la", "les", "un", "une", "des", "je", "tu", "il", "elle", "nous",
        "vous", "comment", "quoi", "où", "pourquoi", "est", "sont", "être"
    }
    
    # Combine words from BOTH queries to match filenames in any language
    combined_query = raw_query + " " + corrected_query
    query_words = [w for w in combined_query.lower().split() if w.isalnum() and len(w) > 2 and w not in stops]
    
    logger.info(f"Form Discovery: raw='{raw_query[:50]}...', corrected='{corrected_query[:50]}...' -> extracted words: {query_words}")
    
    if not query_words:
        logger.info("Form Discovery: No query words after stop-word filtering")
        return []

    found_forms = []
    
    try:
        # 3. Simple Search in Documents Table
        res = supabase.table("documents").select("filename").eq("company_id", company_id).eq("is_active", True).execute()
        all_files = [d['filename'] for d in res.data]
        
        logger.info(f"Form Discovery: {len(all_files)} active files in company")
        
        candidates = []
        for fname in all_files:
            fname_lower = fname.lower()
            # Match if filename relates to the query topic (at least one word match)
            # Also check if query word is PART of any word in filename (substring match)
            matching_words = [w for w in query_words if w in fname_lower]
            if matching_words:
                candidates.append(fname)
                logger.info(f"Form Discovery: '{fname}' matched by words: {matching_words}")
        
        logger.info(f"Form Discovery: {len(candidates)} candidates before limit: {candidates}")
        
        # Limit to 3 most relevant
        candidates = candidates[:3]
        
        # 4. Generate Signed URLs
        for fname in candidates:
            try:
                # Sanitize company_id for storage path (must match upload path)
                safe_company_id = re.sub(r'[^\w\-]', '_', company_id)
                path = f"{safe_company_id}/{fname}"
                
                logger.info(f"Form Discovery: Creating signed URL for path: {path}")
                
                url_res = supabase.storage.from_("documents").create_signed_url(path, 3600)
                
                logger.info(f"Form Discovery: Signed URL response: type={type(url_res).__name__}, has_data={hasattr(url_res, 'data')}")
                
                # Handle all possible SDK response formats (priority order matters!)
                signed_url = None
                
                # Priority 1: Object with .data attribute (newer Supabase SDK - most common)
                if hasattr(url_res, 'data') and url_res.data:
                    data = url_res.data
                    if isinstance(data, dict):
                        signed_url = data.get("signedUrl") or data.get("signedURL") or data.get("signed_url")
                    elif isinstance(data, str) and data.startswith("http"):
                        signed_url = data
                    logger.info(f"Form Discovery: Extracted from .data attribute: {bool(signed_url)}")
                
                # Priority 2: Direct dict response (older SDK format)
                if not signed_url and isinstance(url_res, dict):
                    signed_url = url_res.get("signedURL") or url_res.get("signedUrl") or url_res.get("signed_url")
                    if not signed_url and isinstance(url_res.get("data"), dict):
                        signed_url = url_res["data"].get("signedUrl") or url_res["data"].get("signedURL")
                    logger.info(f"Form Discovery: Extracted from dict: {bool(signed_url)}")
                
                # Priority 3: Direct string (unlikely but handle it)
                if not signed_url and isinstance(url_res, str) and url_res.startswith("http"):
                    signed_url = url_res
                    logger.info(f"Form Discovery: Extracted from string: {bool(signed_url)}")
                
                logger.info(f"Form Discovery: Final signed_url: {signed_url[:80] if signed_url else 'NONE'}...")


                # Build download URL
                if signed_url:
                    separator = "&" if "?" in signed_url else "?"
                    download_url = f"{signed_url}{separator}download={fname}"
                    found_forms.append({"filename": fname, "url": download_url})
                    logger.info(f"Form Discovery: Successfully added form link for {fname}")
                else:
                    logger.warning(f"Form Discovery: Could not extract signed URL from response for {fname}")
                    
            except Exception as e:
                logger.warning(f"Form Discovery: Failed to sign URL for {fname}: {e}")
                
    except Exception as e:
        logger.error(f"Form Discovery: Database query failed: {e}")
        
    logger.info(f"Form Discovery: Returning {len(found_forms)} forms")
    return found_forms
