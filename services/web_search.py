"""
Web Search Service using DuckDuckGo.
Zero-cost alternative with caching to avoid rate limits.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid issues if not installed
_ddgs_available = None


def _check_ddgs():
    """Check if duckduckgo-search is available."""
    global _ddgs_available
    if _ddgs_available is None:
        try:
            from duckduckgo_search import DDGS
            _ddgs_available = True
        except ImportError:
            logger.warning("duckduckgo-search not installed, web search disabled")
            _ddgs_available = False
    return _ddgs_available


def search_web(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default 3)
        
    Returns:
        List of dicts with 'title', 'body', 'url' keys
    """
    if not _check_ddgs():
        return []
    
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            formatted = [
                {
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("href", r.get("link", ""))
                }
                for r in results
            ]
            logger.info(f"Web search: returned {len(formatted)} results for '{query[:30]}...'")
            return formatted
            
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return []


def format_web_results_as_context(results: List[Dict]) -> str:
    """
    Format web search results as context string for LLM.
    
    Args:
        results: List of search results from search_web()
        
    Returns:
        Formatted string for LLM context
    """
    if not results:
        return ""
    
    context_parts = ["--- WEB SEARCH RESULTS ---"]
    
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        body = r.get("body", "")[:300]  # Truncate for token efficiency
        url = r.get("url", "")
        
        context_parts.append(f"\n[Web{i}] {title}")
        context_parts.append(f"{body}")
        if url:
            context_parts.append(f"Source: {url}")
    
    return "\n".join(context_parts)


# Streamlit cached version
try:
    import streamlit as st
    
    @st.cache_data(ttl=300, show_spinner=False)
    def cached_web_search(query: str, max_results: int = 3) -> List[Dict]:
        """Cached web search (5-minute TTL)."""
        return search_web(query, max_results)
        
except ImportError:
    def cached_web_search(query: str, max_results: int = 3) -> List[Dict]:
        """Fallback without caching."""
        return search_web(query, max_results)
