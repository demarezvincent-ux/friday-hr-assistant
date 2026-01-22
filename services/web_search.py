import streamlit as st
from duckduckgo_search import DDGS
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Perform a live web search using DuckDuckGo.
    Zero cost and no API key required.
    """
    # Validate inputs
    if not query or not isinstance(query, str):
        logger.warning("Web Search: Invalid query provided")
        return []
    
    # Limit query length and max results
    if len(query) > 500:
        query = query[:500]
    
    max_results = min(max(max_results, 1), 10)  # Ensure 1-10 results
    
    results = []
    try:
        with DDGS() as ddgs:
            # text search
            response = ddgs.text(query, max_results=max_results)
            for r in response:
                results.append({
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                    "url": r.get("href", "")
                })
        logger.info(f"Web Search: found {len(results)} results for '{query}'")
    except Exception as e:
        logger.error(f"Web Search error: {e}")
    
    return results

@st.cache_data(ttl=300) # 5 minute cache
def cached_web_search(query: str, max_results: int = 3) -> List[Dict]:
    """Cached version of web search to respect rate limits and speed up UI."""
    return search_web(query, max_results=max_results)

def format_web_results_as_context(results: List[Dict]) -> str:
    """Format web results into a context string for the LLM."""
    if not results:
        return ""
    
    context_parts = []
    for i, res in enumerate(results, 1):
        context_parts.append(f"-- WEB SOURCE {i}: {res['title']} ({res['url']}) --\n{res['content']}")
    
    return "\n\n".join(context_parts)
