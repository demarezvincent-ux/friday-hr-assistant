"""
Semantic Query Cache
Caches similar queries to reduce API calls (~40% reduction).
Uses vector similarity to find cached responses.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Cache responses for semantically similar queries.
    Uses Supabase pgvector for similarity search.
    
    Reduces API calls by returning cached responses for similar questions.
    TTL: 24 hours (configurable).
    Similarity threshold: 95% (very high to avoid incorrect cache hits).
    """
    
    def __init__(self, supabase, ttl_hours: int = 24, similarity_threshold: float = 0.95):
        """
        Initialize the semantic cache.
        
        Args:
            supabase: Supabase client instance
            ttl_hours: Time-to-live in hours (default 24h)
            similarity_threshold: Minimum similarity for cache hit (default 0.95)
        """
        self.supabase = supabase
        self.ttl = timedelta(hours=ttl_hours)
        self.threshold = similarity_threshold
    
    def get_cached_response(
        self, 
        query_embedding: List[float], 
        company_id: str
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Find cached response for a similar query.
        
        Args:
            query_embedding: Vector embedding of the current query
            company_id: Company ID to filter by
            
        Returns:
            Tuple of (response, sources) if cache hit, None otherwise
        """
        try:
            result = self.supabase.rpc("match_cached_queries", {
                "p_query_embedding": query_embedding,
                "p_company_id": company_id,
                "p_match_threshold": self.threshold,
                "p_match_count": 1
            }).execute()
            
            if result.data and len(result.data) > 0:
                cache_entry = result.data[0]
                logger.info(f"Cache HIT: similarity={cache_entry.get('similarity', 0):.3f}")
                return (
                    cache_entry.get("response", ""),
                    cache_entry.get("sources", [])
                )
            
            logger.debug("Cache MISS")
            return None
            
        except Exception as e:
            # Cache failures should not break the app
            logger.warning(f"Cache lookup failed (non-critical): {e}")
            return None
    
    def cache_response(
        self, 
        query: str, 
        query_embedding: List[float], 
        response: str,
        sources: List[str],
        company_id: str
    ) -> bool:
        """
        Store a response in the cache.
        
        Args:
            query: Original query text
            query_embedding: Vector embedding of the query
            response: LLM response to cache
            sources: Source documents used
            company_id: Company ID
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            self.supabase.table("query_cache").insert({
                "query_text": query,
                "query_embedding": query_embedding,
                "response": response,
                "sources": sources,
                "company_id": company_id
            }).execute()
            
            logger.info(f"Cached response for query: {query[:50]}...")
            return True
            
        except Exception as e:
            # Cache failures should not break the app
            logger.warning(f"Cache store failed (non-critical): {e}")
            return False
    
    def clear_expired(self, company_id: Optional[str] = None) -> int:
        """
        Clear expired cache entries (older than TTL).
        
        Args:
            company_id: Optional company ID to filter by
            
        Returns:
            Number of entries deleted
        """
        try:
            cutoff = (datetime.now() - self.ttl).isoformat()
            query = self.supabase.table("query_cache").delete().lt(
                "created_at", cutoff
            )
            
            if company_id:
                query = query.eq("company_id", company_id)
            
            result = query.execute()
            count = len(result.data) if result.data else 0
            
            logger.info(f"Cleared {count} expired cache entries")
            return count
            
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
            return 0


def is_cache_available(supabase) -> bool:
    """
    Check if the cache table exists in Supabase.
    Call this on startup to determine if caching is enabled.
    """
    try:
        result = supabase.table("query_cache").select("id").limit(1).execute()
        return True
    except Exception:
        return False
