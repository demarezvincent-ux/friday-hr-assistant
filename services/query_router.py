"""
Query Router - Classifies user intent.
Intent A: DATABASE - Internal HR document search
Intent B: WEB - External comparisons, news, trends
Intent C: CHITCHAT - Greetings and casual conversation
"""

import re
import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    DATABASE = "database"
    WEB = "web"
    CHITCHAT = "chitchat"


class QueryRouter:
    """
    Lightweight query classifier using heuristics.
    No LLM calls to save rate limits.
    """
    
    CHITCHAT_PATTERNS = [
        r"^(hi|hello|hey|bonjour|hallo|dag|salut)\b",
        r"^how are you",
        r"^thank(s| you)",
        r"^(good morning|good afternoon|good evening|goedemorgen|bonsoir)",
        r"^(bye|goodbye|au revoir|tot ziens)",
        r"^what('s| is) your name",
        r"^who are you",
    ]
    
    WEB_INDICATORS = [
        "compare", "versus", "vs", "vs.", 
        "latest", "news", "trend", "trending",
        "2024", "2025", "2026", "2027",
        "market", "industry", "benchmark", "competitor",
        "best practice", "external", "outside",
        "salary survey", "industry standard"
    ]
    
    # HR-specific terms that should always go to database
    DATABASE_INDICATORS = [
        "policy", "procedure", "our", "my", "employee", 
        "vacation", "leave", "sick", "holiday", "benefits",
        "contract", "handbook", "guide", "internal", "company"
    ]
    
    def __init__(self):
        """Initialize the router."""
        pass
    
    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify query intent using heuristics.
        
        Args:
            query: User's raw query
            
        Returns:
            Tuple of (QueryIntent, confidence_score)
        """
        lower_query = query.lower().strip()
        
        # Rule 1: Chit chat patterns (high confidence)
        for pattern in self.CHITCHAT_PATTERNS:
            if re.search(pattern, lower_query):
                logger.info(f"Router: CHITCHAT detected (pattern match)")
                return QueryIntent.CHITCHAT, 0.95
        
        # Count web vs database indicators
        web_score = sum(1 for ind in self.WEB_INDICATORS if ind in lower_query)
        db_score = sum(1 for ind in self.DATABASE_INDICATORS if ind in lower_query)
        
        # Rule 2: Strong web indicators
        if web_score >= 2 and web_score > db_score:
            logger.info(f"Router: WEB detected (score: {web_score})")
            return QueryIntent.WEB, min(0.7 + (web_score * 0.1), 0.95)
        
        # Rule 3: Any database indicators or default
        if db_score > 0 or web_score == 0:
            logger.info(f"Router: DATABASE detected (score: {db_score})")
            return QueryIntent.DATABASE, min(0.6 + (db_score * 0.1), 0.95)
        
        # Rule 4: Weak web signal, still search DB first
        logger.info(f"Router: DATABASE (default, weak web signal)")
        return QueryIntent.DATABASE, 0.5


# Convenience function
def route_query(query: str) -> Tuple[QueryIntent, float]:
    """Route a query to the appropriate search strategy."""
    router = QueryRouter()
    return router.classify(query)
