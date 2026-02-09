import re
import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    CHITCHAT = "chitchat"
    DATABASE = "database"
    WEB = "web"
    LEGAL = "legal"

class QueryRouter:
    """
    Heuristic-based router to classify user intent without LLM calls.
    Optimized for speed and zero cost.
    """
    
    def __init__(self):
        # Patterns for "small talk" and greetings
        self.chitchat_patterns = [
            r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening|yo)\b",
            r"^how are you\??$",
            r"^who are you\??$",
            r"^what can you do\??$",
            r"^thanks?|thank you\b",
            r"^bye|goodbye|see you\b"
        ]
        
        # Patterns for web search intent (news, current events, etc.)
        self.web_patterns = [
            r"\b(news|latest|current|today|weather|stock|price|market|world|global)\b",
            r"\b(who is the|what is the current|search the web for)\b",
            r"\b(2024|2025|2026)\b"  # Future or current years often imply web search
        ]
        
        # Patterns for legal/labor law intent (Belgian context)
        self.legal_patterns = [
            r"\b(cao|cct|pc\s*\d+|paritair\s*comit[eé])\b",
            r"\b(arbeidsrecht|arbeidswet|arbeidsovereenkomst)\b",
            r"\b(ontslag|opzeg|opzegtermijn|dismissal)\b",
            r"\b(loon|salaris|indexatie|indexering|wage|salary)\b",
            r"\b(vakantie|verlof|leave|holiday|recuperatie)\b",
            r"\b(cnt|nar|nationale\s*arbeidsraad)\b",
            r"\b(collective\s*agreement|labor\s*law|employment\s*law)\b",
            r"\b(wet\s+betreffende|koninklijk\s+besluit|kb)\b",
            # Article-level patterns for Belgian law citations
            r"\b(art\.?\s*\d+|artikel\s*\d+)\b",
            r"\b(arbeidsovereenkomstenwet|welzijnswet|loonbeschermingswet)\b",
            r"\b(feestdagenwet|vakantiewet|cao.?wet)\b",
            # Generic Dutch/French legal terms - CRITICAL for catching "wat zegt de wet"
            r"\b(wet|wetgeving|wettelijk|wetboek)\b",
            r"\b(recht|rechten|rechtspositie)\b",
            r"\b(loi|légal|législation|droit)\b",
            r"\b(wettelijk\s+kader|legal\s+framework)\b",
            r"\b(zegt\s+de\s+wet|selon\s+la\s+loi|according\s+to\s+law)\b",
            # English legal terms
            r"\b(law|laws|legal|legally|legislation)\b",
            r"\b(statutory|statute|regulation|regulations)\b"
        ]

    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classifies the query into an intent.
        Returns (Intent, Confidence)
        """
        clean_query = query.strip().lower()
        
        # 1. Check Chitchat
        for pattern in self.chitchat_patterns:
            if re.search(pattern, clean_query):
                return QueryIntent.CHITCHAT, 0.9
        
        # 2. Check Web
        for pattern in self.web_patterns:
            if re.search(pattern, clean_query):
                return QueryIntent.WEB, 0.8
        
        # 3. Check Legal (Belgian labor law terms)
        for pattern in self.legal_patterns:
            if re.search(pattern, clean_query):
                return QueryIntent.LEGAL, 0.85
        
        # 4. Default to Database Search
        # If the query is long or contains complex nouns, it's likely a DB search
        if len(clean_query.split()) > 3:
            return QueryIntent.DATABASE, 0.7
            
        return QueryIntent.DATABASE, 0.5

def route_query(query: str) -> QueryIntent:
    router = QueryRouter()
    intent, _ = router.classify(query)
    return intent
