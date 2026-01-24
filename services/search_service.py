"""
Search Intelligence Engine
LLM-powered query analysis for typo correction, synonym expansion, and language detection.
"""

import asyncio
import logging
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError
from services.agentic.rate_limiter import get_groq_limiter

logger = logging.getLogger(__name__)


class SearchParams(BaseModel):
    """Structured output from the Intelligence Engine."""
    corrected_natural_query: str = Field(
        description="Corrected, natural language query for vector embedding"
    )
    fts_search_string: str = Field(
        description="Pipe-delimited search terms for full-text search (e.g., 'bike | bicycle | fiets')"
    )


class SearchIntelligence:
    """
    LLM-powered query analyzer that:
    1. Fixes typos and expands abbreviations
    2. Generates multilingual synonyms (EN/NL/FR)
    3. Outputs structured search parameters
    """

    SYSTEM_PROMPT = """You are an HR Search Optimization Engine for a multilingual knowledge base (Dutch, English, French).

Your task: Transform user queries into optimal search parameters.

RULES:
1. FIX typos and expand abbreviations
2. Generate COMPREHENSIVE multilingual synonyms (EN, NL, FR)
3. Include COMPOUND words in ALL forms (with and without spaces):
   - "coffeemachine" AND "coffee machine" AND "koffiemachine" AND "koffie machine"
   - "wachtwoord" AND "password" AND "mot de passe" AND "paswoord" AND "code"
4. Include common HR synonyms and related terms
5. Use pipe (|) separator for FTS terms

EXAMPLES:

Input: "koffiemachine wachtwoord"
Output JSON:
{
  "corrected_natural_query": "What is the password for the coffee machine?",
  "fts_search_string": "coffeemachine | coffee machine | koffiemachine | koffie machine | machine café | wachtwoord | password | paswoord | code | mot de passe"
}


Input: "vakantiedagen"
Output JSON:
{
  "corrected_natural_query": "What is the vacation days policy?",
  "fts_search_string": "vakantiedagen | vakantie dagen | vacation | holiday | leave | congé | verlof | days | jours"
}

Input: "where is printer"
Output JSON:
{
  "corrected_natural_query": "Where is the printer located?",
  "fts_search_string": "printer | printers | imprimante | printing | afdrukken | locatie | location | emplacement | waar | where | où"
}

IMPORTANT: Output ONLY valid JSON, no markdown or explanation."""

    def __init__(self, groq_api_key: str, timeout: float = 2.0):
        """
        Initialize the Intelligence Engine.

        Args:
            groq_api_key: API key for Groq
            timeout: Max seconds to wait for LLM response (default 2s)
        """
        self.api_key = groq_api_key
        self.timeout = timeout
        self.model = "llama-3.1-8b-instant"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    async def analyze_query(self, raw_query: str) -> SearchParams:
        """
        Analyze a raw user query and return structured search parameters.

        Args:
            raw_query: The original user input (may contain typos, no spaces, etc.)

        Returns:
            SearchParams with corrected query and FTS string.
            Falls back to raw query on any failure.
        """
        # Validate input
        if not raw_query or not isinstance(raw_query, str):
            return self._fallback_params(str(raw_query) if raw_query else "")
        
        # Limit query length to prevent abuse
        if len(raw_query) > 1000:
            logger.warning(f"Query too long: {len(raw_query)} chars, truncating")
            raw_query = raw_query[:1000]
        
        # Normalize input
        normalized = raw_query.strip()
        if not normalized:
            return self._fallback_params(raw_query)

        try:
            # Run the LLM call in a thread to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._call_groq_sync,
                normalized
            )
            return result

        except asyncio.TimeoutError:
            logger.warning("Intelligence Engine: timeout, using fallback")
            return self._fallback_params(raw_query)
        except Exception as e:
            logger.warning(f"Intelligence Engine: error ({e}), using fallback")
            return self._fallback_params(raw_query)

    def _call_groq_sync(self, query: str) -> SearchParams:
        """Synchronous Groq API call."""
        # Rate Limit check
        get_groq_limiter().wait_if_needed()
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": query}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"}
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Parse and validate with Pydantic
                import json
                parsed = json.loads(content)
                params = SearchParams(**parsed)
                
                # Automatically expand compound words (add no-space versions)
                expanded_terms = self._expand_compound_words(params.fts_search_string)
                params = SearchParams(
                    corrected_natural_query=params.corrected_natural_query,
                    fts_search_string=expanded_terms
                )
                
                logger.info(f"Intelligence Engine: success - corrected='{params.corrected_natural_query[:50]}...'")
                return params


            elif response.status_code == 429:
                logger.warning("Intelligence Engine: rate limited")
                return self._fallback_params(query)
            else:
                logger.warning(f"Intelligence Engine: HTTP {response.status_code}")
                return self._fallback_params(query)

        except requests.Timeout:
            logger.warning("Intelligence Engine: request timeout")
            return self._fallback_params(query)
        except (ValidationError, KeyError, ValueError) as e:
            logger.warning(f"Intelligence Engine: parse error ({e})")
            return self._fallback_params(query)
        except Exception as e:
            logger.warning(f"Intelligence Engine: unexpected error ({e})")
            return self._fallback_params(query)

    def _expand_compound_words(self, fts_string: str) -> str:
        """
        Expand compound words to include both spaced and non-spaced versions.
        e.g., 'coffee machine' -> 'coffee machine | coffeemachine'
        """
        terms = [t.strip() for t in fts_string.split('|')]
        expanded = set(terms)
        
        for term in terms:
            # If term has spaces, add version without spaces
            if ' ' in term:
                expanded.add(term.replace(' ', ''))
            # If term is long enough and has no spaces, try to find natural splits
            # (this is less reliable, so we just ensure we have both forms)
        
        return ' | '.join(sorted(expanded))



    def _fallback_params(self, raw_query: str) -> SearchParams:
        """
        Generate fallback parameters when LLM fails.
        Uses raw query for both fields to ensure search still works.
        """
        # Split concatenated words heuristically for FTS
        # e.g., "bikepolicy" -> "bikepolicy | bike | policy"
        normalized = raw_query.strip().lower()
        fts_terms = [normalized]
        
        # Add the raw query as-is
        if raw_query != normalized:
            fts_terms.append(raw_query)

        return SearchParams(
            corrected_natural_query=raw_query,
            fts_search_string=" | ".join(fts_terms)
        )


# Convenience function for sync usage
def analyze_query_sync(raw_query: str, groq_api_key: str, timeout: float = 2.0) -> SearchParams:
    """
    Synchronous wrapper for query analysis.
    Use this in Streamlit or other sync contexts.
    """
    engine = SearchIntelligence(groq_api_key, timeout)
    return asyncio.run(engine.analyze_query(raw_query))
