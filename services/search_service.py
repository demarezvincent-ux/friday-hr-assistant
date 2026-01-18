"""
Search Intelligence Engine
LLM-powered query analysis for typo correction, synonym expansion, and language detection.
"""

import asyncio
import logging
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError

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
3. Include both COMPOUND and SPLIT versions of terms:
   - "koffiemachine" AND "koffie machine" AND "coffee machine"
   - "wachtwoord" AND "password" AND "mot de passe" AND "paswoord" AND "code"
4. Include common HR synonyms and related terms
5. Use pipe (|) separator for FTS terms

EXAMPLES:

Input: "koffiemachine wachtwoord"
Output JSON:
{
  "corrected_natural_query": "What is the password for the coffee machine?",
  "fts_search_string": "koffiemachine | koffie machine | coffee machine | machine café | wachtwoord | password | paswoord | code | mot de passe"
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
