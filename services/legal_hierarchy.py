"""
Legal Hierarchy Service
Tiered search (law > sector > company) with conflict detection.
"""

import logging
from typing import List, Dict, Tuple, Optional

from supabase import Client

from services.rag_controller import sanitize_fts_query

logger = logging.getLogger(__name__)

# Tier definitions
TIER_LAW = 1
TIER_SECTOR = 2
TIER_COMPANY = 3

TIER_LABELS = {
    TIER_LAW: ("⚖️ TIER 1: FEDERAL LAW", "highest authority, always applies"),
    TIER_SECTOR: ("📋 TIER 2: SECTOR AGREEMENT", "applies to this sector"),
    TIER_COMPANY: ("📄 TIER 3: COMPANY POLICY", "internal rules, lowest authority"),
}

# How many results to keep per tier
TIER_BUDGETS = {
    TIER_LAW: 5,
    TIER_SECTOR: 3,
    TIER_COMPANY: 2,
}


async def search_by_tier(
    query: str,
    query_embedding: List[float],
    supabase: Client,
    fts_string: str = "",
    source_filter: Optional[str] = None,
    match_count: int = 10,
) -> Dict[int, List[dict]]:
    """
    Run the tiered RPC and bucket results by legal_tier.
    Returns {1: [docs], 2: [docs], 3: [docs]}.
    Falls back to the legacy flat RPC if the tiered one doesn't exist.
    """
    safe_fts = sanitize_fts_query(fts_string or query)

    rpc_params = {
        "query_embedding": query_embedding,
        "text_search_query": safe_fts,
        "match_threshold": 0.15,
        "match_count": match_count,
        "source_filter": source_filter,
        "tier_filter": None,
    }

    try:
        result = supabase.rpc("match_legal_documents_tiered", rpc_params).execute()
    except Exception as e:
        logger.warning(f"Tiered RPC failed ({e}), falling back to legacy")
        return await _fallback_flat_search(query, query_embedding, supabase, safe_fts, source_filter, match_count)

    if not result.data:
        logger.info("Tiered search: no matches")
        return {TIER_LAW: [], TIER_SECTOR: [], TIER_COMPANY: []}

    buckets: Dict[int, List[dict]] = {TIER_LAW: [], TIER_SECTOR: [], TIER_COMPANY: []}
    for doc in result.data:
        tier = doc.get("legal_tier", TIER_COMPANY)
        if tier in buckets:
            if len(buckets[tier]) < TIER_BUDGETS.get(tier, 5):
                buckets[tier].append(doc)

    logger.info(
        f"Tiered search: law={len(buckets[TIER_LAW])}, "
        f"sector={len(buckets[TIER_SECTOR])}, "
        f"company={len(buckets[TIER_COMPANY])}"
    )
    return buckets


async def _fallback_flat_search(
    query, query_embedding, supabase, safe_fts, source_filter, match_count
) -> Dict[int, List[dict]]:
    """Fall back to match_legal_documents and infer tier from metadata."""
    rpc_params = {
        "query_embedding": query_embedding,
        "text_search_query": safe_fts,
        "match_threshold": 0.15,
        "match_count": match_count,
        "source_filter": source_filter,
    }

    try:
        result = supabase.rpc("match_legal_documents", rpc_params).execute()
    except Exception:
        return {TIER_LAW: [], TIER_SECTOR: [], TIER_COMPANY: []}

    buckets: Dict[int, List[dict]] = {TIER_LAW: [], TIER_SECTOR: [], TIER_COMPANY: []}
    for doc in (result.data or []):
        tier = _infer_tier(doc)
        if len(buckets[tier]) < TIER_BUDGETS.get(tier, 5):
            buckets[tier].append(doc)

    return buckets


def _infer_tier(doc: dict) -> int:
    """Infer tier from metadata when the DB doesn't have legal_tier yet."""
    source = doc.get("metadata", {}).get("source", "")
    category = doc.get("metadata", {}).get("category", "")

    if source == "BELGIAN_LAW" or category == "legal_foundation":
        return TIER_LAW
    if any(source.startswith(p) for p in ("CAO", "PC", "CCT", "NAR", "CNT")):
        return TIER_SECTOR
    return TIER_COMPANY


def _format_doc(doc: dict, tier: int) -> str:
    """Format a single doc with tier-appropriate header."""
    metadata = doc.get("metadata", {})
    source = metadata.get("source", "Legal")

    if source == "BELGIAN_LAW":
        law_name = metadata.get("law_name", "")
        law_date = metadata.get("law_date", "")
        article = metadata.get("article_number", "")
        title = metadata.get("title", "")
        chapter = metadata.get("chapter", "")

        header = f"[CITE THIS: {article} van de {law_name} ({law_date})]"
        if title or chapter:
            parts = [p for p in [title, chapter] if p]
            header += f"\n{' > '.join(parts)}"
        content = doc.get("content", "")
        return f"{header}\n{content}"

    topic = metadata.get("topic", "")
    effective_date = metadata.get("effective_date", "")
    header = f"SOURCE: {source}"
    if topic:
        header += f" ({topic})"
    if effective_date:
        header += f" [Effective: {effective_date}]"
    content = doc.get("summary") or doc.get("content", "")[:1500]
    return f"{header}\n{content}"


def assemble_tiered_context(
    buckets: Dict[int, List[dict]],
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Build the LLM context string with explicit hierarchy headers.

    Returns:
        (context_str, law_sources, sector_sources, company_sources)
    """
    sections = []
    law_sources: List[str] = []
    sector_sources: List[str] = []
    company_sources: List[str] = []

    source_lists = {
        TIER_LAW: law_sources,
        TIER_SECTOR: sector_sources,
        TIER_COMPANY: company_sources,
    }

    for tier in (TIER_LAW, TIER_SECTOR, TIER_COMPANY):
        docs = buckets.get(tier, [])
        if not docs:
            continue

        label, desc = TIER_LABELS[tier]
        section_parts = [f"=== {label} ({desc}) ==="]

        for doc in docs:
            section_parts.append(_format_doc(doc, tier))

            # Build source label
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Legal")
            if source == "BELGIAN_LAW":
                slabel = f"{metadata.get('article_number', '')} {metadata.get('law_name', '')}"
            else:
                topic = metadata.get("topic", "")
                slabel = f"{source} - {topic}" if topic else source

            if slabel not in source_lists[tier]:
                source_lists[tier].append(slabel)

        sections.append("\n".join(section_parts))

    # Detect conflicts: if tier 2/3 mentions same topic as tier 1, add marker
    if law_sources and (sector_sources or company_sources):
        sections.insert(0,
            "⚠️ NOTE: Multiple authority levels found. "
            "Federal law (Tier 1) takes precedence over sector and company rules "
            "unless the sector/company rule is MORE FAVORABLE to the employee."
        )

    context_str = "\n\n".join(sections)
    logger.info(
        f"Tiered context: {len(law_sources)} law, "
        f"{len(sector_sources)} sector, "
        f"{len(company_sources)} company sources"
    )
    return context_str, law_sources, sector_sources, company_sources
