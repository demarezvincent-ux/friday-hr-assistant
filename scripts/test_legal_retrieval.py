import sys
import os
import asyncio
from supabase import create_client

# Add parent directory to path
sys.path.insert(0, os.getcwd())

from services.unified_config import get_config
from scripts.legal_scraper import BelgianLegalScraper
from services.legal_hierarchy import search_by_tier, assemble_tiered_context, TIER_LAW, TIER_SECTOR, TIER_COMPANY

async def test_retrieval():
    print("=" * 60)
    print("Testing Legal Hierarchy Retrieval")
    print("=" * 60)
    
    # Init scraper to get embedding function
    scraper = BelgianLegalScraper(dry_run=True)
    
    # Init Supabase
    url = get_config("SUPABASE_URL")
    key = get_config("SUPABASE_KEY")
    supabase = create_client(url, key)
    
    query = "ontslag tijdens ziekte"
    print(f"\nQuery: '{query}'")
    embedding = scraper.get_embedding(query)
    
    if not embedding:
        print("❌ FAILURE: Could not generate embedding")
        return
    
    # Test tiered search
    buckets = await search_by_tier(query, embedding, supabase, match_count=10)
    
    print(f"\n--- TIER 1: Federal Law ({len(buckets[TIER_LAW])} results) ---")
    for doc in buckets[TIER_LAW]:
        source = doc.get("metadata", {}).get("source", "?")
        article = doc.get("metadata", {}).get("article_number", "")
        print(f"  ⚖️  {source} {article}")
    
    print(f"\n--- TIER 2: Sector ({len(buckets[TIER_SECTOR])} results) ---")
    for doc in buckets[TIER_SECTOR]:
        source = doc.get("metadata", {}).get("source", "?")
        print(f"  📋 {source}")
    
    print(f"\n--- TIER 3: Company ({len(buckets[TIER_COMPANY])} results) ---")
    for doc in buckets[TIER_COMPANY]:
        source = doc.get("metadata", {}).get("source", "?")
        print(f"  📄 {source}")
    
    # Test context assembly
    context, law_src, sector_src, company_src = assemble_tiered_context(buckets)
    
    print(f"\n--- ASSEMBLED CONTEXT (first 500 chars) ---")
    print(context[:500])
    
    # Verify tier ordering in context
    tier1_pos = context.find("TIER 1")
    tier2_pos = context.find("TIER 2")
    tier3_pos = context.find("TIER 3")
    
    print(f"\n--- VERIFICATION ---")
    if tier1_pos >= 0 and (tier2_pos < 0 or tier1_pos < tier2_pos):
        print("✅ Tier 1 (Law) appears before Tier 2 (Sector)")
    elif tier1_pos < 0 and tier2_pos < 0 and tier3_pos < 0:
        print("⚠️  No tier headers found (maybe no legal_knowledge data?)")
    else:
        print("❌ Tier ordering is wrong!")
    
    total = len(law_src) + len(sector_src) + len(company_src)
    print(f"✅ Sources: {len(law_src)} law, {len(sector_src)} sector, {len(company_src)} company ({total} total)")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
