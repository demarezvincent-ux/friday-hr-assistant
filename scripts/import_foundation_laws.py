#!/usr/bin/env python3
"""
Import Belgian Labor Law Foundation Documents

Foundation laws are core legal texts that Friday references alongside company 
policies to provide legally-grounded HR answers.

Usage:
    # Test extraction without saving
    python scripts/import_foundation_laws.py --dry-run
    
    # Import to database
    python scripts/import_foundation_laws.py
    
    # Check results
    python -c "from supabase import create_client; import os; \\
        s = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY']); \\
        r = s.table('legal_knowledge').select('*').eq('metadata->>category', 'legal_foundation').execute(); \\
        print(f'Found {len(r.data)} foundation laws')"

Foundation Laws Imported:
- Arbeidswet 1971: Working time, rest periods, overtime
- Arbeidsovereenkomstenwet 1978: Employment contracts, dismissal, protections

These laws are tagged with:
- category: "legal_foundation" 
- priority: 10 (ranked higher in search results)
- is_critical: True (used for important legal references)
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.legal_scraper import BelgianLegalScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FOUNDATION LAWS REGISTRY
# Add new laws here to import them
# ═══════════════════════════════════════════════════════════════════════════════

FOUNDATION_LAWS: List[Dict] = [
    {
        "url": "https://www.ejustice.just.fgov.be/eli/wet/1971/03/16/1971031602/justel",
        "title": "Arbeidswet 1971",
        "numac": "1971031602",
        "topic": "Working Hours",
        "description": "Working time, rest periods, overtime, night work regulations"
    },
    {
        "url": "https://www.ejustice.just.fgov.be/eli/wet/1978/07/03/1978070303/justel",
        "title": "Arbeidsovereenkomstenwet 1978",
        "numac": "1978070303",
        "topic": "Employment Contracts",
        "description": "Employment contracts, dismissal, notice periods, worker protections"
    },
]


def extract_law_text(url: str, headers: Dict) -> Optional[str]:
    """
    Fetch and extract law text from eJustice HTML page.
    
    Args:
        url: eJustice URL to fetch
        headers: HTTP headers for request
        
    Returns:
        Extracted text or None if extraction failed
    """
    try:
        resp = requests.get(url, headers=headers, timeout=40)
        
        if resp.status_code != 200:
            logger.error(f"   ✗ HTTP {resp.status_code}")
            return None
        
        print(f"   ✓ Fetched HTML ({resp.status_code} OK)")
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Try multiple content selectors (eJustice structure)
        content_div = (
            soup.find('div', class_='loi-body') or
            soup.find('div', id='mainContent') or
            soup.find('article', class_='loi') or
            soup.find('main') or
            soup.find('div', class_='content') or
            soup.find('body')  # Ultimate fallback
        )
        
        if not content_div:
            logger.error("   ✗ No content container found in HTML")
            return None
        
        # Extract clean text
        text = content_div.get_text(separator='\n', strip=True)
        
        if not text:
            logger.error("   ✗ Extracted text is empty")
            return None
        
        # Validate length (laws are long documents)
        if len(text) < 1000:
            logger.warning(f"   ⚠️ Text too short ({len(text)} chars) - might be error page")
            # Don't return None - let caller decide
        
        print(f"   ✓ Extracted: {len(text):,} characters")
        return text
        
    except requests.exceptions.Timeout:
        logger.error("   ✗ Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"   ✗ Request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"   ✗ Extraction failed: {e}")
        return None


def import_foundation_laws(dry_run: bool = False) -> Dict[str, int]:
    """
    Import all foundation laws from eJustice to Supabase.
    
    Args:
        dry_run: If True, don't save to database
        
    Returns:
        Stats dict with 'imported', 'failed', 'skipped' counts
    """
    stats = {"imported": 0, "failed": 0, "skipped": 0}
    
    # Initialize scraper (reuse existing infrastructure)
    try:
        scraper = BelgianLegalScraper(dry_run=dry_run)
    except Exception as e:
        logger.error(f"Failed to initialize scraper: {e}")
        print(f"\n❌ Could not initialize scraper: {e}")
        print("   Check that SUPABASE_URL and SUPABASE_KEY are set")
        return stats
    
    print("=" * 70)
    print("IMPORTING BELGIAN LABOR LAW FOUNDATION DOCUMENTS")
    if dry_run:
        print("🔸 DRY RUN MODE - No database changes will be made")
    print("=" * 70)
    print()
    
    for law in FOUNDATION_LAWS:
        print(f"📄 Processing: {law['title']}")
        print(f"   URL: {law['url']}")
        print(f"   Topic: {law['topic']}")
        
        # Rate limiting
        scraper.polite_sleep()
        
        # Extract text from HTML
        text = extract_law_text(law['url'], scraper.headers)
        
        if not text:
            print(f"   ❌ Failed to extract content")
            stats["failed"] += 1
            print()
            continue
        
        # Check for duplicates
        content_hash = scraper.get_document_hash(text)
        if scraper.is_in_db(content_hash):
            print(f"   ⚠️ Already exists in database (skipping)")
            stats["skipped"] += 1
            print()
            continue
        
        # Build metadata for foundation law
        metadata = {
            "category": "legal_foundation",
            "source_domain": "ejustice.just.fgov.be",
            "is_active": True,
            "title": law["title"],
            "numac": law["numac"],
            "topic": law["topic"],
            "description": law["description"],
            "is_critical": True,
            "document_type": "foundation_law",
            "priority": 10
        }
        
        # Save to database
        try:
            if scraper.save_to_supabase(text, law['url'], "FOUNDATION_LAW", metadata):
                print(f"   ✓ Saved to database")
                print(f"   ✅ Imported successfully!")
                stats["imported"] += 1
            else:
                print(f"   ❌ Failed to save to database")
                stats["failed"] += 1
        except Exception as e:
            logger.error(f"Database error: {e}")
            print(f"   ❌ Database error: {e}")
            stats["failed"] += 1
        
        print()
    
    # Summary
    print("=" * 70)
    print("IMPORT COMPLETE")
    print(f"✅ Imported: {stats['imported']}")
    print(f"⚠️ Skipped: {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📊 Total: {len(FOUNDATION_LAWS)}")
    print("=" * 70)
    
    return stats


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Import Belgian Labor Law Foundation Documents to Friday"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test extraction without saving to database"
    )
    
    args = parser.parse_args()
    
    stats = import_foundation_laws(dry_run=args.dry_run)
    
    # Exit with error if all failed
    if stats["failed"] > 0 and stats["imported"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
