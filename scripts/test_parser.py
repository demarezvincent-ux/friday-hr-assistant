
import sys
import os
import re
import requests
import logging
from bs4 import BeautifulSoup

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.legal_scraper import BelgianLegalScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parser():
    law_info = {"name": "Arbeidswet", "short": "AW", "date": "1971-03-16", "numac": "1971031602", "type": "wet"}
    eli_url = "https://www.ejustice.just.fgov.be/eli/wet/1971/03/16/1971031602/justel"
    
    print(f"Fetching {eli_url}...")
    resp = requests.get(eli_url, timeout=30)
    if resp.status_code != 200:
        print(f"Failed to fetch: {resp.status_code}")
        return

    scraper = BelgianLegalScraper(dry_run=True)
    print("Parsing articles...")
    try:
        articles = scraper._parse_law_articles(resp.text, law_info, eli_url)
        print(f"Successfully found {len(articles)} articles.")
        if articles:
            print(f"First article: {articles[0]['article_number']}")
            print(f"Text snippet: {articles[0]['text'][:100]}...")
    except Exception as e:
        print(f"Parser crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parser()
