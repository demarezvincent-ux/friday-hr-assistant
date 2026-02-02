#!/usr/bin/env python3
"""
FRIDAY Legal Brain - Belgian Law Scraper
Weekly pipeline to scrape, analyze, and store Belgian labor law.

Targets:
- PC 200 CAOs (werk.belgie.be)
- CNT/NAR national agreements (cnt-nar.be)
- Federal legislation (ejustice.fgov.be)

Usage:
    python legal_scraper.py --target all
    python legal_scraper.py --target pc200 --dry-run
    python legal_scraper.py --target pc200 --limit 5
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from supabase import create_client

# Optional: OCR support
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# PDF extraction
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")


class BelgianLegalScraper:
    """
    Main scraper class for Belgian labor law.
    Handles PDF download, text extraction, AI analysis, and Supabase storage.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'nl-BE,nl;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.google.be/'
        }
        self.request_delay = 1.5  # Polite scraping: 1.5s between requests
        
        # Initialize clients (skip in dry-run for standalone testing)
        if not self.dry_run:
            if not all([SUPABASE_URL, SUPABASE_KEY]):
                raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")
            self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            self.supabase = None
        
        self.groq_key = GROQ_API_KEY
        self.hf_key = HF_API_KEY
        
        # Stats
        self.stats = {"scraped": 0, "new": 0, "skipped": 0, "errors": 0}

    # --- Utility Methods ---
    
    def get_document_hash(self, content) -> str:
        """Create MD5 fingerprint for deduplication. Accepts bytes or string."""
        if isinstance(content, bytes):
            return hashlib.md5(content).hexdigest()
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def is_in_db(self, content_hash: str) -> bool:
        """Check if document already exists in database."""
        if self.dry_run or not self.supabase:
            return False
        try:
            result = self.supabase.table("legal_knowledge").select("id").eq(
                "content_hash", content_hash
            ).limit(1).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.warning(f"DB check failed: {e}")
            return False

    def polite_sleep(self):
        """Rate limiting to avoid overwhelming government servers."""
        time.sleep(self.request_delay)

    # --- PDF Extraction ---
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, use_ocr_fallback: bool = True) -> Tuple[str, str]:
        """
        Extract text from PDF bytes.
        Returns: (text, extraction_method)
        """
        if not PDF_AVAILABLE:
            logger.error("pdfplumber not installed")
            return "", "error"
        
        text_parts = []
        
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        if text_parts:
            return "\n\n".join(text_parts), "text_layer"
        
        # OCR fallback for scanned PDFs
        if use_ocr_fallback and OCR_AVAILABLE:
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(pdf_bytes, dpi=200)
                ocr_text = []
                for img in images:
                    ocr_text.append(pytesseract.image_to_string(img, lang='nld+fra+eng'))
                return "\n\n".join(ocr_text), "ocr"
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
        
        return "", "failed"

    # --- AI Analysis ---
    
    def ai_analyze_law(self, text: str) -> Optional[Dict]:
        """
        Use Groq (Llama 3) to extract structured metadata from legal text.
        """
        if not self.groq_key:
            logger.warning("No GROQ_API_KEY, skipping AI analysis")
            return None
        
        prompt = f"""You are a Belgian labor law expert. Analyze this CAO/legal text and extract metadata.

Output ONLY valid JSON (no markdown, no explanation):
{{
    "topic": "One of: Wages/Leave/Dismissal/Working Hours/Benefits/Safety/Training/Other",
    "effective_date": "YYYY-MM-DD or null if unknown",
    "summary_dutch": "1-2 sentence summary in Dutch",
    "summary_english": "1-2 sentence summary in English",
    "is_critical": true if this affects wages/dismissal/major benefits, false otherwise,
    "paritair_comite": "PC number like '200' or null if not specified"
}}

Text (first 6000 chars):
{text[:6000]}
"""
        
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                # Clean potential markdown wrapping
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                return json.loads(content)
            else:
                logger.warning(f"Groq API error: {resp.status_code}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
        
        return None

    # --- Embeddings ---
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using HuggingFace Inference API."""
        if not self.hf_key:
            logger.warning("No HF_API_KEY, skipping embedding generation")
            return None
        
        model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        clean_text = text.replace("\n", " ").strip()[:5000]  # Limit input size
        
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=self.hf_key)
            embeddings = client.feature_extraction(clean_text, model=model_id)
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return embeddings[0] if isinstance(embeddings[0], list) else embeddings
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    # --- Storage ---
    
    def save_to_supabase(
        self, 
        text: str, 
        source_url: str, 
        source_type: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store document in legal_knowledge table."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would store: {source_url[:60]}...")
            return True
        
        if not self.supabase:
            return False
        
        content_hash = self.get_document_hash(text)
        
        # Check duplicate
        if self.is_in_db(content_hash):
            logger.info(f"Skipping duplicate: {source_url[:50]}...")
            self.stats["skipped"] += 1
            return False
        
        # Build metadata
        doc_metadata = {
            "source": source_type,
            "source_url": source_url,
            **(metadata or {})
        }
        
        # AI analysis
        ai_meta = self.ai_analyze_law(text)
        if ai_meta:
            doc_metadata.update(ai_meta)
        
        # Generate embedding
        embedding = self.get_embedding(text[:3000])  # Use summary portion for embedding
        
        # Build record
        record = {
            "content": text,
            "summary": ai_meta.get("summary_english") if ai_meta else None,
            "metadata": doc_metadata,
            "content_hash": content_hash,
            "embedding": embedding
        }
        
        try:
            self.supabase.table("legal_knowledge").insert(record).execute()
            logger.info(f"Stored: {source_type} - {source_url[:50]}...")
            self.stats["new"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            self.stats["errors"] += 1
            return False

    # --- PC 200 Scraper ---
    
    def fetch_pc_200_updates(self, limit: int = 0) -> List[str]:
        """
        Scrapes PC 200 from multiple sources with fallbacks.
        Primary: sfonds200.be homepage (WORKS)
        Fallback: werk.belgie.be (structure changes frequently)
        """
        logger.info("=== Scraping PC 200 via Multiple Sources ===")
        
        # These URLs have been tested - homepage approach works best
        start_urls = [
            "https://www.sfonds200.be/nl/",  # WORKS - Found PDFs here
            "https://werk.belgie.be/nl/themas/paritaire-comites"  # Fallback - list page
        ]
        scraped_urls = []

        for base_url in start_urls:
            try:
                logger.info(f"Trying: {base_url}")
                response = requests.get(base_url, headers=self.headers, timeout=20)
                if response.status_code != 200:
                    logger.warning(f"PC 200: {base_url} returned {response.status_code}")
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')
                # Find all PDF links
                links = soup.find_all('a', href=lambda h: h and h.endswith('.pdf'))
                
                if not links:
                    logger.info(f"No PDFs found at {base_url}")
                    continue
                
                logger.info(f"Found {len(links)} PDF links at {base_url}")

                count = 0
                for link in links:
                    if limit > 0 and count >= limit:
                        break

                    href = link.get('href', '')
                    if not href:
                        continue
                    
                    # Build full URL
                    if href.startswith('http'):
                        full_url = href
                    elif href.startswith('/'):
                        domain = base_url.split('/')[0] + '//' + base_url.split('/')[2]
                        full_url = domain + href
                    else:
                        full_url = base_url.rstrip('/') + '/' + href

                    # Download and process
                    try:
                        self.polite_sleep()
                        pdf_resp = requests.get(full_url, headers=self.headers, timeout=30)
                        if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                            # Check duplication via content hash
                            content_hash = self.get_document_hash(pdf_resp.content)
                            if self.is_in_db(content_hash):
                                logger.info(f"Skipping duplicate: {full_url[:60]}...")
                                self.stats["skipped"] += 1
                                continue
                            
                            text, method = self.extract_text_from_pdf(pdf_resp.content)
                            if text and len(text) > 100:
                                if self.save_to_supabase(text, full_url, "PC 200"):
                                    scraped_urls.append(full_url)
                                    count += 1
                                    logger.info(f"Processed new PC 200 doc: {full_url[:60]}...")
                    except Exception as e:
                        logger.warning(f"Failed to process {full_url}: {e}")
                        self.stats["errors"] += 1
                
                # If we found documents, stop trying other sources
                if scraped_urls:
                    break

            except Exception as e:
                logger.warning(f"Error with {base_url}: {e}")
        
        return scraped_urls

    # --- CNT/NAR Scraper ---
    
    def fetch_cnt_nar_updates(self, limit: int = 0) -> List[str]:
        """
        ✅ FIXED: Bypasses CNT Firewall by using Federal Database Mirror.
        Stable Mirror URL: https://werk.belgie.be/nl/themas/paritaire-comites/nationale-arbeidsraad/cao-nar
        """
        logger.info("=== Scraping CNT/NAR via Federal Mirror ===")
        url = "https://werk.belgie.be/nl/themas/paritaire-comites/nationale-arbeidsraad/cao-nar"
        scraped_urls = []

        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code != 200:
                logger.warning(f"CNT Mirror returned {response.status_code}")
                return scraped_urls

            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.select('div.main-content a[href$=".pdf"]')
            # Fallback: if no links found with that selector, try all PDF links
            if not links:
                links = soup.find_all('a', href=lambda h: h and h.endswith('.pdf'))

            count = 0
            for link in links:
                if limit > 0 and count >= limit:
                    break

                href = link.get('href', '')
                if not href:
                    continue
                    
                full_url = f"https://werk.belgie.be{href}" if href.startswith('/') else href

                # Download and process
                try:
                    self.polite_sleep()
                    pdf_resp = requests.get(full_url, headers=self.headers, timeout=30)
                    if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                        # Check duplication via content hash
                        content_hash = self.get_document_hash(pdf_resp.content)
                        if self.is_in_db(content_hash):
                            logger.info(f"Skipping duplicate: {full_url[:60]}...")
                            self.stats["skipped"] += 1
                            continue
                        
                        text, method = self.extract_text_from_pdf(pdf_resp.content)
                        if text and len(text) > 100:
                            if self.save_to_supabase(text, full_url, "CNT/NAR"):
                                scraped_urls.append(full_url)
                                count += 1
                                logger.info(f"Processed new CNT doc: {full_url[:60]}...")
                except Exception as e:
                    logger.warning(f"Failed to process {full_url}: {e}")
                    self.stats["errors"] += 1

        except Exception as e:
            logger.error(f"CNT Scraping failed: {e}")
            self.stats["errors"] += 1
        
        return scraped_urls

    # --- Federal Law Scraper ---
    
    def fetch_federal_law_updates(self, limit: int = 0) -> List[str]:
        """
        Scrape federal labor legislation from Justel/e-Justice.
        Target: http://www.ejustice.just.fgov.be/cgi_loi/change_lg.pl
        
        Focus on labor-related laws (arbeidsrecht, wet betreffende de arbeidsovereenkomsten, etc.)
        """
        logger.info("=== Scraping Federal Labor Laws ===")
        
        # Key labor law search terms
        search_terms = [
            "arbeidsovereenkomsten",
            "arbeidswet",
            "loonbescherming",
            "vakantiegeld",
            "opzegtermijn"
        ]
        
        base_url = "http://www.ejustice.just.fgov.be"
        scraped_urls = []
        
        for term in search_terms[:2]:  # Limit initial scope
            try:
                search_url = f"{base_url}/cgi_loi/loi_a.pl"
                params = {
                    "language": "nl",
                    "cheression": term,
                    "table_name": "wet",
                    "cn": ""
                }
                
                self.polite_sleep()
                resp = requests.get(search_url, params=params, headers=self.headers, timeout=15)
                
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    
                    # Find law links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if 'loi_a.pl' in href or 'wet' in href.lower():
                            full_url = href if href.startswith('http') else f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                            
                            if limit > 0 and len(scraped_urls) >= limit:
                                break
                            
                            # Fetch the actual law text
                            self.polite_sleep()
                            law_resp = requests.get(full_url, headers=self.headers, timeout=15)
                            if law_resp.status_code == 200:
                                law_soup = BeautifulSoup(law_resp.text, 'html.parser')
                                # Extract text from the main content area
                                content_div = law_soup.find('div', class_='loi-body') or law_soup.find('body')
                                if content_div:
                                    text = content_div.get_text(separator='\n', strip=True)
                                    if text and len(text) > 200:
                                        self.save_to_supabase(text, full_url, "FEDERAL_LAW")
                                        scraped_urls.append(full_url)
                                        self.stats["scraped"] += 1
            
            except Exception as e:
                logger.warning(f"Federal law search for '{term}' failed: {e}")
        
        return scraped_urls

    # --- Main Runner ---
    
    def run(self, target: str = "all", limit: int = 0):
        """Run the scraper pipeline."""
        logger.info(f"Starting Legal Brain scraper - Target: {target}, Limit: {limit}")
        
        if target in ["all", "pc200"]:
            self.fetch_pc_200_updates(limit)
        
        if target in ["all", "cnt"]:
            self.fetch_cnt_nar_updates(limit)
        
        if target in ["all", "federal"]:
            self.fetch_federal_law_updates(limit)
        
        logger.info("=" * 50)
        logger.info(f"Scraping complete! Stats: {self.stats}")
        return self.stats


def main():
    parser = argparse.ArgumentParser(description="FRIDAY Legal Brain Scraper")
    parser.add_argument(
        "--target", 
        choices=["all", "pc200", "cnt", "federal"], 
        default="all",
        help="Which sources to scrape"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0,
        help="Max documents per source (0=unlimited)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Don't store to database, just log what would be scraped"
    )
    
    args = parser.parse_args()
    
    try:
        scraper = BelgianLegalScraper(dry_run=args.dry_run)
        stats = scraper.run(target=args.target, limit=args.limit)
        
        # Exit with error ONLY if we had failures AND no success at all
        # Success = new documents OR correctly skipped duplicates
        if stats["errors"] > 0 and (stats["new"] + stats["skipped"]) == 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Scraper failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
