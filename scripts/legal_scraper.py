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
import random
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
        """Rate limiting with randomized delay to avoid bot detection."""
        delay = random.uniform(1.5, 3.5)
        time.sleep(delay)

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

    # --- 2-Step Crawler Helper ---
    
    def extract_content_from_page(self, page_url: str, source_tag: str) -> bool:
        """
        2-Step Crawler: Extract content from a sub-page.
        Step 1: Look for PDF download links
        Step 2: If no PDF, scrape the main HTML content
        
        Returns True if content was successfully processed.
        """
        try:
            self.polite_sleep()
            resp = requests.get(page_url, headers=self.headers, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"Sub-page returned {resp.status_code}: {page_url[:60]}...")
                return False
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Step 1: Look for PDF download links
            pdf_links = soup.select('span.file a[href$=".pdf"]')
            if not pdf_links:
                pdf_links = soup.find_all('a', href=lambda h: h and '.pdf' in h.lower())
            
            if pdf_links:
                # Found PDF - download and process
                href = pdf_links[0].get('href', '')
                if href:
                    if href.startswith('/'):
                        pdf_url = f"https://werk.belgie.be{href}"
                    elif href.startswith('http'):
                        pdf_url = href
                    else:
                        pdf_url = f"https://werk.belgie.be/{href}"
                    
                    try:
                        pdf_resp = requests.get(pdf_url, headers=self.headers, timeout=30)
                        if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                            content_hash = self.get_document_hash(pdf_resp.content)
                            if self.is_in_db(content_hash):
                                logger.info(f"Skipping duplicate PDF: {pdf_url[:50]}...")
                                self.stats["skipped"] += 1
                                return True  # Still considered success
                            
                            text, method = self.extract_text_from_pdf(pdf_resp.content)
                            if text and len(text) > 100:
                                if self.save_to_supabase(text, pdf_url, source_tag):
                                    logger.info(f"Processed PDF: {pdf_url[:50]}...")
                                    return True
                    except Exception as e:
                        logger.warning(f"PDF download failed for {pdf_url[:50]}...: {e}")
            
            # Step 2: No PDF found - scrape HTML content
            content_div = soup.select_one('div.content') or soup.select_one('div.main-content') or soup.select_one('article')
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if text and len(text) > 200:
                    content_hash = self.get_document_hash(text)
                    if self.is_in_db(content_hash):
                        logger.info(f"Skipping duplicate HTML: {page_url[:50]}...")
                        self.stats["skipped"] += 1
                        return True
                    
                    if self.save_to_supabase(text, page_url, source_tag):
                        logger.info(f"Processed HTML content: {page_url[:50]}...")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to extract from {page_url[:50]}...: {e}")
            self.stats["errors"] += 1
            return False

    # --- PC 200 Scraper (2-Step Crawler Walk) ---
    
    def fetch_pc_200_updates(self, limit: int = 0) -> List[str]:
        """
        2-Step Crawler for PC 200 via Federal Government (FOD Werk).
        Step 1: Fetch index page and find content links
        Step 2: For each link, extract PDFs or HTML content
        
        Target: https://werk.belgie.be/nl/themas/paritaire-comites/pc-200-aanvullend-paritair-comite-voor-de-bedienden
        """
        logger.info("=== Scraping PC 200 via 2-Step Crawler ===")
        
        index_url = "https://werk.belgie.be/nl/themas/paritaire-comites/pc-200-aanvullend-paritair-comite-voor-de-bedienden"
        scraped_urls = []
        
        try:
            logger.info(f"Fetching index: {index_url}")
            resp = requests.get(index_url, headers=self.headers, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"PC 200 index returned {resp.status_code}")
                # Fallback to sfonds200
                return self._fallback_sfonds200(limit)
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find content links in main area
            all_links = soup.select('div.main-content a[href], article a[href], div.content a[href]')
            if not all_links:
                all_links = soup.find_all('a', href=True)
            
            # Filter for content pages
            content_links = []
            for link in all_links:
                href = link.get('href', '')
                if not href:
                    continue
                # Must contain /nl/, not be mailto, not be anchor
                if '/nl/' in href and 'mailto:' not in href and not href.startswith('#'):
                    if href.startswith('/'):
                        full_url = f"https://werk.belgie.be{href}"
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    if full_url not in content_links:
                        content_links.append(full_url)
            
            logger.info(f"Found {len(content_links)} content links")
            
            # Process first N links
            max_links = min(10, len(content_links)) if limit == 0 else min(limit, len(content_links))
            count = 0
            
            for link_url in content_links[:max_links]:
                if limit > 0 and count >= limit:
                    break
                
                try:
                    if self.extract_content_from_page(link_url, "PC 200"):
                        scraped_urls.append(link_url)
                        count += 1
                except Exception as e:
                    logger.warning(f"Error processing {link_url[:50]}...: {e}")
                    self.stats["errors"] += 1
            
        except Exception as e:
            logger.error(f"PC 200 crawl failed: {e}")
            self.stats["errors"] += 1
            # Try fallback
            return self._fallback_sfonds200(limit)
        
        return scraped_urls
    
    def _fallback_sfonds200(self, limit: int = 0) -> List[str]:
        """Fallback: Direct PDF scraping from sfonds200 homepage."""
        logger.info("Trying fallback: sfonds200 homepage")
        scraped_urls = []
        
        try:
            resp = requests.get("https://www.sfonds200.be/nl/", headers=self.headers, timeout=20)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                links = soup.find_all('a', href=lambda h: h and h.endswith('.pdf'))
                
                count = 0
                for link in links:
                    if limit > 0 and count >= limit:
                        break
                    
                    href = link.get('href', '')
                    if href.startswith('/'):
                        full_url = f"https://www.sfonds200.be{href}"
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    try:
                        self.polite_sleep()
                        pdf_resp = requests.get(full_url, headers=self.headers, timeout=30)
                        if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                            content_hash = self.get_document_hash(pdf_resp.content)
                            if self.is_in_db(content_hash):
                                self.stats["skipped"] += 1
                                continue
                            
                            text, method = self.extract_text_from_pdf(pdf_resp.content)
                            if text and len(text) > 100:
                                if self.save_to_supabase(text, full_url, "PC 200"):
                                    scraped_urls.append(full_url)
                                    count += 1
                    except Exception as e:
                        logger.warning(f"Fallback PDF error: {e}")
                        self.stats["errors"] += 1
                        
        except Exception as e:
            logger.warning(f"Fallback failed: {e}")
        
        return scraped_urls

    # --- CNT/NAR Scraper (2-Step Crawler Walk) ---
    
    def fetch_cnt_nar_updates(self, limit: int = 0) -> List[str]:
        """
        2-Step Crawler for CNT/NAR via Federal Mirror (FOD Werk).
        Bypasses cnt-nar.be firewall by using government mirror.
        
        Target: https://werk.belgie.be/nl/themas/paritaire-comites/nationale-arbeidsraad
        """
        logger.info("=== Scraping CNT/NAR via 2-Step Federal Mirror ===")
        
        index_url = "https://werk.belgie.be/nl/themas/paritaire-comites/nationale-arbeidsraad"
        scraped_urls = []
        
        try:
            logger.info(f"Fetching CNT index: {index_url}")
            resp = requests.get(index_url, headers=self.headers, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"CNT index returned {resp.status_code}")
                return scraped_urls
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find links containing "cao" or "advies" keywords
            all_links = soup.find_all('a', href=True)
            content_links = []
            
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()
                
                # Look for CAO/regulation related links
                if any(kw in href.lower() or kw in text for kw in ['cao', 'advies', 'collectieve', 'arbeidsovereenkomst']):
                    if '/nl/' in href and 'mailto:' not in href:
                        if href.startswith('/'):
                            full_url = f"https://werk.belgie.be{href}"
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        if full_url not in content_links:
                            content_links.append(full_url)
            
            logger.info(f"Found {len(content_links)} CNT content links")
            
            # Process first N links
            max_links = min(10, len(content_links)) if limit == 0 else min(limit, len(content_links))
            count = 0
            
            for link_url in content_links[:max_links]:
                if limit > 0 and count >= limit:
                    break
                
                try:
                    if self.extract_content_from_page(link_url, "CNT/NAR"):
                        scraped_urls.append(link_url)
                        count += 1
                except Exception as e:
                    logger.warning(f"Error processing {link_url[:50]}...: {e}")
                    self.stats["errors"] += 1
            
        except Exception as e:
            logger.error(f"CNT crawl failed: {e}")
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

    # --- FPS Employment News Scraper (Task 1) ---
    
    def fetch_fps_employment_news(self, limit: int = 0) -> List[str]:
        """
        Scrape FPS Employment (FOD Werk) news feed.
        Central hub for CAOs, reforms, and worker rights.
        
        Target: https://werk.belgie.be/nl/nieuws
        Keywords: CAO, Arbeid, Loon, Verlof, Indexering
        """
        logger.info("=== Scraping FPS Employment News (werk.belgie.be) ===")
        
        news_url = "https://werk.belgie.be/nl/nieuws"
        scraped_urls = []
        keywords = ['cao', 'arbeid', 'loon', 'verlof', 'indexering', 'collectieve', 'arbeidsovereenkomst', 'ontslag']
        
        try:
            self.polite_sleep()
            resp = requests.get(news_url, headers=self.headers, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"FPS News returned {resp.status_code}")
                return scraped_urls
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find news articles using Drupal pattern
            articles = soup.select('article.node--type-news')
            if not articles:
                articles = soup.select('div.views-row, li.views-row')
            
            logger.info(f"Found {len(articles)} news articles")
            
            count = 0
            for article in articles:
                if limit > 0 and count >= limit:
                    break
                
                # Find the ACTUAL article link (starts with /nl/nieuws/), not social share
                link = article.find('a', href=lambda h: h and h.startswith('/nl/nieuws/'))
                if not link:
                    # Try other content links
                    link = article.find('a', href=lambda h: h and (
                        h.startswith('/nl/themas/') or 
                        h.startswith('/nl/publicaties/')
                    ))
                
                if not link:
                    continue
                
                href = link.get('href', '')
                
                # Build full URL
                full_url = f"https://werk.belgie.be{href}"
                
                if full_url in scraped_urls:
                    continue
                
                # Drill down into article page
                try:
                    if self.extract_content_from_page(full_url, "FPS_NEWS"):
                        scraped_urls.append(full_url)
                        count += 1
                except Exception as e:
                    logger.warning(f"Error processing FPS article: {e}")
                    self.stats["errors"] += 1
            
            logger.info(f"Processed {count} FPS Employment news items")
            
        except Exception as e:
            logger.error(f"FPS Employment scraping failed: {e}")
            self.stats["errors"] += 1
        
        return scraped_urls

    # --- Official Gazette Scraper (Task 2) ---
    
    def fetch_official_gazette_updates(self, limit: int = 0) -> List[str]:
        """
        Scrape Belgian Official Gazette (Belgisch Staatsblad).
        Captures Royal Decrees (KB) and Laws (Wet) when published.
        
        Primary: staatsbladmonitor.be (aggregator)
        Fallback: ejustice.just.fgov.be/staatsblad
        """
        logger.info("=== Scraping Official Gazette (Belgisch Staatsblad) ===")
        
        # Use easier-to-scrape aggregator
        gazette_urls = [
            "http://www.ejustice.just.fgov.be/cgi/summary.pl",  # Official summary
            "https://www.ejustice.just.fgov.be/doc/rech_n.htm"  # Search
        ]
        scraped_urls = []
        
        try:
            # Try to get recent labor law publications
            # ejustice search with labor keywords
            search_terms = ['arbeidsovereenkomst', 'arbeidswet', 'RSZ', 'ONSS']
            
            for term in search_terms:
                if limit > 0 and len(scraped_urls) >= limit:
                    break
                
                self.polite_sleep()
                
                # ejustice search endpoint
                search_url = f"http://www.ejustice.just.fgov.be/cgi/api_rech.pl?lg=n&numac=&text={term}"
                
                try:
                    resp = requests.get(search_url, headers=self.headers, timeout=20)
                    if resp.status_code != 200:
                        continue
                    
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    
                    # Find result links
                    result_links = soup.find_all('a', href=re.compile(r'numac='))
                    
                    for link in result_links[:3]:  # Top 3 per keyword
                        if limit > 0 and len(scraped_urls) >= limit:
                            break
                        
                        href = link.get('href', '')
                        if not href:
                            continue
                        
                        if href.startswith('/'):
                            full_url = f"http://www.ejustice.just.fgov.be{href}"
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        if full_url in scraped_urls:
                            continue
                        
                        try:
                            if self.extract_content_from_page(full_url, "GAZETTE"):
                                scraped_urls.append(full_url)
                        except Exception as e:
                            logger.warning(f"Gazette article error: {e}")
                            self.stats["errors"] += 1
                
                except Exception as e:
                    logger.warning(f"Gazette search for '{term}' failed: {e}")
            
            logger.info(f"Processed {len(scraped_urls)} Official Gazette items")
            
        except Exception as e:
            logger.error(f"Official Gazette scraping failed: {e}")
            self.stats["errors"] += 1
        
        return scraped_urls

    # --- Social Security News Scraper (Task 3) ---
    
    def fetch_social_security_news(self, limit: int = 0) -> List[str]:
        """
        Scrape Social Security (RSZ/ONSS) news and instructions.
        Covers payroll taxes and administrative instructions.
        
        Target: https://www.socialsecurity.be/site_nl/civil/general/news/index.htm
        Focus: Administrative Instructions (bible for payroll processing)
        """
        logger.info("=== Scraping Social Security News ===")
        
        news_url = "https://www.socialsecurity.be/site_nl/civil/general/news/index.htm"
        scraped_urls = []
        
        try:
            self.polite_sleep()
            resp = requests.get(news_url, headers=self.headers, timeout=20)
            if resp.status_code != 200:
                logger.warning(f"Social Security news returned {resp.status_code}")
                return scraped_urls
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find news items (standard HTML list or divs)
            news_items = soup.select('div.news-item, ul.news-list li, article, div.content a[href]')
            if not news_items:
                # Fallback: find all internal links
                news_items = soup.find_all('a', href=True)
            
            logger.info(f"Found {len(news_items)} social security items")
            
            # Focus on administrative instructions keywords
            priority_keywords = ['administratieve', 'instructie', 'bijdrage', 'RSZ', 'ONSS', 'werkgever']
            
            count = 0
            max_items = min(5, limit if limit > 0 else 5)  # Max 5 items as per spec
            
            for item in news_items:
                if count >= max_items:
                    break
                
                if item.name == 'a':
                    link = item
                else:
                    link = item.find('a', href=True)
                
                if not link:
                    continue
                
                href = link.get('href', '')
                title = link.get_text(strip=True).lower()
                
                # Skip if obviously not relevant
                if 'mailto:' in href or href.startswith('#'):
                    continue
                
                # Build full URL
                if href.startswith('/'):
                    full_url = f"https://www.socialsecurity.be{href}"
                elif href.startswith('http'):
                    full_url = href
                elif href.startswith('..'):
                    # Relative path handling
                    full_url = f"https://www.socialsecurity.be/site_nl/{href.replace('../', '')}"
                else:
                    continue
                
                if full_url in scraped_urls:
                    continue
                
                try:
                    if self.extract_content_from_page(full_url, "SOCIAL_SECURITY"):
                        scraped_urls.append(full_url)
                        count += 1
                except Exception as e:
                    logger.warning(f"Social Security article error: {e}")
                    self.stats["errors"] += 1
            
            logger.info(f"Processed {count} Social Security news items")
            
        except Exception as e:
            logger.error(f"Social Security scraping failed: {e}")
            self.stats["errors"] += 1
        
        return scraped_urls

    # --- Main Runner ---
    
    def run(self, target: str = "all", limit: int = 0):
        """Run the scraper pipeline."""
        logger.info(f"Starting Legal Brain scraper - Target: {target}, Limit: {limit}")
        
        # Legacy scrapers (kept for backwards compatibility)
        if target in ["all", "pc200"]:
            self.fetch_pc_200_updates(limit)
        
        if target in ["all", "cnt"]:
            self.fetch_cnt_nar_updates(limit)
        
        if target in ["all", "federal"]:
            self.fetch_federal_law_updates(limit)
        
        # New government news scrapers (2026 reforms)
        if target in ["all", "fps"]:
            self.fetch_fps_employment_news(limit)
        
        if target in ["all", "gazette"]:
            self.fetch_official_gazette_updates(limit)
        
        if target in ["all", "socialsecurity"]:
            self.fetch_social_security_news(limit)
        
        logger.info("=" * 50)
        logger.info(f"Scraping complete! Stats: {self.stats}")
        return self.stats


def main():
    parser = argparse.ArgumentParser(description="FRIDAY Legal Brain Scraper")
    parser.add_argument(
        "--target", 
        choices=["all", "pc200", "cnt", "federal", "fps", "gazette", "socialsecurity"], 
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
