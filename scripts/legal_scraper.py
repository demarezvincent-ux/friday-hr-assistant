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
from urllib.parse import urljoin, urlparse
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

# Add parent directory to path for imports if run from scripts/
if __name__ == "__main__" and os.path.dirname(os.path.abspath(__file__)).endswith('scripts'):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.unified_config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")
# Try both GROQ_API_KEY and FIXED_GROQ_KEY
GROQ_API_KEY = get_config("GROQ_API_KEY") or get_config("FIXED_GROQ_KEY")
HF_API_KEY = get_config("HF_API_KEY")

# --- Belgian Law Registry for Article-Level Scraping ---
BELGIAN_LAW_REGISTRY = [
    {"name": "Arbeidsovereenkomstenwet", "short": "AOW", "date": "1978-07-03", "numac": "1978070303", "type": "wet"},
    {"name": "Arbeidswet", "short": "AW", "date": "1971-03-16", "numac": "1971031602", "type": "wet"},
    {"name": "Welzijnswet", "short": "WW", "date": "1996-08-04", "numac": "1996012650", "type": "wet"},
    {"name": "CAO-wet", "short": "CAOW", "date": "1968-12-05", "numac": "1968120503", "type": "wet"},
    {"name": "Loonbeschermingswet", "short": "LBW", "date": "1965-04-12", "numac": "1965041207", "type": "wet"},
    {"name": "Feestdagenwet", "short": "FDW", "date": "1974-04-04", "numac": "1974040401", "type": "wet"},
    {"name": "Jaarlijkse Vakantiewet", "short": "JVW", "date": "1971-06-28", "numac": "1971062801", "type": "wet"},
]


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
        self.request_delay = 2.0  # Polite scraping: 2.0s between requests
        
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
        delay = random.uniform(2.0, 4.0)
        time.sleep(delay)

    def is_valid_url(self, url: str) -> bool:
        """
        Filter out URLs that should not be scraped.
        Returns False for social media, share links, images, and navigation elements.
        """
        if not url:
            return False
        
        url_lower = url.lower()
        
        # Social media blacklist
        social_patterns = [
            'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com',
            'youtube.com', 'pinterest.com', 'whatsapp.com', 't.co/',
            'sharer.php', 'share?', '/share/', 'intent/tweet'
        ]
        
        # Action/utility patterns to skip
        action_patterns = [
            '/print/', '/print?', '/rss', '/zoeken', '/search',
            '/feed', '/mailto:', 'javascript:', '#main-content',
            '/login', '/register', '/sitemap', '/contact'
        ]
        
        # Media file extensions (skip images/videos)
        media_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.ico', '.mp4', '.mp3')
        
        # Check social patterns
        for pattern in social_patterns:
            if pattern in url_lower:
                logger.debug(f"Skipping social URL: {url[:60]}...")
                return False
        
        # Check action patterns
        for pattern in action_patterns:
            if pattern in url_lower:
                logger.debug(f"Skipping action URL: {url[:60]}...")
                return False
        
        # Check media extensions
        if url_lower.endswith(media_extensions):
            logger.debug(f"Skipping media URL: {url[:60]}...")
            return False
        
        return True

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
    
    def _get_fallback_metadata(self) -> Dict:
        """
        Return default metadata when AI analysis fails.
        Ensures documents always have searchable metadata fields.
        """
        return {
            "topic": "Other",
            "effective_date": None,
            "summary_english": "Legal document",
            "summary_dutch": "Juridisch document",
            "is_critical": False,
            "paritair_comite": None
        }

    def _clean_ai_response(self, content: str) -> Optional[Dict]:
        """
        Try multiple strategies to extract JSON from AI response.
        Handles empty responses, markdown wrapping, and embedded JSON.
        """
        if not content or not content.strip():
            logger.debug("AI returned empty response")
            return None
        
        content = content.strip()
        
        # Strategy 1: Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Strip markdown code fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', content)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract JSON object with regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Try to find JSON between first { and last }
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(content[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass
        
        return None

    def ai_analyze_law(self, text: str) -> Dict:
        """
        Use Groq (Llama 3) to extract structured metadata from legal text.
        Always returns a valid metadata dict (fallback if AI fails).
        """
        fallback = self._get_fallback_metadata()
        
        if not self.groq_key:
            logger.debug("No GROQ_API_KEY, using fallback metadata")
            return fallback
        
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
                content = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                parsed = self._clean_ai_response(content)
                if parsed:
                    # Merge with fallback to ensure all fields exist
                    return {**fallback, **parsed}
                else:
                    logger.debug("Could not parse AI response, using fallback")
            else:
                logger.warning(f"Groq API error: {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning("Groq API timeout, using fallback metadata")
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
        
        return fallback

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
            logger.debug(f"Skipping duplicate: {source_url[:50]}...")
            self.stats["skipped"] += 1
            return False
        
        # Build metadata
        doc_metadata = {
            "source": source_type,
            "source_url": source_url,
            **(metadata or {})
        }
        
        # AI analysis (always returns valid metadata with fallback)
        ai_meta = self.ai_analyze_law(text)
        doc_metadata.update(ai_meta)
        
        # Generate embedding
        embedding = self.get_embedding(text[:3000])  # Use summary portion for embedding
        
        # Build record
        record = {
            "content": text,
            "summary": ai_meta.get("summary_english"),
            "metadata": doc_metadata,
            "content_hash": content_hash,
            "embedding": embedding
        }
        
        try:
            self.supabase.table("legal_knowledge").insert(record).execute()
            logger.info(f"✓ Stored: {source_type} - {source_url[:50]}...")
            self.stats["new"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            self.stats["errors"] += 1
            return False

    # --- Recursive Spider (Phase 1 Engine Upgrade) ---
    
    def recursive_crawl(
        self, 
        entry_url: str, 
        source_tag: str,
        max_depth: int = 2,
        domain_filter: str = None,
        limit: int = 0
    ) -> List[str]:
        """
        Recursive crawler with depth control.
        Follows internal links within target domain up to max_depth.
        
        Args:
            entry_url: Starting URL
            source_tag: Tag for metadata (e.g., FOD_THEMAS)
            max_depth: Maximum crawl depth (default 2)
            domain_filter: Only follow links containing this (e.g., '/nl/')
            limit: Max documents to process (0=unlimited)
        """
        # urljoin and urlparse already imported at module level
        
        visited = set()
        queue = [(entry_url, 0)]  # (url, depth)
        scraped_urls = []
        base_domain = urlparse(entry_url).netloc
        
        logger.info(f"=== Recursive Crawl: {entry_url} (max_depth={max_depth}) ===")
        
        while queue:
            if limit > 0 and len(scraped_urls) >= limit:
                break
            
            url, depth = queue.pop(0)
            
            # Skip if visited or too deep
            if url in visited or depth > max_depth:
                continue
            visited.add(url)
            
            try:
                self.polite_sleep()
                resp = requests.get(url, headers=self.headers, timeout=30)
                if resp.status_code != 200:
                    continue
                
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Check if it's a PDF
                if url.lower().endswith('.pdf'):
                    # Process PDF directly
                    pdf_text, method = self.extract_text_from_pdf(resp.content)
                    if pdf_text and len(pdf_text) > 100:
                        # Extract title from URL
                        title = url.split('/')[-1].replace('.pdf', '')
                        metadata = {
                            "category": "legal_knowledge",
                            "source_domain": base_domain,
                            "is_active": True,
                            "title": title,
                            "crawl_depth": depth
                        }
                        if self.save_to_supabase(pdf_text, url, source_tag, metadata):
                            scraped_urls.append(url)
                            logger.info(f"[Depth {depth}] PDF: {url[:60]}...")
                    continue
                
                # Extract main content for HTML pages
                main_content = (
                    soup.select_one('#main-content') or
                    soup.select_one('.region-content') or
                    soup.select_one('main') or
                    soup.select_one('article') or
                    soup.select_one('.content')
                )
                
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                    if text and len(text) > 300:
                        # Extract title
                        title_tag = soup.find('h1') or soup.find('title')
                        title = title_tag.get_text(strip=True) if title_tag else url.split('/')[-1]
                        
                        metadata = {
                            "category": "legal_knowledge",
                            "source_domain": base_domain,
                            "is_active": True,
                            "title": title[:200],
                            "crawl_depth": depth
                        }
                        
                        content_hash = self.get_document_hash(text)
                        if not self.is_in_db(content_hash):
                            if self.save_to_supabase(text, url, source_tag, metadata):
                                scraped_urls.append(url)
                                logger.info(f"[Depth {depth}] HTML: {url[:60]}...")
                
                # Find child links (only if not at max depth)
                if depth < max_depth:
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        
                        # Skip empty, anchors, mailto, external
                        if not href or href.startswith('#') or 'mailto:' in href:
                            continue
                        
                        # Build absolute URL
                        full_url = urljoin(url, href)
                        parsed = urlparse(full_url)
                        
                        # Must be same domain
                        if parsed.netloc != base_domain:
                            continue
                        
                        # Apply URL blacklist filter (social, media, navigation)
                        if not self.is_valid_url(full_url):
                            continue
                        
                        # Apply domain filter (e.g., '/nl/')
                        if domain_filter and domain_filter not in full_url:
                            continue
                        
                        # Add to queue
                        if full_url not in visited:
                            queue.append((full_url, depth + 1))
                
            except Exception as e:
                logger.warning(f"Crawl error at {url[:50]}...: {e}")
                self.stats["errors"] += 1
        
        logger.info(f"Recursive crawl complete: {len(scraped_urls)} documents from {len(visited)} pages")
        return scraped_urls

    # --- FOD Werk Themas Scraper (Phase 2) ---
    
    def fetch_fod_themas(self, limit: int = 0) -> List[str]:
        """
        Recursive crawl of FOD Werk Themas.
        Entry: https://werk.belgie.be/nl/themas
        Goal: Capture "Verklarende nota's" on contracts, holidays, well-being.
        """
        logger.info("=== Scraping FOD Werk Themas (Recursive) ===")
        
        entry_url = "https://werk.belgie.be/nl/themas"
        return self.recursive_crawl(
            entry_url=entry_url,
            source_tag="FOD_THEMAS",
            max_depth=2,
            domain_filter="/nl/",
            limit=limit
        )

    # --- NAR CAOs Scraper (Phase 2) ---
    
    def fetch_nar_caos(self, limit: int = 0) -> List[str]:
        """
        Scrape NAR (CNT) CAO documents.
        
        NOTE: cnt-nar.be now uses JavaScript to load PDF links dynamically,
        making it unscrapable with static requests. We use the government mirror
        at werk.belgie.be which provides static HTML with CAO information.
        
        Primary: werk.belgie.be NAR section (static HTML)
        Fallback: Direct recursive crawl of available PDFs
        """
        logger.info("=== Scraping NAR CAO Registry (via Government Mirror) ===")
        
        scraped_urls = []
        
        # Use government mirror - CAO/NAR information pages (verified January 2026)
        mirror_urls = [
            "https://werk.belgie.be/nl/themas/paritaire-comites-en-collectieve-arbeidsovereenkomsten-caos/collectieve",
            "https://werk.belgie.be/nl/over-de-fod/dienstverlening/afschriften-van-cao",
            "https://werk.belgie.be/nl/themas/arbeidsovereenkomsten"
        ]
        
        cao_keywords = ['cao', 'collectieve', 'arbeidsovereenkomst', 'overeenkomst', 'advies']
        
        try:
            for mirror_url in mirror_urls:
                if limit > 0 and len(scraped_urls) >= limit:
                    break
                    
                self.polite_sleep()
                resp = requests.get(mirror_url, headers=self.headers, timeout=20)
                
                if resp.status_code != 200:
                    logger.warning(f"NAR mirror returned {resp.status_code}: {mirror_url}")
                    continue
                
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Find all links - look for PDFs and CAO-related pages
                all_links = soup.find_all('a', href=True)
                
                for link in all_links:
                    if limit > 0 and len(scraped_urls) >= limit:
                        break
                    
                    href = link.get('href', '')
                    link_text = link.get_text(strip=True).lower()
                    
                    # Skip navigation, mailto, anchors
                    if not href or href.startswith('#') or 'mailto:' in href:
                        continue
                    
                    # Check if CAO-related
                    is_cao_related = any(kw in href.lower() or kw in link_text for kw in cao_keywords)
                    is_pdf = '.pdf' in href.lower()
                    
                    if not (is_cao_related or is_pdf):
                        continue
                    
                    # Build full URL
                    full_url = urljoin(mirror_url, href)
                    
                    if full_url in scraped_urls:
                        continue
                    
                    # Process PDFs directly
                    if is_pdf:
                        try:
                            self.polite_sleep()
                            pdf_resp = requests.get(full_url, headers=self.headers, timeout=30)
                            if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                                pdf_text, method = self.extract_text_from_pdf(pdf_resp.content)
                                if pdf_text and len(pdf_text) > 100:
                                    # Extract CAO number from filename
                                    filename = href.split('/')[-1]
                                    cao_match = re.search(r'cao[_-]?(\d+)', filename.lower())
                                    cao_number = cao_match.group(1) if cao_match else filename.replace('.pdf', '')
                                    
                                    metadata = {
                                        "category": "legal_knowledge",
                                        "source_domain": "werk.belgie.be",
                                        "is_active": True,
                                        "title": f"CAO {cao_number}",
                                        "cao_number": cao_number
                                    }
                                    if self.save_to_supabase(pdf_text, full_url, "NAR_CAO", metadata):
                                        scraped_urls.append(full_url)
                                        logger.info(f"Processed CAO PDF: {cao_number}")
                        except Exception as e:
                            logger.warning(f"Failed to process CAO PDF: {e}")
                            self.stats["errors"] += 1
                    
                    # For HTML pages, use the 2-step crawler
                    elif '/nl/' in full_url:
                        try:
                            if self.extract_content_from_page(full_url, "NAR_CAO"):
                                scraped_urls.append(full_url)
                                logger.info(f"Processed CAO page: {full_url[:60]}...")
                        except Exception as e:
                            logger.warning(f"Failed to process CAO page: {e}")
                            self.stats["errors"] += 1
            
            logger.info(f"NAR scraping complete: {len(scraped_urls)} documents")
            
        except Exception as e:
            logger.error(f"NAR CAO scraping failed: {e}")
            self.stats["errors"] += 1
        
        return scraped_urls

    # --- 2-Step Crawler Helper ---
    
    def extract_content_from_page(self, page_url: str, source_tag: str) -> bool:
        """
        2-Step Crawler: Extract content from a sub-page.
        Step 1: Check if URL is valid (not social/nav)
        Step 2: Look for PDF download links
        Step 3: If no PDF, scrape the main HTML content
        
        Returns True if content was successfully processed.
        """
        # Filter out invalid URLs before processing
        if not self.is_valid_url(page_url):
            return False
        
        try:
            self.polite_sleep()
            resp = requests.get(page_url, headers=self.headers, timeout=30)
            
            # Handle different status codes appropriately
            if resp.status_code == 404:
                logger.debug(f"Page not found (404): {page_url[:60]}...")
                return False
            elif resp.status_code == 400:
                logger.debug(f"Bad request (400): {page_url[:60]}...")
                return False
            elif resp.status_code != 200:
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
                                logger.debug(f"Skipping duplicate PDF: {pdf_url[:50]}...")
                                self.stats["skipped"] += 1
                                return True  # Still considered success
                            
                            text, method = self.extract_text_from_pdf(pdf_resp.content)
                            if text and len(text) > 100:
                                if self.save_to_supabase(text, pdf_url, source_tag):
                                    logger.info(f"✓ Processed PDF: {pdf_url[:50]}...")
                                    return True
                    except requests.exceptions.Timeout:
                        logger.warning(f"PDF download timeout: {pdf_url[:50]}...")
                    except Exception as e:
                        logger.warning(f"PDF download failed for {pdf_url[:50]}...: {e}")
            
            # Step 2: No PDF found - scrape HTML content using expanded selectors
            # Try selectors in order of specificity (Drupal/government site patterns)
            content_selectors = [
                'div.field--name-body',
                'article.node--type-news',
                'div.node--type-page',
                'main#main-content',
                'div.region-content',
                'article',
                'div.content',
                '#block-system-main',
                'div[role="main"]',
                'div.main-content'
            ]
            
            content_div = None
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    break
            
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                if text and len(text) > 300:  # Increased from 200 to 300
                    content_hash = self.get_document_hash(text)
                    if self.is_in_db(content_hash):
                        logger.debug(f"Skipping duplicate HTML: {page_url[:50]}...")
                        self.stats["skipped"] += 1
                        return True
                    
                    if self.save_to_supabase(text, page_url, source_tag):
                        logger.info(f"✓ Processed HTML: {page_url[:50]}...")
                        return True
            
            return False
        
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout: {page_url[:50]}...")
            self.stats["errors"] += 1
            return False
        except Exception as e:
            logger.warning(f"Failed to extract from {page_url[:50]}...: {e}")
            self.stats["errors"] += 1
            return False

    # --- PC 200 Scraper (sfonds200.be) ---
    
    def fetch_pc_200_updates(self, limit: int = 0) -> List[str]:
        """
        Scrape PC 200 CAOs from Sector Fund.
        Uses sfonds200.be main page and sectorinformatie pages for PDFs
        """
        logger.info("=== Scraping PC 200 CAOs (sfonds200.be) ===")
        
        scraped_urls = []
        
        # Pages to scan for PDFs
        pages_to_scan = [
            "https://www.sfonds200.be/nl/",
            "https://www.sfonds200.be/nl/sectorinformatie/cao-s/",
            "https://www.sfonds200.be/nl/sectorinformatie/arbeidsvoorwaarden/",
            "https://www.sfonds200.be/nl/sectorinformatie/verloning/",
        ]
        
        all_pdf_urls = set()
        
        try:
            for page_url in pages_to_scan:
                try:
                    logger.info(f"Scanning: {page_url}")
                    self.polite_sleep()
                    resp = requests.get(page_url, headers=self.headers, timeout=30)
                    
                    if resp.status_code != 200:
                        continue
                    
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    
                    # Find all PDF links
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if '.pdf' in href.lower():
                            if href.startswith('/'):
                                full_url = f"https://www.sfonds200.be{href}"
                            elif href.startswith('http'):
                                full_url = href
                            else:
                                full_url = f"https://www.sfonds200.be/{href}"
                            
                            if self.is_valid_url(full_url):
                                all_pdf_urls.add(full_url)
                
                except Exception as e:
                    logger.warning(f"Error scanning {page_url}: {e}")
            
            logger.info(f"Found {len(all_pdf_urls)} total PDFs across sfonds200")
            
            count = 0
            for pdf_url in list(all_pdf_urls):
                if limit > 0 and count >= limit:
                    break
                
                try:
                    self.polite_sleep()
                    pdf_resp = requests.get(pdf_url, headers=self.headers, timeout=40)
                    
                    if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                        content_hash = self.get_document_hash(pdf_resp.content)
                        
                        if self.is_in_db(content_hash):
                            logger.debug(f"Skipping duplicate PDF")
                            self.stats["skipped"] += 1
                            continue
                        
                        text, method = self.extract_text_from_pdf(pdf_resp.content)
                        
                        if text and len(text) > 100:
                            filename = pdf_url.split('/')[-1]
                            
                            metadata = {
                                "category": "legal_knowledge",
                                "source_domain": "sfonds200.be",
                                "is_active": True,
                                "title": f"PC 200 - {filename[:50]}",
                                "paritair_comite": "200"
                            }
                            
                            if self.save_to_supabase(text, pdf_url, "PC 200", metadata):
                                scraped_urls.append(pdf_url)
                                count += 1
                                logger.info(f"✓ PC 200 PDF: {filename[:40]}")
                
                except Exception as e:
                    logger.warning(f"Failed to process PDF: {e}")
                    self.stats["errors"] += 1
            
            logger.info(f"PC 200 scraping complete: {count} documents")
            return scraped_urls
            
        except Exception as e:
            logger.error(f"PC 200 scraping failed: {e}")
            self.stats["errors"] += 1
            return scraped_urls

    # --- CNT/NAR Scraper (cnt-nar.be - Verified Feb 2026) ---
    
    def fetch_cnt_nar_updates(self, limit: int = 0) -> List[str]:
        """
        Scrape CNT/NAR (National Labor Council) CAOs.
        NEW: Uses cnt-nar.be official document directory (verified Feb 2026)
        """
        logger.info("=== Scraping CNT/NAR CAOs (cnt-nar.be) ===")
        
        index_url = "https://cnt-nar.be/nl/documents/cao-nummer"
        scraped_urls = []
        
        try:
            logger.info(f"Fetching CNT/NAR index: {index_url}")
            self.polite_sleep()
            resp = requests.get(index_url, headers=self.headers, timeout=30)
            
            if resp.status_code != 200:
                logger.warning(f"CNT/NAR index returned {resp.status_code}")
                return scraped_urls
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find CAO links (cnt-nar.be uses structured document pages)
            all_links = soup.find_all('a', href=True)
            cao_links = []
            
            for link in all_links:
                href = link.get('href', '')
                link_text = link.get_text(strip=True).lower()
                
                # Filter for actual CAO document links
                if 'cao' in href.lower() or 'cao' in link_text:
                    if '/nl/documents/' in href or href.startswith('/'):
                        if href.startswith('/'):
                            full_url = f"https://cnt-nar.be{href}"
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        if full_url not in cao_links and self.is_valid_url(full_url):
                            cao_links.append(full_url)
            
            logger.info(f"Found {len(cao_links)} CNT/NAR CAO links")
            
            count = 0
            max_links = min(limit if limit > 0 else 20, len(cao_links))
            
            for cao_url in cao_links[:max_links]:
                if limit > 0 and count >= limit:
                    break
                
                try:
                    if self.extract_content_from_page(cao_url, "CNT/NAR"):
                        scraped_urls.append(cao_url)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to process CNT/NAR page: {e}")
                    self.stats["errors"] += 1
            
            logger.info(f"CNT/NAR scraping complete: {count} documents")
            return scraped_urls
            
        except Exception as e:
            logger.error(f"CNT/NAR scraping failed: {e}")
            self.stats["errors"] += 1
            return scraped_urls

    # --- Federal Law Scraper (FOD Werk mirrors) ---
    
    def fetch_federal_law_updates(self, limit: int = 0) -> List[str]:
        """
        Scrape federal labor laws using FOD Werk thematic pages.
        These pages link to key labor legislation with readable content.
        """
        logger.info("=== Scraping Federal Labor Laws ===")
        
        scraped_urls = []
        
        # FOD Werk thematic pages that link to federal laws (verified working URLs)
        law_pages = [
            ("https://werk.belgie.be/nl/themas/arbeidsovereenkomsten", "Arbeidsovereenkomsten"),
            ("https://werk.belgie.be/nl/themas/arbeidsreglementering", "Arbeidsreglementering"),
            ("https://werk.belgie.be/nl/themas/verloning", "Verloning"),
            ("https://werk.belgie.be/nl/themas/welzijn-op-het-werk", "Welzijn op het werk"),
            ("https://werk.belgie.be/nl/themas/feestdagen-en-verloven", "Feestdagen en verloven"),
            ("https://werk.belgie.be/nl/themas/paritaire-comites-en-collectieve-arbeidsovereenkomsten-caos", "Paritaire comites en CAOs"),
            ("https://werk.belgie.be/nl/themas/herstructurering", "Herstructurering"),
        ]
        
        try:
            count = 0
            for page_url, topic in law_pages:
                if limit > 0 and count >= limit:
                    break
                
                try:
                    logger.info(f"Fetching: {topic}")
                    self.polite_sleep()
                    resp = requests.get(page_url, headers=self.headers, timeout=30)
                    
                    if resp.status_code != 200:
                        logger.warning(f"Topic {topic} returned {resp.status_code}")
                        continue
                    
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    
                    # Find main content
                    content_div = (
                        soup.select_one('div.field--name-body') or
                        soup.select_one('article.node') or
                        soup.select_one('div.content') or
                        soup.select_one('main')
                    )
                    
                    if content_div:
                        text = content_div.get_text(separator='\n', strip=True)
                        
                        if text and len(text) > 300:
                            content_hash = self.get_document_hash(text)
                            
                            if self.is_in_db(content_hash):
                                logger.debug(f"Skipping duplicate: {topic}")
                                self.stats["skipped"] += 1
                                continue
                            
                            metadata = {
                                "category": "legal_knowledge",
                                "source_domain": "werk.belgie.be",
                                "is_active": True,
                                "title": f"Federal Law - {topic}",
                                "document_type": "federal_law",
                                "topic": topic
                            }
                            
                            if self.save_to_supabase(text, page_url, "FEDERAL_LAW", metadata):
                                scraped_urls.append(page_url)
                                count += 1
                                logger.info(f"✓ Federal Law: {topic}")
                    
                    # Also find and process sub-page links
                    sub_links = content_div.find_all('a', href=True) if content_div else []
                    for link in sub_links[:3]:  # Limit sub-pages per topic
                        if limit > 0 and count >= limit:
                            break
                        
                        href = link.get('href', '')
                        if href.startswith('/nl/') and self.is_valid_url(f"https://werk.belgie.be{href}"):
                            sub_url = f"https://werk.belgie.be{href}"
                            if sub_url not in scraped_urls:
                                if self.extract_content_from_page(sub_url, "FEDERAL_LAW"):
                                    scraped_urls.append(sub_url)
                                    count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to fetch topic {topic}: {e}")
                    self.stats["errors"] += 1
            
            logger.info(f"Federal laws scraping complete: {count} documents")
            return scraped_urls
            
        except Exception as e:
            logger.error(f"Federal law scraping failed: {e}")
            self.stats["errors"] += 1
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

    # --- FOD Publications Scraper (werk.belgie.be - Verified Feb 2026) ---
    
    def fetch_fod_publications(self, limit: int = 0) -> List[str]:
        """
        Scrape FOD Werk official publications.
        Uses werk.belgie.be publications directory
        """
        logger.info("=== Scraping FOD Publications (werk.belgie.be) ===")
        
        pub_url = "https://werk.belgie.be/nl/publicaties"
        scraped_urls = []
        
        try:
            logger.info(f"Fetching publications: {pub_url}")
            self.polite_sleep()
            resp = requests.get(pub_url, headers=self.headers, timeout=30)
            
            if resp.status_code != 200:
                logger.warning(f"Publications page returned {resp.status_code}")
                return scraped_urls
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Find main content area
            main_content = (
                soup.select_one('main') or
                soup.select_one('#main-content') or
                soup.select_one('.region-content') or
                soup
            )
            
            # Find publication links in main content
            pub_links = []
            
            for link in main_content.find_all('a', href=True):
                href = link.get('href', '')
                
                # Only internal /nl/publicaties links
                if href.startswith('/nl/publicaties/') and href != '/nl/publicaties':
                    full_url = f"https://werk.belgie.be{href}"
                    
                    if self.is_valid_url(full_url) and full_url not in pub_links:
                        pub_links.append(full_url)
            
            # If no publicaties links, try broader approach
            if not pub_links:
                for link in main_content.find_all('a', href=True):
                    href = link.get('href', '')
                    if href.startswith('/nl/') and href not in ['/nl/publicaties', '/nl/']:
                        full_url = f"https://werk.belgie.be{href}"
                        if self.is_valid_url(full_url) and full_url not in pub_links:
                            pub_links.append(full_url)
            
            logger.info(f"Found {len(pub_links)} publication links")
            
            count = 0
            max_links = min(limit if limit > 0 else 15, len(pub_links))
            
            for pub_link in pub_links[:max_links]:
                if limit > 0 and count >= limit:
                    break
                
                try:
                    if self.extract_content_from_page(pub_link, "FOD_PUBLICATION"):
                        scraped_urls.append(pub_link)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to process publication: {e}")
                    self.stats["errors"] += 1
            
            logger.info(f"FOD Publications complete: {count} documents")
            return scraped_urls
            
        except Exception as e:
            logger.error(f"FOD Publications scraping failed: {e}")
            self.stats["errors"] += 1
            return scraped_urls

    # --- Belgian Law Article Scraper (eJustice ELI) ---
    
    def fetch_law_articles(self, limit: int = 0) -> List[str]:
        """
        Scrape article-level text from core Belgian labor laws via eJustice ELI URLs.
        
        Each article is stored as a separate document for precise citation.
        No AI analysis needed - we already know the metadata from the HTML structure.
        """
        logger.info("=== Scraping Belgian Law Articles (eJustice) ===")
        
        scraped_urls = []
        laws_processed = 0
        
        for law_info in BELGIAN_LAW_REGISTRY:
            if limit > 0 and laws_processed >= limit:
                break
            
            # Build ELI URL: https://www.ejustice.just.fgov.be/eli/wet/YYYY/MM/DD/numac/justel
            date_parts = law_info["date"].split("-")
            eli_url = f"https://www.ejustice.just.fgov.be/eli/{law_info['type']}/{date_parts[0]}/{date_parts[1]}/{date_parts[2]}/{law_info['numac']}/justel"
            
            logger.info(f"Fetching {law_info['name']} from {eli_url}")
            
            try:
                self.polite_sleep()
                resp = requests.get(eli_url, headers=self.headers, timeout=30)
                
                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch {law_info['name']}: HTTP {resp.status_code}")
                    self.stats["errors"] += 1
                    continue
                
                # Parse articles from HTML
                articles = self._parse_law_articles(resp.text, law_info, eli_url)
                
                if not articles:
                    logger.warning(f"No articles found in {law_info['name']}")
                    continue
                
                logger.info(f"Found {len(articles)} articles in {law_info['name']}")
                
                # Store each article
                for article in articles:
                    if self._save_law_article(article, law_info, eli_url):
                        scraped_urls.append(f"{eli_url}#{article['article_number']}")
                
                laws_processed += 1
                self.stats["scraped"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {law_info['name']}: {e}")
                self.stats["errors"] += 1
        
        logger.info(f"Law article scraping complete: {len(scraped_urls)} articles from {laws_processed} laws")
        return scraped_urls
    
    def _parse_law_articles(self, html: str, law_info: Dict, source_url: str) -> List[Dict]:
        """
        Parse eJustice HTML to extract individual articles with hierarchy context.
        
        eJustice pages structure:
        - Uses <a name="Art.X"></a> anchors to mark article starts
        - Articles are in the "Tekst" section (after table of contents)
        - Hierarchy: TITEL > HOOFDSTUK > Afdeling > Onderafdeling
        """
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        
        # Find all article anchors - these are the reliable markers
        # Format: <a name="Art.1"></a>, <a name="Art.2bis"></a>, <a name="Art.119.1"></a>
        article_anchors = soup.find_all('a', attrs={'name': re.compile(r'^Art\.\d+(?:\.\d+)?(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?$', re.IGNORECASE)})
        
        if not article_anchors:
            logger.warning(f"No article anchors found in {law_info['name']}")
            return []
        
        logger.debug(f"Found {len(article_anchors)} article anchors")
        
        # Track current hierarchy
        current_title = None
        current_chapter = None
        current_section = None
        
        # Regex patterns for hierarchy detection
        title_pattern = re.compile(r'TITEL\s+[IVXLCDM]+\.?\s*[-.]?\s*.*', re.IGNORECASE)
        chapter_pattern = re.compile(r'HOOFDSTUK\s+[IVXLCDM]+\.?\s*[-.]?\s*.*', re.IGNORECASE)
        section_pattern = re.compile(r'Afdeling\s+\d+\.?\s*[-.]?\s*.*', re.IGNORECASE)
        
        # Process each article anchor
        for i, anchor in enumerate(article_anchors):
            # Extract full article number from anchor name (e.g., "Art.119.1" -> "119.1", "Art.2bis" -> "2bis")
            anchor_name = anchor.get('name', '')
            # Match patterns: Art.1, Art.2bis, Art.119.1, Art.119.11
            art_match = re.match(r'Art\.(\d+(?:\.[\d]+)?(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)', anchor_name, re.IGNORECASE)
            if not art_match:
                continue
            
            article_number = f"Art. {art_match.group(1)}"
            
            # Find the text between this anchor and the next one
            # We need to walk the DOM from this anchor to the next
            text_parts = []
            current_node = anchor.next_sibling
            
            # Find the next article anchor to know where to stop
            next_anchor = article_anchors[i + 1] if i + 1 < len(article_anchors) else None
            
            while current_node:
                # Stop if we reach the next article anchor
                if next_anchor and current_node == next_anchor:
                    break
                
                # Check if we've passed the next anchor (by checking if it's a parent/sibling relationship issue)
                if next_anchor:
                    try:
                        # If the current node contains or is the next anchor, stop
                        if hasattr(current_node, 'find') and current_node.find('a', attrs={'name': next_anchor.get('name')}):
                            break
                    except:
                        pass
                
                # Extract text from this node
                if hasattr(current_node, 'get_text'):
                    text_parts.append(current_node.get_text(separator=' ', strip=True))
                elif isinstance(current_node, str):
                    text_parts.append(current_node.strip())
                
                # Move to next sibling or parent's next sibling
                current_node = current_node.next_sibling
                
                # Limit iterations to prevent infinite loops
                if len(text_parts) > 500:
                    break
            
            # Join and clean the article text
            article_text = ' '.join(text_parts)
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            
            # Skip if article is too short (just the marker)
            if len(article_text) < 30:
                continue
            
            # Check for hierarchy markers in the text before this article
            # Look at previous siblings for TITEL, HOOFDSTUK, Afdeling
            prev_node = anchor.previous_sibling
            check_count = 0
            while prev_node and check_count < 20:
                if hasattr(prev_node, 'get_text'):
                    prev_text = prev_node.get_text(strip=True)
                    if title_pattern.match(prev_text):
                        current_title = prev_text
                    elif chapter_pattern.match(prev_text):
                        current_chapter = prev_text
                    elif section_pattern.match(prev_text):
                        current_section = prev_text
                prev_node = prev_node.previous_sibling
                check_count += 1
            
            # Clean article text: remove amendment markers like <W 1985-07-17/41, art. 1, 010>
            clean_text = re.sub(r'<[^>]+>', '', article_text)  # Remove HTML-like tags in text
            clean_text = re.sub(r'\[[^\]]+\]', '', clean_text)  # Remove bracketed references
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
            
            articles.append({
                "article_number": article_number,
                "text": clean_text,
                "title": current_title,
                "chapter": current_chapter,
                "section": current_section,
                "subsection": None
            })
        
        return articles

    
    def _save_law_article(self, article: Dict, law_info: Dict, eli_url: str) -> bool:
        """
        Store a single law article in the legal_knowledge table.
        
        Skips AI analysis since literal law text doesn't need topic extraction.
        Content includes contextual header for embedding quality.
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would store: {article['article_number']} from {law_info['name']}")
            return True
        
        if not self.supabase:
            return False
        
        # Build content with contextual header
        hierarchy_parts = [p for p in [article.get('title'), article.get('chapter'), article.get('section')] if p]
        hierarchy_str = " > ".join(hierarchy_parts) if hierarchy_parts else ""
        
        content = f"""{law_info['name']} ({law_info['date']})
{hierarchy_str}

{article['article_number']}
{article['text']}"""
        
        content_hash = self.get_document_hash(content)
        
        # Check duplicate
        if self.is_in_db(content_hash):
            logger.debug(f"Skipping duplicate: {article['article_number']} {law_info['short']}")
            self.stats["skipped"] += 1
            return False
        
        # Build metadata
        metadata = {
            "source": "BELGIAN_LAW",
            "category": "primary_legislation",
            "law_name": law_info["name"],
            "law_short": law_info["short"],
            "law_date": law_info["date"],
            "article_number": article["article_number"],
            "title": article.get("title"),
            "chapter": article.get("chapter"),
            "section": article.get("section"),
            "language": "nl",
            "eli_url": eli_url,
            "source_url": f"{eli_url}#{article['article_number'].replace(' ', '')}",
            "is_active": True
        }
        
        # Generate embedding directly (no AI analysis)
        embedding = self.get_embedding(content[:3000])
        
        # Build record
        record = {
            "content": content,
            "summary": f"{article['article_number']} - {law_info['name']}",
            "metadata": metadata,
            "content_hash": content_hash,
            "embedding": embedding
        }
        
        try:
            self.supabase.table("legal_knowledge").insert(record).execute()
            logger.info(f"✓ Stored: {article['article_number']} {law_info['short']}")
            self.stats["new"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to store {article['article_number']}: {e}")
            self.stats["errors"] += 1
            return False

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
        
        if target in ["all", "publications"]:
            self.fetch_fod_publications(limit)
        
        # Recursive crawlers (high-volume)
        if target in ["all", "themas"]:
            self.fetch_fod_themas(limit)
        
        if target in ["all", "nar"]:
            self.fetch_nar_caos(limit)
        
        # Article-level law scraping (eJustice)
        if target in ["all", "laws"]:
            self.fetch_law_articles(limit)
        
        logger.info("=" * 50)
        logger.info(f"Scraping complete! Stats: {self.stats}")
        return self.stats


def main():
    parser = argparse.ArgumentParser(description="FRIDAY Legal Brain Scraper")
    parser.add_argument(
        "--target", 
        choices=["all", "pc200", "cnt", "federal", "fps", "gazette", "socialsecurity", "themas", "nar", "publications", "laws"], 
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
