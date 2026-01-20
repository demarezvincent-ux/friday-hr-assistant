"""
Vision Service for Visual RAG
Provides image-to-text functionality using Groq's Llama 4 Scout Vision model.
Includes strict rate limiting to prevent lockouts.

Rate Limits for meta-llama/llama-4-scout-17b-16e-instruct (Free Tier):
- 30 RPM (requests per minute)
- 1K RPD (requests per day)
- 30K TPM (tokens per minute)
- 500K TPD (tokens per day)
"""

import time
import base64
import logging
import requests
from typing import Optional, List, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)

# Rate limiter state
_last_vision_call = 0.0
# Conservative rate limit: 20 RPM (3 seconds between calls) to stay well within 30 RPM
_VISION_COOLDOWN_SECONDS = 3.0


def _rate_limit_vision():
    """Enforce rate limit to prevent Groq lockouts."""
    global _last_vision_call
    now = time.time()
    elapsed = now - _last_vision_call
    if elapsed < _VISION_COOLDOWN_SECONDS:
        sleep_time = _VISION_COOLDOWN_SECONDS - elapsed
        logger.info(f"Vision rate limiter: sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)
    _last_vision_call = time.time()


def describe_image(
    image_bytes: bytes,
    groq_api_key: str,
    prompt: str = "Describe this image in detail for a search index. Mention colors, objects, text, and any important visual elements."
) -> str:
    """
    Send an image to Groq's Llama 4 Scout Vision model and get a text description.
    
    Args:
        image_bytes: Raw image data (PNG, JPG, etc.)
        groq_api_key: Your Groq API key
        prompt: The question to ask about the image
        
    Returns:
        A text description of the image, or a placeholder if it fails.
    """
    _rate_limit_vision()
    
    try:
        # Preprocess image: resize if too large, convert to JPEG for compatibility
        from PIL import Image
        from io import BytesIO
        
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB (removes alpha channel, handles palette images)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Resize if too large (Groq has limits around 20MB, but smaller is faster)
        max_dimension = 1024
        if max(img.size) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logger.info(f"Vision: Resized image to {img.size}")
        
        # Convert to JPEG bytes
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        processed_bytes = buffer.getvalue()
        
        # Convert image to base64
        base64_image = base64.b64encode(processed_bytes).decode('utf-8')
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            description = response.json()['choices'][0]['message']['content']
            logger.info(f"Vision: Successfully described image ({len(description)} chars)")
            return description
        elif response.status_code == 429:
            logger.warning("Vision: Rate limited by Groq (429). Returning placeholder.")
            return "[Image description temporarily unavailable due to rate limits]"
        else:
            # Log the full error response for debugging
            error_detail = response.text[:500] if response.text else "No details"
            logger.warning(f"Vision: Groq returned {response.status_code}: {error_detail}")
            return "[Image description unavailable]"
            
    except Exception as e:
        logger.error(f"Vision: Error describing image: {e}")
        return "[Image description unavailable]"


def extract_images_from_pdf(file) -> List[Tuple[int, bytes]]:
    """
    Extract embedded images from a PDF file using PyMuPDF (low memory).
    
    Args:
        file: A file-like object (e.g., uploaded file)
        
    Returns:
        List of tuples: (page_number, image_bytes)
    """
    images = []
    try:
        import fitz  # PyMuPDF
        
        file.seek(0)
        pdf_bytes = file.read()
        file.seek(0)
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append((page_num + 1, image_bytes))
                    logger.info(f"PDF: Extracted image from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"PDF: Failed to extract image {img_index} from page {page_num + 1}: {e}")
        
        doc.close()
        
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed. Skipping PDF image extraction.")
    except Exception as e:
        logger.error(f"PDF image extraction failed: {e}")
    
    return images


def extract_images_from_docx(file) -> List[Tuple[str, bytes]]:
    """
    Extract embedded images from a Word document.
    
    Args:
        file: A file-like object (e.g., uploaded file)
        
    Returns:
        List of tuples: (image_name, image_bytes)
    """
    images = []
    try:
        from zipfile import ZipFile
        from io import BytesIO
        
        file.seek(0)
        docx_bytes = file.read()
        file.seek(0)
        
        # Supported image extensions (including vector formats)
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.emf', '.wmf']
        
        with ZipFile(BytesIO(docx_bytes)) as zf:
            # Debug: Log all files in word/media/
            media_files = [n for n in zf.namelist() if 'media' in n.lower()]
            logger.info(f"DOCX: Found {len(media_files)} files in media folders: {media_files[:10]}...")
            
            for name in zf.namelist():
                # Check both word/media/ and other media locations
                if 'media/' in name.lower():
                    ext = '.' + name.lower().split('.')[-1] if '.' in name else ''
                    if ext in image_extensions:
                        image_bytes = zf.read(name)
                        images.append((name, image_bytes))
                        logger.info(f"DOCX: Extracted image {name} ({len(image_bytes)} bytes)")
        
        if not images:
            logger.warning("DOCX: No images found in word/media/ folder")
        
    except Exception as e:
        logger.error(f"DOCX image extraction failed: {e}")
    
    return images


def extract_images_from_pptx(file) -> List[Tuple[str, bytes]]:
    """
    Extract embedded images from a PowerPoint presentation.
    
    Args:
        file: A file-like object (e.g., uploaded file)
        
    Returns:
        List of tuples: (image_name, image_bytes)
    """
    images = []
    try:
        from zipfile import ZipFile
        from io import BytesIO
        
        file.seek(0)
        pptx_bytes = file.read()
        file.seek(0)
        
        with ZipFile(BytesIO(pptx_bytes)) as zf:
            for name in zf.namelist():
                if name.startswith('ppt/media/') and any(name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                    image_bytes = zf.read(name)
                    images.append((name, image_bytes))
                    logger.info(f"PPTX: Extracted image {name}")
        
    except Exception as e:
        logger.error(f"PPTX image extraction failed: {e}")
    
    return images


def get_visual_context(file, groq_api_key: str, max_images: int = 20) -> str:
    """
    Extract and describe all images from a document.
    
    This is the main entry point for Visual RAG.
    
    Args:
        file: Uploaded file object
        groq_api_key: Your Groq API key
        max_images: Maximum number of images to process (to limit API calls)
        
    Returns:
        A string containing all image descriptions, ready to be appended to document text.
    """
    ext = file.name.lower().split('.')[-1]
    images = []
    
    # Extract images based on file type
    if ext == 'pdf':
        images = extract_images_from_pdf(file)
    elif ext == 'docx':
        images = extract_images_from_docx(file)
    elif ext == 'pptx':
        images = extract_images_from_pptx(file)
    else:
        logger.info(f"No image extraction support for .{ext} files")
        return ""
    
    if not images:
        logger.info("No images found in document")
        return ""
    
    # Limit number of images to process
    if len(images) > max_images:
        logger.info(f"Limiting image processing from {len(images)} to {max_images}")
        images = images[:max_images]
    
    # Describe each image
    descriptions = []
    for i, img_data in enumerate(images):
        location, image_bytes = img_data
        logger.info(f"Describing image {i+1}/{len(images)} from {location}")
        description = describe_image(image_bytes, groq_api_key)
        
        if ext == 'pdf':
            descriptions.append(f"[Image on page {location}]: {description}")
        else:
            descriptions.append(f"[Image {i+1}]: {description}")
    
    if descriptions:
        visual_context = "\n\n=== VISUAL CONTENT ===\n" + "\n\n".join(descriptions)
        logger.info(f"Generated visual context: {len(visual_context)} chars from {len(descriptions)} images")
        return visual_context
    
    return ""
