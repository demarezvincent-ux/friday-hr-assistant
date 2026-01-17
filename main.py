import streamlit as st
from supabase import create_client, Client
import requests
import pdfplumber
import docx
import re
import time
import os
import datetime
import uuid
import logging
from huggingface_hub import InferenceClient
from services.rag_controller import get_context_with_strategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION & PREMIUM STYLING ---
st.set_page_config(page_title="FRIDAY", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* ============================================
       FRIDAY - Premium Design System v2.0
       ============================================ */
    
    /* 1. Typography */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #1A3C34;
        --primary-hover: #0F2921;
        --primary-gradient: linear-gradient(135deg, #1A3C34 0%, #0F2921 100%);
        --bg-primary: #F5F5F7;
        --bg-secondary: #FAFAFA;
        --bg-white: #FFFFFF;
        --sidebar-bg: rgba(245, 245, 247, 0.85);
        --text-primary: #1D1D1F;
        --text-secondary: #5C5C61;
        --text-tertiary: #86868B;
        --error: #FF3B30;
        --success: #34C759;
        --divider: rgba(0, 0, 0, 0.08);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.12);
        --transition-normal: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }

    /* ============================================
       2. GLASS-MORPHISM SIDEBAR
       ============================================ */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.3);
        width: 300px !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.06);
    }

    /* Sidebar content spacing - prevent overlap */
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1rem;
    }

    /* General Button Styling */
    [data-testid="stSidebar"] .stButton button {
        border: none;
        text-align: left;
        padding: 12px 16px;
        height: 44px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 500;
        transition: var(--transition-smooth);
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* --- ACTIVE STATE (Selected) --- */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: var(--primary-gradient) !important;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(26, 60, 52, 0.4);
        border: none !important;
    }

    /* White text on active buttons */
    div[data-testid="stSidebar"] button[kind="primary"] *,
    div[data-testid="stSidebar"] button[kind="primary"] p,
    div[data-testid="stSidebar"] button[kind="primary"] span,
    div[data-testid="stSidebar"] button[kind="primary"] div {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* --- INACTIVE STATE (Not Selected) --- */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #1D1D1F !important;
        border: 1px solid transparent !important;
    }

    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(26, 60, 52, 0.08) !important;
        transform: translateX(4px);
    }

    /* Black text on inactive buttons */
    div[data-testid="stSidebar"] button[kind="secondary"] *,
    div[data-testid="stSidebar"] button[kind="secondary"] p,
    div[data-testid="stSidebar"] button[kind="secondary"] span,
    div[data-testid="stSidebar"] button[kind="secondary"] div {
        color: #1D1D1F !important;
        -webkit-text-fill-color: #1D1D1F !important;
    }

    /* Sidebar Dividers */
    [data-testid="stSidebar"] hr {
        margin: 20px 0;
        border: none;
        border-top: 1px solid var(--divider);
    }

    /* Section headers */
    [data-testid="stSidebar"] h3 {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-tertiary);
        margin: 20px 0 12px 0;
        padding-left: 4px;
    }

    /* ============================================
       3. MAIN UI COMPONENTS
       ============================================ */
    
    /* Hero/Greeting */
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 56px; font-weight: 700; color: var(--text-primary);
        line-height: 1.1; letter-spacing: -1.5px; margin-bottom: 16px;
    }
    .greeting-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 48px; font-weight: 800; color: var(--text-primary);
        margin-bottom: 12px; letter-spacing: -1px;
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: var(--bg-white); border: 1.5px solid #D2D2D7;
        border-radius: 12px; height: 52px; padding: 0 18px; font-size: 17px;
        transition: var(--transition-smooth);
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 4px rgba(26, 60, 52, 0.1);
    }
    .stChatInput > div > div > textarea {
        background-color: var(--bg-primary); border-radius: 14px;
        min-height: 60px; padding: 18px 56px 18px 20px; font-size: 17px;
        transition: var(--transition-smooth);
    }
    .stChatInput > div > div > textarea:focus {
        background-color: var(--bg-white); border-color: var(--primary) !important;
        box-shadow: 0 0 0 5px rgba(26, 60, 52, 0.12) !important;
    }

    /* Chat Messages with enhanced animations */
    [data-testid="stChatMessage"][data-testid*="user"] > div {
        background: var(--primary-gradient); color: white;
        border-radius: 20px 20px 4px 20px; padding: 14px 20px; max-width: 70%;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] > div {
        background-color: var(--bg-white); color: var(--text-primary);
        border-radius: 18px 18px 18px 4px; padding: 16px 20px; max-width: 70%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }

    /* Source Tags */
    .source-tag {
        font-size: 13px; font-weight: 600; background-color: var(--bg-white);
        border: 1px solid #D2D2D7; padding: 8px 14px; border-radius: 99px;
        color: var(--text-secondary); display: inline-flex; align-items: center;
        gap: 6px; margin-right: 8px; margin-top: 8px;
        transition: var(--transition-normal);
        cursor: pointer;
    }
    .source-tag:hover {
        background-color: var(--bg-primary);
        border-color: var(--primary);
        color: var(--primary);
        transform: translateY(-2px);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 3px dashed #D2D2D7; background-color: var(--bg-secondary);
        border-radius: 16px; padding: 3rem;
        transition: var(--transition-smooth);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background-color: rgba(26, 60, 52, 0.02);
    }
    .file-item {
        display: flex; align-items: center; padding: 14px 18px;
        border-radius: 10px; border: 1px solid transparent; 
        transition: var(--transition-smooth);
        gap: 12px;
    }
    .file-item:hover { 
        background-color: rgba(26, 60, 52, 0.04); 
        transform: translateX(4px);
    }

    /* Login Container */
    .login-container {
        background: var(--bg-white); border-radius: 24px; padding: 56px 48px;
        max-width: 480px; margin: 0 auto; 
        box-shadow: 0 16px 64px rgba(0, 0, 0, 0.12);
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* ============================================
       4. ENHANCED ANIMATIONS
       ============================================ */
    @keyframes fadeIn { 
        from { opacity: 0; } 
        to { opacity: 1; } 
    }
    @keyframes fadeInUp { 
        from { opacity: 0; transform: translateY(20px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    @keyframes slideInUp { 
        from { opacity: 0; transform: translateY(16px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-16px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Custom Loading Spinner */
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.15); opacity: 0.6; }
    }
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-6px); }
    }
    
    .thinking-spinner {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 0;
    }
    .thinking-spinner span {
        width: 8px;
        height: 8px;
        background: var(--primary);
        border-radius: 50%;
        animation: bounce 1.4s ease-in-out infinite;
    }
    .thinking-spinner span:nth-child(2) { animation-delay: 0.16s; }
    .thinking-spinner span:nth-child(3) { animation-delay: 0.32s; }
    
    .thinking-text {
        font-size: 15px;
        color: var(--text-secondary);
        font-style: italic;
        margin-left: 8px;
    }
    
    /* Apply animations */
    .stApp { animation: fadeIn 0.5s ease-out; }
    
    [data-testid="stChatMessage"] {
        animation: slideInUp 0.35s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }
    
    /* Button hover - professional, no jittering */
    button {
        transition: var(--transition-smooth);
    }
    button:hover {
        transform: scale(1.015);
    }
    button:active {
        transform: scale(0.98);
    }
    
    /* Logo */
    .logo-animated { 
        transition: transform 0.3s ease;
    }
    .logo-animated:hover {
        transform: scale(1.03);
    }

    /* ============================================
       5. DOCUMENT VIEWER MODAL
       ============================================ */
    .doc-viewer {
        background: var(--bg-white);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--divider);
        font-family: 'Inter', monospace;
        font-size: 14px;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    
    /* Web source styling */
    .web-source {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* ============================================
       6. PREVENT TEXT/ICON OVERLAP
       ============================================ */
    /* Ensure proper spacing in sidebar buttons */
    [data-testid="stSidebar"] .stButton button > div {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
    }
    
    /* Delete buttons - smaller, aligned */
    [data-testid="stSidebar"] button[key*="del_"] {
        min-width: 32px;
        max-width: 32px;
        padding: 8px;
        justify-content: center;
    }
    
    /* Recent chat items - proper truncation */
    .chat-item-title {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 180px;
    }
</style>
""", unsafe_allow_html=True)


# --- SECRETS HANDLING ---
def get_secret(key_name):
    if key_name in os.environ: return os.environ[key_name]
    try:
        if key_name in st.secrets: return st.secrets[key_name]
    except:
        pass
    return None

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
FIXED_GROQ_KEY = get_secret("FIXED_GROQ_KEY")
HF_API_KEY = get_secret("HF_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, FIXED_GROQ_KEY, HF_API_KEY]):
    st.error("❌ Missing API Keys. Please check your Secrets or Environment Variables.")
    st.stop()

# --- 2. STATE ---
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "company_id" not in st.session_state: st.session_state.company_id = None
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
if "view" not in st.session_state: st.session_state.view = "chat"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- 3. BACKEND LOGIC ---

def get_embeddings_batch(texts):
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    client = InferenceClient(token=HF_API_KEY)
    clean_texts = [t.replace("\n", " ").strip() for t in texts]
    backoff_times = [2, 4, 8, 16]

    for wait_time in backoff_times:
        try:
            embeddings = client.feature_extraction(clean_texts, model=model_id)
            if hasattr(embeddings, "tolist"): return embeddings.tolist()
            return embeddings
        except:
            time.sleep(wait_time)
    return None

def sanitize_filename(filename):
    name = filename.replace(" ", "_")
    return re.sub(r'[^a-zA-Z0-9._-]', '', name)

def normalize_query(query):
    """Normalize query but keep original intent."""
    return query.strip().lower()

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        clean_table = [[(str(cell) if cell else "").replace("\n", " ") for cell in row] for row in table]
                        if clean_table:
                            try:
                                header = "| " + " | ".join(clean_table[0]) + " |"
                                sep = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"
                                body = "\n".join(["| " + " | ".join(row) + " |" for row in clean_table[1:]])
                                text += f"\n{header}\n{sep}\n{body}\n\n"
                            except: pass
                page_text = page.extract_text()
                if page_text: text += page_text + "\n"
    except: return ""
    return text

# --- NEW: Excel and PowerPoint handlers ---
def extract_text_from_excel(file) -> str:
    """Extract text from .xlsx files using pandas."""
    try:
        import pandas as pd
        file.seek(0)
        dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        text_parts = []
        for sheet_name, df in dfs.items():
            text_parts.append(f"## Sheet: {sheet_name}\n")
            # Convert to markdown table format
            if not df.empty:
                text_parts.append(df.to_markdown(index=False))
        file.seek(0)
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Excel extraction failed: {e}")
        return ""

def extract_text_from_pptx(file) -> str:
    """Extract text from .pptx files."""
    try:
        from pptx import Presentation
        file.seek(0)
        prs = Presentation(file)
        text_parts = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = [f"## Slide {slide_num}"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            text_parts.append("\n".join(slide_text))
        file.seek(0)
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"PPTX extraction failed: {e}")
        return ""

def extract_file_metadata(file) -> dict:
    """Extract metadata from file properties."""
    metadata = {
        "title": file.name,
        "created_date": None,
        "author": None
    }
    try:
        file.seek(0)
        if file.name.endswith(".docx"):
            doc = docx.Document(file)
            props = doc.core_properties
            metadata["author"] = props.author if props.author else None
            metadata["created_date"] = str(props.created) if props.created else None
            metadata["title"] = props.title if props.title else file.name
            file.seek(0)
        elif file.name.endswith(".pptx"):
            from pptx import Presentation
            prs = Presentation(file)
            props = prs.core_properties
            metadata["author"] = props.author if props.author else None
            metadata["created_date"] = str(props.created) if props.created else None
            file.seek(0)
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
    return metadata

# --- IMPROVED: Recursive Character Splitter ---
def recursive_chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """
    Recursive Character Splitter: Split by paragraphs first, then sentences.
    Preserves semantic meaning better than fixed-window chunking.
    """
    if not text:
        return []
    
    # Step 1: Split by double newlines (paragraphs)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) + 2 < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too large, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk = ""
                for sent in sentences:
                    if len(sentence_chunk) + len(sent) + 1 < chunk_size:
                        sentence_chunk += sent + " "
                    else:
                        if sentence_chunk:
                            chunks.append(sentence_chunk.strip())
                        sentence_chunk = sent + " "
                if sentence_chunk:
                    current_chunk = sentence_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Legacy alias for compatibility
def smart_chunking(text, chunk_size=500, overlap=100):
    return recursive_chunk_text(text, chunk_size, overlap)

# --- CACHING for embeddings ---
@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_embeddings(text_tuple: tuple):
    """Cache embeddings (tuple for hashability)."""
    return get_embeddings_batch(list(text_tuple))

# --- DATABASE OPERATIONS ---

def register_document(filename, company_id, metadata=None):
    try:
        doc_data = {"company_id": company_id, "filename": filename, "is_active": True}
        if metadata:
            doc_data["title"] = metadata.get("title")
            doc_data["author"] = metadata.get("author")
        supabase.table("documents").insert(doc_data).execute()
        return True
    except Exception as e:
        logger.warning(f"Register document failed: {e}")
        return False

def check_if_document_exists(filename, company_id):
    try:
        res = supabase.table("documents").select("id").eq("company_id", company_id).eq("filename", filename).execute()
        return len(res.data) > 0
    except: return False

def get_all_documents(company_id):
    try:
        return supabase.table("documents").select("*").eq("company_id", company_id).order('is_active', desc=True).order('created_at', desc=True).execute().data
    except: return []

def toggle_document_status(filename, company_id, current_status):
    try:
        supabase.table("documents").update({"is_active": not current_status}).eq("company_id", company_id).eq("filename", filename).execute()
        return True
    except: return False

# --- IMPROVED: Cascade Delete ---
def delete_document(filename, company_id):
    """Cascade delete: file record + chunks + storage bucket."""
    try:
        # 1. Delete from documents table
        supabase.table("documents").delete().eq("company_id", company_id).eq("filename", filename).execute()
        
        # 2. Delete all vector chunks (both filter syntaxes for compatibility)
        try:
            supabase.table("document_chunks").delete().eq("metadata->>company_id", company_id).eq("metadata->>filename", filename).execute()
        except:
            supabase.table("document_chunks").delete().filter("metadata->>company_id", "eq", company_id).filter("metadata->>filename", "eq", filename).execute()
        
        # 3. Delete from storage bucket
        try:
            supabase.storage.from_("documents").remove([f"{company_id}/{filename}"])
        except Exception as e:
            logger.info(f"Storage delete (may not exist): {e}")
        
        logger.info(f"Cascade delete successful: {filename}")
        return True
    except Exception as e:
        logger.error(f"Cascade delete failed: {e}")
        return False

# --- IMPROVED: Document processor with new file types ---
def process_and_store_document(file, company_id, force_overwrite=False):
    """Process and store document with support for PDF, DOCX, XLSX, PPTX."""
    clean_name = sanitize_filename(file.name)
    
    if check_if_document_exists(clean_name, company_id):
        if not force_overwrite:
            return "exists"
        else:
            delete_document(clean_name, company_id)

    # Extract text based on file type
    text = ""
    file_metadata = {"title": clean_name}
    
    try:
        file_ext = file.name.lower()
        
        if file_ext.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file_ext.endswith(".docx"):
            file_metadata = extract_file_metadata(file)
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_ext.endswith(".xlsx"):
            text = extract_text_from_excel(file)
        elif file_ext.endswith(".pptx"):
            file_metadata = extract_file_metadata(file)
            text = extract_text_from_pptx(file)
        else:
            return "unsupported"
            
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return "error"
    
    if not text or len(text.strip()) < 10:
        return "empty"

    # Upload to storage
    try:
        file.seek(0)
        supabase.storage.from_("documents").upload(f"{company_id}/{clean_name}", file.read(), {"upsert": "true"})
    except Exception as e:
        logger.warning(f"Storage upload failed (continuing): {e}")

    # Use recursive chunking
    chunks = recursive_chunk_text(text)
    if not chunks:
        return "empty"
    
    # Process in batches with progress
    for i in range(0, len(chunks), 20):
        batch = chunks[i:i+20]
        vectors = get_embeddings_batch(batch)
        
        if vectors:
            payload = []
            for j, vec in enumerate(vectors):
                if isinstance(vec, list) and len(vec) > 300:
                    chunk_metadata = {
                        "company_id": company_id, 
                        "filename": clean_name, 
                        "is_active": True,
                        "chunk_index": i + j,
                        "title": file_metadata.get("title"),
                        "author": file_metadata.get("author")
                    }
                    payload.append({
                        "content": batch[j],
                        "metadata": chunk_metadata,
                        "embedding": vec
                    })
            if payload:
                supabase.table("document_chunks").insert(payload).execute()
    
    register_document(clean_name, company_id, file_metadata)
    return "success"

# --- [FIX 2] INTELLIGENT SEARCH EXPANSION ---
def get_relevant_context(query, company_id):
    normalized_query = normalize_query(query)
    search_query = normalized_query
    
    try:
        # EXPANSION PROMPT: Explicitly ask for split and joined variations
        # This fixes "koffie machine" vs "koffiemachine" vs "coffee machine"
        expansion_prompt = (
            f"You are a search optimizer. User Query: '{normalized_query}'. "
            f"Generate a boolean search string that includes: "
            f"1. English, Dutch, and French translations. "
            f"2. CRITICAL: For every compound word, output BOTH the joined version (e.g. 'koffiemachine') AND the split version (e.g. 'coffee machine'). "
            f"Output ONLY the terms separated by ' OR '."
        )
        
        expansion_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for model in expansion_models:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
                json={"model": model, "messages": [{"role": "user", "content": expansion_prompt}], "temperature": 0.1, "max_tokens": 100},
                timeout=10
            )
            if resp.status_code == 200:
                search_query = resp.json()['choices'][0]['message']['content']
                break
            elif resp.status_code == 429: continue
            else: break
    except: pass

    # Get Vector Embedding for the ORIGINAL normalized query (to capture intent)
    vectors = get_embeddings_batch([normalized_query])
    if not vectors: return "", []

    try:
        # Use the EXPANDED query for Full Text Search (FTS) to catch the English keywords
        params = {"query_embedding": vectors[0], "match_threshold": 0.15, "match_count": 15, "filter_company_id": company_id, "query_text": search_query}
        res = supabase.rpc("match_documents_hybrid", params).execute()
        context_str = ""
        sources = []
        for m in res.data:
            context_str += f"-- SOURCE: {m['metadata']['filename']} --\n{m['content']}\n\n"
            if m['metadata']['filename'] not in sources: sources.append(m['metadata']['filename'])
        return context_str, sources
    except: return "", []

def ask_groq(context, history, query):
    """Ask Groq LLM with precision citation support."""
    system_prompt = """You are FRIDAY, an expert HR assistant.
Answer questions based ONLY on the provided CONTEXT.
IMPORTANT: Reference sources using [Doc1], [Doc2], etc. when citing information.
If the context contains Markdown tables, format your response appropriately.
If you don't know the answer based on the context, say so clearly."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (last 4 exchanges)
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add context if available
    if context:
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
    
    messages.append({"role": "user", "content": query})

    # Try models with fallback
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    backoff_times = [1, 2, 4]
    
    for model in models:
        for wait_time in backoff_times:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
                    json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 1500},
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()['choices'][0]['message']['content']
                elif resp.status_code == 429:
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error: {resp.status_code}"
            except requests.Timeout:
                return "⚠️ Request timed out. Please try again."
            except Exception as e:
                logger.error(f"Groq API error: {e}")
                return "Connection error. Please try again."
    
    return "⚠️ High traffic. Please try again in a moment."

# --- CHAT HISTORY & PERSISTENCE ---

def load_chat_history(chat_id):
    try: return supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute().data
    except: return []

def save_message(chat_id, role, content, company_id, sources=None):
    try: 
        supabase.table("messages").insert({
            "chat_id": chat_id, "role": role, "content": content, 
            "sources": sources, "company_id": company_id 
        }).execute()
    except: pass

def get_recent_chats(company_id):
    try:
        res = supabase.table("messages").select("chat_id, content, created_at").eq("company_id", company_id).order("created_at", desc=True).limit(50).execute()
        seen_ids = set()
        unique_chats = []
        for msg in res.data:
            if msg['chat_id'] not in seen_ids:
                seen_ids.add(msg['chat_id'])
                unique_chats.append({"id": msg['chat_id'], "title": msg['content'][:30] + "..."})
        return unique_chats[:10]
    except: return []

def delete_chat(chat_id, company_id):
    try:
        supabase.table("messages").delete().eq("chat_id", chat_id).eq("company_id", company_id).execute()
        return True
    except: return False

def get_dynamic_greeting():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12: return "Good morning."
    elif 12 <= hour < 17: return "Good afternoon."
    elif 17 <= hour < 21: return "Good evening."
    else: return "Hello."

# --- UI PAGES ---
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100px; margin-bottom: 28px;"><span class="logo-animated" style="font-family: \'Playfair Display\', serif; font-size: 48px; font-weight: 800; color: #1A3C34; letter-spacing: -1.2px;">Friday</span></div>', unsafe_allow_html=True)

        # Navigation
        st.markdown("### Menu")
        # [FIX 1] Button selection logic is correct here, CSS handles the color
        if st.button("◉  Chat", use_container_width=True, type="primary" if st.session_state.view == "chat" else "secondary"):
            st.session_state.view = "chat"; st.rerun()
        if st.button("◎  Documents", use_container_width=True, type="primary" if st.session_state.view == "documents" else "secondary"):
            st.session_state.view = "documents"; st.rerun()

        st.markdown("---")

        if st.session_state.view == "chat":
            if st.button("＋  New Chat", use_container_width=True, type="secondary"): 
                create_new_chat(); st.rerun()

            st.markdown("### Recent Chats")
            recent = get_recent_chats(st.session_state.company_id)
            if not recent:
                st.caption("No conversations yet")
            else:
                for chat in recent:
                    is_active = st.session_state.current_chat_id == chat['id']
                    with st.container():
                        col1, col2 = st.columns([6, 1])
                        with col1:
                            # Dynamic button type ensures CSS triggers White or Black text
                            btn_type = "primary" if is_active else "secondary"
                            if st.button(f"{chat['title']}", key=f"chat_{chat['id']}", use_container_width=True, type=btn_type):
                                st.session_state.current_chat_id = chat['id']
                                st.rerun()
                        with col2:
                            if st.button("×", key=f"del_{chat['id']}"):
                                delete_chat(chat['id'], st.session_state.company_id)
                                if st.session_state.current_chat_id == chat['id']: create_new_chat()
                                st.rerun()

        st.markdown("---")
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("↪  Sign Out", use_container_width=True): 
            st.session_state.clear(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def create_new_chat(): st.session_state.current_chat_id = str(uuid.uuid4())

def handle_query(query):
    """Handle user query with custom loading spinner and elite RAG pipeline."""
    save_message(st.session_state.current_chat_id, "user", query, st.session_state.company_id)
    with st.chat_message("user", avatar="👤"):
        st.write(query)
    
    with st.chat_message("assistant", avatar="⚡"):
        message_placeholder = st.empty()
        
        # Custom thinking spinner with animated dots
        spinner_html = '''
        <div style="display: flex; align-items: center; padding: 12px 0;">
            <div class="thinking-spinner">
                <span></span><span></span><span></span>
            </div>
            <span class="thinking-text">Thinking...</span>
        </div>
        '''
        message_placeholder.markdown(spinner_html, unsafe_allow_html=True)
        
        history = load_chat_history(st.session_state.current_chat_id)
        
        # Elite RAG Pipeline with Query Router + Reranker
        context, sources = get_context_with_strategy(
            raw_query=query,
            company_id=st.session_state.company_id,
            supabase=supabase,
            groq_api_key=FIXED_GROQ_KEY,
            get_embeddings_fn=get_embeddings_batch,
            hf_api_key=HF_API_KEY,  # Enable HuggingFace reranker
            match_count=40,
            top_k=5
        )
        
        response = ask_groq(context, history, query)
        
        save_message(st.session_state.current_chat_id, "assistant", response, st.session_state.company_id, sources)
        message_placeholder.empty()
    
    st.rerun()

def chat_page():
    if not st.session_state.current_chat_id: create_new_chat()
    history = load_chat_history(st.session_state.current_chat_id)

    if not history:
        greeting = get_dynamic_greeting()
        st.markdown(f"""
        <div style="text-align: center; margin-top: 80px; margin-bottom: 48px;">
            <h1 class="greeting-title">{greeting}</h1>
            <p class="greeting-subtitle">How can FRIDAY help you with HR tasks today?</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in history:
        with st.chat_message(msg["role"], avatar="⚡" if msg["role"] == "assistant" else None):
            st.write(msg["content"])
            if msg["sources"]:
                tags = "".join([f"<div class='source-tag'>📄 {s}</div>" for s in msg["sources"]])
                st.markdown(f"<div class='source-container'>{tags}</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Ask FRIDAY anything..."):
        handle_query(prompt)

def documents_page():
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h1 style="font-family: 'Playfair Display', serif; font-size: 42px; font-weight: 800;">Knowledge Base</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("### Upload Documents")
        # Support for all file types
        uploaded_files = st.file_uploader(
            "Drag and drop files here", 
            type=["pdf", "docx", "xlsx", "pptx"], 
            accept_multiple_files=True,
            help="Supported: PDF, Word, Excel, PowerPoint"
        )
        
        c_check, c_btn = st.columns([1, 1])
        with c_check:
            force_overwrite = st.checkbox("Overwrite existing?")
        with c_btn:
            if uploaded_files and st.button("Start Indexing", type="primary", use_container_width=True):
                # Process all files first, then show results (bulk upload fix)
                results = {"success": 0, "exists": 0, "error": 0, "empty": 0, "unsupported": 0}
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                for idx, f in enumerate(uploaded_files):
                    status_container.text(f"Processing {f.name}...")
                    res = process_and_store_document(f, st.session_state.company_id, force_overwrite)
                    results[res] = results.get(res, 0) + 1
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()
                status_container.empty()
                
                # Show summary
                if results["success"] > 0:
                    st.success(f"✅ {results['success']} document(s) indexed successfully!")
                if results["exists"] > 0:
                    st.info(f"ℹ️ {results['exists']} document(s) already exist (skipped)")
                if results["error"] > 0:
                    st.error(f"❌ {results['error']} document(s) failed to process")
                if results["empty"] > 0:
                    st.warning(f"⚠️ {results['empty']} document(s) were empty")
                
                time.sleep(2)
                st.rerun()

    with col2:
        docs = get_all_documents(st.session_state.company_id)
        st.markdown(f"### Indexed Files ({len(docs)})")
        if not docs: st.info("No documents yet.")
        else:
            for doc in docs:
                status_color = "#34C759" if doc['is_active'] else "#86868B"
                with st.container():
                    c1, c2, c3 = st.columns([6, 1, 1])
                    with c1: st.markdown(f"<div class='file-item'><span style='color:{status_color}; margin-right:10px'>●</span>{doc['filename']}</div>", unsafe_allow_html=True)
                    with c2: 
                        if st.button("⏸", key=f"arch_{doc['id']}"): 
                            toggle_document_status(doc['filename'], st.session_state.company_id, doc['is_active']); st.rerun()
                    with c3:
                        if st.button("×", key=f"del_{doc['id']}"):
                            delete_document(doc['filename'], st.session_state.company_id); st.rerun()

# --- 5. AUTHENTICATION ---
def handle_login():
    """Callback for login form - handles authentication state."""
    pw = st.session_state.get("login_password", "")
    if not pw:
        st.session_state.login_error = "Please enter an access code"
        return
    
    try:
        res = supabase.table('clients').select("*").eq('access_code', pw).execute()
        if res.data:
            st.session_state.authenticated = True
            st.session_state.company_id = res.data[0]['company_id']
            st.session_state.login_error = None
        else:
            st.session_state.login_error = "Invalid Code"
    except Exception as e:
        logger.error(f"Login error: {e}")
        st.session_state.login_error = "Login Error"

def login_page():
    """Render the login page with fixed single-click authentication."""
    # Initialize login error state
    if "login_error" not in st.session_state:
        st.session_state.login_error = None
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 48px;">
            <div style="font-family: 'Playfair Display', serif; font-size: 72px; font-weight: 800; color: #1A3C34;">Friday</div>
            <h1 class="hero-title" style="font-size: 42px;">Your Intelligent<br><em>HR Companion</em></h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Use text_input with key for callback access
        pw = st.text_input("Access Code", type="password", key="login_password")
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Button with on_click callback for immediate state update
        if st.button("Sign In", use_container_width=True, type="primary", on_click=handle_login):
            # The callback already handled authentication
            if st.session_state.authenticated:
                st.rerun()
        
        # Show error if present
        if st.session_state.login_error:
            st.error(st.session_state.login_error)

if not st.session_state.authenticated: login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat": chat_page()
    elif st.session_state.view == "documents": documents_page()