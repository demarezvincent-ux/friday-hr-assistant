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
import pandas as pd
from pptx import Presentation
from huggingface_hub import InferenceClient
from services.rag_controller import get_context_with_strategy

# Setup logging for debugging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- 1. CONFIGURATION & PREMIUM STYLING ---
st.set_page_config(page_title="FRIDAY", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* ============================================
       FRIDAY - Premium Design System v3.0
       ============================================ */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --primary: #1A3C34;
        --primary-light: #2A5248;
        --background: #F9F9F7;
        --sidebar-bg: #F0F0EE;
        --text-primary: #1A3C34;
        --text-secondary: #5C6F68;
        --white: #FFFFFF;
        --border: #E6E6E3;
    }

    /* Global Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        background-color: var(--background);
    }
    
    h1, h2, h3, h4, h5, h6, .playfair {
        font-family: 'Playfair Display', serif !important;
        color: var(--text-primary) !important;
    }

    /* Streamlit App Container */
    .stApp {
        background-color: var(--background);
    }

    /* ============================================
       SIDEBAR
       ============================================ */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid rgba(26, 60, 52, 0.06);
    }
    
    [data-testid="stSidebar"] hr {
        margin: 24px 0;
        border-color: rgba(26, 60, 52, 0.1) !important;
    }

    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent;
        color: var(--text-secondary);
        border: none;
        text-align: left;
        padding-left: 12px;
        font-weight: 500;
        transition: all 0.2s ease;
        justify-content: flex-start;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        color: var(--primary);
        background-color: rgba(26, 60, 52, 0.04);
        transform: translateX(4px);
    }

    /* Selected state for Sidebar Buttons handled by 'type="primary"' in Python 
       but let's override the primary style to fit the theme */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        color: var(--white) !important;
        box-shadow: 0 4px 12px rgba(26, 60, 52, 0.15);
        border-radius: 8px;
    }

    /* ============================================
       MAIN CONTENT & CHAT
       ============================================ */
    
    /* Input Fields */
    .stTextInput > div > div > input, 
    .stChatInput > div > div > textarea {
        background-color: var(--white);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text-primary);
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    
    .stTextInput > div > div > input:focus,
    .stChatInput > div > div > textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(26, 60, 52, 0.1);
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        gap: 1.5rem;
    }

    /* User Message */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: transparent; 
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: transparent;
    }

    /* Avatars */
    [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {
        border-radius: 50%;
        background-color: var(--primary) !important;
        color: white;
        padding: 4px;
    }

    /* ============================================
       FILE LIST & ICONS
       ============================================ */
    .file-list-card {
        background: var(--white);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }

    /* Center align control buttons - targeting the Documents page columns */
    /* This targets buttons inside the columns used for the file list */
    [data-testid="stVerticalBlock"] [data-testid="column"] button {
        margin: 0 auto;
        display: block;
    }

    /* Status Dot */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--primary);
    }

    /* ============================================
       ANIMATIONS
       ============================================ */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    
    .element-container, .stChatMessage {
        animation: fadeIn 0.4s ease-out;
    }

    .logo-animated {
        animation: fadeIn 1s ease-out;
    }

    /* Thinking Spinner */
    .thinking-spinner { display: flex; gap: 6px; padding: 12px 0; align-items: center; }
    .thinking-spinner span {
        width: 6px; height: 6px; background: var(--primary);
        border-radius: 50%; opacity: 0.6;
        animation: pulse 1.4s infinite ease-in-out;
    }
    .thinking-spinner span:nth-child(1) { animation-delay: -0.32s; }
    .thinking-spinner span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes pulse { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #D2D2D7; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #86868B; }

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

def extract_text_from_excel(file) -> str:
    """Extract text from .xlsx files using pandas."""
    try:
        file.seek(0)
        dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        text_parts = []
        for sheet_name, df in dfs.items():
            text_parts.append(f"## Sheet: {sheet_name}\n")
            if not df.empty:
                text_parts.append(df.to_markdown(index=False))
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Excel extraction failed: {e}")
        return ""

def extract_text_from_pptx(file) -> str:
    """Extract text from .pptx files slides."""
    try:
        file.seek(0)
        prs = Presentation(file)
        text_parts = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = [f"## Slide {slide_num}"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            text_parts.append("\n".join(slide_text))
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"PPTX extraction failed: {e}")
        return ""

def extract_file_metadata(file) -> dict:
    """Extract metadata from file properties."""
    meta = {"title": file.name, "created_date": None, "author": None}
    try:
        file.seek(0)
        fname_lower = file.name.lower()
        if fname_lower.endswith(".docx"):
            doc = docx.Document(file)
            props = doc.core_properties
            meta["author"] = props.author if props.author else None
            meta["title"] = props.title if props.title else file.name
        elif fname_lower.endswith(".pptx"):
            prs = Presentation(file)
            props = prs.core_properties
            meta["author"] = props.author if props.author else None
            meta["title"] = props.title if props.title else file.name
        elif fname_lower.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                md = pdf.metadata
                if md:
                    meta["author"] = md.get("Author")
                    meta["title"] = md.get("Title") if md.get("Title") else file.name
    except: pass
    file.seek(0)
    return meta

def recursive_chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """Recursive Character Splitter for semantic preservation."""
    if not text: return []
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        if len(current_chunk) + len(para) + 2 < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            if len(para) > chunk_size:
                parts = [para[i:i+chunk_size] for i in range(0, len(para), chunk_size-overlap)]
                chunks.extend(parts)
                current_chunk = ""
            else:
                current_chunk = para + "\n\n"
    if current_chunk.strip(): chunks.append(current_chunk.strip())
    return chunks

# Legacy alias
def smart_chunking(text, chunk_size=500, overlap=100):
    return recursive_chunk_text(text, chunk_size, overlap)

# --- DATABASE OPERATIONS ---

def register_document(filename, company_id, metadata=None):
    try:
        doc_data = {"company_id": company_id, "filename": filename, "is_active": True}
        if metadata:
            if metadata.get("title"):
                doc_data["title"] = metadata.get("title")
            if metadata.get("author"):
                doc_data["author"] = metadata.get("author")
        result = supabase.table("documents").insert(doc_data).execute()
        logger.info(f"Document registered: {filename}, result: {result.data}")
        return True
    except Exception as e:
        logger.error(f"Failed to register document {filename}: {e}")
        return False

def check_if_document_exists(filename, company_id):
    try:
        res = supabase.table("documents").select("id").eq("company_id", company_id).eq("filename", filename).execute()
        return len(res.data) > 0
    except: return False

def get_all_documents(company_id):
    try:
        result = supabase.table("documents").select("*").eq("company_id", company_id).order('is_active', desc=True).order('created_at', desc=True).execute()
        logger.info(f"Fetched {len(result.data)} documents for company {company_id}")
        return result.data
    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}")
        return []

def toggle_document_status(filename, company_id, current_status):
    try:
        supabase.table("documents").update({"is_active": not current_status}).eq("company_id", company_id).eq("filename", filename).execute()
        return True
    except: return False

def delete_document(filename, company_id):
    try:
        supabase.table("documents").delete().eq("company_id", company_id).eq("filename", filename).execute()
        supabase.table("document_chunks").delete().eq("metadata->>company_id", company_id).eq("metadata->>filename", filename).execute()
        try: supabase.storage.from_("documents").remove([f"{company_id}/{filename}"])
        except: pass
        return True
    except: return False

def process_and_store_document(file, company_id, force_overwrite=False):
    """Process and store document with support for PDF, DOCX, XLSX, PPTX."""
    clean_name = sanitize_filename(file.name)
    logger.info(f"Starting to process document: {clean_name}")
    
    if check_if_document_exists(clean_name, company_id):
        if not force_overwrite: 
            logger.info(f"Document already exists: {clean_name}")
            return "exists"
        else: 
            logger.info(f"Overwriting existing document: {clean_name}")
            delete_document(clean_name, company_id)

    text = ""
    file_metadata = {"title": clean_name}
    try:
        ext = file.name.lower().split('.')[-1]
        logger.info(f"Processing file type: {ext}")
        if ext == "pdf":
            file_metadata = extract_file_metadata(file)
            text = extract_text_from_pdf(file)
        elif ext == "docx":
            file_metadata = extract_file_metadata(file)
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "xlsx": text = extract_text_from_excel(file)
        elif ext == "pptx":
            file_metadata = extract_file_metadata(file)
            text = extract_text_from_pptx(file)
        else: 
            logger.warning(f"Unsupported file type: {ext}")
            return "unsupported"
    except Exception as e:
        logger.error(f"Extraction failed for {clean_name}: {e}")
        return "error"
    
    if not text or len(text.strip()) < 10: 
        logger.warning(f"Document empty or too short: {clean_name}")
        return "empty"
    
    logger.info(f"Extracted {len(text)} characters from {clean_name}")

    try:
        file.seek(0)
        supabase.storage.from_("documents").upload(f"{company_id}/{clean_name}", file.read(), {"upsert": "true"})
    except Exception as e:
        logger.warning(f"Storage upload failed (non-critical): {e}")

    chunks = recursive_chunk_text(text)
    logger.info(f"Created {len(chunks)} chunks from {clean_name}")
    
    chunks_stored = 0
    for i in range(0, len(chunks), 20):
        batch = chunks[i:i+20]
        vectors = get_embeddings_batch(batch)
        if vectors:
            payload = []
            for j, vec in enumerate(vectors):
                if isinstance(vec, list) and len(vec) > 300:
                    payload.append({
                        "content": batch[j],
                        "metadata": {
                            "company_id": company_id, "filename": clean_name, "is_active": True,
                            "title": file_metadata.get("title"), "author": file_metadata.get("author")
                        },
                        "embedding": vec
                    })
            if payload: 
                try:
                    supabase.table("document_chunks").insert(payload).execute()
                    chunks_stored += len(payload)
                except Exception as e:
                    logger.error(f"Failed to store chunks batch: {e}")
        else:
            logger.warning(f"Embedding generation returned None for batch starting at {i}")
    
    logger.info(f"Stored {chunks_stored} chunks for {clean_name}")
    
    if register_document(clean_name, company_id, file_metadata):
        logger.info(f"Successfully registered document: {clean_name}")
        return "success"
    return "error"

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
    """Ask Groq LLM with focus on real document name citations."""
    system_prompt = """You are FRIDAY, an expert HR assistant.
Answer questions based ONLY on the provided CONTEXT.
IMPORTANT: Cite the source filename directly in your response (e.g., "according to internal_policy.pdf") 
rather than using [Doc1]. Reference filenames exactly as they appear in context headers.
If you don't know the answer, say so. Format tables in Markdown if present."""
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
    if context: messages.append({"role": "user", "content": f"CONTEXT (Filenames in headers):\n{context}"})
    messages.append({"role": "user", "content": query})

    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    for model in models:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
                json={"model": model, "messages": messages, "temperature": 0.1},
                timeout=30
            )
            if resp.status_code == 200: return resp.json()['choices'][0]['message']['content']
            elif resp.status_code == 429: continue
        except: pass
    return "⚠️ Service currently limited. Please try again."

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
        st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100px; margin-bottom: 28px;"><span class="logo-animated" style="font-family: \'Playfair Display\', serif; font-size: 56px; font-weight: 800; color: #1A3C34; letter-spacing: -1.2px;">Friday</span></div>', unsafe_allow_html=True)

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
    save_message(st.session_state.current_chat_id, "user", query, st.session_state.company_id)
    with st.chat_message("user", avatar="👤"): st.write(query)
    
    with st.chat_message("assistant", avatar="⚡"):
        msg_placeholder = st.empty()
        # Custom thinking spinner
        spinner_html = '''
        <div class="thinking-spinner">
            <span></span><span></span><span></span>
            <span class="thinking-text">Friday is thinking...</span>
        </div>
        '''
        msg_placeholder.markdown(spinner_html, unsafe_allow_html=True)
        
        history = load_chat_history(st.session_state.current_chat_id)
        
        # Elite RAG Pipeline
        import asyncio
        context, all_sources = asyncio.run(get_context_with_strategy(
            raw_query=query,
            company_id=st.session_state.company_id,
            supabase=supabase,
            groq_api_key=FIXED_GROQ_KEY,
            get_embeddings_fn=get_embeddings_batch,
            hf_api_key=HF_API_KEY,
            top_k=5
        ))
        
        response = ask_groq(context, history, query)
        
        # Improved source citation: check for filename (with or without extension) in response
        cited_sources = []
        response_lower = response.lower()
        for src in all_sources:
            # Check for full filename or basename without extension
            src_lower = src.lower()
            src_base = src.rsplit('.', 1)[0].lower() if '.' in src else src_lower
            # Also check for partial matches (at least 70% of filename)
            if (src_lower in response_lower or 
                src_base in response_lower or
                src.replace('_', ' ').lower() in response_lower):
                cited_sources.append(src)
        
        # If no sources were explicitly cited but we have context sources, show top 2 as "used"
        if not cited_sources and all_sources:
            cited_sources = all_sources[:2]
        
        save_message(st.session_state.current_chat_id, "assistant", response, st.session_state.company_id, cited_sources)
        msg_placeholder.empty()
        st.markdown(response)
        
        if cited_sources:
            st.markdown('<div class="source-container" style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;">', unsafe_allow_html=True)
            for src in cited_sources:
                st.markdown(f'<span class="source-tag" style="background: rgba(26, 60, 52, 0.06); color: #1A3C34; padding: 6px 12px; border-radius: 16px; font-size: 13px; font-weight: 500;">📄 {src}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
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
        <h1 style="font-family: 'Playfair Display', serif; font-size: 48px; font-weight: 700; color: #1A3C34;">Knowledge Base</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "PDF, Word, Excel, PPTX", 
            type=["pdf", "docx", "xlsx", "pptx"], 
            accept_multiple_files=True
        )
        
        c_check, c_btn = st.columns([1, 1])
        with c_check: force_overwrite = st.checkbox("Overwrite existing?")
        with c_btn:
            if uploaded_files and st.button("Start Indexing", type="primary", use_container_width=True):
                # Bulk upload results
                results = {"success": 0, "exists": 0, "error": 0}
                progress_bar = st.progress(0)
                status = st.empty()
                
                for idx, f in enumerate(uploaded_files):
                    status.text(f"Indexing {f.name}...")
                    res = process_and_store_document(f, st.session_state.company_id, force_overwrite)
                    if res == "success": results["success"] += 1
                    elif res == "exists": results["exists"] += 1
                    else: results["error"] += 1
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status.empty(); progress_bar.empty()
                if results["success"] > 0: st.success(f"✅ {results['success']} files indexed successfully!")
                if results["exists"] > 0: st.info(f"ℹ️ {results['exists']} files already existed.")
                if results["error"] > 0: st.error(f"❌ {results['error']} files failed to process.")
                # Give database time to fully commit before refreshing
                time.sleep(1)
                st.rerun()

    with col2:
        docs = get_all_documents(st.session_state.company_id)
        st.markdown(f"### Indexed Files ({len(docs)})")
        if not docs: st.info("No documents indexed yet.")
        else:
            for doc in docs:
                status_icon = "🟢" if doc['is_active'] else "⚪"
                file_ext = doc['filename'].split('.')[-1].upper()
                # Truncate long filenames for display
                display_name = doc['filename']
                if len(display_name) > 40:
                    display_name = display_name[:37] + "..."
                
                # Use columns for clean layout: icon, name, status, actions
                col_status, col_name, col_toggle, col_delete = st.columns([0.5, 4, 0.5, 0.5])
                
                with col_status:
                    st.markdown(f"<div style='padding-top: 8px;'>{status_icon}</div>", unsafe_allow_html=True)
                
                with col_name:
                    st.markdown(f'''
                    <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
                        <span style="font-weight: 500; color: #1A3C34;">{file_ext}</span>
                        <span title="{doc['filename']}" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{display_name}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_toggle:
                    if st.button("⏸" if doc['is_active'] else "▶", key=f"pause_{doc['id']}", help="Pause/Resume"):
                        toggle_document_status(doc['filename'], st.session_state.company_id, doc['is_active']); st.rerun()
                
                with col_delete:
                    if st.button("🗑", key=f"del_doc_{doc['id']}", help="Delete"):
                        delete_document(doc['filename'], st.session_state.company_id); st.rerun()
                
                st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

# --- 5. AUTHENTICATION ---
def handle_login():
    """Callback for login form - handles authentication state properly."""
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
        else: st.session_state.login_error = "Invalid Code"
    except: st.session_state.login_error = "Connection Error"

def login_page():
    if "login_error" not in st.session_state: st.session_state.login_error = None
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 48px;">
            <div style="font-family: 'Playfair Display', serif; font-size: 72px; font-weight: 800; color: #1A3C34;">Friday</div>
            <h1 class="hero-title" style="font-size: 42px;">Your Intelligent<br><em>HR Companion</em></h1>
        </div>
        """, unsafe_allow_html=True)
        
        pw = st.text_input("Access Code", type="password", key="login_password")
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        if st.button("Sign In", use_container_width=True, type="primary", on_click=handle_login):
            if st.session_state.authenticated: st.rerun()
        
        if st.session_state.login_error: st.error(st.session_state.login_error)

if not st.session_state.authenticated: login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat": chat_page()
    elif st.session_state.view == "documents": documents_page()