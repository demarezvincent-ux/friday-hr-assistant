import streamlit as st
import asyncio
from supabase import create_client, Client
import requests
import pdfplumber
import docx
import re
import time
import os
import logging
import pandas as pd
from pptx import Presentation
from huggingface_hub import InferenceClient

# Import Services
from services.rag_controller import get_context_with_strategy
from services.agentic.rate_limiter import get_huggingface_limiter, get_groq_limiter

# --- LOGGING & CONFIG ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="FRIDAY", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UTILS: SECRETS & INIT ---
def get_secret(key_name):
    """Get secret from environment or Streamlit secrets with validation."""
    if not key_name or not isinstance(key_name, str): return None
    if key_name in os.environ: return os.environ[key_name].strip()
    try:
        if key_name in st.secrets: return st.secrets[key_name].strip()
    except Exception: pass
    return None

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
FIXED_GROQ_KEY = get_secret("FIXED_GROQ_KEY")
HF_API_KEY = get_secret("HF_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, FIXED_GROQ_KEY, HF_API_KEY]):
    st.error("❌ Critical Error: Missing API Keys. Please check your secrets.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- STATE MANAGEMENT ---
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "company_id" not in st.session_state: st.session_state.company_id = None
if "messages" not in st.session_state: st.session_state.messages = []
if "view" not in st.session_state: st.session_state.view = "chat"

# --- LOAD STYLES ---
def load_css():
    try:
        with open("assets/style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ Style file not found. UI might look unstyled.")

load_css()

# --- BACKEND FUNCTIONS (Embeddings, Processing) ---

def get_embeddings_batch(texts):
    """Generate embeddings using HuggingFace Inference API with rate limiting."""
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    client = InferenceClient(token=HF_API_KEY)
    clean_texts = [t.replace("\n", " ").strip() for t in texts]
    backoff_times = [2, 4, 8, 16]
    
    hf_limiter = get_huggingface_limiter()
    hf_limiter.wait_if_needed()

    for wait_time in backoff_times:
        try:
            embeddings = client.feature_extraction(clean_texts, model=model_id)
            if hasattr(embeddings, "tolist"): return embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.warning(f"Embedding failed, retrying in {wait_time}s: {e}")
            time.sleep(wait_time)
    return None

def sanitize_filename(filename):
    if not filename: return "unknown_file"
    name = filename[:100].replace(" ", "_")
    name = re.sub(r'[/\\:*?"<>|]', '', name)
    name = re.sub(r'\.\.', '', name)
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
    if name.startswith('.'): name = 'file_' + name[1:]
    return name

def recursive_chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
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

def process_and_store_document(file, company_id, force_overwrite=False):
    """Refactored document processor with Vision support."""
    try:
        clean_name = sanitize_filename(file.name)
        
        # Check Exists
        res = supabase.table("documents").select("id").eq("company_id", company_id).eq("filename", clean_name).execute()
        if res.data and not force_overwrite:
            return "exists"
        
        text = ""
        ext = file.name.lower().split('.')[-1]
        
        # Text Extraction
        try:
            if ext == "pdf":
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text() or ""
                        text += t + "\n"
            elif ext == "docx":
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif ext == "xlsx":
                dfs = pd.read_excel(file, sheet_name=None)
                for sheet, df in dfs.items():
                    text += f"Sheet: {sheet}\n{df.to_markdown()}\n"
            elif ext == "pptx":
                prs = Presentation(file)
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"): text += shape.text + "\n"
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return "error"
            
        if not text: return "empty"

        # === VISUAL RAG INTEGRATION ===
        try:
            from services.vision_service import get_visual_context
            file.seek(0)
            visual_context = get_visual_context(file, FIXED_GROQ_KEY, max_images=15)
            if visual_context:
                text = text + "\n\n=== VISUAL CONTEXT ===\n" + visual_context
                logger.info(f"Added visual context: {len(visual_context)} chars")
        except Exception as e:
            logger.warning(f"Visual context extraction failed: {e}")
        # ==============================

        # Upload to Storage
        file.seek(0)
        safe_cid = re.sub(r'[^\w\-]', '_', company_id)
        ctypes = {"pdf":"application/pdf", "docx":"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "xlsx":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "pptx":"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
        supabase.storage.from_("documents").upload(
            f"{safe_cid}/{clean_name}", 
            file.read(), 
            {"upsert": "true", "contentType": ctypes.get(ext, "application/octet-stream")}
        )

        # Chunk & Embed
        chunks = recursive_chunk_text(text)
        payload = []
        
        # Process in batches of 20
        for i in range(0, len(chunks), 20):
            batch = chunks[i:i+20]
            vectors = get_embeddings_batch(batch)
            if vectors:
                for j, vec in enumerate(vectors):
                    if j < len(batch): # Safety check
                        payload.append({
                            "content": batch[j],
                            "metadata": {"company_id": company_id, "filename": clean_name, "is_active": True},
                            "embedding": vec
                        })
        
        # Insert in chunks of 50
        if payload:
            for i in range(0, len(payload), 50):
                sub_payload = payload[i:i+50]
                supabase.table("document_chunks").insert(sub_payload).execute()

        # Register Doc
        supabase.table("documents").insert({
            "company_id": company_id, "filename": clean_name, "is_active": True
        }).execute()
        
        return "success"
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return "error"

def delete_document(filename, company_id):
    try:
        supabase.table("documents").delete().eq("company_id", company_id).eq("filename", filename).execute()
        supabase.table("document_chunks").delete().eq("metadata->>company_id", company_id).eq("metadata->>filename", filename).execute()
        safe_cid = re.sub(r'[^\w\-]', '_', company_id)
        supabase.storage.from_("documents").remove([f"{safe_cid}/{filename}"])
        return True
    except: return False

# --- RAG LOGIC --- 

async def ask_friday(user_query, history):
    """Main chat handler."""
    try:
        context_str, sources = await get_context_with_strategy(
            raw_query=user_query,
            company_id=st.session_state.company_id,
            supabase=supabase,
            groq_api_key=FIXED_GROQ_KEY,
            get_embeddings_fn=get_embeddings_batch, 
            hf_api_key=HF_API_KEY,
            match_count=250,
            top_k=7,
            use_cache=True
        )
        
        system_prompt = """You are FRIDAY, an expert multilingual HR assistant.
        CRITICAL RULES:
        1. LANGUAGE: Respond in the SAME language as the user's query.
        2. SOURCES: Answer ONLY based on the provided CONTEXT.
        3. TONE: Professional but conversational.
        4. CITATIONS: Attribute information to source files.
        5. FORMS: If "RECOMMENDED FORMS" are listed in context, provide the links.
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-4:]: 
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        if context_str:
            messages.append({"role": "user", "content": f"CONTEXT:\n{context_str}\n\nUSER QUESTION: {user_query}"})
        else:
            messages.append({"role": "user", "content": user_query})

        # 3. Call LLM (Groq)
        model = "llama-3.3-70b-versatile"
        try:
            # Enforce Rate Limit for Free Tier
            get_groq_limiter().wait_if_needed()
            
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
                json={"model": model, "messages": messages, "temperature": 0.1},
                timeout=30
            ) 
            if resp.status_code == 200:
                answer = resp.json()['choices'][0]['message']['content']
                return answer, sources
            return "Connection error.", []
        except Exception as e:
            return "Thinking error.", []

    except Exception as e:
        logger.error(f"RAG Error: {e}")
        return "Search error.", []

# --- UI RENDERING ---

def login_screen():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>⚡ FRIDAY</h1>", unsafe_allow_html=True)
        st.markdown("<div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
        company_id = st.text_input("Company ID", placeholder="Enter your company ID...")
        if st.button("Enter Workspace", type="primary", use_container_width=True):
            if company_id:
                st.session_state.authenticated = True
                st.session_state.company_id = company_id
                st.rerun()
            else:
                st.warning("Please enter a Company ID.")
        
        # Help text
        st.markdown("<div style='text-align:center; margin-top:20px; color:#999; font-size:12px;'>Protected System • Authorized Use Only</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ FRIDAY")
        st.markdown(f"<div style='margin-bottom: 20px; color: #5C6F68; font-size: 0.9rem;'>Workspace: <b>{st.session_state.company_id}</b></div>", unsafe_allow_html=True)
        
        if st.button("💬 Chat Assistant", key="nav_chat", type="primary" if st.session_state.view == "chat" else "secondary"):
            st.session_state.view = "chat"
            st.rerun()
            
        if st.button("📂 Documents", key="nav_docs", type="primary" if st.session_state.view == "documents" else "secondary"):
            st.session_state.view = "documents"
            st.rerun()
            
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.company_id = None
            st.session_state.messages = []
            st.rerun()

def render_documents_page():
    st.title("Knowledge Base")
    st.markdown("Manage your company's documents here.")
    
    with st.container():
        st.markdown("### Upload New Document")
        uploaded_file = st.file_uploader("Drag and drop PDF, DOCX, XLSX, or PPTX", type=["pdf", "docx", "xlsx", "pptx"])
        if uploaded_file:
            if st.button("Process & Index", type="primary"):
                with st.spinner("Processing document... including visual analysis..."):
                    result = process_and_store_document(uploaded_file, st.session_state.company_id)
                    if result == "success":
                        st.success(f"✅ Indexed {uploaded_file.name} successfully!")
                        time.sleep(1)
                        st.rerun()
                    elif result == "exists":
                        st.warning("Document already exists.")
                    else:
                        st.error("Failed to process document.")

    st.markdown("---")
    st.markdown("### Active Documents")
    
    try:
        res = supabase.table("documents").select("*").eq("company_id", st.session_state.company_id).order('created_at', desc=True).execute()
        docs = res.data or []
    except: docs = []

    if not docs:
        st.info("No documents found.")
        return

    for doc in docs:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"<div class='file-card'><span class='icon'>📄</span> <span class='title'>{doc['filename']}</span></div>", unsafe_allow_html=True)
        with c2:
            if st.button("🗑️", key=f"del_{doc['id']}", help="Delete"):
                delete_document(doc['filename'], st.session_state.company_id)
                st.rerun()

def render_chat_page():
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>How can I help you today?</h1>", unsafe_allow_html=True)

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                # Correctly rendering sources with style
                sources_html = "".join([f"<span class='source-citation'>{s}</span>" for s in msg["sources"]])
                st.markdown(f"<div style='margin-top: 8px;'>{sources_html}</div>", unsafe_allow_html=True)

    # Input
    if prompt := st.chat_input("Ask about policies, procedures, or forms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("""
                <div class="thinking-container">
                    <div class="dot-pulse"></div>
                    <span style="font-size: 0.9rem; color: #5C6F68;">Thinking...</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Use asyncio.run for the async call
            response_text, sources = asyncio.run(ask_friday(prompt, st.session_state.messages))
            
            placeholder.markdown(response_text)
            
            if sources:
                sources_html = "".join([f"<span class='source-citation'>{s}</span>" for s in sources])
                st.markdown(f"<div style='margin-top: 8px;'>{sources_html}</div>", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": response_text, "sources": sources})

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not st.session_state.authenticated:
        login_screen()
    else:
        render_sidebar()
        if st.session_state.view == "documents":
            render_documents_page()
        else:
            render_chat_page()