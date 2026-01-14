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
from huggingface_hub import InferenceClient

# --- 1. CONFIGURATION & PREMIUM STYLING ---
st.set_page_config(page_title="FRIDAY", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* ============================================
       FRIDAY - Premium Design System
       ============================================ */
    
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #1A3C34;
        --primary-gradient: linear-gradient(135deg, #1A3C34 0%, #0F2921 100%);
        --bg-primary: #F5F5F7;
        --bg-secondary: #FAFAFA;
        --bg-white: #FFFFFF;
        --text-primary: #1D1D1F;
        --text-secondary: #5C5C61;
        --text-tertiary: #86868B;
        --border-color: #D2D2D7;
    }

    /* Global Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }

    /* ============================================
       SIDEBAR NAVIGATION (FIXED FOR DYNAMIC COLORS)
       ============================================ */
    [data-testid="stSidebar"] {
        background-color: var(--bg-primary);
        border-right: 1px solid rgba(0,0,0,0.08);
    }

    /* General Button Styling */
    [data-testid="stSidebar"] .stButton button {
        background-color: transparent;
        border: none;
        text-align: left;
        padding: 12px 16px;
        height: 44px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }

    /* --- ACTIVE STATE (Selected) --- */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: var(--primary-gradient) !important;
        box-shadow: 0 4px 12px rgba(26, 60, 52, 0.35);
    }

    /* FORCE WHITE TEXT for Active State */
    div[data-testid="stSidebar"] button[kind="primary"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* --- INACTIVE STATE (Not Selected) --- */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(0, 0, 0, 0.05) !important;
    }

    /* FORCE BLACK TEXT for Inactive State */
    div[data-testid="stSidebar"] button[kind="secondary"] * {
        color: #1D1D1F !important;
        fill: #1D1D1F !important;
        -webkit-text-fill-color: #1D1D1F !important;
    }

    /* Sidebar Headers */
    [data-testid="stSidebar"] h3 {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: var(--text-tertiary) !important;
        text-transform: uppercase;
        margin: 28px 0 12px 0 !important;
    }

    /* ============================================
       MAIN CONTENT STYLING
       ============================================ */
    
    /* Inputs */
    .stTextInput input, .stChatInput textarea {
        background-color: var(--bg-white);
        border: 1.5px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    .stTextInput input:focus, .stChatInput textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(26, 60, 52, 0.1) !important;
    }

    /* Chat Bubbles */
    [data-testid="stChatMessage"][data-testid*="user"] > div {
        background: var(--primary-gradient);
        color: white;
        border-radius: 20px 20px 4px 20px;
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] > div {
        background-color: var(--bg-primary);
        border-radius: 18px 18px 18px 4px;
    }

    /* Source Tags */
    .source-tag {
        font-size: 12px;
        background: white;
        border: 1px solid var(--border-color);
        padding: 4px 10px;
        border-radius: 20px;
        color: var(--text-secondary);
        display: inline-block;
        margin-right: 5px;
        margin-top: 5px;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-color);
        background: var(--bg-secondary);
        padding: 2rem;
        border-radius: 12px;
    }

    /* Login Box */
    .login-container {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

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
    backoff_times = [2, 4, 8]

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
    """
    FIX 2: Smart Normalization.
    This converts Dutch variations to the English equivalent found in the document.
    This ensures that 'koffiemachine' finds 'coffee machine'.
    """
    normalized = query.lower().strip()
    
    # Map all coffee machine variations (Dutch/English/Compound) to the English term
    # \b matches word boundaries, \s* matches optional spaces, [-]? matches optional hyphen
    if re.search(r'\b(koffie|coffee)\s*[-]?\s*machine\b', normalized) or 'koffiemachine' in normalized or 'coffeemachine' in normalized:
        normalized = normalized.replace('koffiemachine', 'coffee machine')
        normalized = normalized.replace('coffeemachine', 'coffee machine')
        normalized = re.sub(r'\bkoffie\s*[-]?\s*machine\b', 'coffee machine', normalized)
    
    # Other common compound fixes
    normalized = normalized.replace('wacht woord', 'wachtwoord').replace('password', 'wachtwoord')
    
    return normalized

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: text += page_text + "\n"
    except: return ""
    return text

def smart_chunking(text, chunk_size=500, overlap=100):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# --- DATABASE OPERATIONS ---

def register_document(filename, company_id):
    try:
        supabase.table("documents").insert({"company_id": company_id, "filename": filename, "is_active": True}).execute()
        return True
    except: return False

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

def delete_document(filename, company_id):
    try:
        supabase.table("documents").delete().eq("company_id", company_id).eq("filename", filename).execute()
        supabase.table("document_chunks").delete().eq("metadata->>company_id", company_id).eq("metadata->>filename", filename).execute()
        try: supabase.storage.from_("documents").remove([f"{company_id}/{filename}"])
        except: pass
        return True
    except: return False

def process_and_store_document(file, company_id, force_overwrite=False):
    clean_name = sanitize_filename(file.name)
    if check_if_document_exists(clean_name, company_id):
        if not force_overwrite: return "exists"
        else: delete_document(clean_name, company_id)

    text = ""
    try:
        if file.name.endswith(".pdf"): text = extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
    except: return "error"
    if not text: return "empty"

    try:
        file.seek(0)
        supabase.storage.from_("documents").upload(f"{company_id}/{clean_name}", file.read(), {"upsert": "true"})
    except: pass

    chunks = smart_chunking(text)
    bar = st.progress(0)
    for i in range(0, len(chunks), 20):
        batch = chunks[i:i+20]
        vectors = get_embeddings_batch(batch)
        if vectors:
            payload = []
            for j, vec in enumerate(vectors):
                if isinstance(vec, list) and len(vec) > 300:
                    payload.append({
                        "content": batch[j],
                        "metadata": {"company_id": company_id, "filename": clean_name, "is_active": True},
                        "embedding": vec
                    })
            if payload: supabase.table("document_chunks").insert(payload).execute()
        bar.progress(min((i+20)/len(chunks), 1.0))
    bar.empty()
    register_document(clean_name, company_id)
    return "success"

def get_relevant_context(query, company_id):
    # 1. Apply the strict normalization (koffiemachine -> coffee machine)
    search_query = normalize_query(query)
    
    # 2. Embed the normalized query
    vectors = get_embeddings_batch([search_query])
    if not vectors: return "", []

    try:
        # 3. Search with slightly lower threshold to catch semantic matches
        params = {
            "query_embedding": vectors[0], 
            "match_threshold": 0.12, # Slightly lowered to catch more relevant content
            "match_count": 15, 
            "filter_company_id": company_id, 
            "query_text": search_query
        }
        res = supabase.rpc("match_documents_hybrid", params).execute()
        context_str = ""
        sources = []
        for m in res.data:
            context_str += f"-- SOURCE: {m['metadata']['filename']} --\n{m['content']}\n\n"
            if m['metadata']['filename'] not in sources: sources.append(m['metadata']['filename'])
        return context_str, sources
    except: return "", []

def ask_groq(context, history, query):
    system_prompt = "You are FRIDAY, an expert HR assistant. Answer strictly based on CONTEXT. If unknown, say so."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
    if context: messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
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
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            elif resp.status_code == 429:
                continue
        except: pass
    
    return "⚠️ All AI models are busy. Please try again."

# --- CHAT HISTORY & UI ---

def load_chat_history(chat_id):
    try: return supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute().data
    except: return []

def save_message(chat_id, role, content, company_id, sources=None):
    try: 
        supabase.table("messages").insert({
            "chat_id": chat_id, "role": role, "content": content, "sources": sources, "company_id": company_id 
        }).execute()
    except: pass

def get_recent_chats(company_id):
    try:
        res = supabase.table("messages").select("chat_id, content").eq("company_id", company_id).order("created_at", desc=True).limit(50).execute()
        seen, unique = set(), []
        for msg in res.data:
            if msg['chat_id'] not in seen:
                seen.add(msg['chat_id'])
                unique.append({"id": msg['chat_id'], "title": msg['content'][:30] + "..."})
        return unique[:10]
    except: return []

def delete_chat(chat_id, company_id):
    try:
        supabase.table("messages").delete().eq("chat_id", chat_id).eq("company_id", company_id).execute()
        return True
    except: return False

def render_sidebar():
    with st.sidebar:
        st.markdown('<div style="text-align:center; margin-bottom: 28px;"><h1 style="font-family:\'Playfair Display\'; font-size:42px;">Friday</h1></div>', unsafe_allow_html=True)
        
        st.markdown("### Menu")
        # Logic to switch button types
        chat_type = "primary" if st.session_state.view == "chat" else "secondary"
        doc_type = "primary" if st.session_state.view == "documents" else "secondary"

        if st.button("◉  Chat", use_container_width=True, type=chat_type):
            st.session_state.view = "chat"; st.rerun()
        if st.button("◎  Documents", use_container_width=True, type=doc_type):
            st.session_state.view = "documents"; st.rerun()

        st.markdown("---")

        if st.session_state.view == "chat":
            if st.button("＋  New Chat", use_container_width=True, type="secondary"): 
                st.session_state.current_chat_id = str(uuid.uuid4()); st.rerun()

            st.markdown("### Recent Chats")
            recent = get_recent_chats(st.session_state.company_id)
            for chat in recent:
                col1, col2 = st.columns([6, 1])
                with col1:
                    btn_type = "primary" if st.session_state.current_chat_id == chat['id'] else "secondary"
                    if st.button(chat['title'], key=f"c_{chat['id']}", use_container_width=True, type=btn_type):
                        st.session_state.current_chat_id = chat['id']; st.rerun()
                with col2:
                    if st.button("×", key=f"d_{chat['id']}"):
                        delete_chat(chat['id'], st.session_state.company_id)
                        st.rerun()

        st.markdown("---")
        if st.button("↪  Sign Out", use_container_width=True): 
            st.session_state.clear(); st.rerun()

def chat_page():
    if not st.session_state.current_chat_id: st.session_state.current_chat_id = str(uuid.uuid4())
    history = load_chat_history(st.session_state.current_chat_id)

    if not history:
        st.markdown(f"<h1 style='text-align: center; margin-top: 80px; font-family: Playfair Display;'>Good Afternoon.</h1>", unsafe_allow_html=True)

    for msg in history:
        with st.chat_message(msg["role"], avatar="⚡" if msg["role"] == "assistant" else None):
            st.write(msg["content"])
            if msg["sources"]:
                st.markdown("".join([f"<span class='source-tag'>📄 {s}</span>" for s in msg["sources"]]), unsafe_allow_html=True)

    if prompt := st.chat_input("Ask FRIDAY..."):
        save_message(st.session_state.current_chat_id, "user", prompt, st.session_state.company_id)
        with st.chat_message("user"): st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Thinking..."):
                context, sources = get_relevant_context(prompt, st.session_state.company_id)
                response = ask_groq(context, history, prompt)
                st.write(response)
                save_message(st.session_state.current_chat_id, "assistant", response, st.session_state.company_id, sources)
        st.rerun()

def documents_page():
    st.title("Knowledge Base")
    uploaded = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded and st.button("Index Files"):
        for f in uploaded:
            process_and_store_document(f, st.session_state.company_id)
        st.success("Uploaded!"); time.sleep(1); st.rerun()

    docs = get_all_documents(st.session_state.company_id)
    for doc in docs:
        c1, c2, c3 = st.columns([6, 1, 1])
        c1.write(f"📄 {doc['filename']}")
        if c3.button("×", key=f"del_{doc['id']}"):
            delete_document(doc['filename'], st.session_state.company_id)
            st.rerun()

# --- AUTH ---
def login_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    with st.form("login"):
        st.markdown("<h1 style='text-align: center; font-family: Playfair Display;'>FRIDAY</h1>", unsafe_allow_html=True)
        pw = st.text_input("Access Code", type="password")
        if st.form_submit_button("Enter", type="primary", use_container_width=True):
            res = supabase.table('clients').select("*").eq('access_code', pw).execute()
            if res.data:
                st.session_state.authenticated = True
                st.session_state.company_id = res.data[0]['company_id']
                st.rerun()
            else: st.error("Invalid Code")

if not st.session_state.authenticated: login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat": chat_page()
    elif st.session_state.view == "documents": documents_page()