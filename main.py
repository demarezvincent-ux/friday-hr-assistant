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
    
    /* 1. Typography */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #1A3C34;
        --primary-hover: #0F2921;
        --primary-gradient: linear-gradient(135deg, #1A3C34 0%, #0F2921 100%);
        --bg-primary: #F5F5F7;
        --bg-secondary: #FAFAFA;
        --bg-white: #FFFFFF;
        --sidebar-bg: #F5F5F7;
        --text-primary: #1D1D1F;
        --text-secondary: #5C5C61;
        --text-tertiary: #86868B;
        --error: #FF3B30;
        --divider: rgba(0, 0, 0, 0.08);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        --transition-normal: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }

    /* ============================================
       2. SIDEBAR & NAVIGATION BUTTONS (FIXED)
       ============================================ */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--divider);
        width: 300px !important;
    }

    /* General Button Styling */
    [data-testid="stSidebar"] .stButton button {
        border: none;
        text-align: left;
        padding: 12px 16px;
        height: 44px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        width: 100%;
    }

    /* --- ACTIVE STATE (Selected) --- 
       Forces background to Green and TEXT TO WHITE */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: var(--primary-gradient) !important;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(26, 60, 52, 0.35);
        border: none !important;
    }

    /* STRICTLY FORCE WHITE TEXT ON ACTIVE BUTTONS */
    div[data-testid="stSidebar"] button[kind="primary"] *,
    div[data-testid="stSidebar"] button[kind="primary"] p,
    div[data-testid="stSidebar"] button[kind="primary"] span,
    div[data-testid="stSidebar"] button[kind="primary"] div {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* --- INACTIVE STATE (Not Selected) --- 
       Forces background to transparent and TEXT TO BLACK */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #1D1D1F !important;
        border: 1px solid transparent !important;
    }

    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(0, 0, 0, 0.05) !important;
        color: #000000 !important;
    }

    /* STRICTLY FORCE BLACK TEXT ON INACTIVE BUTTONS */
    div[data-testid="stSidebar"] button[kind="secondary"] *,
    div[data-testid="stSidebar"] button[kind="secondary"] p,
    div[data-testid="stSidebar"] button[kind="secondary"] span,
    div[data-testid="stSidebar"] button[kind="secondary"] div {
        color: #1D1D1F !important;
        -webkit-text-fill-color: #1D1D1F !important;
    }

    /* Sidebar Dividers */
    [data-testid="stSidebar"] hr {
        margin: 16px 0;
        border-top: 1px solid var(--divider);
    }

    /* ============================================
       3. REST OF UI
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
        border-radius: 10px; height: 52px; padding: 0 16px; font-size: 17px;
    }
    .stChatInput > div > div > textarea {
        background-color: var(--bg-primary); border-radius: 14px;
        min-height: 60px; padding: 18px 56px 18px 20px; font-size: 17px;
    }
    .stChatInput > div > div > textarea:focus {
        background-color: var(--bg-white); border-color: var(--primary) !important;
        box-shadow: 0 0 0 5px rgba(26, 60, 52, 0.12) !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"][data-testid*="user"] > div {
        background: var(--primary-gradient); color: white;
        border-radius: 20px 20px 4px 20px; padding: 14px 20px; max-width: 70%;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] > div {
        background-color: var(--bg-primary); color: var(--text-primary);
        border-radius: 18px 18px 18px 4px; padding: 16px 20px; max-width: 70%;
    }

    /* Source Tags */
    .source-tag {
        font-size: 13px; font-weight: 600; background-color: var(--bg-white);
        border: 1px solid #D2D2D7; padding: 8px 14px; border-radius: 99px;
        color: var(--text-secondary); display: inline-flex; align-items: center;
        margin-right: 8px; margin-top: 8px;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 3px dashed #D2D2D7; background-color: var(--bg-secondary);
        border-radius: 12px; padding: 3rem;
    }
    .file-item {
        display: flex; align-items: center; padding: 14px 18px;
        border-radius: 8px; border: 1px solid transparent; transition: all 0.2s ease;
    }
    .file-item:hover { background-color: rgba(26, 60, 52, 0.04); }

    /* Login */
    .login-container {
        background: white; border-radius: 20px; padding: 56px 48px;
        max-width: 480px; margin: 0 auto; box-shadow: 0 12px 48px rgba(0,0,0,0.12);
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Animations */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes slideInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    
    /* Apply animations */
    .stApp { animation: fadeIn 0.6s ease-out; }
    
    [data-testid="stChatMessage"] {
        animation: slideInUp 0.4s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }
    
    /* Interactive Elements Hover Scale */
    button:hover {
        transform: scale(1.01);
    }
    
    /* Logo - Static now */
    .logo-animated { 
        /* Animation removed as per request */
        transition: transform 0.3s ease;
    }
    .logo-animated:hover {
        transform: scale(1.05); /* Subtle scale on hover instead of bounce */
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
    system_prompt = "You are FRIDAY, an expert HR assistant. Answer strictly based on CONTEXT. Context may have Markdown tables. If unknown, say so."
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
            if resp.status_code == 200: return resp.json()['choices'][0]['message']['content']
            elif resp.status_code == 429: continue
            else: return f"Error: {resp.status_code}"
        except: return "Connection error"
    return "⚠️ High traffic. Please try again."

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
    save_message(st.session_state.current_chat_id, "user", query, st.session_state.company_id)
    with st.chat_message("user", avatar="👤"): st.write(query)
    
    with st.chat_message("assistant", avatar="⚡"):
        message_placeholder = st.empty()
        for i in range(6): 
            message_placeholder.markdown(f"*Thinking{'.' * (i%4)}*")
            time.sleep(0.2)
        
        history = load_chat_history(st.session_state.current_chat_id)
        context, sources = get_relevant_context(query, st.session_state.company_id)
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
        uploaded_files = st.file_uploader("PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
        
        c_check, c_btn = st.columns([1, 1])
        with c_check: force_overwrite = st.checkbox("Overwrite existing?")
        with c_btn:
            if uploaded_files and st.button("Start Indexing", type="primary", use_container_width=True):
                for f in uploaded_files:
                    with st.spinner(f"Processing {f.name}..."):
                        process_and_store_document(f, st.session_state.company_id, force_overwrite)
                    time.sleep(0.5); st.rerun()

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
def login_page():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 48px;">
            <div style="font-family: 'Playfair Display', serif; font-size: 72px; font-weight: 800; color: #1A3C34;">Friday</div>
            <h1 class="hero-title" style="font-size: 42px;">Your Intelligent<br><em>HR Companion</em></h1>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            pw = st.text_input("Access Code", type="password")
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                try:
                    res = supabase.table('clients').select("*").eq('access_code', pw).execute()
                    if res.data:
                        st.session_state.authenticated = True
                        st.session_state.company_id = res.data[0]['company_id']
                        st.rerun()
                    else: st.error("Invalid Code")
                except: st.error("Login Error")

if not st.session_state.authenticated: login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat": chat_page()
    elif st.session_state.view == "documents": documents_page()