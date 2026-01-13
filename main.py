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
    /* 1. Global Font & Background - Warm Paper Theme */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Inter:wght@300;400;500&display=swap');

    .stApp {
        background-color: #FAF9F6; /* Warm Paper */
        font-family: 'Inter', sans-serif;
    }

    /* 2. Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background-color: #F0EFEB;
        border-right: 1px solid #DCDAD5;
    }

    /* 3. Typography - Serif Headers */
    h1, h2, h3 {
        font-family: 'Merriweather', serif !important;
        color: #2C2C2C;
        letter-spacing: -0.5px;
    }

    /* 4. Chat Message Bubbles - Distinct Cards */
    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* 5. Buttons - Soft & Elegant */
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid #E0E0E0;
    }

    /* Primary Button (Sage Green) */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #8FAE8B !important;
        border: none;
        color: white !important;
        box-shadow: 0 4px 6px rgba(143, 174, 139, 0.25);
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #7A9A76 !important;
        transform: translateY(-1px);
    }

    /* 6. Inputs & Text Areas */
    .stTextInput > div > div > input, .stChatInput > div > div > textarea {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        color: #333;
    }

    /* 7. Source Tags */
    .source-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .source-tag {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 4px 12px;
        border-radius: 99px;
        color: #4B5563;
        display: inline-flex;
        align-items: center;
    }
    .source-tag:hover { border-color: #8FAE8B; color: #6B8E68; }

    /* 8. Sidebar History Buttons */
    .history-btn {
        width: 100%;
        text-align: left;
        padding: 8px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 0.85rem;
        cursor: pointer;
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

# --- 3. BACKEND LOGIC (ROBUST VERSION) ---

# [FIX 1] Exponential Backoff
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
    """Normalize query to handle compound word variations (e.g., 'koffie machine' -> 'koffiemachine')."""
    # Common Dutch compound word patterns that should be joined
    compound_patterns = [
        (r'\bkoffie\s+machine\b', 'koffiemachine'),
        (r'\bkoffie\s*-\s*machine\b', 'koffiemachine'),
        (r'\bwacht\s+woord\b', 'wachtwoord'),
        (r'\bwacht\s*-\s*woord\b', 'wachtwoord'),
        (r'\btijd\s+registratie\b', 'tijdregistratie'),
        (r'\bverlof\s+dagen\b', 'verlofdagen'),
        (r'\bwerk\s+uren\b', 'werkuren'),
        (r'\bziekte\s+verlof\b', 'ziekteverlof'),
    ]
    
    normalized = query.lower()
    for pattern, replacement in compound_patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    # Return ONLY the normalized version to prevent FTS dilution/noise
    return normalized.strip()

# [FIX 2] Markdown Table Extraction
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

# [FIX 3] Cross-Lingual Search with Query Normalization
def get_relevant_context(query, company_id):
    # Normalize query to handle compound word variations
    normalized_query = normalize_query(query)
    
    # Start with normalized query as base
    search_query = normalized_query
    
    try:
        # CRITICAL FIX: Ask for FULL queries with OR. This enforces (Password AND Machine) logic per language.
        # Example output: "password coffee machine OR wachtwoord koffiemachine"
        expansion_prompt = f"Translate '{normalized_query}' into English, Dutch, and French fully formed queries. Correct typos. Output ONLY the queries separated by ' OR '."
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": expansion_prompt}], "temperature": 0.1, "max_tokens": 100},
            timeout=10
        )
        if resp.status_code == 200:
            # Use the expanded "OR" query directly
            llm_expansion = resp.json()['choices'][0]['message']['content']
            search_query = llm_expansion
        else:
            print(f"Query expansion failed with status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Query expansion error: {e}")

    vectors = get_embeddings_batch([search_query])
    if not vectors: return "", []

    try:
        # CRITICAL FIX: Use the expanded 'search_query' (with ORs) for FTS.
        # This handles: Multi-lingual, Typos ("coffe"), and Compound words ("koffiemachine")
        # OPTIMIZATION: Reduced match_count to 15 for Rate Limit safety (Token budget ~2.5k instead of 6k)
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

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.1},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        elif resp.status_code == 429:
            return f"Error: Groq Rate Limit (429). Your free tier might be exhausted or you are sending too many requests."
        else:
            return f"Error: Groq API returned status {resp.status_code}. {resp.text[:100]}"
    except Exception as e: 
        return f"Connection error: {str(e)}"

# --- CHAT HISTORY & PERSISTENCE ---

def load_chat_history(chat_id):
    try: return supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute().data
    except: return []

def save_message(chat_id, role, content, company_id, sources=None):
    try: 
        # UPDATED: We now save 'company_id' so we can find these chats later
        supabase.table("messages").insert({
            "chat_id": chat_id, 
            "role": role, 
            "content": content, 
            "sources": sources,
            "company_id": company_id 
        }).execute()
    except: pass

def get_recent_chats(company_id):
    # Retrieve the last 50 messages to find unique chat IDs
    try:
        res = supabase.table("messages").select("chat_id, content, created_at").eq("company_id", company_id).order("created_at", desc=True).limit(50).execute()

        seen_ids = set()
        unique_chats = []

        for msg in res.data:
            if msg['chat_id'] not in seen_ids:
                seen_ids.add(msg['chat_id'])
                # Use the first message we find as the "Title" preview
                unique_chats.append({
                    "id": msg['chat_id'],
                    "title": msg['content'][:30] + "..."
                })
        return unique_chats[:10] # Return top 10 recent chats
    except:
        return []

def delete_chat(chat_id, company_id):
    """Delete a chat and all its messages from the database."""
    try:
        supabase.table("messages").delete().eq("chat_id", chat_id).eq("company_id", company_id).execute()
        return True
    except:
        return False

def get_dynamic_greeting():
    """Return a greeting based on the current time of day."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning."
    elif 12 <= hour < 17:
        return "Good afternoon."
    elif 17 <= hour < 21:
        return "Good evening."
    else:
        return "Hello."

# --- UI PAGES ---
def render_sidebar():
    with st.sidebar:
        st.title("⚡ FRIDAY")
        st.caption(f"ID: {st.session_state.company_id}")
        st.markdown("---")

        # Navigation
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.view == "chat" else "secondary"):
            st.session_state.view = "chat"; st.rerun()
        if st.button("📂 Documents", use_container_width=True, type="primary" if st.session_state.view == "documents" else "secondary"):
            st.session_state.view = "documents"; st.rerun()

        st.markdown("---")

        # Action: New Chat
        if st.session_state.view == "chat":
            if st.button("➕ New Chat", use_container_width=True): 
                create_new_chat(); st.rerun()

            st.markdown("### Recent Chats")
            recent = get_recent_chats(st.session_state.company_id)
            if not recent:
                st.caption("No history found.")
            else:
                for chat in recent:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(f"📝 {chat['title']}", key=f"chat_{chat['id']}", use_container_width=True):
                            st.session_state.current_chat_id = chat['id']
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{chat['id']}", help="Delete chat"):
                            delete_chat(chat['id'], st.session_state.company_id)
                            # If deleting current chat, create new one
                            if st.session_state.current_chat_id == chat['id']:
                                create_new_chat()
                            st.rerun()

        st.markdown("---")
        if st.button("Log Out"): st.session_state.clear(); st.rerun()

def create_new_chat(): st.session_state.current_chat_id = str(uuid.uuid4())

# --- HELPER: HANDLE QUERY EXECUTION WITH LOADING ANIMATION ---
def handle_query(query):
    # Save user msg with COMPANY ID
    save_message(st.session_state.current_chat_id, "user", query, st.session_state.company_id)

    # Show loading animation while processing
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        # Create a placeholder for the typing animation
        message_placeholder = st.empty()
        
        # Show animated loading dots
        loading_frames = ["Thinking.", "Thinking..", "Thinking..."]
        for i in range(6):  # Show animation while processing starts
            message_placeholder.markdown(f"*{loading_frames[i % 3]}*")
            time.sleep(0.3)
        
        # Process the query
        history = load_chat_history(st.session_state.current_chat_id)
        context, sources = get_relevant_context(query, st.session_state.company_id)
        response = ask_groq(context, history, query)

        # Save AI msg with COMPANY ID
        save_message(st.session_state.current_chat_id, "assistant", response, st.session_state.company_id, sources)
        
        # Clear placeholder and show final response
        message_placeholder.empty()
    
    st.rerun()

def chat_page():
    if not st.session_state.current_chat_id: create_new_chat()
    history = load_chat_history(st.session_state.current_chat_id)

    # --- DYNAMIC GREETING (only when no history) ---
    if not history:
        greeting = get_dynamic_greeting()
        st.markdown(f"""
        <div style="text-align: center; margin-top: 15vh; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">{greeting}</h1>
            <p style="color: #666; font-family: 'Inter', sans-serif;">How can FRIDAY help you with HR tasks today?</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CHAT HISTORY RENDER ---
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["sources"]:
                tags = "".join([f"<div class='source-tag'>📄 {s}</div>" for s in msg["sources"]])
                st.markdown(f"<div class='source-container'>{tags}</div>", unsafe_allow_html=True)

    # --- CHAT INPUT ---
    if prompt := st.chat_input("Ask a question..."):
        handle_query(prompt)

def documents_page():
    st.title("📂 Knowledge Base")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload")
        uploaded_files = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"], accept_multiple_files=True)
        force_overwrite = st.checkbox("Overwrite existing files?")
        if uploaded_files and st.button("Index Files", type="primary"):
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    status = process_and_store_document(f, st.session_state.company_id, force_overwrite)
                    if status == "success": st.toast(f"✅ Indexed: {f.name}")
                    elif status == "exists": st.warning(f"⚠️ {f.name} exists.")
                    else: st.error(f"❌ Error: {f.name}")
            time.sleep(1)
            st.rerun()

    with col2:
        st.subheader("Indexed Documents")
        docs = get_all_documents(st.session_state.company_id)
        if not docs: st.info("No documents found.")
        else:
            for doc in docs:
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 1])
                    icon = "🟢" if doc['is_active'] else "🔴"
                    style = "" if doc['is_active'] else "text-decoration: line-through; color: gray;"
                    c1.markdown(f"{icon} <span style='{style}'>{doc['filename']}</span>", unsafe_allow_html=True)

                    if c2.button("📦", key=f"arch_{doc['id']}", help="Archive"):
                        toggle_document_status(doc['filename'], st.session_state.company_id, doc['is_active'])
                        st.rerun()
                    if c3.button("🗑️", key=f"del_{doc['id']}"):
                        delete_document(doc['filename'], st.session_state.company_id)
                        st.rerun()
                    st.divider()

# --- 5. AUTHENTICATION ---
def login_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.title("⚡ FRIDAY Access")
        with st.form("login_form"):
            pw = st.text_input("Access Code", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                try:
                    res = supabase.table('clients').select("*").eq('access_code', pw).execute()
                    if res.data:
                        st.session_state.authenticated = True
                        st.session_state.company_id = res.data[0]['company_id']
                        st.rerun()
                    else: st.error("Invalid Access Code")
                except Exception as e: st.error(f"Login Error: {e}")

if not st.session_state.authenticated: login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat": chat_page()
    elif st.session_state.view == "documents": documents_page()