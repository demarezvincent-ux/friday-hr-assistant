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

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="FRIDAY", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    .stApp { background-color: #F9FAFB; font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #F3F4F6; border-right: 1px solid #E5E7EB; }

    /* Source Tags */
    .source-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .source-tag {
        font-size: 0.75rem; background-color: #FFFFFF; border: 1px solid #E5E7EB;
        padding: 4px 10px; border-radius: 16px; color: #374151; display: flex; align-items: center;
        transition: all 0.2s;
    }
    .source-tag:hover { border-color: #3D6E98; color: #3D6E98; }

    /* Chat Bubbles */
    div[data-testid="stChatMessage"] { background-color: transparent; padding: 1rem 0; }
</style>
""",
            unsafe_allow_html=True)


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
    st.error(
        "❌ Missing API Keys. Please check your Secrets or Environment Variables."
    )
    st.stop()

# --- 2. STATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "company_id" not in st.session_state: st.session_state.company_id = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "view" not in st.session_state: st.session_state.view = "chat"


@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase = init_supabase()

# --- 3. BACKEND LOGIC ---


# [FIX 1] Exponential Backoff for Uploads
def get_embeddings_batch(texts):
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    client = InferenceClient(token=HF_API_KEY)
    clean_texts = [t.replace("\n", " ").strip() for t in texts]

    # Wait 2s -> 4s -> 8s -> 16s
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


# [FIX 2] Table Extraction to Markdown
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                # 1. Extract Tables first
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # Convert list-of-lists to Markdown Table
                        # Filter out None and replace newlines in cells
                        clean_table = [[
                            (str(cell) if cell else "").replace("\n", " ")
                            for cell in row
                        ] for row in table]
                        if clean_table:
                            try:
                                header = "| " + " | ".join(
                                    clean_table[0]) + " |"
                                sep = "| " + " | ".join(
                                    ["---"] * len(clean_table[0])) + " |"
                                body = "\n".join([
                                    "| " + " | ".join(row) + " |"
                                    for row in clean_table[1:]
                                ])
                                text += f"\n{header}\n{sep}\n{body}\n\n"
                            except:
                                pass

                # 2. Extract regular text
                page_text = page.extract_text()
                if page_text: text += page_text + "\n"
    except:
        return ""
    return text


def smart_chunking(text, chunk_size=1000, overlap=200):
    if not text: return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


# --- DATABASE REGISTRY ---


def register_document(filename, company_id):
    try:
        supabase.table("documents").insert({
            "company_id": company_id,
            "filename": filename,
            "is_active": True
        }).execute()
        return True
    except:
        return False


def check_if_document_exists(filename, company_id):
    try:
        res = supabase.table("documents").select("id").eq(
            "company_id", company_id).eq("filename", filename).execute()
        return len(res.data) > 0
    except:
        return False


def get_all_documents(company_id):
    try:
        return supabase.table("documents").select("*").eq(
            "company_id",
            company_id).order('is_active',
                              desc=True).order('created_at',
                                               desc=True).execute().data
    except:
        return []


def toggle_document_status(filename, company_id, current_status):
    try:
        supabase.table("documents").update({
            "is_active": not current_status
        }).eq("company_id", company_id).eq("filename", filename).execute()
        return True
    except:
        return False


def delete_document(filename, company_id):
    try:
        supabase.table("documents").delete().eq("company_id", company_id).eq(
            "filename", filename).execute()
        supabase.table("document_chunks").delete().eq(
            "metadata->>company_id", company_id).eq("metadata->>filename",
                                                    filename).execute()
        try:
            supabase.storage.from_("documents").remove(
                [f"{company_id}/{filename}"])
        except:
            pass
        return True
    except:
        return False


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
    except:
        return "error"

    if not text: return "empty"

    # Upload raw file (optional)
    try:
        file.seek(0)
        supabase.storage.from_("documents").upload(
            f"{company_id}/{clean_name}", file.read(), {"upsert": "true"})
    except:
        pass

    # Chunking & Embedding
    chunks = smart_chunking(text)
    batch_size = 20
    bar = st.progress(0)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = get_embeddings_batch(batch)

        if vectors:
            payload = []
            for j, vec in enumerate(vectors):
                if isinstance(vec, list) and len(vec) > 300:
                    payload.append({
                        "content": batch[j],
                        "metadata": {
                            "company_id": company_id,
                            "filename": clean_name,
                            "is_active": True
                        },
                        "embedding": vec
                    })
            if payload:
                supabase.table("document_chunks").insert(payload).execute()
        bar.progress(min((i + batch_size) / len(chunks), 1.0))

    bar.empty()
    register_document(clean_name, company_id)
    return "success"


# [FIX 3] Cross-Lingual Search (Query Expansion)
def get_relevant_context(query, company_id):
    # Step A: Expand Query (English + Dutch + French Keywords)
    search_query = query
    try:
        expansion_prompt = f"""
        Translate the following user query into a list of keywords in English, Dutch, and French.
        Combine them into a single string.
        User Query: "{query}"
        Output ONLY the keywords.
        """
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{
                    "role": "user",
                    "content": expansion_prompt
                }],
                "temperature": 0.1,
                "max_tokens": 100
            },
            timeout=3)
        if resp.status_code == 200:
            search_query = resp.json()['choices'][0]['message']['content']
    except:
        pass  # Fallback to original if Groq fails

    # Step B: Vector Search with Expanded Query
    vectors = get_embeddings_batch([search_query])
    if not vectors: return "", []

    try:
        params = {
            "query_embedding": vectors[0],
            "match_threshold": 0.15,
            "match_count": 8,
            "filter_company_id": company_id,
            "query_text": query  # Placeholder
        }
        # Uses the 'match_documents_hybrid' SQL function
        res = supabase.rpc("match_documents_hybrid", params).execute()

        context_str = ""
        sources = []
        for m in res.data:
            context_str += f"-- SOURCE: {m['metadata']['filename']} --\n{m['content']}\n\n"
            if m['metadata']['filename'] not in sources:
                sources.append(m['metadata']['filename'])

        return context_str, sources
    except:
        return "", []


def ask_groq(context, history, query):
    system_prompt = """
    You are FRIDAY, an expert HR assistant.
    1. Answer strictly based on the provided CONTEXT.
    2. The context may contain Markdown tables.
    3. If the answer is not in the context, say "I couldn't find that information."
    """
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if context:
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
    messages.append({"role": "user", "content": query})

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.1
            })
        return resp.json()['choices'][0]['message']['content']
    except:
        return "Connection error."


# --- CHAT HISTORY DB ---
def load_chat_history(chat_id):
    try:
        return supabase.table("messages").select("*").eq(
            "chat_id", chat_id).order("created_at").execute().data
    except:
        return []


def save_message(chat_id, role, content, sources=None):
    try:
        supabase.table("messages").insert({
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "sources": sources
        }).execute()
    except:
        pass


# --- 4. UI PAGES ---


def render_sidebar():
    with st.sidebar:
        st.title("⚡ FRIDAY")
        st.caption(f"ID: {st.session_state.company_id}")
        st.markdown("---")
        if st.button("💬 Chat",
                     use_container_width=True,
                     type="primary"
                     if st.session_state.view == "chat" else "secondary"):
            st.session_state.view = "chat"
            st.rerun()
        if st.button("📂 Documents",
                     use_container_width=True,
                     type="primary"
                     if st.session_state.view == "documents" else "secondary"):
            st.session_state.view = "documents"
            st.rerun()

        st.markdown("---")
        if st.session_state.view == "chat":
            if st.button("➕ New Chat", use_container_width=True):
                create_new_chat()
                st.rerun()

        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()


def create_new_chat():
    st.session_state.current_chat_id = str(uuid.uuid4())


def documents_page():
    st.title("📂 Knowledge Base")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload")
        uploaded_files = st.file_uploader("Upload PDF/DOCX",
                                          type=["pdf", "docx"],
                                          accept_multiple_files=True)
        force_overwrite = st.checkbox("Overwrite existing files?")

        if uploaded_files and st.button("Index Files", type="primary"):
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    status = process_and_store_document(
                        f, st.session_state.company_id, force_overwrite)
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
                    style = "" if doc[
                        'is_active'] else "text-decoration: line-through; color: gray;"
                    c1.markdown(
                        f"{icon} <span style='{style}'>{doc['filename']}</span>",
                        unsafe_allow_html=True)

                    if c2.button("📦",
                                 key=f"arch_{doc['id']}",
                                 help="Archive/Unarchive"):
                        toggle_document_status(doc['filename'],
                                               st.session_state.company_id,
                                               doc['is_active'])
                        st.rerun()

                    if c3.button("🗑️", key=f"del_{doc['id']}"):
                        delete_document(doc['filename'],
                                        st.session_state.company_id)
                        st.rerun()
                    st.divider()


def chat_page():
    if not st.session_state.current_chat_id: create_new_chat()
    history = load_chat_history(st.session_state.current_chat_id)

    if not history:
        st.markdown(
            "<h2 style='text-align: center; color: #333;'>Good day.</h2>",
            unsafe_allow_html=True)

    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["sources"]:
                tags = "".join([
                    f"<div class='source-tag'>📄 {s}</div>"
                    for s in msg["sources"]
                ])
                st.markdown(f"<div class='source-container'>{tags}</div>",
                            unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question..."):
        save_message(st.session_state.current_chat_id, "user", prompt)
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context, sources = get_relevant_context(
                    prompt, st.session_state.company_id)
                response = ask_groq(context, history, prompt)
                st.write(response)
                if sources:
                    tags = "".join([
                        f"<div class='source-tag'>📄 {s}</div>" for s in sources
                    ])
                    st.markdown(f"<div class='source-container'>{tags}</div>",
                                unsafe_allow_html=True)
                save_message(st.session_state.current_chat_id, "assistant",
                             response, sources)


# --- 5. AUTH (FIXED BUG) ---
def login_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.title("⚡ FRIDAY Access")
        with st.form("login_form"):
            pw = st.text_input("Access Code", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                # [FIX 4] Clean Logic: Check DB -> Set State -> Rerun immediately
                try:
                    res = supabase.table('clients').select("*").eq(
                        'access_code', pw).execute()
                    if res.data:
                        st.session_state.authenticated = True
                        st.session_state.company_id = res.data[0]['company_id']
                        st.rerun()  # Immediate rerun on success
                    else:
                        st.error(
                            "Invalid Access Code"
                        )  # Only shows if submit was clicked AND failed
                except Exception as e:
                    st.error(f"Login Error: {e}")


if not st.session_state.authenticated:
    login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat": chat_page()
    elif st.session_state.view == "documents": documents_page()
