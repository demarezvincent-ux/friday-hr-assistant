import streamlit as st
from supabase import create_client, Client
import requests
import pdfplumber  # CRITICAL: Ensures tables are read correctly
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
if "chats" not in st.session_state: st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
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

    for attempt in range(3):
        try:
            embeddings = client.feature_extraction(clean_texts, model=model_id)
            if hasattr(embeddings, "tolist"): return embeddings.tolist()
            return embeddings
        except:
            time.sleep(2)
    return None


def sanitize_filename(filename):
    name = filename.replace(" ", "_")
    return re.sub(r'[^a-zA-Z0-9._-]', '', name)


def smart_chunking(text, chunk_size=1000, overlap=200):
    """
    Splits text by character count with overlap to preserve table context.
    Does NOT split by punctuation, preserving table formatting.
    """
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


# --- DATABASE REGISTRY FUNCTIONS ---


def register_document(filename, company_id):
    try:
        supabase.table("documents").insert({
            "company_id": company_id,
            "filename": filename
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
        res = supabase.table("documents").select("*").eq(
            "company_id", company_id).order('created_at', desc=True).execute()
        return res.data
    except:
        return []


def delete_document(filename, company_id):
    try:
        supabase.table("documents").delete().eq("company_id", company_id).eq(
            "filename", filename).execute()
        supabase.table("document_chunks").delete().eq(
            "metadata->>company_id", company_id).eq("metadata->>filename",
                                                    filename).execute()
        try:
            path = f"{company_id}/{filename}"
            supabase.storage.from_("documents").remove([path])
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
        if file.name.endswith(".pdf"):
            # Using pdfplumber for better table extraction
            with pdfplumber.open(file) as pdf:
                pages = [p.extract_text(layout=True) or "" for p in pdf.pages]
                text = "\n".join(pages)
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return "error"

    if not text: return "empty"

    # Optional: Upload raw file
    try:
        file.seek(0)
        supabase.storage.from_("documents").upload(
            f"{company_id}/{clean_name}", file.read(), {"upsert": "true"})
    except:
        pass

    chunks = smart_chunking(text, chunk_size=1000, overlap=200)
    batch_size = 20
    progress_bar = st.progress(0)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = get_embeddings_batch(batch)

        if vectors:
            payload = []
            for j, vector in enumerate(vectors):
                if isinstance(vector, list) and len(vector) > 300:
                    payload.append({
                        "content": batch[j],
                        "metadata": {
                            "company_id": company_id,
                            "filename": clean_name
                        },
                        "embedding": vector
                    })
            if payload:
                supabase.table("document_chunks").insert(payload).execute()

        progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))

    progress_bar.empty()
    register_document(clean_name, company_id)
    return "success"


def get_relevant_context(query, company_id):
    vectors = get_embeddings_batch([query])
    if not vectors: return "", []

    try:
        params = {
            "query_embedding": vectors[0],
            "match_threshold":
            0.15,  # <--- FIXED: Lowered from 0.35 to 0.15 to match original behavior
            "match_count": 8,  # <--- INCREASED: Fetch more chunks to be safe
            "filter_company_id": company_id
        }
        response = supabase.rpc("match_documents", params).execute()

        context_str = ""
        sources = []
        for m in response.data:
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
    2. The context may contain tables formatted as text. Look closely at line alignment.
    3. If the answer is not in the context, say "I couldn't find that information."
    """

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if context:
        messages.append({
            "role": "user",
            "content": f"CONTEXT SNIPPETS:\n{context}"
        })

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

            chat_ids = list(st.session_state.chats.keys())
            if chat_ids:
                st.caption("Recent Chats")
                for cid in reversed(chat_ids[-5:]):
                    msgs = st.session_state.chats[cid]
                    title = msgs[0][
                        'content'][:20] + "..." if msgs else "New Chat"
                    if st.button(f"📝 {title}", key=cid):
                        st.session_state.current_chat_id = cid
                        st.rerun()

        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()


def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = []
    st.session_state.current_chat_id = new_id


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
                    elif status == "exists":
                        st.warning(f"⚠️ {f.name} exists. Enable Overwrite.")
                    else:
                        st.error(f"❌ Error: {f.name}")
            time.sleep(1)
            st.rerun()

    with col2:
        st.subheader("Indexed Documents")
        docs = get_all_documents(st.session_state.company_id)
        if not docs: st.info("No documents found.")
        else:
            for doc in docs:
                with st.container():
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(f"📄 **{doc['filename']}**")
                    if c2.button("🗑️", key=f"del_{doc['id']}", help="Delete"):
                        if delete_document(doc['filename'],
                                           st.session_state.company_id):
                            st.toast("Deleted")
                            time.sleep(0.5)
                            st.rerun()
                    st.divider()


def chat_page():
    if not st.session_state.current_chat_id: create_new_chat()
    current_messages = st.session_state.chats[st.session_state.current_chat_id]

    hour = datetime.datetime.now().hour
    greeting = "Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 18 else "Good evening"

    if not current_messages:
        st.markdown(
            f"<h2 style='text-align: center; color: #333;'>{greeting}.</h2>",
            unsafe_allow_html=True)

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                tags_html = "".join([
                    f"<div class='source-tag'>📄 {s}</div>"
                    for s in msg["sources"]
                ])
                st.markdown(f"<div class='source-container'>{tags_html}</div>",
                            unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question..."):
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chats[st.session_state.current_chat_id].append(
            user_msg)
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context, sources = get_relevant_context(
                    prompt, st.session_state.company_id)
                response = ask_groq(context, current_messages, prompt)
                st.write(response)
                if sources:
                    tags_html = "".join([
                        f"<div class='source-tag'>📄 {s}</div>" for s in sources
                    ])
                    st.markdown(
                        f"<div class='source-container'>{tags_html}</div>",
                        unsafe_allow_html=True)

        assistant_msg = {
            "role": "assistant",
            "content": response,
            "sources": sources
        }
        st.session_state.chats[st.session_state.current_chat_id].append(
            assistant_msg)


# --- 5. AUTH ---
def login_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.title("⚡ FRIDAY Access")
        with st.form("login"):
            pw = st.text_input("Access Code", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                try:
                    res = supabase.table('clients').select("*").eq(
                        'access_code', pw).execute()
                    if res.data:
                        st.session_state.authenticated = True
                        st.session_state.company_id = res.data[0]['company_id']
                        st.rerun()
                    else:
                        st.error("Invalid Code")
                except:
                    st.error("Login Error")


if not st.session_state.authenticated: login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat": chat_page()
    elif st.session_state.view == "documents": documents_page()
