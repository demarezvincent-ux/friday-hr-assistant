import streamlit as st
from supabase import create_client, Client
import requests
from PyPDF2 import PdfReader
import docx
import re
import time
import os
import datetime
import uuid
from huggingface_hub import InferenceClient

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="FRIDAY", page_icon="⚡", layout="wide")

# Custom CSS for "Perplexity-like" feel
st.markdown("""
<style>
    /* General Fonts & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    .stApp { background-color: #F9FAFB; font-family: 'Inter', sans-serif; }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F3F4F6;
        border-right: 1px solid #E5E7EB;
    }

    /* Source Tags (Perplexity Style) */
    .source-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
        margin-top: 4px;
    }
    .source-tag {
        font-size: 0.75rem;
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 4px 10px;
        border-radius: 16px;
        color: #374151;
        font-weight: 500;
        display: flex;
        align-items: center;
        transition: all 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .source-tag:hover {
        border-color: #3D6E98;
        color: #3D6E98;
    }

    /* Message Bubbles */
    div[data-testid="stChatMessage"] {
        background-color: transparent; 
        padding: 1rem 0;
    }
    div[data-testid="stChatMessage"][data-testid="user"] {
        background-color: transparent;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""",
            unsafe_allow_html=True)

# --- SECRETS HANDLING ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    FIXED_GROQ_KEY = st.secrets["FIXED_GROQ_KEY"]
    HF_API_KEY = st.secrets["HF_API_KEY"]
except:
    try:
        SUPABASE_URL = os.environ["SUPABASE_URL"]
        SUPABASE_KEY = os.environ["SUPABASE_KEY"]
        FIXED_GROQ_KEY = os.environ["FIXED_GROQ_KEY"]
        HF_API_KEY = os.environ["HF_API_KEY"]
    except:
        st.error(
            "❌ Critical Error: API Keys missing. Please configure Secrets.")
        st.stop()

# --- 2. STATE & DATABASE INIT ---

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "company_id" not in st.session_state: st.session_state.company_id = None
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Dictionary of chat sessions
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "view" not in st.session_state:
    st.session_state.view = "chat"  # 'chat' or 'documents'


@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        return None


supabase = init_supabase()

# --- 3. CORE LOGIC (BACKEND) ---


def get_embeddings_batch(texts):
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    client = InferenceClient(token=HF_API_KEY)
    clean_texts = [t.replace("\n", " ").strip() for t in texts]

    # Retry logic for HuggingFace cold starts
    for attempt in range(3):
        try:
            embeddings = client.feature_extraction(clean_texts, model=model_id)
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return embeddings
        except Exception as e:
            time.sleep(2)
    return None


def sanitize_filename(filename):
    name = filename.replace(" ", "_")
    return re.sub(r'[^a-zA-Z0-9._-]', '', name)


def semantic_chunking(text, chunk_size=500):
    # Split by sentence endings to keep context intact
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and len(
                current_chunk) > 0:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks


def delete_document(filename, company_id):
    try:
        # 1. Remove from Storage
        path = f"{company_id}/{filename}"
        supabase.storage.from_("documents").remove([path])

        # 2. Remove embeddings from Database
        # Note: This requires your DB to have metadata->>filename matching.
        # If your vector table doesn't allow easy metadata filtering, this part might fail silently
        # but the storage delete is the most important for the user.
        supabase.table("document_chunks").delete().eq(
            "metadata->>company_id", company_id).eq("metadata->>filename",
                                                    filename).execute()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def process_and_store_document(file, company_id, force_overwrite=False):
    # Check if exists first
    clean_name = sanitize_filename(file.name)
    path = f"{company_id}/{clean_name}"

    # Check existence
    try:
        existing = supabase.storage.from_("documents").list(path=company_id)
        existing_names = [f['name'] for f in existing]

        if clean_name in existing_names:
            if not force_overwrite:
                return "exists"  # Signal to UI that it exists
            else:
                # If overwriting, first delete old chunks
                delete_document(clean_name, company_id)
    except:
        pass

    # Extract Text
    text = ""
    try:
        if file.name.endswith(".pdf"):
            pdf = PdfReader(file)
            text = "".join([page.extract_text() for page in pdf.pages])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
    except:
        return "error"

    if not text: return "empty"

    # Upload File to Storage
    try:
        file.seek(0)
        supabase.storage.from_("documents").upload(path, file.read(),
                                                   {"upsert": "true"})
    except Exception as e:
        pass  # Handle storage error gracefully

    # Chunk & Embed
    chunks = semantic_chunking(text)
    batch_size = 20

    # Using a placeholder for progress to not block UI
    progress_text = st.empty()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = get_embeddings_batch(batch)

        if vectors:
            payload = []
            for j, vector in enumerate(vectors):
                # Sanity check vector length
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

        progress_text.caption(
            f"Processing {file.name}: {min(100, int((i/len(chunks))*100))}%")

    progress_text.empty()
    return "success"


def get_relevant_context(query, company_id):
    vectors = get_embeddings_batch([query])
    if not vectors: return "", []

    try:
        params = {
            "query_embedding": vectors[0],
            "match_threshold": 0.35,  # Strictness
            "match_count": 6,
            "filter_company_id": company_id
        }
        response = supabase.rpc("match_documents", params).execute()

        # Build context string and list of sources
        matches = response.data
        context_str = ""
        sources = []

        for m in matches:
            context_str += f"-- SOURCE: {m['metadata']['filename']} --\n{m['content']}\n\n"
            if m['metadata']['filename'] not in sources:
                sources.append(m['metadata']['filename'])

        return context_str, sources
    except Exception as e:
        print(e)
        return "", []


def ask_groq(context, history, query):
    system_prompt = """
    You are FRIDAY, an expert HR assistant.
    1. USE ONLY the provided CONTEXT SNIPPETS to answer.
    2. If the answer is not in the context, say "I couldn't find that information in the documents."
    3. Do not mention "context snippets" in your output, just answer naturaly.
    4. Keep tone professional and concise.
    """

    # We construct messages specifically for this turn to ensure context isolation
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent history (last 2 turns) for conversation flow, but NOT previous contexts
    # We filter history to only include user/assistant content, stripping metadata
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Inject CURRENT context immediately before the query
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
                "temperature": 0.1  # Low temp for factual accuracy
            })
        return resp.json()['choices'][0]['message']['content']
    except:
        return "I'm having trouble connecting to my brain right now."


# --- 4. UI COMPONENTS ---


def render_sidebar():
    with st.sidebar:
        st.title("⚡ FRIDAY")
        st.caption(f"Workspace: {st.session_state.company_id}")

        st.markdown("---")

        # Navigation
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

        # Chat History Management
        if st.session_state.view == "chat":
            st.subheader("History")
            if st.button("➕ New Chat", use_container_width=True):
                create_new_chat()
                st.rerun()

            # Show list of past chats
            chat_ids = list(st.session_state.chats.keys())
            if chat_ids:
                for cid in reversed(chat_ids[-5:]):  # Show last 5
                    # Get first message as title or default
                    msgs = st.session_state.chats[cid]
                    title = msgs[0][
                        'content'][:25] + "..." if msgs else "Empty Chat"
                    if st.button(f"Draft: {title}", key=cid):
                        st.session_state.current_chat_id = cid
                        st.rerun()

        # Logout
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()


def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = []
    st.session_state.current_chat_id = new_id


# --- 5. PAGES ---


def documents_page():
    st.title("📂 Knowledge Base")
    st.markdown("Manage the documents FRIDAY uses to answer your questions.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Drop PDF or DOCX here",
                                          type=["pdf", "docx"],
                                          accept_multiple_files=True)
        force_overwrite = st.checkbox("Overwrite files if they already exist?")

        if uploaded_files and st.button("Process Files", type="primary"):
            for f in uploaded_files:
                with st.spinner(f"Reading {f.name}..."):
                    status = process_and_store_document(
                        f, st.session_state.company_id, force_overwrite)
                    if status == "success":
                        st.toast(f"✅ Indexed: {f.name}")
                    elif status == "exists":
                        st.warning(
                            f"⚠️ {f.name} already exists. Check 'Overwrite' to replace it."
                        )
                    else:
                        st.error(f"❌ Failed to process {f.name}")
            time.sleep(1)
            st.rerun()

    with col2:
        st.subheader("Parsed Files")
        # Fetch current files
        try:
            files = supabase.storage.from_("documents").list(
                path=st.session_state.company_id)
            # Filter out empty folder placeholder
            files = [
                f for f in files if f['name'] != '.emptyFolderPlaceholder'
            ]

            if not files:
                st.info("No documents found.")
            else:
                for f in files:
                    # File Card
                    with st.container():
                        c1, c2 = st.columns([4, 1])
                        c1.markdown(f"📄 **{f['name']}**")
                        if c2.button("🗑️",
                                     key=f"del_{f['name']}",
                                     help="Delete"):
                            if delete_document(f['name'],
                                               st.session_state.company_id):
                                st.toast("Deleted")
                                time.sleep(0.5)
                                st.rerun()
                        st.divider()
        except Exception as e:
            st.error("Could not load file list.")


def chat_page():
    # Ensure chat session exists
    if not st.session_state.current_chat_id:
        create_new_chat()

    current_messages = st.session_state.chats[st.session_state.current_chat_id]

    # Greeting based on time
    hour = datetime.datetime.now().hour
    greeting = "Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 18 else "Good evening"

    if not current_messages:
        st.markdown(
            f"<h2 style='text-align: center; color: #333;'>{greeting}, welcome to FRIDAY.</h2>",
            unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: center; color: #666;'>Ask me anything about your documents.</p>",
            unsafe_allow_html=True)

    # Render Chat History
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # Render Sources visually distinct (Perplexity style)
            if "sources" in msg and msg["sources"]:
                # Generate HTML for badges
                tags_html = "".join([
                    f"<div class='source-tag'>📄 {s}</div>"
                    for s in msg["sources"]
                ])
                st.markdown(f"<div class='source-container'>{tags_html}</div>",
                            unsafe_allow_html=True)

    # Input Area
    if prompt := st.chat_input("Ask a question..."):
        # 1. Add User Message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chats[st.session_state.current_chat_id].append(
            user_msg)
        with st.chat_message("user"):
            st.write(prompt)

        # 2. Process Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                # Get Context & Sources
                context_text, sources = get_relevant_context(
                    prompt, st.session_state.company_id)

                # Get LLM Response
                response_text = ask_groq(context_text, current_messages,
                                         prompt)

                # Display Answer
                st.write(response_text)

                # Display Sources immediately
                if sources:
                    tags_html = "".join([
                        f"<div class='source-tag'>📄 {s}</div>" for s in sources
                    ])
                    st.markdown(
                        f"<div class='source-container'>{tags_html}</div>",
                        unsafe_allow_html=True)

        # 3. Save Assistant Message
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "sources":
            sources  # Store ONLY sources used for this specific answer
        }
        st.session_state.chats[st.session_state.current_chat_id].append(
            assistant_msg)


# --- 6. AUTHENTICATION & MAIN ROUTER ---


def login_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.title("⚡ FRIDAY Access")
        with st.form("login_form"):
            password = st.text_input("Company Access Code", type="password")
            submit = st.form_submit_button("Enter Workspace",
                                           use_container_width=True)

            if submit:
                if not supabase:
                    st.error("Database Error")
                else:
                    try:
                        res = supabase.table('clients').select("*").eq(
                            'access_code', password).execute()
                        if res.data:
                            st.session_state.authenticated = True
                            st.session_state.company_id = res.data[0][
                                'company_id']
                            st.rerun()
                        else:
                            st.error("Invalid Code")
                    except Exception as e:
                        st.error(f"Login Error: {e}")


# Main Router
if not st.session_state.authenticated:
    login_page()
else:
    render_sidebar()
    if st.session_state.view == "chat":
        chat_page()
    elif st.session_state.view == "documents":
        documents_page()
