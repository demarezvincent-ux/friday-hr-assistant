import streamlit as st
from supabase import create_client, Client
import requests
from PyPDF2 import PdfReader
import docx
import io
import re
import time
import os
import datetime
from huggingface_hub import InferenceClient

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="FRIDAY", page_icon="⚡", layout="centered")

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
        st.error("❌ Geen API Keys gevonden! Voeg ze toe aan Secrets.")
        st.stop()

# --- DATABASE SETUP ---
@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        return None

supabase = init_supabase()

# --- 2. API FUNCTIONS ---

def get_embeddings_batch(texts):
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    client = InferenceClient(token=HF_API_KEY)
    # Newlines verwijderen voor betere embeddings
    clean_texts = [t.replace("\n", " ").strip() for t in texts]

    for attempt in range(3):
        try:
            embeddings = client.feature_extraction(clean_texts, model=model_id)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            return embeddings
        except Exception as e:
            if "503" in str(e) or "loading" in str(e).lower():
                time.sleep(2)
                continue
            else:
                return None
    return None

def ask_groq(context, chat_history, question):
    if not context:
        context = "No relevant documents found. Answer generally."

    system_prompt = """
    You are FRIDAY, an intelligent HR assistant.
    Tone: Professional, calm, Claude-like.
    Instructions: 
    1. Answer strictly based on the provided CONTEXT SNIPPETS.
    2. Answer in the SAME language as the user.
    3. Be concise and to the point.
    """

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT SNIPPETS:\n{context}"}]

    # --- CRUCIAAL: History opschonen (geen 'sources' meesturen naar Groq) ---
    for msg in chat_history[-4:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    messages.append({"role": "user", "content": question})

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.3
            })

        if resp.status_code != 200:
            return f"Error {resp.status_code}: {resp.text}"

        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# --- 3. DOCUMENT PROCESSING ---

def sanitize_filename(filename):
    name = filename.replace(" ", "_")
    return re.sub(r'[^a-zA-Z0-9._-]', '', name)

def semantic_chunking(text, chunk_size=500):
    """
    Splits text op basis van leestekens (. ! ?) zodat zinnen heel blijven.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and len(current_chunk) > 0:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_and_store_document(file, company_id):
    text = ""
    try:
        if file.name.endswith(".pdf"):
            pdf = PdfReader(file)
            text = "".join([page.extract_text() for page in pdf.pages])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
    except:
        return False

    if not text: return False

    # Upload raw file naar Storage
    try:
        file.seek(0)
        path = f"{company_id}/{sanitize_filename(file.name)}"
        supabase.storage.from_("documents").upload(path, file.read(), {"upsert": "true"})
    except:
        pass

    # Chunking met de nieuwe logica
    chunks = semantic_chunking(text, chunk_size=500)
    batch_size = 20
    progress_bar = st.progress(0)

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        vectors = get_embeddings_batch(batch_chunks)

        if vectors:
            data_payload = []
            for j, vector in enumerate(vectors):
                if isinstance(vector, list) and len(vector) > 300:
                    data_payload.append({
                        "content": batch_chunks[j],
                        "metadata": {
                            "company_id": company_id,
                            "filename": file.name
                        },
                        "embedding": vector
                    })

            if data_payload:
                try:
                    supabase.table("document_chunks").insert(data_payload).execute()
                except Exception as e:
                    print(f"DB Error: {e}")

        progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))

    return True

def get_relevant_context(query, company_id):
    vectors = get_embeddings_batch([query])
    if not vectors or len(vectors) == 0: return "", []

    query_vector = vectors[0]

    try:
        params = {
            "query_embedding": query_vector,
            "match_threshold": 0.1,  # <--- AANGEPAST: Lager gezet (was 0.3)
            "match_count": 5,        # <--- AANGEPAST: Meer resultaten (was 4)
            "filter_company_id": company_id
        }
        response = supabase.rpc("match_documents", params).execute()

        context_text = "\n".join([f"-- Snippet --\n{match['content']}\n" for match in response.data])

        # Unieke bronnen verzamelen
        sources = list(set([match['metadata']['filename'] for match in response.data]))

        return context_text, sources
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return "", []

# --- 4. APP UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Inter:wght@400;600&display=swap');
    .stApp { background-color: #FAF9F6; color: #2D2D2D; }
    h1, h2, h3 { font-family: 'Merriweather', serif !important; color: #1A1A1A; }
    div.stButton > button {
        background-color: #3D6E98 !important; color: white !important;
        border-radius: 20px !important; border: none !important;
        box-shadow: 0 2px 5px rgba(61, 110, 152, 0.2);
    }
    div[data-testid="stChatMessage"][data-testid="user"] { background-color: #F0EFEB !important; border-radius: 12px; font-family: 'Inter'; }
    section[data-testid="stSidebar"] { background-color: #F7F7F5; border-right: 1px solid #EAEAEA; }

    .source-tag {
        font-size: 0.8rem;
        background-color: #e0e0e0;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 5px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "company_id" not in st.session_state: st.session_state.company_id = None

# --- LOGIN ---
def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Welcome to FRIDAY</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            password = st.text_input("Access Code", type="password", label_visibility="collapsed")
            submit_button = st.form_submit_button("Enter Workspace", use_container_width=True)

            if submit_button:
                if not supabase:
                    st.error("Database connection failed")
                else:
                    try:
                        response = supabase.table('clients').select("*").eq('access_code', password).execute()
                        if response.data:
                            st.session_state.authenticated = True
                            st.session_state.company_id = response.data[0]['company_id']
                            st.rerun()
                        else:
                            st.error("Invalid access code")
                    except Exception as e:
                        st.error("Login failed. Please try again.")

# --- MAIN APP ---
def main_app():
    with st.sidebar:
        st.markdown(f"### {st.session_state.company_id}")
        st.markdown("---")

        if "hf_" not in HF_API_KEY:
            st.error("⚠️ Please add your Hugging Face API Key!")
            st.stop()

        active_files = []
        try:
            files = supabase.storage.from_("documents").list(path=st.session_state.company_id)
            active_files = [f['name'] for f in files if f['name'] != '.emptyFolderPlaceholder']

            st.markdown("**Knowledge Base**")
            for f in active_files:
                st.markdown(f"<div style='padding: 8px; background: white; margin-bottom: 5px; border-radius: 5px; border: 1px solid #eee; font-size: 0.85rem;'>📄 {f}</div>", unsafe_allow_html=True)
        except:
            pass

        with st.expander("Upload New Files"):
            up_files = st.file_uploader("Index Documents", type=["pdf", "docx"], accept_multiple_files=True)
            if up_files and st.button("Process & Index"):
                for f in up_files:
                    clean_name = sanitize_filename(f.name)
                    if clean_name in active_files:
                        st.toast(f"Skipping {f.name} (Already Indexed)", icon="⏭️")
                        continue

                    with st.spinner(f"Indexing {f.name}..."):
                        success = process_and_store_document(f, st.session_state.company_id)
                        if success:
                            st.toast(f"Indexed {f.name}", icon="✅")
                            active_files.append(clean_name)
                        else:
                            st.toast(f"Failed {f.name}", icon="❌")
                time.sleep(1)
                st.rerun()

        # --- RESET BUTTON (Handig voor testen) ---
        with st.expander("⚠️ Danger Zone"):
            if st.button("Hard Reset (Delete All)", type="primary"):
                try:
                    supabase.table("document_chunks").delete().neq("id", 0).execute()
                    files = supabase.storage.from_("documents").list(path=st.session_state.company_id)
                    if files:
                        file_paths = [f"{st.session_state.company_id}/{f['name']}" for f in files]
                        supabase.storage.from_("documents").remove(file_paths)
                    st.toast("Alles gewist!", icon="🗑️")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("Log Out", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.company_id = None
            st.rerun()

    # DYNAMIC GREETING
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning"
    elif 12 <= current_hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    st.markdown(f"<h3>{greeting}. How can I help you?</h3>", unsafe_allow_html=True)

    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.caption(f"📚 Sources: {', '.join(msg['sources'])}")

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                relevant_context, sources = get_relevant_context(prompt, st.session_state.company_id)

                # --- DEBUG: KIJKEN WAT HIJ VINDT ---
                with st.expander("🔍 Debug: Bekijk gevonden context"):
                    st.write(relevant_context if relevant_context else "❌ Geen relevante tekst gevonden (Vectorscore te laag)")

                resp = ask_groq(relevant_context, st.session_state.messages, prompt)
                st.markdown(resp)

                if sources:
                    st.caption(f"📚 Sources: {', '.join(sources)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": resp,
            "sources": sources
        })

if st.session_state.authenticated: main_app()
else: login_page()