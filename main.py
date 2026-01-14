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
    
    /* 1. Typography - Playfair Display for headings, Inter for body */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        /* Brand Colors - Deep Forest Green */
        --primary: #1A3C34;
        --primary-hover: #0F2921;
        --primary-gradient: linear-gradient(135deg, #1A3C34 0%, #0F2921 100%);
        
        /* Neutral Colors */
        --bg-primary: #F5F5F7;
        --bg-secondary: #FAFAFA;
        --bg-white: #FFFFFF;
        --sidebar-bg: #F5F5F7;
        
        /* Text Colors */
        --text-primary: #1D1D1F;
        --text-secondary: #5C5C61;
        --text-tertiary: #86868B;
        
        /* Primary Tints */
        --primary-light: rgba(26, 60, 52, 0.08);
        --primary-lighter: rgba(26, 60, 52, 0.04);
        --text-on-primary: #FFFFFF;
        
        /* Border & Divider */
        --border-color: #D2D2D7;
        --border-light: rgba(0, 0, 0, 0.06);
        --divider: rgba(0, 0, 0, 0.08);
        
        /* Semantic Colors */
        --error: #FF3B30;
        --warning: #FF9500;
        --success: #34C759;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.08);
        
        /* Transitions */
        --transition-fast: all 0.15s ease;
        --transition-normal: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Global Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        min-height: 100vh;
    }

    /* ============================================
       2. SIDEBAR - Apple Light Style
       ============================================ */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--divider);
        width: 300px !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 20px 16px;
    }

    /* Sidebar Text - Dark on Light */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }

    /* Section Headers in Sidebar */
    [data-testid="stSidebar"] h3 {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: var(--text-tertiary) !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin: 28px 0 12px 0 !important;
    }
    
    /* Sidebar Navigation Buttons */
    [data-testid="stSidebar"] .stButton button {
        background-color: transparent;
        color: var(--text-primary) !important;
        border: none;
        text-align: left;
        padding: 12px 16px;
        height: 44px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(0, 0, 0, 0.05);
        transform: translateX(4px);
    }

    /* Active Navigation Button - Selected State */
    div[data-testid="stSidebar"] button[kind="primary"],
    div[data-testid="stSidebar"] button[kind="primary"]:hover,
    div[data-testid="stSidebar"] button[kind="primary"]:focus,
    div[data-testid="stSidebar"] button[kind="primary"]:active {
        background: var(--primary-gradient) !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        border-left: none !important;
        padding-left: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(26, 60, 52, 0.35);
        transform: scale(1.02);
    }
    
    /* Force white text on ALL elements inside active sidebar button */
    div[data-testid="stSidebar"] button[kind="primary"] *,
    div[data-testid="stSidebar"] button[kind="primary"] p,
    div[data-testid="stSidebar"] button[kind="primary"] span,
    div[data-testid="stSidebar"] button[kind="primary"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Inactive Navigation Button - Not Selected State */
    div[data-testid="stSidebar"] button[kind="secondary"],
    div[data-testid="stSidebar"] button[kind="secondary"]:hover,
    div[data-testid="stSidebar"] button[kind="secondary"]:focus {
        background-color: transparent !important;
        color: #1D1D1F !important;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"] *,
    div[data-testid="stSidebar"] button[kind="secondary"] p,
    div[data-testid="stSidebar"] button[kind="secondary"] span,
    div[data-testid="stSidebar"] button[kind="secondary"] div {
        color: #1D1D1F !important;
        -webkit-text-fill-color: #1D1D1F !important;
    }
    
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: rgba(0, 0, 0, 0.06) !important;
        transform: translateX(4px);
    }

    /* Sidebar Dividers */
    [data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid var(--divider);
        margin: 16px 0;
    }

    /* Recent Chats Styling */
    .recent-chat-item {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: var(--transition-fast);
    }
    
    .recent-chat-item:hover {
        background-color: rgba(0, 0, 0, 0.06);
        transform: translateX(2px);
        transition: all 0.2s ease;
    }
    
    .recent-chat-item.active {
        background-color: rgba(0, 191, 165, 0.08);
        border-left: 3px solid var(--primary);
    }

    /* Logout Button - Destructive Style */
    [data-testid="stSidebar"] .logout-btn button {
        color: var(--error) !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .logout-btn button:hover {
        background-color: rgba(255, 59, 48, 0.1) !important;
    }

    /* ============================================
       3. MAIN CONTENT AREA
       ============================================ */
    
    /* Hero/Greeting Headers */
    .hero-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 56px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
        letter-spacing: -1.5px;
        margin-bottom: 16px;
    }
    
    .hero-subtitle {
        font-size: 19px;
        font-weight: 400;
        color: #86868B;
        line-height: 1.4;
        margin-top: 8px;
    }

    .greeting-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 48px;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 12px;
        letter-spacing: -1px;
    }
    
    .greeting-subtitle {
        font-size: 19px;
        font-weight: 400;
        color: var(--text-secondary);
        line-height: 1.4;
    }

    /* ============================================
       4. FORM ELEMENTS - Apple Style
       ============================================ */
    
    /* Text Inputs */
    .stTextInput > div > div > input {
        background-color: var(--bg-white);
        border: 1.5px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
        height: 52px;
        padding: 0 16px;
        font-size: 17px;
        box-shadow: var(--shadow-sm);
        transition: var(--transition-normal);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        border-width: 2.5px !important;
        box-shadow: 0 0 0 6px rgba(26, 60, 52, 0.12) !important;
        transform: translateY(-1px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-tertiary) !important;
        opacity: 1;
    }
    
    /* Hide helper text */
    .stTextInput > div > div > div[data-baseweb="form-control-meta"] {
        display: none;
    }

    /* Chat Input */
    .stChatInput > div > div > textarea {
        background-color: var(--bg-primary);
        border: 1.5px solid transparent;
        border-radius: 14px;
        color: var(--text-primary);
        min-height: 60px;
        padding: 18px 56px 18px 20px;
        font-size: 17px;
        transition: var(--transition-normal);
    }

    .stChatInput > div > div > textarea:focus {
        background-color: var(--bg-white);
        border-color: var(--primary) !important;
        border-width: 2px !important;
        box-shadow: 0 0 0 5px rgba(26, 60, 52, 0.12) !important;
    }
    
    .stChatInput > div > div > textarea::placeholder {
        color: var(--text-tertiary) !important;
    }

    /* Primary Buttons */
    .stButton > button[kind="primary"],
    button[kind="primary"] {
        background: var(--primary-gradient) !important;
        color: var(--text-on-primary) !important;
        border: none !important;
        border-radius: 10px;
        height: 52px;
        font-size: 17px;
        font-weight: 600;
        padding: 0 24px;
        box-shadow: 0 2px 8px rgba(26, 60, 52, 0.3);
        transition: var(--transition-normal);
    }

    .stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(26, 60, 52, 0.35) !important;
        filter: brightness(1.05);
    }

    .stButton > button[kind="primary"]:active,
    button[kind="primary"]:active {
        transform: translateY(0);
    }

    /* Secondary Buttons */
    .stButton > button[kind="secondary"] {
        background-color: var(--bg-white);
        color: var(--primary);
        border: 1.5px solid var(--border-color);
        border-radius: 10px;
        height: 44px;
        font-size: 15px;
        font-weight: 500;
        transition: var(--transition-fast);
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: var(--primary);
        background-color: rgba(0, 191, 165, 0.04);
    }

    /* Remove focus outlines */
    *:focus-visible {
        outline: none !important;
    }

    /* ============================================
       5. CHAT MESSAGES - Modern Bubbles
       ============================================ */
    
    /* User Messages */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: transparent;
    }
    
    [data-testid="stChatMessage"][data-testid*="user"] > div {
        background: var(--primary-gradient);
        color: white;
        border-radius: 20px 20px 4px 20px;
        padding: 14px 20px;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(0, 191, 165, 0.2);
    }

    /* Assistant Messages */
    [data-testid="stChatMessage"][data-testid*="assistant"] > div {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        border-radius: 18px 18px 18px 4px;
        padding: 16px 20px;
        max-width: 70%;
        line-height: 1.6;
    }

    /* ============================================
       6. SOURCE TAGS
       ============================================ */
    .source-container { 
        display: flex; 
        flex-wrap: wrap; 
        gap: 8px; 
        margin-top: 12px; 
    }
    
    .source-tag {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', sans-serif;
        font-size: 13px;
        font-weight: 600;
        background-color: var(--bg-white);
        border: 1px solid var(--border-color);
        padding: 8px 14px;
        border-radius: 99px;
        color: var(--text-secondary);
        display: inline-flex;
        align-items: center;
        transition: var(--transition-fast);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    .source-tag:hover { 
        border-color: var(--primary); 
        color: var(--primary); 
        background-color: rgba(26, 60, 52, 0.04);
    }

    /* ============================================
       7. LOGIN PAGE SPECIFIC
       ============================================ */
    .login-container {
        background: linear-gradient(135deg, #FFFFFF 0%, #FAFAFA 100%);
        border-radius: 20px;
        padding: 56px 48px;
        max-width: 480px;
        margin: 0 auto;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
        border: 1px solid var(--border-light);
    }

    .logo-gradient {
        font-family: 'Playfair Display', serif !important;
        font-size: 42px;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1.5px;
    }

    /* ============================================
       8. FILE UPLOADER - Documents Page
       ============================================ */
    [data-testid="stFileUploader"] {
        border: 3px dashed var(--border-color);
        background-color: var(--bg-secondary);
        border-radius: 12px;
        padding: 3rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background-color: rgba(26, 60, 52, 0.06);
        transform: scale(1.02);
    }

    /* File List Cards */
    .file-card {
        background: var(--bg-white);
        border-radius: 12px;
        padding: 8px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-light);
    }
    
    .file-item {
        display: flex;
        align-items: center;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 6px;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    
    .file-item:hover {
        background-color: rgba(26, 60, 52, 0.04);
        border-color: var(--border-color);
        transform: translateX(4px);
    }

    /* File Type Icons */
    .file-icon-pdf { color: #FF3B30; font-size: 18px; font-weight: 700; }
    .file-icon-docx { color: #007AFF; font-size: 18px; font-weight: 700; }
    .file-icon-default { color: var(--text-tertiary); font-size: 18px; }

    /* iOS-style Toggle */
    .ios-toggle input[type="checkbox"] {
        appearance: none;
        width: 52px;
        height: 32px;
        background-color: #E5E5EA;
        border-radius: 16px;
        position: relative;
        cursor: pointer;
        transition: var(--transition-normal);
    }
    
    .ios-toggle input[type="checkbox"]:checked {
        background-color: var(--primary);
    }
    
    .ios-toggle input[type="checkbox"]::before {
        content: '';
        position: absolute;
        width: 28px;
        height: 28px;
        background-color: white;
        border-radius: 50%;
        top: 2px;
        left: 2px;
        transition: var(--transition-normal);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .ios-toggle input[type="checkbox"]:checked::before {
        transform: translateX(20px);
    }

    /* ============================================
       9. ANIMATIONS - Enhanced with smooth transitions
       ============================================ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(26, 60, 52, 0.2); }
        50% { box-shadow: 0 0 20px rgba(26, 60, 52, 0.4); }
    }

    /* Apply animations to elements */
    .stMarkdown {
        animation: fadeIn 0.4s ease-out forwards;
    }
    
    .stButton {
        animation: fadeIn 0.3s ease-out forwards;
    }
    
    .stButton button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
    }
    
    .stButton button:active {
        transform: translateY(0) scale(0.98);
    }
    
    .stChatMessage {
        animation: fadeInUp 0.4s ease-out forwards;
    }
    
    /* Sidebar animations */
    [data-testid="stSidebar"] {
        animation: fadeInLeft 0.5s ease-out forwards;
    }
    
    /* Chat input animation */
    .stChatInput {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    /* File uploader animation */
    [data-testid="stFileUploader"] {
        animation: scaleIn 0.4s ease-out forwards;
    }
    
    /* Hero/greeting animation */
    .greeting-title, .hero-title {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    .greeting-subtitle, .hero-subtitle {
        animation: fadeInUp 0.6s ease-out 0.1s forwards;
        opacity: 0;
    }
    
    /* Source tags animation */
    .source-tag {
        animation: scaleIn 0.3s ease-out forwards;
        transition: all 0.25s ease !important;
    }
    
    .source-tag:hover {
        transform: translateY(-2px) scale(1.02);
    }
    
    /* Recent chat items animation */
    .recent-chat-item {
        animation: fadeInLeft 0.3s ease-out forwards;
    }
    
    /* Login container animation */
    .login-container {
        animation: scaleIn 0.5s ease-out forwards;
    }
    
    /* Input focus animation */
    .stTextInput > div > div > input,
    .stChatInput > div > div > textarea {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Logo float animation */
    .logo-animated {
        animation: float 3s ease-in-out infinite;
    }
    
    .thinking-dots {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Staggered animation delays for list items */
    .stButton:nth-child(1) { animation-delay: 0.05s; }
    .stButton:nth-child(2) { animation-delay: 0.1s; }
    .stButton:nth-child(3) { animation-delay: 0.15s; }
    .stButton:nth-child(4) { animation-delay: 0.2s; }
    .stButton:nth-child(5) { animation-delay: 0.25s; }

    /* ============================================
       10. SCROLLBAR - Minimal Style
       ============================================ */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: var(--border-color);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: var(--text-tertiary);
    }

    /* ============================================
       11. RESPONSIVE BREAKPOINTS
       ============================================ */
    @media (max-width: 768px) {
        .hero-title { font-size: 36px; }
        .greeting-title { font-size: 32px; }
        .login-container { padding: 32px 24px; }
        [data-testid="stSidebar"] { width: 100% !important; }
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
        
        # Model fallback for query expansion
        expansion_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for model in expansion_models:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {FIXED_GROQ_KEY}"},
                json={"model": model, "messages": [{"role": "user", "content": expansion_prompt}], "temperature": 0.1, "max_tokens": 100},
                timeout=10
            )
            if resp.status_code == 200:
                # Use the expanded "OR" query directly
                llm_expansion = resp.json()['choices'][0]['message']['content']
                search_query = llm_expansion
                break
            elif resp.status_code == 429:
                # Rate limit - try next model
                continue
            else:
                print(f"Query expansion failed with status {resp.status_code}: {resp.text}")
                break
    except Exception as e:
        # Any other error: Silently fall back to normalized query
        pass

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
    """Main LLM call with automatic fallback on rate limits."""
    system_prompt = "You are FRIDAY, an expert HR assistant. Answer strictly based on CONTEXT. Context may have Markdown tables. If unknown, say so."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
    if context: messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
    messages.append({"role": "user", "content": query})

    # Model fallback chain: Primary -> Fallback (higher rate limits)
    models = [
        "llama-3.3-70b-versatile",    # Primary: Best quality (1K req/day, 100K tokens/day)
        "llama-3.1-8b-instant"         # Fallback: 14x more requests (14.4K/day, 500K tokens/day)
    ]
    
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
                # Rate limit hit - try next model in the chain
                continue
            else:
                return f"Error: Groq API returned status {resp.status_code}. {resp.text[:100]}"
        except Exception as e: 
            return f"Connection error: {str(e)}"
    
    # All models exhausted
    return "⚠️ All AI models are rate-limited. Please try again in a few minutes."

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
        # Logo Section - Centered
        st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100px; margin-bottom: 28px;"><span class="logo-animated" style="font-family: \'Playfair Display\', serif; font-size: 48px; font-weight: 800; color: #1A3C34; letter-spacing: -1.2px; text-shadow: 0 2px 8px rgba(26, 60, 52, 0.15);">Friday</span></div>', unsafe_allow_html=True)

        # Navigation Menu
        st.markdown("### Menu")
        if st.button("◉  Chat", use_container_width=True, type="primary" if st.session_state.view == "chat" else "secondary"):
            st.session_state.view = "chat"; st.rerun()
        if st.button("◎  Documents", use_container_width=True, type="primary" if st.session_state.view == "documents" else "secondary"):
            st.session_state.view = "documents"; st.rerun()

        st.markdown("---")

        # Chat-specific actions
        if st.session_state.view == "chat":
            if st.button("＋  New Chat", use_container_width=True, type="secondary"): 
                create_new_chat(); st.rerun()

            st.markdown("### Recent Chats")
            recent = get_recent_chats(st.session_state.company_id)
            if not recent:
                st.caption("No conversations yet")
            else:
                for chat in recent:
                    # Show more of the title
                    display_title = (chat['title'][:45] + '…') if len(chat['title']) > 45 else chat['title']
                    is_active = st.session_state.current_chat_id == chat['id']
                    
                    with st.container():
                        col1, col2 = st.columns([6, 1])
                        with col1:
                            btn_type = "primary" if is_active else "secondary"
                            if st.button(f"{display_title}", key=f"chat_{chat['id']}", help=chat['title'], use_container_width=True, type=btn_type):
                                st.session_state.current_chat_id = chat['id']
                                st.rerun()
                        with col2:
                            if st.button("×", key=f"del_{chat['id']}", help="Delete chat"):
                                delete_chat(chat['id'], st.session_state.company_id)
                                if st.session_state.current_chat_id == chat['id']:
                                    create_new_chat()
                                st.rerun()

        # Logout at bottom
        st.markdown("---")
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("↪  Sign Out", use_container_width=True): 
            st.session_state.clear(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def create_new_chat(): st.session_state.current_chat_id = str(uuid.uuid4())

# --- HELPER: HANDLE QUERY EXECUTION WITH LOADING ANIMATION ---
def handle_query(query):
    # Save user msg with COMPANY ID
    save_message(st.session_state.current_chat_id, "user", query, st.session_state.company_id)

    # Show loading animation while processing
    with st.chat_message("user", avatar="👤"):
        st.write(query)
    
    with st.chat_message("assistant", avatar="⚡"):
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
        <div style="text-align: center; margin-top: 80px; margin-bottom: 48px; max-width: 600px; margin-left: auto; margin-right: auto;">
            <h1 class="greeting-title">{greeting}</h1>
            <p class="greeting-subtitle">How can FRIDAY help you with HR tasks today?</p>
        </div>
        """, unsafe_allow_html=True)

    # --- CHAT HISTORY RENDER ---
    for msg in history:
        if msg["role"] == "assistant":
            avatar = "⚡"
        else:
            avatar = None # Use default or we can add a custom image path here
            
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            if msg["sources"]:
                tags = "".join([f"<div class='source-tag'>📄 {s}</div>" for s in msg["sources"]])
                st.markdown(f"<div class='source-container'>{tags}</div>", unsafe_allow_html=True)

    # --- CHAT INPUT ---
    if prompt := st.chat_input("Ask FRIDAY anything..."):
        handle_query(prompt)

def documents_page():
    # Page Header with subtitle
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h1 style="font-family: 'Playfair Display', serif; font-size: 42px; font-weight: 800; color: #1D1D1F; margin-bottom: 12px;">Knowledge Base</h1>
        <p style="font-size: 17px; color: #6E6E73;">Upload documents to enhance FRIDAY's knowledge</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<h3 style='font-size: 20px; font-weight: 600; color: #1D1D1F; margin-bottom: 20px;'>Upload Documents</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Drag and drop PDF/DOCX files here", type=["pdf", "docx"], accept_multiple_files=True)
        
        c_check, c_btn = st.columns([1, 1])
        with c_check:
            force_overwrite = st.checkbox("Overwrite existing files?")
        with c_btn:
            if uploaded_files:
                if st.button("Start Indexing", type="primary", use_container_width=True):
                    for f in uploaded_files:
                        with st.spinner(f"Processing {f.name}..."):
                            status = process_and_store_document(f, st.session_state.company_id, force_overwrite)
                            if status == "success": st.toast(f"✅ Indexed: {f.name}")
                            elif status == "exists": st.warning(f"⚠️ {f.name} exists.")
                            else: st.error(f"❌ Error: {f.name}")
                    time.sleep(1)
                    st.rerun()

    with col2:
        docs = get_all_documents(st.session_state.company_id)
        file_count = len(docs) if docs else 0
        st.markdown(f"<h3 style='font-size: 20px; font-weight: 600; color: #1D1D1F; margin-bottom: 20px;'>Indexed Files <span style='color: #86868B; font-weight: 400;'>({file_count})</span></h3>", unsafe_allow_html=True)
        
        if not docs: 
            st.info("No documents uploaded yet.")
        else:
            for doc in docs:
                # Determine file type icon and color
                filename = doc['filename']
                if filename.lower().endswith('.pdf'):
                    file_icon = "◉"
                    icon_class = "file-icon-pdf"
                elif filename.lower().endswith('.docx'):
                    file_icon = "◎"
                    icon_class = "file-icon-docx"
                else:
                    file_icon = "○"
                    icon_class = "file-icon-default"
                
                status_indicator = "●" if doc['is_active'] else "○"
                status_color = "#34C759" if doc['is_active'] else "#86868B"
                opacity = "1" if doc['is_active'] else "0.6"
                
                with st.container():
                    c1, c2, c3 = st.columns([6, 1, 1])
                    with c1:
                        st.markdown(f"""
                        <div class="file-item" style="opacity: {opacity};">
                            <span class="{icon_class}" style="margin-right: 12px; font-size: 16px;">{file_icon}</span>
                            <span style="font-weight: 500; color: #1D1D1F; font-size: 15px; flex: 1; overflow: hidden; text-overflow: ellipsis;" title="{filename}">{filename}</span>
                            <span style="color: {status_color}; font-size: 10px; margin-left: 8px;">{status_indicator}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with c2:
                        if st.button("⏸", key=f"arch_{doc['id']}", help="Toggle active status"):
                            toggle_document_status(doc['filename'], st.session_state.company_id, doc['is_active'])
                            st.rerun()
                    with c3:
                        if st.button("×", key=f"del_{doc['id']}", help="Delete document"):
                            delete_document(doc['filename'], st.session_state.company_id)
                            st.rerun()

# --- 5. AUTHENTICATION ---
def login_page():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Hero Section with clean headline
        st.markdown("""
        <div style="text-align: center; margin-bottom: 48px;">
            <div style="font-family: 'Playfair Display', serif; font-size: 72px; font-weight: 800; color: #1A3C34; margin-bottom: 24px; letter-spacing: -2px;">Friday</div>
            <h1 class="hero-title" style="font-size: 42px;">Your Intelligent<br><em style="font-style: italic;">HR Companion</em></h1>
            <p class="hero-subtitle">Streamline HR tasks with AI-powered assistance</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            pw = st.text_input("Access Code", type="password", placeholder="Enter your access code")
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
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