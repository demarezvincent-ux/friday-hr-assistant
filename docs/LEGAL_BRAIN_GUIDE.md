# 🧠 FRIDAY Legal Brain: User Guide

This guide explains how the automated legal knowledge pipeline works, how to run it, and how to troubleshoot it.

---

## 🏗️ Architecture

The Legal Brain is designed to autonomously keep FRIDAY updated with Belgian labor law. It runs completely free via GitHub Actions (no server required).

```mermaid
graph TD
    Cron[GitHub Cron (Weekly)] -->|Triggers| Scraper[Python Scraper]
    Manual[Manual Run] -->|Triggers| Scraper
    
    Scraper -->|1. Fetch| Web[Government Sites]
    Web -->|PC 200| Scraper
    Web -->|CNT/NAR| Scraper
    Web -->|Federal Law| Scraper
    
    Scraper -->|2. Extract| Text[PDF Text / OCR]
    
    Scraper -->|3. Analyze| AI[Groq (Llama 3)]
    AI -->|Metadata| Scraper
    
    Scraper -->|4. Store| DB[(Supabase)]
    
    User[User Query] -->|LEGAL Intent| RAG[RAG Controller]
    RAG <-->|Hybrid Search| DB
```

### Components
1.  **Scraper (`scripts/legal_scraper.py`)**: The engine. It visits sites, downloads PDFs, extracts text (with OCR fallback), analyzes content using AI, and saves it.
2.  **Database (`legal_knowledge` table)**: Specialized table in Supabase with vector embeddings for semantic search.
3.  **RAG Controller**: When you ask a legal question, FRIDAY now searches *this* database alongside your company docs.

---

## 🚀 Setup & Configuration

### 1. Supabase (One-time)
You must apply the schema to your Supabase database.
1.  Open [Supabase Dashboard](https://supabase.com/dashboard).
2.  Go to **SQL Editor**.
3.  Open `sql/legal_knowledge_schema.sql` from this repo.
4.  Paste the content and click **Run**.

### 2. GitHub Secrets (Crucial)
For the pipeline to run on GitHub, set these secrets in **Settings > Secrets and variables > Actions**:
- `SUPABASE_URL`: Your Supabase project URL.
- `SUPABASE_KEY`: Your `service_role` key (allows writing to DB).
- `GROQ_API_KEY`: For AI analysis.
- `HF_API_KEY`: For generating vector embeddings.

---

## 🛠️ How to Run

### Option A: From GitHub (Easy)
1.  Go to the **Actions** tab in your repository.
2.  Select **"Weekly Legal Brain Transplant"**.
3.  Click **Run workflow**.
4.  (Optional) Choose a specific target (e.g., `federal`) or limit documents.

### Option B: Local Command Line (Dev)
Prerequisite: `pip install -r scripts/legal_requirements.txt`

```bash
# 1. Test configuration (Dry run - no storage)
python3 scripts/legal_scraper.py --target federal --dry-run

# 2. Run PC 200 scraper (limit to 5 recent docs)
python3 scripts/legal_scraper.py --target pc200 --limit 5

# 3. Run everything
python3 scripts/legal_scraper.py --target all
```

---

## 🔍 Troubleshooting Scrapers

Government websites are notorious for changing URLs. If the scraper finds 0 documents:

1.  **Check the logs**:
    ```
    WARNING - Failed to fetch URL: 404
    ```
    This means the URL in `legal_scraper.py` is dead.

2.  **Fixing a Broken URL**:
    - **PC 200**: The scraper now targets `https://www.sfonds200.be/nl/` (Homepage) because deep links change frequently. If this fails, visit the site and update `start_urls` in `fetch_pc_200_updates`.
    - **CNT**: Targets `https://cnt-nar.be/nl`. Deep linking is difficult due to redirects.
    - Copy the new URL.
    - Edit `scripts/legal_scraper.py`.

3.  **DNS Errors**:
    - Ensure your machine has internet access.
    - If `cnt-nar.be` fails, their server might be down or blocking automated requests. The script will just log an error and continue.

## 🤖 The "Brain" Logic

When you ask: *"What is the indexing for 2026?"*

1.  **Routing**: The new `QueryRouter` sees keywords like "indexering", "PC 200". It flags this as `QueryIntent.LEGAL`.
2.  **Search**: `rag_controller.py` searches **BOTH**:
    - `legal_knowledge` (The laws we scraped)
    - `document_chunks` (Your uploaded company policy)
3.  **Synthesis**: It presents the Law *first*, then your Company Policy, allowing the AI to say: *"The law requires 2.21% indexation, and your company policy confirms this applies to all employees."*
