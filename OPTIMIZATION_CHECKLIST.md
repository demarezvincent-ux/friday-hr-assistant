# Friday Optimization Checklist

Quick reference for implementation tracking.

---

## ✅ Tier 1: Quick Wins (1-2 days)

### [ ] 1. Dual-Source Citation UI (3-4 hours) ⚠️ CRITICAL
- [ ] Modify `services/rag_controller.py` to return `legal_sources` and `company_sources` separately
- [ ] Update `main.py` `render_sources_html()` function
- [ ] Add ⚖️ icon with green badges for legal sources
- [ ] Add 📄 icon with blue badges for company sources
- [ ] Test with queries that return both source types

### [ ] 2. Document Freshness Warnings (2-3 hours)
- [ ] Add `created_at` check in `rag_controller.py` context assembly
- [ ] Create yellow warning banner HTML template
- [ ] Add health dashboard page showing:
  - [ ] Total document count
  - [ ] Oldest document date
  - [ ] Last legal scrape date
  - [ ] Health indicator (green/yellow/red)

### [ ] 3. Rate Limiter Improvements (2 hours)
- [ ] Update `services/agentic/rate_limiter.py`:
  - [ ] Replace simple delay with token bucket algorithm
  - [ ] Add X-RateLimit-Remaining header tracking
  - [ ] Implement exponential backoff with jitter for 429 errors
- [ ] Test with burst queries

### [ ] 4. Error Handling & User Feedback (3 hours)
- [ ] Add structured logging with context (company_id, query, error_type)
- [ ] Replace bare `except:` with specific exceptions:
  - [ ] `main.py` process_and_store_document()
  - [ ] `services/rag_controller.py` get_context_with_strategy()
  - [ ] `services/search_service.py` analyze_query()
- [ ] Add user-friendly error messages in UI
- [ ] Test error scenarios (file too large, API timeout, invalid format)

---

## ✅ Tier 2: Strategic Enhancements (3-7 days)

### [ ] 5. Expand Legal Coverage (1 day)
- [ ] PC 124 (Construction) - 2 hours
  - [ ] Find URL for PC 124 on werk.belgie.be
  - [ ] Add to `scripts/legal_scraper.py` BELGIAN_LAW_URLS
  - [ ] Test scraping and AI extraction
  - [ ] Verify embeddings stored correctly
- [ ] PC 140.03 (Transport) - 2 hours
- [ ] PC 302 (Hospitality) - 2 hours
- [ ] PC 111 (Metals) - 2 hours

### [ ] 6. Semantic Cache Activation (4-6 hours)
- [ ] Verify `query_cache` table exists in Supabase
- [ ] Enable caching in `rag_controller.py` (already default, verify active)
- [ ] Add cache metrics to dashboard:
  - [ ] Hit rate percentage
  - [ ] Cache size (number of entries)
  - [ ] Oldest entry timestamp
- [ ] Implement cleanup job:
  - [ ] Option A: PostgreSQL cron extension
  - [ ] Option B: Python scheduled task (APScheduler)
- [ ] Set TTL to 24 hours

### [ ] 7. Advanced Search Diagnostics (1 day)
- [ ] Add admin mode toggle in sidebar (check if user is admin)
- [ ] When debug mode enabled, show under each response:
  - [ ] Original query
  - [ ] Corrected query (from Intelligence Engine)
  - [ ] FTS search string
  - [ ] Number of candidates retrieved
  - [ ] Top 5 reranker scores
  - [ ] Cache hit/miss status
- [ ] Create `failed_queries` table in Supabase
- [ ] Store queries with 0 results for analysis

### [ ] 8. Batch Document Processing (2-3 days)
- [ ] Choose approach:
  - [ ] Option A: Celery + Redis queue (full-featured)
  - [ ] Option B: asyncio parallel processing (lightweight)
- [ ] Implement chosen approach:
  - [ ] Upload files to storage immediately
  - [ ] Process embeddings in background
  - [ ] Store processing status in documents table
- [ ] Update UI to show processing status ('Indexing...' badge)
- [ ] Add progress indicators
- [ ] Test with 10+ document bulk upload

---

## ✅ Tier 3: Advanced Features (1-2 weeks)

### [ ] 9. Slack Bot Integration (1 week) ⚠️ HIGH PRIORITY
- [ ] Phase 1: Slack Bot (4-5 days)
  - [ ] Install Slack Bolt for Python (`pip install slack-bolt`)
  - [ ] Create Slack app in Slack API portal
  - [ ] Configure bot permissions (chat:write, im:history, im:read)
  - [ ] Create bot listener for DM events
  - [ ] Connect bot to existing `rag_controller.py` backend
  - [ ] Format responses with Slack blocks (sections, buttons)
  - [ ] Add slash command `/friday help`
  - [ ] Test in private Slack workspace
  - [ ] Deploy to Railway/Render free tier
- [ ] Phase 2: Teams Bot (2-3 days)
  - [ ] Install Bot Framework SDK
  - [ ] Create Teams app manifest
  - [ ] Implement similar DM flow

### [ ] 10. Workflow Action Bridges (2-3 days) ⚠️ HIGH PRIORITY
- [ ] Create `action_links` table in Supabase:
  - [ ] Columns: id, company_id, keyword, action_label, url, created_at
  - [ ] Add indexes on company_id and keyword
- [ ] Create admin UI page for managing action links
- [ ] Modify `rag_controller.py` to detect keyword matches
- [ ] Update `main.py` to render action buttons below responses
- [ ] Test with common workflows:
  - [ ] vacation → Request Time Off
  - [ ] payslip → View Payslips
  - [ ] bike lease → Start Application
- [ ] Document onboarding process for HR managers

### [ ] 11. Multi-Document Conversation Context (1 week)
- [ ] Implement conversation memory:
  - [ ] Store last N messages in session state
  - [ ] Add conversation_id to tracking
- [ ] Build follow-up detector:
  - [ ] Check for pronouns (it, that, this, they)
  - [ ] Check for incomplete questions
- [ ] Create query rewriter:
  - [ ] Use LLM to rewrite follow-up as standalone query
  - [ ] Include previous context for rewriting
- [ ] Test conversation flows:
  - [ ] Q1: "vacation days?" → Q2: "what about part-timers?"
  - [ ] Q1: "bike lease?" → Q2: "how much does it cost?"
- [ ] Add "Start New Conversation" button to reset context

---

## 🔧 Bonus: Engineer Manual Search Tool (Optional)

### [ ] Phase 1: MVP (1 week)
- [ ] Create `equipment_manuals` table with extended metadata:
  - [ ] manufacturer, model, equipment_type, section_number
- [ ] Add "Engineering Manuals" mode toggle in sidebar
- [ ] Create separate upload flow for equipment manuals
- [ ] Add filters in UI:
  - [ ] Search by manufacturer
  - [ ] Search by equipment type
- [ ] Test with sample equipment manuals (HVAC, electrical)

### [ ] Phase 2: Enhanced Search (1 week)
- [ ] Improve chunking logic:
  - [ ] Preserve part numbers (regex detection)
  - [ ] Preserve section numbers
  - [ ] Keep diagrams with associated text
- [ ] Add OCR fallback for scanned PDFs:
  - [ ] Use PyMuPDF's OCR capabilities
  - [ ] Fall back to OCR if text extraction fails
- [ ] Expand Intelligence Engine with technical terms:
  - [ ] compressor failure = compressor not starting, motor dead
  - [ ] circuit breaker = fuse, overload protection
- [ ] Test with 10+ equipment manuals

### [ ] Phase 3: Mobile Optimization (1 week)
- [ ] Responsive design for phone/tablet:
  - [ ] Adjust CSS for mobile breakpoints
  - [ ] Optimize button sizes for touch
- [ ] Add voice input (Web Speech API):
  - [ ] Add microphone button
  - [ ] Convert speech to text
  - [ ] Submit query automatically
- [ ] Implement offline mode:
  - [ ] Cache common queries (localStorage)
  - [ ] Show offline indicator
  - [ ] Queue queries when offline, sync when online

---

## 📊 Testing & Quality

### [ ] Test Coverage
- [ ] Install pytest and pytest-asyncio
- [ ] Unit tests:
  - [ ] `sanitize_filename()`
  - [ ] `recursive_chunk_text()`
  - [ ] `normalize_query()`
  - [ ] `sanitize_fts_query()`
- [ ] Integration tests:
  - [ ] RAG pipeline end-to-end
  - [ ] Document upload flow
  - [ ] Query → Context → Response
- [ ] Mock tests:
  - [ ] Groq API calls
  - [ ] HuggingFace API calls
  - [ ] Supabase calls

### [ ] Code Quality Refactoring
- [ ] Add type hints to all functions
- [ ] Centralize magic numbers in config file:
  - [ ] chunk_size, overlap, match_count, match_threshold
  - [ ] timeout values, retry counts
- [ ] Create DatabaseService class to reduce Supabase query duplication
- [ ] Replace bare `except:` with specific exception types
- [ ] Add docstrings to all functions

### [ ] CI/CD Pipeline
- [ ] Create `.github/workflows/test.yml`:
  - [ ] Run pytest on every push
  - [ ] Check code formatting (black)
  - [ ] Run type checking (mypy)
- [ ] Add pre-commit hooks:
  - [ ] Format with black
  - [ ] Lint with flake8
- [ ] Set up deployment automation (optional)

---

## 🔐 Security Hardening

### [ ] User-Level Rate Limiting
- [ ] Add rate limit table: user_id, window_start, query_count
- [ ] Implement 100 queries/hour limit per company
- [ ] Add rate limit exceeded message in UI

### [ ] File Upload Security
- [ ] Add virus scanning (ClamAV):
  - [ ] Install ClamAV daemon
  - [ ] Scan file before processing
  - [ ] Quarantine infected files
- [ ] Verify .gitignore includes:
  - [ ] .streamlit/secrets.toml
  - [ ] .env
  - [ ] *.key files

### [ ] Monitoring & Observability
- [ ] Set up Sentry for error tracking:
  - [ ] Install sentry-sdk
  - [ ] Add DSN to secrets
  - [ ] Configure error sampling
- [ ] Add basic metrics tracking:
  - [ ] Query latency (P50, P95, P99)
  - [ ] Cache hit rate
  - [ ] Error rate by type
  - [ ] Documents processed per day
- [ ] Set up alerts for critical errors

---

## 📈 Success Metrics to Track

### Week 1-2
- [ ] Dual-source citation implemented and tested
- [ ] Interactive + AppTweak pilot agreements signed

### Week 3-6
- [ ] Slack bot deployed to at least 1 pilot customer
- [ ] Workflow action bridges active with 5+ common workflows
- [ ] Both pilots using Friday 3+ times per week

### Week 7-10
- [ ] 2 pilot customers converted to paid (€199/month)
- [ ] 3 new paying customers acquired
- [ ] €1,000 MRR achieved

---

## 💡 Tips for Implementation

1. **Start with Tier 1** - Quick wins build momentum
2. **Test incrementally** - Don't batch changes
3. **Keep main.py stable** - Most changes are in services/
4. **Version your database** - Use migrations for schema changes
5. **Monitor in production** - Add logging before optimizing
6. **Get user feedback early** - Deploy to staging for pilots

---

**Status:** Ready for implementation
**Last Updated:** February 11, 2026
