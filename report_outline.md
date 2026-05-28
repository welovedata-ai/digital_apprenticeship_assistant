# Report Outline: StandardsBot — A 24/7 AI Teaching Assistant for Digital Apprenticeship KSBs

**Suggested length:** 8–10 pages
**Word count target:** ~3,000–4,000 words (excluding code listings)

---

## 1. Introduction (~300 words)

- Overview: StandardsBot as a 24/7 AI teaching assistant for digital
  apprenticeship learners, not a general standards search engine.
- Problem statement: apprentices often struggle to interpret the technical
  language of KSBs in Skills England standards; progress reviews and
  portfolio building can feel intimidating without support.
- Solution: a hybrid RAG chatbot that retrieves content from real Skills
  England pages, explains KSBs in plain English, gives workplace examples,
  suggests evidence ideas, and supports reflection — available any time.
- Educational use case justification: why apprentices were chosen as the
  primary target user.
- Assessment boundary: the system is designed to support learning, not to
  certify achievement. The bot never states that a KSB has been met.
- Report structure overview.

---

## 2. Chosen Website and Justification (~300 words)

- Description of Skills England as the authoritative publisher of
  apprenticeship standards in England.
- Why it was selected:
  - Structured, consistent page layout across standards.
  - Publicly accessible, non-login-gated content.
  - Contains the exact KSB text that apprentices are assessed against.
  - 15 Digital and AI standards were selected as the most relevant for
    the target user group.
- Screenshot placeholder: example Skills England standard page.
- Note on JavaScript rendering and why Playwright was required.
- Scope limitations: 15 pages from a larger catalogue.

---

## 3. Scraping Methodology (~350 words)

- Tool choice: Playwright (headless Chromium) and why `requests` + BS4
  alone was insufficient for this site.
- Ethical scraping principles:
  - Fixed URL list; no auto-crawling.
  - Descriptive User-Agent identifying this as a student research project.
  - 2-second delay between requests.
  - Skip-on-re-run to avoid unnecessary server load.
- Data saved per page: URL, title, HTTP status code, page type,
  raw HTML, extracted text, timestamp.
- File format: one JSON file per page in `data/raw/`.
- Challenges: JavaScript-rendered content, network timeouts.
- Screenshot placeholder: terminal output from `python -m src.scraper`.

---

## 4. Data Cleaning and KSB Extraction (~450 words)

- Overview of `src/cleaner.py`.
- HTML cleaning steps:
  1. Removing non-content tags (script, style, nav, footer, etc.).
  2. Removing HTML comments.
  3. Removing elements with navigation/chrome CSS classes.
  4. Extracting text with `get_text(separator="\n")`.
  5. Normalising whitespace and removing consecutive duplicate lines.
- **Structured KSB extraction** (new for this version):
  - Why structured extraction matters for a teaching assistant —
    knowing which items are K, S, or B enables more targeted answers
    and KSB-referenced Q&A generation.
  - Approach: regex-based extraction from cleaned text targeting
    patterns like `K1: description`, `S2\ntext`, `B3: ...`.
  - Fields extracted: level, duration, occupational summary, duties,
    knowledge items, skill items, behaviour items.
  - Graceful fallback: if a section cannot be found, the field is left
    empty and full cleaned text remains available.
- Example of structured output for one standard (table or code block).
- Code snippet: the `extract_ksb_structure()` function.

---

## 5. Synthetic Q&A Generation — Teaching Focus (~400 words)

- Rationale: pre-generating teaching Q&A pairs anchored to specific
  KSBs allows fast, high-quality answers to common apprentice questions
  without an LLM call at retrieval time.
- **Apprentice-focused prompt design:**
  - System prompt defines an expert educator role generating teaching
    content for learners.
  - Questions cover: KSB explanation, workplace examples, evidence ideas,
    reflection, misconceptions, portfolio/professional discussion.
  - Strict grounding rules: no invented KSBs, no claims of achievement.
  - Assessment boundary embedded in the prompt.
- New CSV columns:
  - `ksb_type` (knowledge / skill / behaviour / general / unknown).
  - `ksb_reference` (K1, S3, B2, etc. — blank if not identified).
- Model: `gpt-4o-mini` at temperature 0.4.
- Post-processing: automatic removal of pairs with "not specified in
  the source" answers (identified as low quality during testing).
- Example Q&A pairs table (3–5 examples across different types).
- Quantity: ~125–150 pairs across 15 documents.

---

## 6. Vector Database and Embeddings (~350 words)

- Tool: ChromaDB with persistent local storage (`data/chroma_db/`).
- Embedding model: `text-embedding-3-small` — low cost, good semantic
  quality for English-language educational content.
- Two collections:
  - `doc_chunks`: 800-char overlapping chunks of cleaned page text.
    Overlap of 100 chars ensures KSB items are rarely split across chunks.
  - `qa_pairs`: question text from `qa_dataset.csv` with full teaching
    answer and KSB metadata stored alongside.
- Distance metric: cosine similarity (`hnsw:space: cosine`).
- Build behaviour: collections are built once via
  `python -m src.vector_store`; `--force` triggers a rebuild.
  This prevents unnecessary API calls on subsequent app launches.
- Module-level singletons in `retriever.py` ensure the Chroma client
  and embedding function are constructed once per process, not per query.

---

## 7. Hybrid Retrieval for Teaching Support (~400 words)

- Design rationale: two-stage hybrid balances the precision of curated
  KSB teaching Q&A with the breadth of raw document chunks.
- **Stage 1 — Q&A semantic search:**
  - Query is embedded and compared against `qa_pairs`.
  - Threshold set at 0.88 (high) to prevent generic questions like
    "What level is this?" matching the wrong standard.
  - If confidence ≥ 0.88, the pre-generated teaching answer is returned
    with source citation and KSB reference.
- **Stage 2 — Document chunk fallback:**
  - Top-8 chunks returned so cross-standard questions
    (e.g. "Which Level 3 standards cover cyber security?") have
    broad enough context.
- **Tier 3 — LLM general knowledge fallback:**
  - Triggered when max chunk confidence < 0.45.
  - Response is clearly labelled in the UI as general knowledge, not
    from scraped content.
  - Maintains the assessment boundary in the fallback prompt.
- Threshold selection rationale and impact on precision vs recall.
- Flow diagram placeholder.

---

## 8. Learning Modes and Teaching Prompt Design (~350 words)

- Five learning modes selectable from the sidebar:
  - **Explain simply** — plain-English breakdown for new learners.
  - **Give workplace examples** — concrete digital workplace scenarios.
  - **Suggest evidence ideas** — portfolio and professional discussion tips.
  - **Quiz me** — a short test question to check understanding.
  - **Help me reflect** — coaching-style questions for progress reviews.
- How modes are implemented: each mode appends a focused instruction
  to the user content passed to the LLM; the system prompt and
  retrieved context remain the same.
- The assessment boundary: how the system prompt prevents the bot from
  claiming an apprentice has achieved or passed a KSB, and why this is
  ethically important for an educational tool.
- Example response comparison: same query answered in "Explain simply"
  vs "Suggest evidence ideas" mode.

---

## 9. Streamlit Application (~300 words)

- Features:
  - Persistent chat interface with message history (conversational memory).
  - Learning mode selector (sidebar selectbox).
  - Hero banner with robot image and topic pills.
  - Six apprentice-style suggested questions.
  - Source cards with KSB reference badge where identified.
  - "How this answer was retrieved" expandable panel.
  - Sample Q&A browser filtered by question type.
  - Assessment disclaimer in sidebar.
  - Clear conversation button.
- Session state: messages stored as dicts including role, content,
  sources, retrieval info, mode used, and fallback flag.
- Screenshot placeholder: app home page.
- Screenshot placeholder: a Q&A match answer with KSB reference badge.
- Screenshot placeholder: a quiz-mode response.

---

## 10. Limitations and Future Improvements (~300 words)

### Limitations

- **Scope:** 15 pages only. Out-of-scope standards fall through to LLM
  general knowledge or receive "not in knowledge base" responses.
- **KSB extraction accuracy:** regex-based extraction is best-effort;
  unusual page formatting may prevent items being identified.
- **No formal evaluation:** answer quality was assessed manually during
  development; no RAGAS or automated RAG evaluation was applied.
- **Static snapshot:** scraped content is not updated automatically.
- **Assessment boundary:** the bot cannot verify actual evidence or sign
  off any assessment criteria; professional judgement is always required.

### Future Improvements

- Scheduled re-scraping to keep content current.
- LLM-based KSB extraction (more reliable than regex for complex layouts).
- Cross-encoder re-ranking for higher-precision retrieval.
- Streamed responses for a more responsive user experience.
- Portfolio evidence builder: structured workflow for apprentices to
  capture and organise evidence against specific KSBs.
- Multilingual support for learners whose first language is not English.
- Integration with e-portfolio platforms (OneFile, Aptem, etc.).

---

## 11. Learning Reflections (~300 words)

- What was learned about JavaScript-rendered scraping and why Playwright
  was necessary for this particular website.
- The challenge of balancing retrieval precision (high threshold) vs
  recall (enough results for cross-standard queries).
- How the learning mode feature changed the prompt design: the same
  retrieved context can produce very different, equally valid teaching
  responses depending on the mode instruction.
- Reflection on the assessment boundary: why it matters to build ethical
  constraints into the system prompt, not just the UI copy.
- What would be done differently with more time: earlier experimentation
  with structured KSB extraction, formal RAG evaluation metrics.

---

## Appendix A — Architecture Diagram

*[Placeholder — insert diagram showing:
scraper → cleaner (+ KSB extraction) → qa_generator → vector_store →
retriever (Stage 1: Q&A | Stage 2: doc_chunks | Stage 3: LLM fallback) →
chatbot (+ learning mode) → Streamlit UI]*

---

## Appendix B — Streamlit Screenshots

*[Placeholder — insert screenshots of:
1. App home page with hero, suggested questions and mode selector
2. An "Explain simply" answer with source citation and KSB badge
3. A "Quiz me" response
4. The sidebar Q&A browser
5. A general knowledge fallback response with amber banner]*

---

## References

- Skills England. (2024). *Apprenticeship standards*. <https://skillsengland.education.gov.uk/>
- OpenAI. (2024). *GPT-4o mini model card*. <https://platform.openai.com/docs/models>
- Chroma. (2024). *ChromaDB documentation*. <https://docs.trychroma.com/>
- Playwright. (2024). *Playwright for Python*. <https://playwright.dev/python/>
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Institute for Apprenticeships and Technical Education. (2023). *Apprenticeship standards guidance*.
