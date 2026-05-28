# Digital Apprenticeship Assistant

A Python 3.10+ Streamlit chatbot that helps apprentices on digital
apprenticeship programmes understand their Knowledge, Skills and Behaviours
(KSBs) from Skills England standards — available any time, in plain English.

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/t2GH9TDuAjc/maxresdefault.jpg)](https://youtu.be/t2GH9TDuAjc)

---

## Purpose

Apprentices often struggle to make sense of the dense, technical language used
in apprenticeship standards. This assistant acts as a supportive study companion:
it explains what each KSB means, gives real workplace examples, suggests evidence
ideas for portfolios and professional discussions, quizzes learners on their
understanding, and supports reflection ahead of progress reviews.

**Primary users:** Apprentices on digital programmes.

**Important boundary:** This tool is for learning support only. It does **not**
tell apprentices they have achieved, passed, or met a KSB. It explains what they
*may* need to demonstrate and always directs learners to check with their trainer,
employer or assessor for any assessment-related questions.

---

## Standards Covered (15)

| Standard | Level |
|----------|-------|
| Software Developer | 4 |
| Business Analyst | 4 |
| Data Analyst | 4 |
| DevOps Engineer | 4 |
| Data Engineer | 5 |
| Machine Learning Engineer | 7 |
| Digital and Technology Solutions Specialist | 7 |
| Data Technician | 3 |
| Digital Support Technician | 3 |
| IT Solutions Technician | 3 |
| Software Development Technician | 3 |
| Cyber Security Technician | 3 |
| AI Leadership: AI Strategy and Opportunity | unit |
| AI Leadership: AI Adoption, Procurement and Governance | unit |
| AI Leadership: AI Delivery and Organisational Transformation | unit |

All pages are from **Skills England** — <https://skillsengland.education.gov.uk/>

---

## Project Structure

```
standards_bot/
├── assets/
│   ├── banner.png              ← hero banner image (top of UI)
│   └── robot.png               ← sidebar logo and browser-tab icon
├── data/
│   ├── raw/                    ← one JSON file per scraped page (gitignored)
│   ├── processed/              ← cleaned text + KSB structure (gitignored)
│   ├── chroma_db/              ← ChromaDB vector store (gitignored)
│   └── qa_dataset.csv          ← generated teaching Q&A pairs (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py               ← all settings in one place
│   ├── scraper.py              ← Playwright headless scraper
│   ├── cleaner.py              ← HTML → clean text + KSB extraction
│   ├── qa_generator.py         ← apprentice-focused Q&A generation
│   ├── vector_store.py         ← ChromaDB collection builder
│   ├── retriever.py            ← two-stage hybrid retriever
│   └── chatbot.py              ← teaching answer generation + learning modes
├── app.py                      ← Streamlit UI
├── requirements.txt
├── .env                        ← your API key (never committed)
├── .env.example                ← template — copy to .env
├── .gitignore
├── README.md
└── report_outline.md
```

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the Playwright browser (one-time, ~150 MB)

```bash
python -m playwright install chromium
```

### 4. Create your `.env` file

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS / Linux
```

Open `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

---

## Running the Pipeline

Run each stage in order the first time. After that, only re-run a stage if
its source data has changed.

### Stage 1 — Scrape

```bash
python -m src.scraper
```

Downloads the 15 Skills England pages using a headless Chromium browser
(needed because the site relies on JavaScript to render content). Saves one
JSON file per page to `data/raw/`. Already-saved pages are skipped on re-run.

### Stage 2 — Clean and extract KSB structure

```bash
python -m src.cleaner
```

Strips navigation, scripts and other site chrome from the raw HTML. Also
attempts to extract structured KSB data (level, summary, duties, knowledge
items, skill items, behaviour items) using regex. Saves results to
`data/processed/`. Structured extraction degrades gracefully — if a section
cannot be identified, the full cleaned text is still preserved.

### Stage 3 — Generate teaching Q&A pairs

```bash
python -m src.qa_generator
```

Sends each cleaned document to `gpt-4o-mini` with a teaching-focused system
prompt. Generates questions an apprentice would actually ask:

- *What does K3 mean in plain English?*
- *What workplace examples illustrate this skill?*
- *What evidence could I put in my portfolio for S2?*
- *What reflection questions can help me prepare for a progress review?*

Saves ~150 pairs to `data/qa_dataset.csv`. Each row includes `ksb_type` and
`ksb_reference` (e.g. K1, S3, B2) where the model could identify them. Low-
quality pairs where the answer was "not specified in the source" are
automatically filtered out before saving.

> **Cost note:** One API call per page (~15 total). Very low cost at
> `gpt-4o-mini` pricing.

### Stage 4 — Build the vector store

```bash
python -m src.vector_store           # build if not yet built
python -m src.vector_store --force   # drop and fully rebuild
```

Embeds document chunks and Q&A questions using `text-embedding-3-small` and
stores them in two ChromaDB collections persisted to `data/chroma_db/`. Only
needs to run once per dataset version. Use `--force` after regenerating the
Q&A dataset or re-cleaning documents.

### Stage 5 — Run the app

```bash
python -m streamlit run app.py
```

Open <http://localhost:8501> in your browser (or use `--server.port 8502` if
that port is taken).

---

## How Hybrid Retrieval Works

Every question the apprentice asks goes through a three-tier process:

```
Apprentice question
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 1: Q&A semantic search       │
│  Embed the question and compare     │
│  against ~150 pre-generated         │
│  teaching Q&A pairs.                │
└─────────────────────────────────────┘
        │
  confidence ≥ 88% ──► Return pre-generated teaching answer immediately
        │
  confidence < 88%
        │
        ▼
┌──────────────────────────────────────┐
│  Stage 2: Document chunk search      │
│  Retrieve the top-8 most relevant    │
│  chunks from the cleaned page text.  │
│  GPT-4o-mini synthesises a teaching  │
│  answer from those chunks.           │
└──────────────────────────────────────┘
        │
  confidence < 45% (nothing useful found)
        │
        ▼
  LLM general knowledge fallback
  (clearly labelled with a warning banner in the UI)
```

The high Q&A threshold (88%) prevents vague matches from returning the wrong
standard's answer. The document chunk fallback ensures the LLM always has
primary source material to work from before resorting to general knowledge.

---

## Learning Modes

The sidebar offers five answer styles that shape how the assistant responds:

| Mode | What it does |
|------|-------------|
| Explain simply | Plain-English breakdown, short sentences, no jargon |
| Give workplace examples | Concrete digital workplace scenarios |
| Suggest evidence ideas | Portfolio and professional discussion tips |
| Quiz me | A short test question or multiple-choice question |
| Help me reflect | Coaching-style reflection prompts (3–5 questions) |

---

## Source Citations

Every response includes a **Sources** panel showing which Skills England page
the answer came from. Where a KSB reference (e.g. K3, S1) was identified, it
appears as a badge on the source card.

An expandable **"How this answer was found"** panel shows whether the response
came from a Q&A match or vector search, along with the confidence score.

---

## Ethical Scraping

- Fixed URL list only — no auto-crawling or link following
- User-Agent string identifies this as a student research project
- 2-second delay between each page request
- Already-saved pages are skipped on re-run
- Data used solely for non-commercial educational purposes

---

## Limitations

- Covers only 15 pages — questions about other standards use an LLM general
  knowledge fallback, clearly flagged in the UI.
- Scraped content reflects the Skills England website at the time of scraping
  and is not updated automatically.
- KSB extraction is regex-based and best-effort; unusual page layouts may
  result in empty structured fields (the full text is always preserved as a
  fallback).
- The assistant never replaces professional assessment judgement.

---

## Future Improvements

- Scheduled re-scraping to keep content current with Skills England updates.
- Expand beyond the initial 15 standards.
- Cross-encoder re-ranking for higher-precision retrieval.
- Streamed token-by-token responses for a more responsive feel.
- User feedback buttons to log answer quality and improve the Q&A dataset.
- Export of study notes or conversation transcripts.
- Dedicated portfolio-evidence builder workflow.
