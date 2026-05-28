"""
Central configuration for StandardsBot.

StandardsBot is a 24/7 AI teaching assistant that helps apprentices on
digital apprenticeship programmes understand their Knowledge, Skills and
Behaviours (KSBs) from Skills England standards.

All paths, model names, and tunable parameters live here so they can be
changed in one place without hunting through multiple files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed list of Skills England URLs to scrape.
# All 15 pages cover Digital and AI apprenticeship standards/units.
# Do NOT auto-crawl — only these pages are in scope.
# ---------------------------------------------------------------------------
STANDARD_URLS = [
    "https://skillsengland.education.gov.uk/apprenticeship-units/AU0009",
    "https://skillsengland.education.gov.uk/apprenticeship-units/AU0010",
    "https://skillsengland.education.gov.uk/apprenticeship-units/AU0011",
    "https://skillsengland.education.gov.uk/apprenticeships/st1398-v1-0",
    "https://skillsengland.education.gov.uk/apprenticeships/st1386-v1-0",
    "https://skillsengland.education.gov.uk/apprenticeships/st0795-v1-1",
    "https://skillsengland.education.gov.uk/apprenticeships/st0120-v1-1",
    "https://skillsengland.education.gov.uk/apprenticeships/st0118-v1-1",
    "https://skillsengland.education.gov.uk/apprenticeships/st0116-v1-2",
    "https://skillsengland.education.gov.uk/apprenticeships/st0128-v1-1",
    "https://skillsengland.education.gov.uk/apprenticeships/st0117-v1-2",
    "https://skillsengland.education.gov.uk/apprenticeships/st0482-v1-0",
    "https://skillsengland.education.gov.uk/apprenticeships/st0505-v1-1",
    "https://skillsengland.education.gov.uk/apprenticeships/st0825-v1-1",
    "https://skillsengland.education.gov.uk/apprenticeships/st0865-v1-1",
]

# ---------------------------------------------------------------------------
# Data directory paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.csv"

# ---------------------------------------------------------------------------
# ChromaDB persistence
# ---------------------------------------------------------------------------
CHROMA_PERSIST_PATH = str(DATA_DIR / "chroma_db")
DOCS_COLLECTION_NAME = "doc_chunks"
QA_COLLECTION_NAME = "qa_pairs"

# ---------------------------------------------------------------------------
# OpenAI models
# ---------------------------------------------------------------------------
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Text chunking
# Chunks are kept at 800 chars with overlap so KSB items are rarely split
# across two separate chunks.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------------
# Hybrid retrieval
# Q/A pairs are searched first. Only if the best match scores at or above
# RETRIEVAL_THRESHOLD is the pre-generated teaching answer returned.
# Below the threshold, the retriever falls through to document chunk search.
# Set deliberately high (0.88) to avoid generic Q/A pairs matching the
# wrong standard.
# ---------------------------------------------------------------------------
RETRIEVAL_THRESHOLD = 0.88

# ---------------------------------------------------------------------------
# Ethical scraping
# ---------------------------------------------------------------------------
SCRAPE_DELAY_SECONDS = 2
USER_AGENT = (
    "StandardsBot/1.0 (educational research project; "
    "non-commercial use; scraping fixed URL list only)"
)

# ---------------------------------------------------------------------------
# Learning mode labels (used by the Streamlit UI and chatbot)
# ---------------------------------------------------------------------------
LEARNING_MODES = [
    "Explain simply",
    "Give workplace examples",
    "Suggest evidence ideas",
    "Quiz me",
    "Help me reflect",
]
