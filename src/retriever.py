"""
Hybrid retriever: two-stage search over the KSB knowledge base.

Stage 1 — Q/A semantic search
    The apprentice's query is compared against the embedded questions in the
    qa_pairs Chroma collection.  If the best match scores at or above
    RETRIEVAL_THRESHOLD, the pre-generated teaching answer is returned
    immediately.  This is the fast path and provides the most targeted,
    apprentice-focused response.

Stage 2 — Document chunk vector search (fallback)
    If no Q/A match is confident enough, the retriever queries the
    doc_chunks collection and returns the top-N raw text chunks for the
    chatbot to synthesise a teaching answer from.

Each result carries:
    context       -- matched question text (qa_match) or document chunk
    answer        -- pre-generated teaching answer (qa_match only)
    method        -- "qa_match" | "vector_search"
    confidence    -- cosine similarity in [0, 1]
    source_title  -- page title from metadata
    source_url    -- canonical URL from metadata
    ksb_type      -- "knowledge" | "skill" | "behaviour" | "general" | ""
    ksb_reference -- KSB code e.g. "K1", "S3", "B2", or "" if unknown
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from src.config import (
    CHROMA_PERSIST_PATH,
    DOCS_COLLECTION_NAME,
    QA_COLLECTION_NAME,
    OPENAI_EMBEDDING_MODEL,
    RETRIEVAL_THRESHOLD,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once per process, reused every query
# ---------------------------------------------------------------------------
_client: Optional[chromadb.PersistentClient] = None
_embedding_fn: Optional[OpenAIEmbeddingFunction] = None


@dataclass
class RetrievalResult:
    """A single result from the hybrid retriever."""

    context: str            # Matched question (qa_match) or document chunk
    method: str             # "qa_match" or "vector_search"
    confidence: float       # Cosine similarity in [0, 1]
    source_title: str       # Human-readable page title
    source_url: str         # Canonical URL
    answer: Optional[str]   # Pre-generated teaching answer (qa_match only)
    ksb_type: str = field(default="")       # knowledge | skill | behaviour | general | unknown
    ksb_reference: str = field(default="")  # e.g. K1, S3, B2 — blank if not identified


def _cosine_similarity(chroma_distance: float) -> float:
    """
    Convert ChromaDB's cosine distance to a [0, 1] similarity score.
    ChromaDB distance is in [0, 2]: 0 = identical, 2 = opposite.
    """
    return round(max(0.0, 1.0 - chroma_distance / 2.0), 4)


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    return _client


def _get_embedding_fn() -> OpenAIEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        _embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=OPENAI_EMBEDDING_MODEL,
        )
    return _embedding_fn


# ---------------------------------------------------------------------------
# Stage 1: Q/A semantic search
# ---------------------------------------------------------------------------

def _qa_search(
    client: chromadb.PersistentClient,
    embedding_fn: OpenAIEmbeddingFunction,
    query: str,
) -> Optional[RetrievalResult]:
    """
    Search the pre-generated Q/A teaching pairs for a confident match.

    Returns a RetrievalResult if confidence >= RETRIEVAL_THRESHOLD,
    otherwise returns None so Stage 2 is triggered.
    """
    try:
        collection = client.get_collection(
            name=QA_COLLECTION_NAME, embedding_function=embedding_fn
        )
    except Exception:
        return None  # Collection not yet built

    if collection.count() == 0:
        return None

    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=["metadatas", "distances", "documents"],
    )

    if not results or not results.get("distances") or not results["distances"][0]:
        return None

    confidence = _cosine_similarity(results["distances"][0][0])

    if confidence < RETRIEVAL_THRESHOLD:
        return None  # Not confident enough — fall through to Stage 2

    meta = results["metadatas"][0][0]
    matched_question = results["documents"][0][0]

    return RetrievalResult(
        context=matched_question,
        method="qa_match",
        confidence=confidence,
        source_title=meta.get("source_title", ""),
        source_url=meta.get("source_page", ""),
        answer=meta.get("answer"),
        ksb_type=meta.get("ksb_type", ""),
        ksb_reference=meta.get("ksb_reference", ""),
    )


# ---------------------------------------------------------------------------
# Stage 2: Document chunk vector search
# ---------------------------------------------------------------------------

def _docs_search(
    client: chromadb.PersistentClient,
    embedding_fn: OpenAIEmbeddingFunction,
    query: str,
    n_results: int = 8,
) -> list[RetrievalResult]:
    """
    Search document chunks for passages relevant to the apprentice's query.

    Returns the top n_results chunks so the chatbot can synthesise a
    teaching answer from multiple parts of the standard.
    """
    try:
        collection = client.get_collection(
            name=DOCS_COLLECTION_NAME, embedding_function=embedding_fn
        )
    except Exception:
        return []

    if collection.count() == 0:
        return []

    actual_n = min(n_results, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=actual_n,
        include=["metadatas", "distances", "documents"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    retrieval_results: list[RetrievalResult] = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieval_results.append(
            RetrievalResult(
                context=doc,
                method="vector_search",
                confidence=_cosine_similarity(distance),
                source_title=meta.get("source_title", ""),
                source_url=meta.get("source_url", ""),
                answer=None,
                ksb_type="",
                ksb_reference="",
            )
        )

    return retrieval_results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def retrieve(query: str, n_doc_results: int = 8) -> list[RetrievalResult]:
    """
    Perform hybrid retrieval for the apprentice's query.

    Stage 1: Q/A teaching pairs — returns immediately if a confident
             match is found (confidence >= RETRIEVAL_THRESHOLD).
    Stage 2: Document chunk search — fallback when no confident Q/A
             match exists. Returns up to n_doc_results chunks for the
             LLM to synthesise a teaching answer from.

    Returns an empty list if neither collection has been built yet.
    """
    client = _get_client()
    embedding_fn = _get_embedding_fn()

    qa_result = _qa_search(client, embedding_fn, query)
    if qa_result is not None:
        return [qa_result]

    return _docs_search(client, embedding_fn, query, n_results=n_doc_results)
