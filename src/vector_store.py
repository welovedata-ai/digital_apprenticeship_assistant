"""
Vector store builder using ChromaDB.

Creates and persists two collections:
  - doc_chunks  : overlapping chunks of cleaned page text
  - qa_pairs    : Q/A dataset questions (answers stored as metadata)

Both collections use OpenAI text-embedding-3-small embeddings and cosine
similarity as the distance metric.

By default, existing collections are left untouched --- the script exits
early if both collections already contain data.  Pass --force to drop
and fully rebuild them.

Run with:
    python -m src.vector_store           # build only if not yet built
    python -m src.vector_store --force   # always drop and rebuild
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from src.config import (
    PROCESSED_DIR,
    QA_DATASET_PATH,
    CHROMA_PERSIST_PATH,
    DOCS_COLLECTION_NAME,
    QA_COLLECTION_NAME,
    OPENAI_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_embedding_function() -> OpenAIEmbeddingFunction:
    """Return a configured OpenAI embedding function for ChromaDB."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=OPENAI_EMBEDDING_MODEL,
    )


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Chunk size and overlap are controlled by CHUNK_SIZE and CHUNK_OVERLAP
    in config.py.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _add_in_batches(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    batch_size: int = 50,
) -> None:
    """Add items to a Chroma collection in batches to respect API limits."""
    total = len(ids)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"    Added items {i + 1}---{end} of {total}")


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------

def build_docs_collection(
    client: chromadb.PersistentClient,
    embedding_fn: OpenAIEmbeddingFunction,
    force: bool = False,
) -> chromadb.Collection:
    """
    Chunk all processed documents and store them in the doc_chunks collection.

    If the collection already contains data and force=False, it is returned
    as-is without any embedding calls.  Pass force=True to drop and rebuild.
    """
    # Return existing collection if it already has content and force is not set
    try:
        existing = client.get_collection(
            name=DOCS_COLLECTION_NAME, embedding_function=embedding_fn
        )
        if existing.count() > 0 and not force:
            print(
                f"  --- '{DOCS_COLLECTION_NAME}' already has {existing.count()} chunk(s). "
                "Skipping rebuild (use --force to override)."
            )
            return existing
    except Exception:
        pass  # Collection does not exist yet --- proceed to build

    processed_files = sorted(PROCESSED_DIR.glob("*.json"))
    if not processed_files:
        print(f"  --- No processed files found in {PROCESSED_DIR}. Skipping docs collection.")
        return client.get_or_create_collection(
            name=DOCS_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # Drop and rebuild
    try:
        client.delete_collection(name=DOCS_COLLECTION_NAME)
        print(f"  Dropped existing '{DOCS_COLLECTION_NAME}' collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=DOCS_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []

    for path in processed_files:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)

        text = doc.get("cleaned_text", "")
        if not text.strip():
            continue

        chunks = chunk_text(text)
        print(f"  {path.name}: {len(chunks)} chunk(s)")

        for j, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}_chunk_{j}"
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metas.append(
                {
                    "source_url": doc.get("url", ""),
                    "source_title": doc.get("title", ""),
                    "chunk_id": chunk_id,
                    "file_name": path.name,
                    "page_type": doc.get("page_type", ""),
                }
            )

    if all_docs:
        print(f"\n  Embedding {len(all_docs)} chunk(s) --- this may take a moment---")
        _add_in_batches(collection, all_ids, all_docs, all_metas)

    return collection


def build_qa_collection(
    client: chromadb.PersistentClient,
    embedding_fn: OpenAIEmbeddingFunction,
    force: bool = False,
) -> chromadb.Collection:
    """
    Embed the question text from qa_dataset.csv and store it in the
    qa_pairs collection. The full answer is stored as metadata so the
    retriever can return it without a second lookup.

    If the collection already contains data and force=False, it is returned
    as-is.  Pass force=True to drop and rebuild.
    """
    # Return existing collection if it already has content and force is not set
    try:
        existing = client.get_collection(
            name=QA_COLLECTION_NAME, embedding_function=embedding_fn
        )
        if existing.count() > 0 and not force:
            print(
                f"  --- '{QA_COLLECTION_NAME}' already has {existing.count()} pair(s). "
                "Skipping rebuild (use --force to override)."
            )
            return existing
    except Exception:
        pass  # Collection does not exist yet --- proceed to build

    if not QA_DATASET_PATH.exists():
        print(f"  --- Q/A dataset not found at {QA_DATASET_PATH}. Skipping qa collection.")
        return client.get_or_create_collection(
            name=QA_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    df = pd.read_csv(QA_DATASET_PATH)
    if df.empty:
        print("  --- Q/A dataset is empty. Skipping qa collection.")
        return client.get_or_create_collection(
            name=QA_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # Drop and rebuild
    try:
        client.delete_collection(name=QA_COLLECTION_NAME)
        print(f"  Dropped existing '{QA_COLLECTION_NAME}' collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=QA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"qa_{i}" for i in range(len(df))]
    questions = df["question"].astype(str).tolist()
    metadatas = [
        {
            "answer": str(row.get("answer", "")),
            "source_page": str(row.get("source_page", "")),
            "source_title": str(row.get("source_title", "")),
            "audience": str(row.get("audience", "")),
            "question_type": str(row.get("question_type", "")),
            # New KSB-specific metadata (may be empty for older datasets)
            "ksb_type": str(row.get("ksb_type", "")),
            "ksb_reference": str(row.get("ksb_reference", "")),
        }
        for _, row in df.iterrows()
    ]

    print(f"\n  Embedding {len(questions)} Q/A question(s)---")
    _add_in_batches(collection, ids, questions, metadatas)

    return collection


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_vector_store(force: bool = False) -> None:
    """
    Build both Chroma collections from the latest processed data.

    Args:
        force: If True, drop and rebuild existing collections.
               If False (default), skip collections that already have data.
    """
    Path(CHROMA_PERSIST_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    embedding_fn = get_embedding_function()

    if force:
        print("--force flag set: existing collections will be dropped and rebuilt.\n")

    print("---" * 60)
    print("Document chunks collection")
    print("---" * 60)
    docs_col = build_docs_collection(client, embedding_fn, force=force)
    print(f"\n--- '{DOCS_COLLECTION_NAME}': {docs_col.count()} chunk(s).")

    print()
    print("---" * 60)
    print("Q/A pairs collection")
    print("---" * 60)
    qa_col = build_qa_collection(client, embedding_fn, force=force)
    print(f"\n--- '{QA_COLLECTION_NAME}': {qa_col.count()} pair(s).")

    print(f"\nVector store ready at: {CHROMA_PERSIST_PATH}")


if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv
    build_vector_store(force=force_rebuild)

