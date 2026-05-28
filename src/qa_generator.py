"""
Synthetic Q/A generation for apprentice teaching support.

Reads each cleaned document from data/processed/ and sends it to
GPT-4o-mini with a teaching-focused prompt that generates questions an
apprentice would actually ask while studying their digital standard.

Questions target:
  - plain-English KSB explanations
  - workplace examples
  - evidence ideas for portfolios and professional discussions
  - reflection prompts for progress reviews
  - common misunderstandings to avoid

Output columns in qa_dataset.csv:
  question        — the apprentice's likely question
  answer          — a supportive teaching answer
  source_page     — URL of the Skills England page
  source_title    — page title
  audience        — always "apprentice" for this dataset
  question_type   — overview | ksb_explanation | workplace_example |
                    evidence_idea | reflection | misconception | general
  ksb_type        — knowledge | skill | behaviour | general | unknown
  ksb_reference   — e.g. K1, S3, B2 (blank if not identifiable)

Run with:
    python -m src.qa_generator
"""

import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config import (
    PROCESSED_DIR,
    QA_DATASET_PATH,
    OPENAI_CHAT_MODEL,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = OpenAI()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert UK apprenticeship educator creating a teaching Q&A dataset \
for apprentices on digital apprenticeship programmes.

Your task is to generate realistic questions an apprentice might ask while \
studying their Knowledge, Skills and Behaviours (KSBs), and to write \
supportive, plain-English answers based only on the source document provided.

Generate questions around:
  - What does this KSB mean in plain English?
  - How could I practise or develop this knowledge, skill or behaviour?
  - What workplace examples could help me understand this KSB?
  - What evidence might show I have this skill in my portfolio?
  - What should I avoid misunderstanding about this KSB?
  - How does this KSB link to my day-to-day role?
  - Can you explain this standard in beginner-friendly language?
  - What reflection questions could help me prepare for a progress review?
  - How could I talk about this KSB in a professional discussion?

Strict rules:
  1. Base answers ONLY on the content provided. Do not invent KSBs or \
assessment requirements.
  2. NEVER state that an apprentice has achieved, passed, or met a KSB. \
Instead say what they may need to demonstrate, and always recommend \
checking with their trainer, employer or assessor.
  3. Write in a warm, supportive, confidence-building tone — like a \
knowledgeable mentor helping a learner.
  4. If a question cannot be answered from the source, the answer must \
start with: "This is not specified in the source document."
  5. Where the question relates to a specific numbered KSB (e.g. K1, S3, \
B2), set ksb_reference to that reference and ksb_type accordingly.
  6. Return ONLY a valid JSON array. No explanation, no markdown fences.

Each array item must follow this exact schema:
{
  "question": "<the apprentice's question>",
  "answer": "<supportive teaching answer based only on the source>",
  "audience": "apprentice",
  "question_type": "overview|ksb_explanation|workplace_example|evidence_idea|reflection|misconception|general",
  "ksb_type": "knowledge|skill|behaviour|general|unknown",
  "ksb_reference": "<e.g. K1, S3, B2, or blank string if not applicable>"
}"""


# ---------------------------------------------------------------------------
# Helper: parse the model response
# ---------------------------------------------------------------------------

def _parse_response(raw_content: str) -> list[dict]:
    """Parse the model's JSON response into a list of Q/A dicts."""
    stripped = raw_content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return value
    return []


# ---------------------------------------------------------------------------
# Per-document generation
# ---------------------------------------------------------------------------

def _build_ksb_summary(doc: dict) -> str:
    """
    Build a concise KSB inventory string to include in the prompt.

    If structured KSBs were extracted by the cleaner, this gives the model
    a ready-made list to generate specific K/S/B-referenced questions from.
    Falls back gracefully if the fields are absent or empty.
    """
    parts: list[str] = []

    if doc.get("level"):
        parts.append(f"Level: {doc['level']}")
    if doc.get("duration"):
        parts.append(f"Typical duration: {doc['duration']}")
    if doc.get("summary"):
        parts.append(f"Occupational summary: {doc['summary'][:400]}")

    for label, key in [("Knowledge items", "knowledge"), ("Skill items", "skills"), ("Behaviour items", "behaviours")]:
        items = doc.get(key, [])
        if items:
            parts.append(f"\n{label}:\n" + "\n".join(f"  {item}" for item in items[:20]))

    return "\n".join(parts)


def generate_qa_for_document(
    title: str,
    url: str,
    text: str,
    doc: dict,
    num_pairs: int = 12,
) -> list[dict]:
    """
    Call the OpenAI API to generate apprentice-focused Q/A pairs for one document.

    The prompt includes both the full cleaned text (truncated if very long)
    and a structured KSB summary so the model can generate specific
    KSB-referenced questions (K1, S3, etc.) where possible.
    """
    max_chars = 7_000
    truncated_text = text[:max_chars] + "\n[... content truncated ...]" if len(text) > max_chars else text

    ksb_summary = _build_ksb_summary(doc)

    user_prompt = (
        f"Standard title: {title}\n"
        f"Source URL: {url}\n\n"
        f"--- Structured KSB data (if available) ---\n"
        f"{ksb_summary if ksb_summary else 'Not available for this page.'}\n\n"
        f"--- Full document text ---\n"
        f"{truncated_text}\n\n"
        f"Generate exactly {num_pairs} diverse apprentice-style Q/A pairs from the "
        f"content above. Cover at least three different question_type values. "
        f"Where possible, reference specific KSBs by their code (K1, S3, B2 etc.). "
        f"Return only the JSON array."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        raw = response.choices[0].message.content
        pairs = _parse_response(raw)

        for pair in pairs:
            pair["source_page"] = url
            pair["source_title"] = title

        return pairs

    except Exception as exc:
        print(f"    OpenAI error for '{title[:50]}': {exc}")
        return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all() -> None:
    """Generate apprentice-focused Q/A pairs for every processed document."""
    processed_files = sorted(PROCESSED_DIR.glob("*.json"))

    if not processed_files:
        print(f"No processed files found in {PROCESSED_DIR}. Run the cleaner first:")
        print("    python -m src.cleaner")
        return

    print(f"StandardsBot Q/A Generator — {len(processed_files)} document(s)\n")
    print("Generating apprentice-focused KSB teaching Q/A pairs...\n")

    pairs_per_doc = max(8, 150 // len(processed_files))
    all_pairs: list[dict] = []

    for index, path in enumerate(processed_files, start=1):
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)

        title = doc.get("title") or path.stem
        url = doc.get("url", "")
        text = doc.get("cleaned_text", "")

        if not text.strip():
            print(f"  [{index:>2}/{len(processed_files)}] Skipping {path.name}: empty content.")
            continue

        k_count = len(doc.get("knowledge", []))
        s_count = len(doc.get("skills", []))
        b_count = len(doc.get("behaviours", []))
        print(
            f"  [{index:>2}/{len(processed_files)}] {title[:55]}\n"
            f"    KSBs found: K={k_count} S={s_count} B={b_count}"
        )

        pairs = generate_qa_for_document(title, url, text, doc, num_pairs=pairs_per_doc)
        all_pairs.extend(pairs)

        print(f"    Generated {len(pairs)} pairs (total so far: {len(all_pairs)})")

        if index < len(processed_files):
            time.sleep(0.5)

    if not all_pairs:
        print("\nNo Q/A pairs generated. Check your OPENAI_API_KEY and processed data.")
        return

    # Remove pairs with unhelpful "not specified" answers
    before = len(all_pairs)
    all_pairs = [
        p for p in all_pairs
        if "not specified in the source" not in str(p.get("answer", "")).lower()
    ]
    removed = before - len(all_pairs)
    if removed:
        print(f"\nFiltered out {removed} low-quality pairs (answer was 'not specified').")

    # Write CSV with all columns including new KSB fields
    QA_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question", "answer", "source_page", "source_title",
        "audience", "question_type", "ksb_type", "ksb_reference",
    ]

    with open(QA_DATASET_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_pairs)

    print(f"\nDone. {len(all_pairs)} Q/A pairs saved to {QA_DATASET_PATH}")


if __name__ == "__main__":
    generate_all()
