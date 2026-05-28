"""
Cleaner for raw scraped HTML.

Loads each JSON file from data/raw/, strips non-content elements
(scripts, styles, navigation, headers, footers) using BeautifulSoup,
then attempts structured KSB extraction from the cleaned text.

Structured extraction tries to identify:
  - standard title and level
  - occupational summary
  - duty list
  - knowledge items (K1, K2, ...)
  - skill items  (S1, S2, ...)
  - behaviour items (B1, B2, ...)

If a section cannot be reliably identified, the field is left as an empty
list and the full cleaned text remains available as a fallback. This means
the pipeline always produces usable output even for unusual page layouts.

Run with:
    python -m src.cleaner
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment

from src.config import RAW_DIR, PROCESSED_DIR

# Tags whose entire content should be removed before text extraction
DISCARD_TAGS = [
    "script", "style", "noscript", "link", "meta",
    "nav", "header", "footer", "aside", "iframe",
]

# CSS class/id fragments that indicate navigation or site chrome
DISCARD_CLASS_FRAGMENTS = [
    "nav", "navigation", "header", "footer", "cookie",
    "breadcrumb", "skip-link", "banner", "sidebar",
    "menu", "search", "social", "share", "print",
]

# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _has_noise_class(tag) -> bool:
    """Return True if the tag looks like site chrome based on its class/id."""
    if not hasattr(tag, "attrs") or not tag.attrs:
        return False
    combined = " ".join(tag.get("class", []) + [tag.get("id", "")]).lower()
    return any(fragment in combined for fragment in DISCARD_CLASS_FRAGMENTS)


def clean_html(html_content: str) -> str:
    """
    Parse raw HTML and return clean, readable plain text.

    Removes scripts, styles, navigation and other site chrome, then
    normalises whitespace and drops consecutive duplicate lines.
    """
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup.find_all(DISCARD_TAGS):
        tag.decompose()

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    for tag in soup.find_all(True):
        if _has_noise_class(tag):
            tag.decompose()

    raw_text = soup.get_text(separator="\n")

    lines = raw_text.splitlines()
    cleaned: list[str] = []
    prev = None
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or line == prev:
            continue
        cleaned.append(line)
        prev = line

    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Structured KSB extraction
# ---------------------------------------------------------------------------
# Skills England pages follow a broadly consistent layout. The cleaned text
# typically contains sections headed "Knowledge", "Skills" and "Behaviours"
# with numbered items (K1, K2 ... S1, S2 ... B1, B2 ...).
#
# This extraction is best-effort: if a section heading or pattern is absent,
# the corresponding list is left empty and the full text is still preserved.
# ---------------------------------------------------------------------------

def _extract_level(text: str) -> str:
    """Extract the apprenticeship level from the cleaned text."""
    match = re.search(r"Level[:\s]+(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else ""


def _extract_duration(text: str) -> str:
    """Extract the typical duration from the cleaned text."""
    match = re.search(r"Typical duration[:\s]+(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_section_items(text: str, prefix: str) -> list[str]:
    """
    Extract numbered KSB items for a given prefix (K, S, or B).

    Looks for patterns like:
        K1\nDescription of the knowledge item
    or
        K1: Description on a single line

    Returns a list of strings in the format "K1: description".
    """
    # Pattern: prefix + number, optionally followed by colon, then text
    pattern = rf"({re.escape(prefix)}\d+)[:\s]+([^\n]+(?:\n(?!{re.escape(prefix)}\d)[^\n]+)*)"
    matches = re.findall(pattern, text)
    items = []
    for ref, body in matches:
        body_clean = re.sub(r"\s+", " ", body).strip()
        if body_clean:
            items.append(f"{ref}: {body_clean}")
    return items


def _extract_duties(text: str) -> list[str]:
    """
    Extract occupational duty items.

    Looks for lines containing "Duty N" or "D\d+" patterns.
    """
    pattern = r"(?:Duty\s+\d+|D\d+)[:\s]+([^\n]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def _extract_summary(text: str) -> str:
    """
    Extract the occupational/overview summary paragraph.

    Takes the first substantive paragraph that follows a heading like
    "Occupation summary" or "Overview of the role".
    """
    match = re.search(
        r"(?:Occupation summary|Overview of the role|Apprenticeship summary)\n+(.+?)(?:\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        summary = re.sub(r"\s+", " ", match.group(1)).strip()
        return summary[:1000]  # cap to avoid including the whole document
    return ""


def extract_ksb_structure(text: str) -> dict:
    """
    Attempt to extract structured KSB fields from the cleaned page text.

    Returns a dict with keys: level, duration, summary, duties,
    knowledge, skills, behaviours.

    All fields degrade gracefully to empty strings / empty lists if the
    content cannot be located.
    """
    return {
        "level": _extract_level(text),
        "duration": _extract_duration(text),
        "summary": _extract_summary(text),
        "duties": _extract_duties(text),
        "knowledge": _extract_section_items(text, "K"),
        "skills": _extract_section_items(text, "S"),
        "behaviours": _extract_section_items(text, "B"),
    }


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def clean_file(raw_path: Path) -> dict | None:
    """
    Load a raw JSON file, clean its HTML, attempt KSB extraction, and
    return a processed document dict.

    Returns None if the file recorded a scrape error or had no HTML.
    """
    with open(raw_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if raw.get("error"):
        print(f"    Warning: Skipping {raw_path.name} — scrape error: {raw['error'][:80]}")
        return None

    html = raw.get("html_content", "")
    if not html.strip():
        print(f"    Warning: Skipping {raw_path.name} — no HTML content.")
        return None

    cleaned_text = clean_html(html)
    if not cleaned_text.strip():
        print(f"    Warning: Skipping {raw_path.name} — cleaning produced empty output.")
        return None

    ksb_structure = extract_ksb_structure(cleaned_text)

    return {
        # Core metadata
        "url": raw.get("url", ""),
        "title": raw.get("title", ""),
        "page_type": raw.get("page_type", "unknown"),
        "scraped_at": raw.get("scraped_at", ""),
        "source_filename": raw_path.name,
        # Full cleaned text (used for chunking and vector search)
        "cleaned_text": cleaned_text,
        "char_count": len(cleaned_text),
        # Structured KSB fields (used for Q/A generation and richer answers)
        "level": ksb_structure["level"],
        "duration": ksb_structure["duration"],
        "summary": ksb_structure["summary"],
        "duties": ksb_structure["duties"],
        "knowledge": ksb_structure["knowledge"],
        "skills": ksb_structure["skills"],
        "behaviours": ksb_structure["behaviours"],
    }


def clean_all() -> None:
    """Process every raw JSON file and write cleaned+structured versions."""
    raw_files = sorted(RAW_DIR.glob("*.json"))

    if not raw_files:
        print(f"No raw files found in {RAW_DIR}. Run the scraper first:")
        print("    python -m src.scraper")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"StandardsBot Cleaner — processing {len(raw_files)} file(s)\n")

    success = 0
    for raw_path in raw_files:
        print(f"  Processing: {raw_path.name}")
        processed = clean_file(raw_path)
        if not processed:
            continue

        out_path = PROCESSED_DIR / raw_path.name
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(processed, fh, ensure_ascii=False, indent=2)

        ksb_counts = (
            f"K:{len(processed['knowledge'])} "
            f"S:{len(processed['skills'])} "
            f"B:{len(processed['behaviours'])}"
        )
        print(f"    Saved {out_path.name} — {processed['char_count']:,} chars | {ksb_counts}")
        success += 1

    print(f"\nDone. {success}/{len(raw_files)} files cleaned to {PROCESSED_DIR}/")


if __name__ == "__main__":
    clean_all()
