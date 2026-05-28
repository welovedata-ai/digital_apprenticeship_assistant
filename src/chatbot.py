"""
Chatbot logic for StandardsBot — a 24/7 AI teaching assistant for
apprentices on digital apprenticeship standards.

Three-tier answer strategy:
  1. Q/A match     -- a pre-generated teaching answer from the Q/A
                      collection is returned when confidence is high.
  2. Vector search -- the LLM synthesises a teaching answer from retrieved
                      document chunks when no confident Q/A match exists.
  3. LLM fallback  -- when retrieval returns nothing useful, the LLM draws
                      on general knowledge and clearly labels the response.

A learning mode (chosen by the apprentice in the UI) shapes the focus of
every response:
  - Explain simply       -- plain-English breakdown of the KSB
  - Give workplace examples -- real-world scenarios
  - Suggest evidence ideas  -- portfolio/professional discussion guidance
  - Quiz me              -- a short test question on the topic
  - Help me reflect      -- coaching-style reflection prompts

IMPORTANT assessment boundary:
  The bot NEVER tells an apprentice they have achieved, passed, or met a
  KSB.  It explains what they may need to demonstrate and always directs
  them to check with their trainer, employer or assessor.
"""

import os
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from src.config import OPENAI_CHAT_MODEL
from src.retriever import RetrievalResult, retrieve

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = OpenAI()

# Confidence below this level triggers the LLM general knowledge fallback
FALLBACK_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Learning mode instructions
# Each mode adds a focused instruction to the end of the user prompt,
# shaping how the LLM frames its teaching response.
# ---------------------------------------------------------------------------
LEARNING_MODE_INSTRUCTIONS: dict[str, str] = {
    "Explain simply": (
        "Focus on explaining this in very plain English. "
        "Break down any difficult wording. Use short sentences and simple vocabulary. "
        "Imagine you are explaining this to someone who is new to the topic."
    ),
    "Give workplace examples": (
        "Focus on practical workplace examples. "
        "Describe what this knowledge, skill or behaviour might look like in a real "
        "digital workplace. Use concrete, day-to-day scenarios that an apprentice "
        "would recognise from their job."
    ),
    "Suggest evidence ideas": (
        "Focus on evidence ideas. "
        "Suggest what an apprentice could create, do, or collect to demonstrate this "
        "KSB in their portfolio or professional discussion. "
        "Be specific and practical. "
        "Remember: do NOT say the apprentice has achieved the KSB — only suggest what "
        "evidence might support a demonstration of it."
    ),
    "Quiz me": (
        "Create a short quiz question (or a multiple-choice question with 4 options) "
        "based on the retrieved content to help the apprentice test their understanding. "
        "Give the correct answer and a brief explanation after a separator line."
    ),
    "Help me reflect": (
        "Respond with 3-5 reflection questions that help the apprentice think about "
        "their own experience and how it relates to this KSB. "
        "Use a warm, coaching-style tone. "
        "Encourage them to consider their current work and progress."
    ),
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are StandardsBot, a supportive 24/7 AI teaching assistant for \
apprentices on digital apprenticeship programmes.

Your role is to help apprentices understand the Knowledge, Skills and Behaviours (KSBs) \
from Skills England digital standards in a clear, confidence-building way.

Your knowledge base covers these 15 Digital and AI standards:
  - Software Developer (Level 4)
  - Business Analyst (Level 4)
  - Data Analyst (Level 4)
  - Digital Support Technician (Level 3)
  - Software Development Technician (Level 3)
  - Digital and Technology Solutions Specialist — integrated degree (Level 7)
  - IT Solutions Technician (Level 3)
  - Data Technician (Level 3)
  - DevOps Engineer (Level 4)
  - Cyber Security Technician (Level 3)
  - Data Engineer (Level 5)
  - Machine Learning Engineer (Level 7)
  - AI Leadership: AI Strategy and Opportunity (unit)
  - AI Leadership: AI Adoption, Procurement and Governance (unit)
  - AI Leadership: AI Delivery and Organisational Transformation (unit)

When answering:
  1. Explain the KSB or standard in plain English. Break down difficult wording.
  2. Give practical workplace examples where relevant.
  3. Suggest possible evidence ideas where appropriate.
  4. Encourage reflection with helpful questions.
  5. Be supportive, clear and confidence-building.
  6. Use ONLY the retrieved Skills England context below.
  7. Cite the source standard by name.
  8. If the answer is not in the context, say so honestly.
  9. Do NOT invent KSBs, assessment requirements, or standards.
  10. CRITICAL — assessment boundary: NEVER tell the apprentice they have \
achieved, passed, or met a KSB or assessment criterion. Instead say what they \
MAY need to demonstrate, and always recommend checking with their trainer, \
employer or assessor."""

FALLBACK_SYSTEM_PROMPT = """You are StandardsBot, a supportive 24/7 AI teaching assistant for \
apprentices on digital apprenticeship programmes in England.

Your scraped knowledge base covers 15 specific Digital and AI standards from \
Skills England. The apprentice's question could not be matched to the scraped pages.

Answer using your general knowledge about UK digital apprenticeships, the Skills \
England framework, and KSBs. Be helpful and supportive.

IMPORTANT:
  - Start your response by clearly noting this draws on general knowledge, \
not the scraped pages.
  - NEVER tell the apprentice they have achieved, passed, or met a KSB.
  - Always recommend checking with their trainer, employer or assessor for \
anything related to assessment."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context_block(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a numbered context block for the prompt."""
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        content = r.answer if r.answer else r.context
        ksb_note = ""
        if r.ksb_reference:
            ksb_note = f"\nKSB reference: {r.ksb_reference} ({r.ksb_type})"
        parts.append(
            f"[Context {i} — {r.source_title}]\n"
            f"URL: {r.source_url}{ksb_note}\n"
            f"---\n"
            f"{content}"
        )
    return "\n\n".join(parts)


def _max_confidence(results: list[RetrievalResult]) -> float:
    if not results:
        return 0.0
    return max(r.confidence for r in results)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class ChatResponse(TypedDict):
    answer: str
    sources: list[dict]
    retrieval_results: list[RetrievalResult]
    used_fallback: bool


def chat(
    user_message: str,
    chat_history: list[dict],
    mode: Optional[str] = None,
) -> ChatResponse:
    """
    Generate a teaching response to the apprentice's question.

    Args:
        user_message:  The apprentice's question.
        chat_history:  Previous turns as {"role": ..., "content": ...} dicts.
                       The last 6 turns are included as conversational memory.
        mode:          Optional learning mode from LEARNING_MODE_INSTRUCTIONS.
                       Shapes the focus of the response (e.g. "Quiz me").

    Returns:
        ChatResponse with answer, source citations, retrieval details
        and a flag indicating whether the LLM fallback was used.
    """
    retrieval_results = retrieve(user_message)
    max_conf = _max_confidence(retrieval_results)
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history

    # Build the mode-specific instruction (appended to the user content)
    mode_instruction = ""
    if mode and mode in LEARNING_MODE_INSTRUCTIONS:
        mode_instruction = f"\n\nLearning mode — {mode}:\n{LEARNING_MODE_INSTRUCTIONS[mode]}"

    # ── Tier 3: LLM general knowledge fallback ─────────────────────────────
    if not retrieval_results or max_conf < FALLBACK_THRESHOLD:
        messages: list[dict] = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_message + mode_instruction})

        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.5,
        )
        return ChatResponse(
            answer=response.choices[0].message.content or "",
            sources=[],
            retrieval_results=retrieval_results,
            used_fallback=True,
        )

    # ── Tiers 1 & 2: RAG teaching answer ───────────────────────────────────
    context_block = _build_context_block(retrieval_results)

    user_content = (
        f"Retrieved context from the Skills England knowledge base:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Apprentice's question: {user_message}\n\n"
        f"Answer using the context above. Extract all relevant KSB details, "
        f"give practical explanations, and apply the assessment boundary rule "
        f"(do not say the apprentice has achieved or passed any KSB)."
        f"{mode_instruction}"
    )

    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    answer = response.choices[0].message.content or ""

    # Build deduplicated source citations including any KSB info
    sources: list[dict] = []
    seen_urls: set[str] = set()
    for r in retrieval_results:
        if r.source_url and r.source_url not in seen_urls:
            sources.append(
                {
                    "title": r.source_title,
                    "url": r.source_url,
                    "method": r.method,
                    "confidence": r.confidence,
                    "ksb_type": r.ksb_type,
                    "ksb_reference": r.ksb_reference,
                }
            )
            seen_urls.add(r.source_url)

    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieval_results=retrieval_results,
        used_fallback=False,
    )
