"""
StandardsBot — Streamlit UI

Digital Apprenticeship Assistant: a 24/7 AI teaching companion for
apprentices studying digital standards from Skills England.

Run with:
    python -m streamlit run app.py
"""

import base64
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).resolve().parent / ".env")

from src.config import (
    CHROMA_PERSIST_PATH,
    LEARNING_MODES,
    QA_DATASET_PATH,
    RAW_DIR,
    RETRIEVAL_THRESHOLD,
)
from src.chatbot import chat

# ── Page config ───────────────────────────────────────────────────────────────
def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_icon_path  = Path(__file__).parent / "assets" / "icon.png"
_robot_path = Path(__file__).parent / "assets" / "robot.png"

# icon.png is the purpose-built small app icon (sidebar + browser tab)
_icon_b64 = _img_b64(_icon_path) if _icon_path.exists() else None

# Browser-tab favicon — use icon.png, fall back to robot.png, then emoji
if _icon_path.exists():
    from PIL import Image as _PILImage
    _page_icon = _PILImage.open(_icon_path)
elif _robot_path.exists():
    from PIL import Image as _PILImage
    _page_icon = _PILImage.open(_robot_path)
else:
    _page_icon = "🎓"

st.set_page_config(
    page_title="Digital Apprenticeship Assistant",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background: #0f1520 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section.main > div { padding-top: 0 !important; }
h1,h2,h3 { font-weight: 700 !important; }

/* ── Sidebar brand ── */
.sb-brand { padding: 0.4rem 0 0.6rem; }
.sb-brand-header {
    display: flex; align-items: center; gap: 0.7rem;
    margin-bottom: 0.7rem;
}
.sb-logo-icon {
    width: 52px; height: 52px; border-radius: 14px; flex-shrink: 0;
    overflow: hidden;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.4),
                0 4px 18px rgba(109,40,217,0.45);
}
.sb-logo-icon img {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
}
.sb-logo-title {
    font-size: 0.93rem; font-weight: 700;
    color: #f1f5f9; line-height: 1.3;
}
.sb-brand-desc {
    font-size: 0.76rem; color: #94a3b8; line-height: 1.65;
    border-left: 2px solid rgba(129,140,248,0.25);
    padding-left: 0.75rem; margin-bottom: 0.5rem;
}
.sb-brand-desc strong { color: #c7d2fe; font-weight: 600; }
.sb-brand-desc ul {
    margin: 0.25rem 0 0.35rem 0.1rem;
    padding-left: 1rem; list-style: disc;
}
.sb-brand-desc ul li { margin-bottom: 0.1rem; }
.sb-brand-link {
    font-size: 0.74rem; color: #818cf8;
    text-decoration: none; display: inline-flex;
    align-items: center; gap: 0.3rem; margin-top: 0.15rem;
}
.sb-brand-link:hover { color: #a5b4fc; text-decoration: underline; }

/* ── Mode selector label ── */
.mode-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #475569; margin: 0 0 0.4rem;
}

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin: 0.5rem 0; }
.stat-card {
    background: #161c2d; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 11px; padding: 0.7rem;
}
.stat-card .si { font-size: 1rem; margin-bottom: 0.2rem; }
.stat-card .sv { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; line-height: 1; }
.stat-card .sl { font-size: 0.64rem; color: #64748b; margin-top: 0.15rem; }

/* ── Standards list ── */
.std-item {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.32rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.79rem; color: #94a3b8;
}
.std-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(109,40,217,0.08); border: 1px solid rgba(109,40,217,0.22);
    border-radius: 10px; padding: 0.65rem 0.85rem;
    font-size: 0.79rem; color: #a78bfa; line-height: 1.6; margin-top: 0.5rem;
}

/* ── Section heading ── */
.sec-heading {
    font-size: 1.15rem; font-weight: 700; color: #f1f5f9;
    margin: 0 0 0.2rem; letter-spacing: -0.2px;
}
.sec-sub { font-size: 0.85rem; color: #64748b; margin: 0 0 1rem; }

/* ── Suggestion cards (buttons) ── */
div[data-testid="stButton"] > button {
    background: #161c2d !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    color: #cbd5e1 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 1.3rem 3.2rem 1.3rem 1.4rem !important;
    width: 100% !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    min-height: 5rem !important;
    position: relative !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(129,140,248,0.1) !important;
    border-color: rgba(129,140,248,0.45) !important;
    color: #f1f5f9 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px rgba(129,140,248,0.14) !important;
}
div[data-testid="stButton"] > button::after {
    content: "→";
    position: absolute; right: 1.2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 1.1rem; color: #4b5563;
    transition: color 0.2s;
}
div[data-testid="stButton"] > button:hover::after { color: #818cf8; }
div[data-testid="stButton"] > button *,
div[data-testid="stButton"] > button > div,
div[data-testid="stButton"] > button > div > p,
div[data-testid="stButton"] > button p {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin: 0 !important;
}

/* ── Clear button override ── */
button[kind="secondary"] {
    font-size: 0.85rem !important;
    min-height: 2.5rem !important;
    padding: 0.5rem 1rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    font-size: 1rem !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    background: #161c2d !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #f1f5f9 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #475569 !important; }

/* ── Mode badge ── */
.mode-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
    color: #818cf8; border-radius: 20px;
    padding: 0.2rem 0.75rem; font-size: 0.73rem; font-weight: 600;
    margin-bottom: 0.6rem;
}

/* ── Source card ── */
.src-card {
    background: #161c2d; border: 1px solid rgba(129,140,248,0.18);
    border-radius: 11px; padding: 0.6rem 1rem;
    margin: 0.25rem 0; font-size: 0.84rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.src-card a { color: #818cf8; text-decoration: none; font-weight: 500; }
.src-card a:hover { text-decoration: underline; color: #a5b4fc; }
.ksb-tag {
    display: inline-block; background: rgba(129,140,248,0.15);
    color: #818cf8; border-radius: 5px;
    padding: 0.1rem 0.45rem; font-size: 0.7rem; font-weight: 700; margin-left: 0.4rem;
}

/* ── General knowledge warning ── */
.gk-banner {
    background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.22);
    border-radius: 11px; padding: 0.7rem 1rem;
    font-size: 0.84rem; color: #fbbf24; margin-bottom: 0.5rem; line-height: 1.5;
}
.gk-banner a { color: #fbbf24; }

/* ── Banner image ── */
[data-testid="stImage"] img {
    border-radius: 20px !important;
    width: 100% !important;
    display: block !important;
}

/* ── Setup warning ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.88rem !important;
}

/* ── Section label (small caps) ── */
.lbl {
    font-size: 0.67rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #475569; margin: 1rem 0 0.5rem;
}

/* ── Divider styling ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1rem 0 !important; }

/* ── Chat message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    padding-bottom: 1rem !important;
    margin-bottom: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Status ────────────────────────────────────────────────────────────────────
raw_count = len(list(RAW_DIR.glob("*.json"))) if RAW_DIR.exists() else 0
qa_count = 0
if QA_DATASET_PATH.exists():
    try:
        qa_count = len(pd.read_csv(QA_DATASET_PATH))
    except Exception:
        pass
chroma_path = Path(CHROMA_PERSIST_PATH)
vector_ready = chroma_path.exists() and any(chroma_path.iterdir())

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    _logo_src = f"data:image/png;base64,{_icon_b64}" if _icon_b64 else ""
    _logo_img = (
        f'<img src="{_logo_src}" alt="Digital Apprenticeship Assistant icon">'
        if _logo_src else '<span style="font-size:1.8rem;line-height:52px;text-align:center;display:block;">🎓</span>'
    )
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-brand-header">
            <div class="sb-logo-icon">{_logo_img}</div>
            <div class="sb-logo-title">Digital Apprenticeship Assistant</div>
        </div>
        <div class="sb-brand-desc">
            <strong>Support Agent for KSBs:</strong><br>
            I answer questions about:
            <ul>
                <li>Knowledge Criteria</li>
                <li>Skills Criteria</li>
                <li>Behaviour Criteria</li>
            </ul>
            All Apprenticeship standards can be found at:<br>
            <a class="sb-brand-link"
               href="https://skillsengland.education.gov.uk/"
               target="_blank">🔗 Skills England website</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Learning mode — top of sidebar, prominent
    st.markdown('<p class="mode-label">How would you like me to respond?</p>', unsafe_allow_html=True)
    selected_mode = st.selectbox(
        label="Answer style",
        options=LEARNING_MODES,
        index=0,
        key="learning_mode",
        label_visibility="collapsed",
    )
    MODE_DESC = {
        "Explain simply":         "Plain-English breakdown, no jargon",
        "Give workplace examples": "What this looks like in a real job",
        "Suggest evidence ideas":  "Ideas for your portfolio or review",
        "Quiz me":                 "Test your understanding with a question",
        "Help me reflect":         "Coaching questions to aid your thinking",
    }
    st.caption(f"📌 {MODE_DESC.get(selected_mode, '')}")
    st.markdown("---")

    # Knowledge base stats
    v_icon = "✅" if vector_ready else "⚠️"
    v_label = "Ready" if vector_ready else "Not built"
    st.markdown(f"""
    <p class="lbl">Knowledge Base</p>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="si">📄</div><div class="sv">{raw_count}</div>
            <div class="sl">Standards</div>
        </div>
        <div class="stat-card">
            <div class="si">💬</div><div class="sv">{qa_count}</div>
            <div class="sl">Q&amp;A Pairs</div>
        </div>
        <div class="stat-card">
            <div class="si">🎯</div><div class="sv">{RETRIEVAL_THRESHOLD:.0%}</div>
            <div class="sl">Match threshold</div>
        </div>
        <div class="stat-card">
            <div class="si">{v_icon}</div><div class="sv" style="font-size:0.82rem;">{v_label}</div>
            <div class="sl">Vector store</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Standards covered (collapsible)
    STANDARDS = [
        ("#3b82f6", "Software Developer"),
        ("#8b5cf6", "Business Analyst"),
        ("#06b6d4", "Data Analyst"),
        ("#10b981", "DevOps Engineer"),
        ("#ef4444", "Cyber Security Technician"),
        ("#f59e0b", "Data Technician"),
        ("#6366f1", "Data Engineer"),
        ("#ec4899", "Machine Learning Engineer"),
        ("#14b8a6", "Digital Support Technician"),
        ("#84cc16", "IT Solutions Technician"),
        ("#f97316", "Software Dev. Technician"),
        ("#a78bfa", "Digital & Tech Solutions Specialist"),
        ("#60a5fa", "AI Leadership (×3 units)"),
    ]
    with st.expander("📚 Standards covered (15)"):
        rows = "".join(
            f'<div class="std-item"><div class="std-dot" style="background:{c};"></div>{n}</div>'
            for c, n in STANDARDS
        )
        st.markdown(rows, unsafe_allow_html=True)

    # Sample Q&A (collapsible)
    with st.expander("📋 Browse sample Q&A pairs"):
        if QA_DATASET_PATH.exists() and qa_count > 0:
            try:
                df_qa = pd.read_csv(QA_DATASET_PATH)
                qtypes = ["All"] + sorted(df_qa["question_type"].dropna().unique().tolist())
                sel = st.selectbox("Filter by type", qtypes, key="qa_type_filter")
                filt = df_qa if sel == "All" else df_qa[df_qa["question_type"] == sel]
                for _, row in filt.sample(min(3, len(filt))).iterrows():
                    ref = str(row.get("ksb_reference", ""))
                    ref_str = f" · **{ref}**" if ref and ref != "nan" else ""
                    st.markdown(f"**Q:** {row['question']}{ref_str}")
                    st.markdown(f"**A:** {str(row['answer'])[:200]}{'...' if len(str(row['answer'])) > 200 else ''}")
                    st.caption(f"{str(row.get('source_title','')).split('/')[0].strip()} · {row.get('question_type','')}")
                    st.markdown("---")
            except Exception as exc:
                st.warning(f"Could not load pairs: {exc}")
        else:
            st.info("Run `python -m src.qa_generator` first.")

    st.markdown("---")

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        '<strong>⚠️ Learning support only</strong><br>'
        'This tool helps you understand your KSBs. It does <strong>not</strong> '
        'replace your trainer, employer or assessor. Always confirm assessment '
        'requirements with your programme team.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── HERO ──────────────────────────────────────────────────────────────────────
banner_path = Path(__file__).parent / "assets" / "banner.png"

if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)
else:
    st.markdown("""
    <div style="text-align:center; padding: 2rem 1rem;">
        <h1 style="font-size:2.4rem; font-weight:800; color:#f1f5f9;">
            Digital Apprenticeship <span style="color:#818cf8;">Assistant</span>
        </h1>
        <p style="color:#94a3b8;">Your 24/7 AI study buddy for digital apprenticeship standards.</p>
    </div>
    """, unsafe_allow_html=True)

# Pipeline warning
if raw_count == 0:
    st.warning(
        "**No data found.** Run the pipeline first: "
        "`python -m src.scraper` → `python -m src.cleaner` → "
        "`python -m src.qa_generator` → `python -m src.vector_store`"
    )

st.markdown("---")

# ── SUGGESTIONS ───────────────────────────────────────────────────────────────
st.markdown("""
<p class="sec-heading">Where would you like to start?</p>
<p class="sec-sub">Click any question below or type your own in the chat box.</p>
""", unsafe_allow_html=True)

SUGGESTIONS = [
    ("📖", "Explain what a Data Analyst apprentice needs to know, in plain English."),
    ("🤖", "What skills does a Machine Learning Engineer need to develop?"),
    ("📁", "How could I evidence a data analysis skill in my portfolio?"),
    ("🔄", "What is the difference between knowledge, skills and behaviours?"),
    ("🏢", "Give me workplace examples for a Digital Support Technician skill."),
    ("💡", "What reflection questions can help me prepare for a progress review?"),
    ("🎯", "How long is a DevOps Engineer apprenticeship and what does it cover?"),
    ("🔐", "What knowledge does a Cyber Security Technician need?"),
]

col_a, col_b = st.columns(2)
for i, (icon, question) in enumerate(SUGGESTIONS):
    col = col_a if i % 2 == 0 else col_b
    if col.button(f"{icon}  {question}", key=f"sug_{i}", use_container_width=True):
        st.session_state["pending_question"] = question

st.markdown("---")

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            used_fallback = msg.get("used_fallback", False)
            sources       = msg.get("sources", [])
            retrieval_info = msg.get("retrieval_info", [])
            msg_mode      = msg.get("mode", "")

            if msg_mode:
                st.markdown(f'<div class="mode-badge">✦ {msg_mode}</div>', unsafe_allow_html=True)

            if used_fallback:
                st.markdown(
                    '<div class="gk-banner">⚠️ <strong>General knowledge response</strong> — '
                    'not drawn from the scraped Skills England pages. Always verify '
                    'important details at <a href="https://skillsengland.education.gov.uk/" '
                    'target="_blank">skillsengland.education.gov.uk</a> and check with your '
                    'trainer, employer or assessor.</div>',
                    unsafe_allow_html=True,
                )

            if sources:
                st.markdown('<p class="lbl" style="margin-top:0.9rem;">Sources</p>', unsafe_allow_html=True)
                for src in sources:
                    title = src["title"].replace(" / Skills England", "").strip()
                    ksb_ref = str(src.get("ksb_reference", ""))
                    ksb_tag = f'<span class="ksb-tag">{ksb_ref}</span>' if ksb_ref and ksb_ref != "nan" else ""
                    st.markdown(
                        f'<div class="src-card">📄 <a href="{src["url"]}" target="_blank">'
                        f'{title}</a>{ksb_tag}</div>',
                        unsafe_allow_html=True,
                    )

            if retrieval_info:
                with st.expander("🔍 How this answer was found"):
                    for info in retrieval_info:
                        badge = "🟢 Q&A match" if info["method"] == "qa_match" else "🔵 Vector search"
                        title = info["title"].replace(" / Skills England", "").strip()
                        ksb_note = f' · {info["ksb_reference"]}' if info.get("ksb_reference") else ""
                        st.caption(f"{badge} · {info['confidence']:.0%} confidence · {title}{ksb_note}")


# ── QUERY HANDLING ────────────────────────────────────────────────────────────
pending = st.session_state.get("pending_question")
if pending:
    del st.session_state["pending_question"]

user_input = st.chat_input("Ask me about your KSBs, standards, evidence ideas...")
query = pending or user_input

if query:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Add `OPENAI_API_KEY=your_key` to your `.env` file.")
        st.stop()
    if not vector_ready:
        st.error("Vector store not ready. Run `python -m src.vector_store` first.")
        st.stop()

    current_mode = st.session_state.get("learning_mode", LEARNING_MODES[0])

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            try:
                result = chat(query, history_for_llm, mode=current_mode)
                answer          = result["answer"]
                sources         = result["sources"]
                retrieval_results = result["retrieval_results"]
                used_fallback   = result.get("used_fallback", False)
            except Exception as exc:
                answer = f"Sorry, I encountered an error: `{exc}`"
                sources, retrieval_results, used_fallback = [], [], False

        st.markdown(f'<div class="mode-badge">✦ {current_mode}</div>', unsafe_allow_html=True)

        if used_fallback:
            st.markdown(
                '<div class="gk-banner">⚠️ <strong>General knowledge response</strong> — '
                'not drawn from the scraped Skills England pages. Always verify '
                'important details at <a href="https://skillsengland.education.gov.uk/" '
                'target="_blank">skillsengland.education.gov.uk</a> and check with your '
                'trainer, employer or assessor.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(answer)

        if sources:
            st.markdown('<p class="lbl" style="margin-top:0.9rem;">Sources</p>', unsafe_allow_html=True)
            for src in sources:
                title = src["title"].replace(" / Skills England", "").strip()
                ksb_ref = str(src.get("ksb_reference", ""))
                ksb_tag = f'<span class="ksb-tag">{ksb_ref}</span>' if ksb_ref and ksb_ref != "nan" else ""
                st.markdown(
                    f'<div class="src-card">📄 <a href="{src["url"]}" target="_blank">'
                    f'{title}</a>{ksb_tag}</div>',
                    unsafe_allow_html=True,
                )

        retrieval_info = [
            {
                "method": r.method,
                "confidence": r.confidence,
                "title": r.source_title,
                "ksb_reference": r.ksb_reference,
            }
            for r in retrieval_results
        ]
        if retrieval_info:
            with st.expander("🔍 How this answer was found"):
                for info in retrieval_info:
                    badge = "🟢 Q&A match" if info["method"] == "qa_match" else "🔵 Vector search"
                    title = info["title"].replace(" / Skills England", "").strip()
                    ksb_note = f' · {info["ksb_reference"]}' if info.get("ksb_reference") else ""
                    st.caption(f"{badge} · {info['confidence']:.0%} confidence · {title}{ksb_note}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_info": retrieval_info,
        "used_fallback": used_fallback,
        "mode": current_mode,
    })
