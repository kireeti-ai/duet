"""
app.py
Streamlit app for biomedical evidence synthesis.
"""

from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from api import run_claim


RESULT_KEY = "latest_result"
ERROR_KEY = "latest_error"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ant-slate-dark: #141413;
            --ant-slate-medium: #3d3d3a;
            --ant-slate-light: #5e5d59;
            --ant-night: #1b1b1a;
            --ant-ivory-light: #faf9f5;
            --ant-ivory-medium: #f0eee6;
            --ant-cloud-light: #d1cfc5;
            --ant-cloud-medium: #b0aea5;
            --ant-accent: #c6613f;
            --ant-white: #ffffff;
        }
        .stApp {
            background: var(--ant-night);
            color: var(--ant-ivory-light);
        }
        .block-container {
            max-width: 880px;
            padding-top: 3rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 {
            color: var(--ant-ivory-light);
            letter-spacing: -0.01em;
            font-weight: 700;
        }
        h1 {
            font-weight: 800;
        }
        .header-card {
            background: linear-gradient(180deg, #232321 0%, #191918 100%);
            border: 1px solid #373632;
            border-left: 4px solid var(--ant-accent);
            border-radius: 14px;
            padding: 0.95rem 1.1rem 0.9rem 1.1rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.18);
        }
        .header-card h1 {
            color: var(--ant-ivory-light) !important;
        }
        .header-title {
            margin: 0;
            color: var(--ant-ivory-light);
            font-size: 2.15rem;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: -0.01em;
            text-shadow: 0 1px 0 rgba(0, 0, 0, 0.15);
        }
        .header-card .app-subtitle {
            color: #c9c5bb;
            margin: 0.45rem 0 0 0;
            font-weight: 700;
        }
        .dataset-note {
            background: #21211f;
            border: 1px solid #3b3a36;
            border-left: 4px solid #788c5d;
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 1rem;
            color: #ece9df;
        }
        .dataset-note p {
            margin: 0.2rem 0;
            line-height: 1.45;
        }
        .dataset-note b {
            color: #faf9f5;
            font-weight: 800;
        }
        .dataset-note .dataset-note-muted {
            color: #cbc7b9;
        }
        div[data-testid="stForm"] {
            background: var(--ant-white);
            border: 1px solid var(--ant-cloud-light);
            border-radius: 14px;
            padding: 1rem 1rem 0.5rem 1rem;
            box-shadow: 0 1px 2px rgba(20, 20, 19, 0.03);
            margin-bottom: 1.2rem;
            color: var(--ant-slate-dark);
        }
        div[data-testid="stForm"] label {
            color: var(--ant-slate-dark);
        }
        div[data-baseweb="textarea"] > div {
            background: linear-gradient(180deg, #f9ede2 0%, #f4eadf 34%, #f0eee6 100%);
            border: 1px solid #dfc8b4;
            border-left: 4px solid var(--ant-accent);
            border-radius: 12px !important;
            box-shadow: none !important;
        }
        div[data-baseweb="textarea"] > div:focus-within {
            border-color: var(--ant-accent) !important;
            box-shadow: 0 0 0 1px var(--ant-accent) !important;
        }
        .stTextArea textarea {
            background: transparent !important;
            border: none !important;
            color: var(--ant-slate-dark);
            line-height: 1.55;
            padding: 0.8rem 0.8rem 0.8rem 0.9rem;
            font-weight: 500;
        }
        .stTextArea textarea::placeholder {
            color: #7a7268;
            opacity: 1;
        }
        .stFormSubmitButton > button {
            background: var(--ant-slate-dark);
            color: var(--ant-ivory-light);
            border: 1px solid var(--ant-slate-dark);
            border-radius: 999px;
            font-weight: 500;
            padding: 0.55rem 1.1rem;
        }
        .stFormSubmitButton > button:hover {
            background: var(--ant-accent);
            border-color: var(--ant-accent);
            color: var(--ant-white);
        }
        .stFormSubmitButton > button:focus {
            box-shadow: 0 0 0 2px rgba(198, 97, 63, 0.25);
        }
        .result-card {
            background: linear-gradient(180deg, #f9ede2 0%, #f4eadf 34%, #f0eee6 100%);
            border: 1px solid #dfc8b4;
            border-left: 4px solid var(--ant-accent);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-top: 0.25rem;
            margin-bottom: 0.75rem;
            color: var(--ant-slate-dark);
        }
        .result-card p, .result-card li {
            line-height: 1.55;
            color: var(--ant-slate-medium);
        }
        .verdict-line {
            margin: 0.18rem 0;
            color: var(--ant-slate-medium);
        }
        .verdict-label {
            font-weight: 800;
            color: var(--ant-slate-dark);
            letter-spacing: 0.01em;
        }
        .verdict-spacer {
            height: 0.45rem;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--ant-cloud-light);
            border-radius: 12px;
            background: var(--ant-white);
            overflow: hidden;
            margin-bottom: 0.6rem;
            color: var(--ant-slate-dark);
        }
        [data-testid="stExpander"] summary {
            background: var(--ant-ivory-medium);
        }
        [data-testid="stCaptionContainer"] {
            color: var(--ant-cloud-medium);
        }
        [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
            color: var(--ant-slate-light);
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_verdict_html(verdict: str) -> str:
    lines = verdict.splitlines()
    rendered: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            rendered.append("<div class='verdict-spacer'></div>")
            continue

        match = re.match(r"^([A-Z][A-Z\s/-]+:)(.*)$", stripped)
        if match:
            label = html.escape(match.group(1))
            content = html.escape(match.group(2).strip())
            if content:
                rendered.append(
                    f"<div class='verdict-line'><span class='verdict-label'>{label}</span> {content}</div>"
                )
            else:
                rendered.append(
                    f"<div class='verdict-line'><span class='verdict-label'>{label}</span></div>"
                )
        else:
            rendered.append(f"<div class='verdict-line'>{html.escape(stripped)}</div>")

    return "".join(rendered)


def _render_retrieved_docs(retrieved: list[dict[str, Any]]) -> None:
    st.subheader("Retrieved evidence")
    if not retrieved:
        st.info("No abstracts were returned.")
        return

    for i, doc in enumerate(retrieved, start=1):
        title = doc.get("title", "Untitled")
        pmid = doc.get("id", "N/A")
        score = doc.get("score")
        rerank_score = doc.get("rerank_score")
        with st.expander(f"{i}. PMID {pmid} — {title}", expanded=False):
            details: list[str] = []
            if isinstance(score, (int, float)):
                details.append(f"retrieval={score:.4f}")
            if isinstance(rerank_score, (int, float)):
                details.append(f"rerank={rerank_score:.4f}")
            if details:
                st.caption(" | ".join(details))
            st.write(doc.get("text", ""))


def main() -> None:
    st.set_page_config(page_title="Equipoise", page_icon="E", layout="centered")
    _inject_styles()

    st.markdown(
        """
        <div class='header-card'>
            <h1 class='header-title'>Equipoise</h1>
            <p class='app-subtitle'>Biomedical claim verification with balanced evidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='dataset-note'>
            <p><b>Data source:</b> AllenAI SciFact</p>
            <p><b>Freshness:</b> Static benchmark snapshot (not live-updating)</p>
            <p><b>Upstream last update:</b> 2021-01-26 02:28:59 UTC</p>
            <p class='dataset-note-muted'>This app uses a fixed SciFact snapshot until a newer dataset is manually downloaded.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if RESULT_KEY not in st.session_state:
        st.session_state[RESULT_KEY] = None
    if ERROR_KEY not in st.session_state:
        st.session_state[ERROR_KEY] = None

    with st.form("claim_form", clear_on_submit=False):
        claim = st.text_area(
            "Enter a biomedical claim",
            placeholder="Example: Vitamin D supplementation improves depression symptoms.",
            height=120,
        )
        analyze_clicked = st.form_submit_button("Analyze claim", type="primary")

    if analyze_clicked:
        if not claim.strip():
            st.session_state[ERROR_KEY] = "Please enter a claim before running."
            st.session_state[RESULT_KEY] = None
        else:
            with st.spinner("Analyzing claim..."):
                try:
                    response = run_claim(claim)
                    st.session_state[RESULT_KEY] = response["output"]
                    st.session_state[ERROR_KEY] = None
                except Exception as error:
                    st.session_state[ERROR_KEY] = f"Pipeline failed: {error}"
                    st.session_state[RESULT_KEY] = None

    if st.session_state[ERROR_KEY]:
        st.error(st.session_state[ERROR_KEY])
        return

    output = st.session_state[RESULT_KEY]
    if not output:
        st.caption("Try a claim to generate verdict and supporting evidence.")
        return

    st.subheader("Verdict")
    verdict_html = _format_verdict_html(output.get("verdict", "No verdict generated."))
    st.markdown(
        f"<div class='result-card'>{verdict_html}</div>",
        unsafe_allow_html=True,
    )

    reformulated = output.get("reformulated_query")
    if reformulated:
        st.caption(f"Reformulated query: {reformulated}")

    _render_retrieved_docs(output.get("retrieved", []))


if __name__ == "__main__":
    main()
