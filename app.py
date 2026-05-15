import streamlit as st
import time
import re
import html
from src.pipeline import run_pipeline

# --- Page Config ---
st.set_page_config(
    page_title="Biomedical Claim Verification",
    page_icon="🔬",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }

    /* FIX 1: single-line input */
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: white;
        border-radius: 12px;
        border: 1px solid #334155;
        padding: 12px 16px;
        font-size: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }

    .verdict-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
    }

    .abstract-card {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }

    .pmid-tag {
        background: #3b82f6;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }

    h1, h2, h3 { color: #ffffff !important; }

    .verdict-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    .badge-supported    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .badge-contradicted { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .badge-mixed        { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .badge-unknown      { background: rgba(148,163,184,0.15);color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

    .citation-badge {
        background: #3b82f6;
        color: white;
        padding: 1px 6px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 0 2px;
        vertical-align: middle;
    }

    /* FIX 2: separate styles for no-contra notice vs actual contra evidence */
    .no-contra-box {
        background: rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px 16px;
        margin-top: 16px;
        font-size: 0.88rem;
        color: #64748b;
        display: flex;
        gap: 10px;
        align-items: flex-start;
    }

    .contra-box {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 16px;
        font-size: 0.95rem;
        color: #fca5a5;
        line-height: 1.6;
    }

    .conf-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 4px;
    }
    .conf-high { background: #4ade80; }
    .conf-med  { background: #fbbf24; }
    .conf-low  { background: #ef4444; }
    .conf-none { background: #475569; }

    .parse-error {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 20px;
        color: #fca5a5;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🔬  RAG")
st.markdown("### Biomedical Claim Verification Engine")

# FIX 1: single-line text_input, not text_area
claim = st.text_input(
    "Enter a biomedical claim to verify:",
    placeholder="e.g., Omega-3 supplementation reduces symptoms of depression",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    verify_clicked = st.button("Verify Claim")

if verify_clicked and claim:
    with st.spinner("Analysing claim against SciFact corpus..."):
        try:
            results = run_pipeline(claim.strip(), method="hybrid")
            v_text = results["verdict"]

            # --- Parse structured sections ---
            verdict_match = re.search(r"VERDICT:\s*([^\n]*)", v_text, re.IGNORECASE)
            verdict_val = verdict_match.group(1).strip() if verdict_match else ""

            summary_match = re.search(
                r"SUMMARY:\s*(.*?)(?=SUPPORTING EVIDENCE:|CONTRADICTING EVIDENCE:|CONFIDENCE:|$)",
                v_text, re.IGNORECASE | re.DOTALL,
            )
            summary_val = summary_match.group(1).strip() if summary_match else ""

            supp_match = re.search(
                r"SUPPORTING EVIDENCE:\s*(.*?)(?=CONTRADICTING EVIDENCE:|CONFIDENCE:|$)",
                v_text, re.IGNORECASE | re.DOTALL,
            )
            supp_val = supp_match.group(1).strip() if supp_match else ""

            contra_match = re.search(
                r"CONTRADICTING EVIDENCE:\s*(.*?)(?=CONFIDENCE:|$)",
                v_text, re.IGNORECASE | re.DOTALL,
            )
            contra_val = contra_match.group(1).strip() if contra_match else ""

            confidence_match = re.search(r"CONFIDENCE:\s*(.*)", v_text, re.IGNORECASE | re.DOTALL)
            confidence_val = confidence_match.group(1).strip() if confidence_match else ""

            # FIX 3: detect parse failure — if no structured sections were found at all,
            # the LLM did not follow the format. Show a clear error instead of
            # dumping raw text into the UI.
            parse_ok = bool(verdict_val or summary_val or supp_val)
            if not parse_ok:
                import textwrap
                st.markdown(textwrap.dedent("""
                <div class="parse-error">
                    <strong>⚠ Response format error</strong><br>
                    The model did not return a structured response. Please try again.
                    If this keeps happening, check that the structured prompt is active in config.
                </div>
                """), unsafe_allow_html=True)
                st.stop()

            # --- Inline citation badges ---
            def make_citation_badges(text: str) -> str:
                """Replace [1], [2,3] etc. with coloured badge spans."""
                def replacer(m):
                    nums = re.findall(r"\d+", m.group(1))
                    return " ".join(f'<span class="citation-badge">{n}</span>' for n in nums)
                escaped = html.escape(text)
                return re.sub(r"\[([0-9\s,]+)\]", replacer, escaped).replace("\n", "<br>")

            summary_html   = make_citation_badges(summary_val)
            supp_html      = make_citation_badges(supp_val)

            # --- Collect cited indices so we know which sources were actually used ---
            full_text = f"{summary_val} {supp_val} {contra_val}"
            used_indices: set[int] = set()
            for m in re.finditer(r"\[([0-9\s,]+)\]", full_text):
                for n in re.findall(r"\d+", m.group(1)):
                    used_indices.add(int(n))

            # FIX 4: if the LLM cited nothing with [N] notation, treat all retrieved
            # sources as used rather than marking them all excluded.
            if not used_indices:
                used_indices = set(range(1, len(results["retrieved"]) + 1))

            # --- Verdict badge ---
            v_lower = verdict_val.lower()
            if "support" in v_lower:
                badge_class, icon = "badge-supported", "✓"
            elif "contradict" in v_lower:
                badge_class, icon = "badge-contradicted", "✕"
            elif any(w in v_lower for w in ("mix", "contest", "insufficient")):
                badge_class, icon = "badge-mixed", "⚖"
            else:
                badge_class, icon = "badge-unknown", "?"

            # --- Confidence dots ---
            c_lower = confidence_val.lower()
            if "high" in c_lower:
                dots = '<span class="conf-dot conf-high"></span>' * 4
            elif "medium" in c_lower or "moderate" in c_lower:
                dots = ('<span class="conf-dot conf-med"></span>' * 2 +
                        '<span class="conf-dot conf-none"></span>' * 2)
            elif "low" in c_lower:
                dots = ('<span class="conf-dot conf-low"></span>' +
                        '<span class="conf-dot conf-none"></span>' * 3)
            else:
                dots = '<span class="conf-dot conf-none"></span>' * 4

            # --- Supporting evidence block (only if non-empty and not "none") ---
            import textwrap
            supp_block = ""
            if supp_val and "none" not in supp_val.lower():
                supp_block = textwrap.dedent(f"""
                <div style="font-size:0.95rem;line-height:1.6;color:#cbd5e1;margin-top:16px;">
                    <strong>Supporting evidence:</strong> {supp_html}
                </div>
                """)

            # FIX 5: contradicting evidence has its own distinct block.
            # "None found" → quiet grey notice.  Actual text → red-tinted box.
            contra_block = ""
            if not contra_val or "none" in contra_val.lower():
                contra_block = textwrap.dedent("""
                <div class="no-contra-box">
                    ℹ No contradicting evidence found among the retrieved abstracts.
                </div>
                """)
            else:
                contra_html = make_citation_badges(contra_val)
                contra_block = textwrap.dedent(f"""
                <div class="contra-box">
                    <strong>Contradicting evidence:</strong> {contra_html}
                </div>
                """)

            # --- Render verdict card ---
            verdict_card_html = f"""
<div class="verdict-card">
<div class="verdict-badge {badge_class}">
<span style="margin-right:6px;">{icon}</span>
{html.escape(verdict_val)}
</div>
<div style="font-size:0.8rem;letter-spacing:1px;color:#94a3b8;margin-top:12px;margin-bottom:4px;text-transform:uppercase;">Claim</div>
<div style="font-size:1.2rem;font-weight:600;margin-bottom:20px;color:white;">"{html.escape(claim.strip())}"</div>
<div style="font-size:1.05rem;line-height:1.6;color:#f1f5f9;">{summary_html}</div>
{supp_block}
{contra_block}
<hr style="opacity:0.1;margin:20px 0;">
<div style="display:flex;align-items:center;color:#94a3b8;font-size:0.9rem;">
<span style="margin-right:10px;">Confidence</span>
<div style="display:flex;align-items:center;margin-right:10px;">{dots}</div>
<span style="color:#cbd5e1;">{html.escape(confidence_val)}</span>
</div>
</div>
"""
            # Ensure absolutely no leading spaces remain to prevent Markdown code block bugs
            verdict_card_html = re.sub(r'^[ \t]+', '', verdict_card_html, flags=re.MULTILINE)
            st.markdown(verdict_card_html, unsafe_allow_html=True)

            # --- Evidence sources ---
            st.markdown(
                f"<div style='font-size:0.75rem;font-weight:600;color:#94a3b8;"
                f"margin-bottom:12px;letter-spacing:1px;'>"
                f"SOURCES RETRIEVED ({len(results['retrieved'])})</div>",
                unsafe_allow_html=True,
            )

            for i, res in enumerate(results["retrieved"]):
                idx = i + 1
                title = res.get("title", f"Source {idx}")
                pmid  = res.get("id", "—")
                body  = res.get("text", "")

                if idx in used_indices:
                    # FIX 6: show the actual paper title, not "Evidence #N"
                    with st.expander(f"[{idx}]  {title[:90]}{'...' if len(title)>90 else ''}", expanded=False):
                        st.markdown(f"""
                        <div class="abstract-card">
                            <p>
                                <span class="pmid-tag">PMID {pmid}</span>
                                <span style="margin-left:10px;font-weight:600;">{html.escape(title)}</span>
                            </p>
                            <p style="font-size:0.9rem;line-height:1.5;opacity:0.9;">
                                {html.escape(body)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding:10px 16px;border-radius:8px;
                                background:rgba(0,0,0,0.15);
                                border:1px solid rgba(255,255,255,0.04);
                                margin-bottom:8px;display:flex;
                                align-items:center;color:#475569;">
                        <span style="margin-right:10px;opacity:0.4;">○</span>
                        <span style="font-size:0.85rem;">
                            [{idx}] {html.escape(title[:80])} — not cited in verdict
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Pipeline error: {e}")

elif verify_clicked and not claim:
    st.warning("Please enter a claim to verify.")

# FIX 7: removed "Equipoise" from footer
st.markdown("<hr style='opacity:0.1;margin-top:50px;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;opacity:0.4;font-size:0.85rem;'>"
    "Biomedical Claim Verification · SciFact Dataset</p>",
    unsafe_allow_html=True,
)