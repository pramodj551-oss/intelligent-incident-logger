# -*- coding: utf-8 -*-
"""
frontend/app.py -- Streamlit SOC Dashboard
Supports Groq (free) and OpenAI providers with live switching.
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from typing import Optional

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="AP Securitas | AI Incident Logger",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Dark SOC theme -----------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #f0f6fc; }
    .stTextArea textarea { background: #161b22 !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; font-family: 'Courier New', monospace; }
    .stTextInput input { background: #161b22 !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #c0392b, #e74c3c); color: white;
        border: none; border-radius: 6px; padding: 0.6rem 2rem;
        font-weight: 700; font-size: 1rem; }
    .badge-high   { background:#c0392b; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; }
    .badge-medium { background:#d68910; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; }
    .badge-low    { background:#1e8449; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; }
    .badge-free   { background:#1f6feb; color:#fff; padding:3px 10px; border-radius:12px;
                    font-size:0.75rem; font-weight:700; }
    .alert-high   { background:#161b22; border-left:4px solid #e74c3c; padding:1rem; border-radius:6px; }
    .alert-medium { background:#161b22; border-left:4px solid #d68910; padding:1rem; border-radius:6px; }
    .alert-low    { background:#161b22; border-left:4px solid #1e8449; padding:1rem; border-radius:6px; }
    .info-card    { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem; margin:0.4rem 0; }
    .field-label  { color:#8b949e; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }
    .field-value  { color:#f0f6fc; font-size:0.95rem; }
    .sop-panel    { background:#0d1f12; border:1px solid #2ea043; border-radius:8px;
                    padding:1.2rem; font-size:0.85rem; font-family:'Courier New',monospace; }
    .cost-zero    { color:#2ea043; font-weight:700; font-size:1.1rem; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# -- Helpers ------------------------------------------------------------------

def badge(level: str) -> str:
    css = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(level, "badge-low")
    return f'<span class="{css}"> {level.upper()}</span>'

def alert_class(level: str) -> str:
    return {"High": "alert-high", "Medium": "alert-medium", "Low": "alert-low"}.get(level, "alert-low")

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


@st.cache_resource(show_spinner=False)
def get_agent(provider: str, api_key: str, embed_mode: str, embed_key: str = ""):
    """
    Cache key = (provider, api_key, embed_mode, embed_key).
    Re-initialises only when any of these change.
    """
    os.environ["LLM_PROVIDER"]   = provider
    os.environ["EMBEDDING_MODE"] = embed_mode

    from src.rag_pipeline import SOPRetriever
    from src.agent import IncidentAgent

    sop_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sop_manual.txt"
    )
    retriever = SOPRetriever(sop_path, openai_api_key=embed_key)
    return IncidentAgent(api_key=api_key, sop_retriever=retriever)


# -- Session state ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "result"  not in st.session_state:
    st.session_state.result  = None


# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## AP Securitas")
    st.markdown("**AI-Powered SOC Operations**")
    st.markdown("---")

    # -- LLM Provider selection -----------------------------------------------
    st.markdown("### LLM Provider")
    provider_choice = st.radio(
        "Choose provider",
        ["Groq (Free)", "OpenAI"],
        index=0,
        label_visibility="collapsed"
    )
    provider = "groq" if provider_choice == "Groq (Free)" else "openai"

    if provider == "groq":
        st.markdown('<span class="badge-free">FREE</span> &nbsp;Llama 3.3 70B via Groq Cloud', unsafe_allow_html=True)
        llm_key = st.text_input(
            "Groq API Key (gsk_...)",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Free at console.groq.com -- no credit card needed"
        )
        st.caption("Get key: console.groq.com -> API Keys")
    else:
        llm_key = st.text_input(
            "OpenAI API Key (sk-...)",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )

    st.markdown("---")

    # -- Embedding mode -------------------------------------------------------
    st.markdown("### Embeddings")
    embed_choice = st.radio(
        "Embedding mode",
        ["Local (Free)", "OpenAI API"],
        index=0,
        label_visibility="collapsed"
    )
    embed_mode = "local" if embed_choice == "Local (Free)" else "openai"

    embed_key = ""
    if embed_mode == "openai":
        embed_key = st.text_input(
            "OpenAI Key for embeddings",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        st.caption("text-embedding-3-small")
    else:
        st.caption("all-MiniLM-L6-v2 (runs locally, no API cost)")

    st.markdown("---")

    # -- Cost indicator -------------------------------------------------------
    st.markdown("### Running Cost")
    llm_cost   = "FREE" if provider == "groq" else "~$0.0002/call"
    embed_cost = "FREE" if embed_mode == "local" else "~$0.00002/run"
    both_free  = (provider == "groq" and embed_mode == "local")

    if both_free:
        st.markdown('<div class="cost-zero">$0.00 per query</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"LLM: `{llm_cost}`")
        st.markdown(f"Embed: `{embed_cost}`")

    st.markdown("---")

    # -- Status ---------------------------------------------------------------
    st.markdown("### Status")
    if llm_key:
        st.markdown(f"{'Groq' if provider == 'groq' else 'OpenAI'} key: configured")
    else:
        st.markdown("API key: missing")

    st.markdown(f"Embed: {embed_mode}")
    st.markdown(f"Reports today: {len(st.session_state.history)}")

    # -- History --------------------------------------------------------------
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent")
        for h in reversed(st.session_state.history[-5:]):
            lvl  = h.get("incident_report", {}).get("threat_level", "Low")
            icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(lvl, "")
            loc  = h.get("incident_report", {}).get("location", "")[:18]
            st.markdown(f"{icon} [{h.get('_ts','')}] {loc}")

        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result  = None
            st.rerun()


# -- Main area ----------------------------------------------------------------
st.markdown("# AP Securitas -- AI Incident Logger")
st.markdown("Natural language in -> Structured report + SOP guidance out.")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    guard_name = st.text_input("Guard Name / ID", placeholder="e.g., Guard Ramesh Kumar -- Post: Gate 2")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    prov_label = "Groq FREE" if provider == "groq" else "OpenAI"
    st.metric("Provider", prov_label)

incident_input = st.text_area(
    "Incident Report (any language)",
    placeholder=(
        "Describe in English, Marathi, Hindi, or mix.\n\n"
        "Example: Gate 2 par ek suspicious black backpack mili hai jo pichhle 20 minute se "
        "koi nahi le gaya. Bag ek pillar ke paas rakhi hai. Kya karna chahiye?"
    ),
    height=140
)

with st.expander("Try sample inputs"):
    samples = {
        "HIGH - Suspicious bag":  "Gate No. 2 ke paas ek abandoned black bag rakhi hai 25 minutes se. Bag se halki tikhi smell aa rahi hai.",
        "HIGH - Fire/Smoke":      "Block B server room mein smoke alarm baj raha hai. Halka dhuan dikh raha hai.",
        "MEDIUM - Unknown person": "Parking lot mein ek unknown male 45 minutes se ghoom raha hai near vehicles. Koi ID nahi hai.",
        "LOW - Lost wallet":       "Main lobby mein ek wallet mila. Visiting card tha andar -- naam: Rajesh Sharma."
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state["_sample"] = text

if "_sample" in st.session_state:
    incident_input = st.session_state.pop("_sample")

st.markdown("---")
submit = st.button("Analyse Incident", type="primary")

if submit:
    if not llm_key:
        key_source = "Groq (console.groq.com)" if provider == "groq" else "OpenAI"
        st.error(f"Please enter your {key_source} API key in the sidebar.")
    elif not incident_input.strip():
        st.warning("Please enter an incident report.")
    else:
        with st.spinner(f"Agent processing via {provider.upper()}..."):
            try:
                agent  = get_agent(provider, llm_key, embed_mode, embed_key)
                result = agent.process(incident_input, guard_name or "Unknown Guard")
                result["_ts"] = ts()
                st.session_state.result  = result
                st.session_state.history.append(result)
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.session_state.result = None


# -- Results ------------------------------------------------------------------
result = st.session_state.result
if result:
    report       = result.get("incident_report", {})
    sop_action   = result.get("sop_action")
    alert_msg    = result.get("alert_message", "")
    protocol_num = result.get("protocol_number")
    threat_level = report.get("threat_level", "Low")
    confidence   = report.get("confidence_score", 0.0)
    model_used   = result.get("llm_model", "unknown")

    st.markdown("## Analysis Results")

    # Header row
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        st.markdown("**Threat Level**")
        st.markdown(badge(threat_level), unsafe_allow_html=True)
    with c2:
        st.markdown("**Protocol**")
        st.markdown(f"`{protocol_num}`" if protocol_num else "--")
    with c3:
        st.markdown("**Immediate Action**")
        st.markdown("YES -- Escalate" if result.get("requires_immediate_action") else "Standard")
    with c4:
        st.markdown("**Confidence / Model**")
        st.markdown(f"`{confidence:.0%}` via `{model_used}`")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Incident Report", "SOP Action Steps", "{ } Raw JSON"])

    with tab1:
        st.markdown(
            f'<div class="{alert_class(threat_level)}">'
            f'<b>AI Directive:</b><br><br>{alert_msg}</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        fields = [
            ("Location",        report.get("location",        "--")),
            ("Object/Person",   report.get("object_involved", "--")),
            ("Summary",         report.get("summary",         "--")),
            ("Reported By",     report.get("reported_by",     "--")),
            ("Incident Time",   report.get("incident_time",   "Not specified")),
            ("Action Taken",    report.get("action_taken",    "None taken yet")),
        ]
        ca, cb = st.columns(2)
        for i, (lbl, val) in enumerate(fields):
            with (ca if i % 2 == 0 else cb):
                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="field-label">{lbl}</div>'
                    f'<div class="field-value">{val}</div></div>',
                    unsafe_allow_html=True
                )

    with tab2:
        if sop_action:
            st.markdown(f"**{protocol_num or 'SOP Match'}**")
            st.markdown(
                f'<div class="sop-panel">{sop_action.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info(
                f"SOP retrieval is triggered for HIGH and MEDIUM threats.\n"
                f"This incident was classified as **{threat_level}**."
            )

    with tab3:
        clean    = {k: v for k, v in result.items() if not k.startswith("_")}
        json_str = json.dumps(clean, indent=2, ensure_ascii=False)
        st.code(json_str, language="json")
        st.download_button(
            "Download JSON",
            data=json_str,
            file_name=f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:0.78rem'>"
    "AP Securitas Pvt. Ltd. | AI-Augmented SOC | Godrej IT Park, Thane | "
    "Built by Pramod Jadhav"
    "</div>",
    unsafe_allow_html=True
)
