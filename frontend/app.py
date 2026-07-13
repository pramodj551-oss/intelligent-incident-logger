# -*- coding: utf-8 -*-
"""
frontend/app.py -- Streamlit SOC Dashboard
Provider switcher: Groq (free) | OpenAI | Hugging Face
Embedding switcher: Local (free) | OpenAI
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from typing import Optional

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="AP Securitas | AI Incident Logger",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background:#0d1117; color:#c9d1d9; }
    section[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #30363d; }
    h1,h2,h3,h4 { color:#f0f6fc; }
    .stTextArea textarea { background:#161b22 !important; color:#c9d1d9 !important;
        border:1px solid #30363d !important; font-family:'Courier New',monospace; }
    .stTextInput input { background:#161b22 !important; color:#c9d1d9 !important;
        border:1px solid #30363d !important; }
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#c0392b,#e74c3c); color:white;
        border:none; border-radius:6px; padding:0.6rem 2rem; font-weight:700; }
    .badge-high   { background:#c0392b; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; }
    .badge-medium { background:#d68910; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; }
    .badge-low    { background:#1e8449; color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; }
    .tag-free     { background:#1f6feb; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.72rem; font-weight:700; }
    .tag-paid     { background:#6e7681; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.72rem; }
    .alert-high   { background:#161b22; border-left:4px solid #e74c3c; padding:1rem; border-radius:6px; margin:0.8rem 0; }
    .alert-medium { background:#161b22; border-left:4px solid #d68910; padding:1rem; border-radius:6px; margin:0.8rem 0; }
    .alert-low    { background:#161b22; border-left:4px solid #1e8449; padding:1rem; border-radius:6px; margin:0.8rem 0; }
    .info-card    { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem; margin:0.4rem 0; }
    .field-label  { color:#8b949e; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }
    .field-value  { color:#f0f6fc; font-size:0.95rem; }
    .sop-panel    { background:#0d1f12; border:1px solid #2ea043; border-radius:8px;
                    padding:1.2rem; font-size:0.85rem; font-family:'Courier New',monospace; }
    .cost-row     { display:flex; gap:8px; align-items:center; margin:2px 0; font-size:0.85rem; }
    hr { border-color:#30363d; }
</style>
""", unsafe_allow_html=True)


# -- Provider metadata --------------------------------------------------------

PROVIDER_META = {
    "groq": {
        "label":       "Groq",
        "tag":         "FREE",
        "tag_class":   "tag-free",
        "model":       "llama-3.3-70b-versatile",
        "env_key":     "GROQ_API_KEY",
        "key_prefix":  "gsk_",
        "help":        "Free key at console.groq.com",
        "note":        "~500 tok/s | Llama 3.3 70B",
    },
    "openai": {
        "label":       "OpenAI",
        "tag":         "PAID",
        "tag_class":   "tag-paid",
        "model":       "gpt-4o-mini",
        "env_key":     "OPENAI_API_KEY",
        "key_prefix":  "sk-",
        "help":        "platform.openai.com/api-keys",
        "note":        "GPT-4o-mini | best accuracy",
    },
    "huggingface": {
        "label":       "Hugging Face",
        "tag":         "FREE",
        "tag_class":   "tag-free",
        "model":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "env_key":     "HUGGINGFACE_API_KEY",
        "key_prefix":  "hf_",
        "help":        "huggingface.co/settings/tokens",
        "note":        "Llama 3.1 8B | GitHub Student Pack",
    },
}


# -- Helpers ------------------------------------------------------------------

def badge(level: str) -> str:
    c = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(level, "badge-low")
    return f'<span class="{c}"> {level.upper()}</span>'

def alert_div(level: str, msg: str) -> str:
    c = {"High": "alert-high", "Medium": "alert-medium", "Low": "alert-low"}.get(level, "alert-low")
    return f'<div class="{c}"><b>AI Directive:</b><br><br>{msg}</div>'

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


@st.cache_resource(show_spinner=False)
def get_agent(provider_name: str, api_key: str, embed_mode: str, embed_key: str = ""):
    """
    Cache key = (provider_name, api_key, embed_mode, embed_key).
    sop_file_path intentionally omitted -- SOPRetriever auto-resolves it
    via _resolve_sop_path() anchored to rag_pipeline.py, reliable on
    local dev, Streamlit Cloud, and Docker.
    """
    from src.providers import get_provider
    from src.rag_pipeline import SOPRetriever
    from src.agent import IncidentAgent

    provider = get_provider(provider_name, api_key)
    os.environ["EMBEDDING_MODE"] = embed_mode
    retriever = SOPRetriever(openai_api_key=embed_key)   # path auto-resolved
    return IncidentAgent(sop_retriever=retriever, provider=provider)


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

    # -- LLM Provider ---------------------------------------------------------
    st.markdown("### LLM Provider")

    provider_options = list(PROVIDER_META.keys())
    provider_labels  = [
        f"{PROVIDER_META[p]['label']}  [{PROVIDER_META[p]['tag']}]"
        for p in provider_options
    ]
    selected_idx     = st.radio(
        "provider_radio",
        range(len(provider_options)),
        format_func=lambda i: provider_labels[i],
        index=0,
        label_visibility="collapsed"
    )
    provider_name = provider_options[selected_idx]
    meta          = PROVIDER_META[provider_name]

    st.caption(meta["note"])

    llm_api_key = st.text_input(
        f"{meta['label']} API Key  ({meta['key_prefix']}...)",
        type="password",
        value=os.getenv(meta["env_key"], ""),
        help=meta["help"]
    )

    st.markdown("---")

    # -- Embedding mode -------------------------------------------------------
    st.markdown("### Embeddings")

    embed_options = ["Local (Free)", "OpenAI API"]
    embed_choice  = st.radio(
        "embed_radio",
        embed_options,
        index=0,
        label_visibility="collapsed"
    )
    embed_mode = "local" if embed_choice == "Local (Free)" else "openai"

    embed_key = ""
    if embed_mode == "local":
        st.caption("all-MiniLM-L6-v2  |  runs on CPU  |  $0")
    else:
        st.caption("text-embedding-3-small  |  high quality")
        embed_key = st.text_input(
            "OpenAI Key for embeddings",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )

    st.markdown("---")

    # -- Live cost display ----------------------------------------------------
    st.markdown("### Cost per query")
    llm_free  = meta["tag"] == "FREE"
    emb_free  = (embed_mode == "local")
    all_free  = llm_free and emb_free

    llm_cost  = "$0.00  FREE" if llm_free else "~$0.0002"
    emb_cost  = "$0.00  FREE" if emb_free else "~$0.00002"

    st.markdown(f"LLM:        `{llm_cost}`")
    st.markdown(f"Embeddings: `{emb_cost}`")
    if all_free:
        st.success("Zero-cost mode active")

    st.markdown("---")

    # -- Status ---------------------------------------------------------------
    st.markdown("### Status")
    key_ok = bool(llm_api_key)
    st.markdown(f"{'Key' if key_ok else 'Key MISSING'}: {'set' if key_ok else 'not set'}")
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
st.markdown("Natural language guard report -> Structured data + SOP action steps.")
st.markdown("---")

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    guard_name = st.text_input("Guard Name / ID", placeholder="e.g., Guard Ramesh Kumar -- Post: Gate 2")
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Provider", meta["label"])
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Model", meta["model"].split("/")[-1][:16])

incident_input = st.text_area(
    "Incident Report  (English / Marathi / Hindi / mix)",
    placeholder=(
        "Example: Gate 2 par ek suspicious black backpack mili hai jo pichhle 20 minute se\n"
        "koi nahi le gaya. Bag ek pillar ke paas rakhi hai. Maine kisi ko paas nahi jaane\n"
        "diya. Kya karna chahiye?"
    ),
    height=140
)

with st.expander("Try sample inputs"):
    samples = {
        "HIGH -- Suspicious bag":      "Gate No. 2 ke paas ek abandoned black bag rakhi hai 25 minutes se. Bag se tikhi smell aa rahi hai.",
        "HIGH -- Fire in server room":  "Block B server room mein smoke alarm baj raha hai. Halka dhuan dikh raha hai.",
        "MEDIUM -- Unknown person":     "Parking lot mein ek unknown male 45 minutes se ghoom raha hai near vehicles. Koi ID nahi hai.",
        "LOW -- Lost wallet":           "Main lobby mein ek wallet mila. Visiting card tha andar -- Rajesh Sharma, ABC Tech.",
    }
    for lbl, txt in samples.items():
        if st.button(lbl, use_container_width=True):
            st.session_state["_sample"] = txt

if "_sample" in st.session_state:
    incident_input = st.session_state.pop("_sample")

st.markdown("---")
submit = st.button("Analyse Incident", type="primary")

if submit:
    if not llm_api_key:
        st.error(f"Please enter your {meta['label']} API key. ({meta['help']})")
    elif not incident_input.strip():
        st.warning("Please enter an incident report.")
    else:
        with st.spinner(f"Processing via {meta['label']} ({meta['model'].split('/')[-1]})..."):
            try:
                agent  = get_agent(provider_name, llm_api_key, embed_mode, embed_key)
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
    prov_used    = result.get("llm_provider", "unknown")

    st.markdown("## Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Threat Level**")
        st.markdown(badge(threat_level), unsafe_allow_html=True)
    with col2:
        st.markdown("**Protocol**")
        st.markdown(f"`{protocol_num}`" if protocol_num else "--")
    with col3:
        st.markdown("**Immediate Action**")
        st.markdown("YES" if result.get("requires_immediate_action") else "Standard")
    with col4:
        st.markdown("**Confidence / Provider**")
        st.markdown(f"`{confidence:.0%}` via `{prov_used}`")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Incident Report", "SOP Action Steps", "{ } Raw JSON"])

    with tab1:
        st.markdown(alert_div(threat_level, alert_msg), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fields = [
            ("Location",      report.get("location",        "--")),
            ("Object/Person", report.get("object_involved", "--")),
            ("Summary",       report.get("summary",         "--")),
            ("Reported By",   report.get("reported_by",     "--")),
            ("Incident Time", report.get("incident_time",   "Not specified")),
            ("Action Taken",  report.get("action_taken",    "None yet")),
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
            st.info(f"SOP retrieval runs for HIGH and MEDIUM threats. This was **{threat_level}**.")

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
    "AP Securitas Pvt. Ltd.  |  AI-Augmented SOC  |  Godrej IT Park, Thane  |  "
    "Built by Pramod Jadhav</div>",
    unsafe_allow_html=True
    )
