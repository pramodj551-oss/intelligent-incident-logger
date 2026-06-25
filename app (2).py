"""
frontend/app.py — Streamlit SOC Dashboard UI
=============================================
Module 1 & 3: Frontend Deployment

Dark-themed Security Operations Dashboard for AP Securitas.
Guards enter incident reports in natural language (any language).
The AI agent returns structured data + SOP guidance instantly.

Run:
    cd incident-logger
    streamlit run frontend/app.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

import streamlit as st
import httpx

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AP Securitas | AI Incident Logger",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS — Dark SOC Theme ───────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    .stApp { background: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

    /* ── Typography ── */
    h1, h2, h3, h4 { color: #f0f6fc; }
    .stTextArea textarea { background: #161b22 !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; font-family: 'Courier New', monospace; }
    .stTextInput input { background: #161b22 !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white; border: none; border-radius: 6px;
        padding: 0.6rem 2rem; font-weight: 700; font-size: 1rem;
        letter-spacing: 0.04em; transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        transform: translateY(-1px); box-shadow: 0 4px 12px rgba(231,76,60,0.4);
    }

    /* ── Threat Badges ── */
    .badge-high   { background:#c0392b; color:#fff; padding:4px 14px; border-radius:20px;
                    font-weight:700; font-size:0.9rem; letter-spacing:0.08em; }
    .badge-medium { background:#d68910; color:#fff; padding:4px 14px; border-radius:20px;
                    font-weight:700; font-size:0.9rem; letter-spacing:0.08em; }
    .badge-low    { background:#1e8449; color:#fff; padding:4px 14px; border-radius:20px;
                    font-weight:700; font-size:0.9rem; letter-spacing:0.08em; }

    /* ── Alert Box ── */
    .alert-box { background:#161b22; border-left:4px solid #e74c3c;
                 padding:1rem 1.2rem; border-radius:6px; margin:0.8rem 0; }
    .alert-box-medium { background:#161b22; border-left:4px solid #d68910;
                        padding:1rem 1.2rem; border-radius:6px; margin:0.8rem 0; }
    .alert-box-low    { background:#161b22; border-left:4px solid #1e8449;
                        padding:1rem 1.2rem; border-radius:6px; margin:0.8rem 0; }

    /* ── Info Cards ── */
    .info-card { background:#161b22; border:1px solid #30363d; border-radius:8px;
                 padding:1rem; margin:0.5rem 0; }
    .field-label { color:#8b949e; font-size:0.8rem; font-weight:600;
                   text-transform:uppercase; letter-spacing:0.08em; margin-bottom:2px; }
    .field-value { color:#f0f6fc; font-size:1rem; }

    /* ── SOP Panel ── */
    .sop-panel { background:#0d1f12; border:1px solid #2ea043;
                 border-radius:8px; padding:1.2rem; font-size:0.88rem;
                 font-family:'Courier New', monospace; color:#c9d1d9; }
    .protocol-tag { background:#2ea043; color:#fff; padding:3px 12px;
                    border-radius:12px; font-size:0.8rem; font-weight:700; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background:#161b22; }
    .stTabs [data-baseweb="tab"]      { color:#8b949e; }
    .stTabs [aria-selected="true"]    { color:#f0f6fc !important; }

    /* ── Dividers ── */
    hr { border-color:#30363d; }

    /* ── History items ── */
    .history-item { background:#161b22; border:1px solid #30363d; border-radius:6px;
                    padding:0.6rem 1rem; margin:0.3rem 0; font-size:0.85rem;
                    cursor:pointer; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────────

def get_threat_badge(level: str) -> str:
    css = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(level, "badge-low")
    return f'<span class="{css}">⬤ {level.upper()}</span>'

def get_alert_class(level: str) -> str:
    return {"High": "alert-box", "Medium": "alert-box-medium", "Low": "alert-box-low"}.get(level, "alert-box-low")

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

@st.cache_resource(show_spinner=False)
def get_agent(openai_key: str):
    """Cache the agent so it initialises only once per session."""
    from src.agent import IncidentAgent
    from src.rag_pipeline import SOPRetriever
    sop_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sop_manual.txt")
    retriever = SOPRetriever(sop_path, openai_key)
    return IncidentAgent(openai_key, retriever)

def call_agent_direct(guard_input: str, guard_name: str, openai_key: str) -> dict:
    """Call the agent directly (no FastAPI hop)."""
    agent = get_agent(openai_key)
    return agent.process(guard_input, guard_name)

def call_api(guard_input: str, guard_name: str, api_url: str) -> dict:
    """Call the FastAPI backend."""
    response = httpx.post(
        f"{api_url}/report",
        json={"guard_input": guard_input, "guard_name": guard_name},
        timeout=60.0
    )
    response.raise_for_status()
    return response.json()["data"]


# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of past results
if "result"  not in st.session_state:
    st.session_state.result = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ AP Securitas")
    st.markdown("**AI-Powered SOC Operations**")
    st.markdown("---")

    # Credentials
    st.markdown("### ⚙️ Configuration")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Your OpenAI API key. Set OPENAI_API_KEY in .env to pre-fill."
    )

    mode = st.radio(
        "Mode",
        ["Direct (no server)", "FastAPI Backend"],
        index=0,
        help="Direct: agent runs inside Streamlit. FastAPI: calls backend server."
    )

    api_url = "http://localhost:8000"
    if mode == "FastAPI Backend":
        api_url = st.text_input("Backend URL", value="http://localhost:8000")

    st.markdown("---")

    # Status
    st.markdown("### 📡 System Status")
    if openai_key:
        st.markdown("🟢 **API Key** Configured")
    else:
        st.markdown("🔴 **API Key** Missing")
    st.markdown(f"🟢 **Mode** {mode}")
    st.markdown(f"🟢 **Time** {datetime.now().strftime('%d %b %Y %H:%M')}")

    st.markdown("---")

    # Incident history
    st.markdown("### 📋 Recent Reports")
    if not st.session_state.history:
        st.caption("No incidents logged yet.")
    else:
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            level = h.get("incident_report", {}).get("threat_level", "Low")
            icon  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")
            loc   = h.get("incident_report", {}).get("location", "Unknown")[:20]
            t     = h.get("_ts", "")
            st.markdown(
                f'<div class="history-item">{icon} [{t}] {loc}</div>',
                unsafe_allow_html=True
            )

    if st.session_state.history:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.result  = None
            st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("# 🚨 AP Securitas — Intelligent Incident Logger")
st.markdown(
    "AI-powered triage: enter any guard report → get structured data + SOP action steps instantly.",
    help="Supports English, Marathi, Hindi, or any mix."
)
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    guard_name  = st.text_input(
        "👤 Guard Name / ID",
        placeholder="e.g., Guard Ramesh Kumar — Post: Gate 2"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Reports Today", len(st.session_state.history))

incident_input = st.text_area(
    "📋 Incident Report (Natural Language)",
    placeholder=(
        "Describe the incident in your own words — English, Marathi, Hindi, or mix.\n\n"
        "Example: Gate 2 par ek suspicious black backpack mili hai jo pichhle 20 minute se koi "
        "nahi le gaya. Bag ek pillar ke paas rakhi hai. Maine kisi ko nahi jaane diya. "
        "Kya karna chahiye?"
    ),
    height=140,
    label_visibility="visible"
)

# ── Sample Inputs ─────────────────────────────────────────────────────────────
with st.expander("💡 Try Sample Reports"):
    samples = {
        "🔴 Suspicious Bag":     "Gate No. 2 ke paas ek abandoned black bag rakhi hai approximately 25 minutes se. Koi malik nahi dikha. Bag se halki si tikhi smell aa rahi hai.",
        "🔴 Fire / Smoke":       "Block B server room mein smoke detect ho raha hai. Fire alarm baj raha hai. Evacuation start karni chahiye?",
        "🟡 Suspicious Person":  "Parking lot mein ek unknown male lagbhag 45 minutes se ghoom raha hai. Wo vehicles ke paas ruk ruk ke dekh raha hai. Uske paas koi ID nahi hai.",
        "🟢 Lost Property":      "Ek visitor ka wallet main lobby mein mila hai. Visitor already chale gaye hain. Name card tha wallet mein."
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state["_sample"] = text

# Pre-fill with sample if clicked
if "_sample" in st.session_state:
    incident_input = st.session_state.pop("_sample")

st.markdown("---")

# ── Submit ────────────────────────────────────────────────────────────────────
submit = st.button("🔍 Analyse Incident", type="primary", use_container_width=False)

if submit:
    if not openai_key:
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar.")
    elif not incident_input.strip():
        st.warning("📝 Please enter an incident report above.")
    else:
        with st.spinner("🤖 AI Agent processing incident..."):
            try:
                if mode == "Direct (no server)":
                    result = call_agent_direct(incident_input, guard_name or "Unknown Guard", openai_key)
                else:
                    result = call_api(incident_input, guard_name or "Unknown Guard", api_url)

                result["_ts"] = ts()
                st.session_state.result  = result
                st.session_state.history.append(result)

            except Exception as e:
                st.error(f"❌ Agent error: {e}")
                st.session_state.result = None


# ── Results Display ───────────────────────────────────────────────────────────
result = st.session_state.result
if result:
    report       = result.get("incident_report", {})
    sop_action   = result.get("sop_action")
    alert_msg    = result.get("alert_message", "No alert generated.")
    immediate    = result.get("requires_immediate_action", False)
    protocol_num = result.get("protocol_number")
    threat_level = report.get("threat_level", "Low")

    st.markdown("## 📊 Analysis Results")

    # ── Header row: threat level + protocol tag ──
    header_col1, header_col2, header_col3 = st.columns([2, 2, 3])
    with header_col1:
        st.markdown("**Threat Level**")
        st.markdown(get_threat_badge(threat_level), unsafe_allow_html=True)
    with header_col2:
        st.markdown("**Protocol**")
        if protocol_num:
            st.markdown(f'<span class="protocol-tag">📋 {protocol_num}</span>', unsafe_allow_html=True)
        else:
            st.markdown("—")
    with header_col3:
        st.markdown("**Immediate Action Required**")
        st.markdown("🔴 **YES — Escalate Now**" if immediate else "🟡 Standard Response")

    st.markdown("---")

    # ── Main tabs ──
    tab1, tab2, tab3 = st.tabs(["📝 Incident Report", "📋 SOP Action Steps", "{ } Raw JSON"])

    with tab1:
        # Alert message at the top
        st.markdown(
            f'<div class="{get_alert_class(threat_level)}">'
            f'<b>🤖 AI Directive for Guard:</b><br><br>{alert_msg}</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Structured report as clean field cards
        fields = [
            ("📍 Location",          report.get("location",        "—")),
            ("🎯 Object / Person",   report.get("object_involved", "—")),
            ("📄 Summary",           report.get("summary",         "—")),
            ("👤 Reported By",       report.get("reported_by",     "Unknown Guard")),
            ("🕐 Incident Time",     report.get("incident_time",   "Not specified")),
            ("⚡ Action Taken",      report.get("action_taken",    "None taken yet")),
        ]

        c1, c2 = st.columns(2)
        for i, (label, value) in enumerate(fields):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="field-label">{label}</div>'
                    f'<div class="field-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with tab2:
        if sop_action:
            st.markdown(f'<span class="protocol-tag">📋 {protocol_num or "SOP Match"}</span>',
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="sop-panel">{sop_action.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info(
                "ℹ️ No SOP retrieval was triggered for this incident.\n\n"
                "SOP lookup runs automatically for HIGH-threat incidents.\n"
                f"This incident was classified as **{threat_level}** threat level."
            )

    with tab3:
        # Clean JSON export view
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        json_str = json.dumps(clean, indent=2, ensure_ascii=False)
        st.code(json_str, language="json")
        st.download_button(
            "⬇️ Download JSON Report",
            data=json_str,
            file_name=f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
    "AP Securitas Pvt. Ltd. &nbsp;|&nbsp; AI-Augmented SOC Operations &nbsp;|&nbsp; "
    "Godrej IT Park, Thane &nbsp;|&nbsp; Built by Pramod Jadhav"
    "</div>",
    unsafe_allow_html=True
)
