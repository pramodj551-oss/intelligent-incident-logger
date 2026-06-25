# 🚨 AP Securitas — Intelligent Incident Logger & SOP Advisor

> **Capstone Project | IIT Patna — Applied AI & ML Essentials Programme (2025–26)**

An end-to-end AI-powered security incident triage system that converts
unstructured guard reports (in any language) into verified structured JSON,
then instantly matches them with relevant Standard Operating Procedures.

---

## 🏗️ Architecture

```
Guard Input (free-form text)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                               │
│                                                                  │
│   [1] extract_data          instructor + Pydantic               │
│         │                   → IncidentReport (location,          │
│         │                     object, threat_level, summary)     │
│         │                                                        │
│         ├──[threat=HIGH]──▶ [2] retrieve_sop                    │
│         │                        ChromaDB + text-embedding-3-small│
│         │                        Semantic match → SOP chunks     │
│         │                        ↓                               │
│         └──[threat≠HIGH]──▶ [3] generate_response               │
│                                  GPT-4o-mini → alert message     │
│                                  AgentResponse assembled         │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
 FastAPI /report endpoint  ←──── Streamlit UI
        │                         (dark SOC dashboard)
        ▼
 AgentResponse JSON
  ├─ incident_report (structured)
  ├─ sop_action (retrieved SOP text)
  ├─ alert_message (actionable directive)
  ├─ requires_immediate_action (bool)
  └─ protocol_number (e.g., "Protocol 101")
```

---

## 📁 Project Structure

```
incident-logger/
├── requirements.txt
├── .env.example             ← Copy to .env and add your OPENAI_API_KEY
├── data/
│   └── sop_manual.txt       ← AP Securitas SOP Knowledge Base (10 protocols)
├── src/
│   ├── schemas.py           ← Pydantic models: IncidentReport, AgentResponse
│   ├── rag_pipeline.py      ← ChromaDB + text-embedding-3-small RAG
│   └── agent.py             ← LangGraph state machine (main orchestrator)
├── backend/
│   └── main.py              ← FastAPI REST API (/report, /chat, /health)
├── frontend/
│   └── app.py               ← Streamlit dark-themed SOC dashboard
└── eval/
    └── eval_harness.py      ← 10-prompt evaluation harness
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/pramodj551-oss/incident-logger.git
cd incident-logger

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here
```

### 3a. Run Frontend Only (Direct Mode — No Server Needed)

```bash
streamlit run frontend/app.py
```
Open http://localhost:8501 — enter your OpenAI key in the sidebar.

### 3b. Run with FastAPI Backend

**Terminal 1 — API:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — UI:**
```bash
streamlit run frontend/app.py
```
In Streamlit sidebar, select **FastAPI Backend** mode and set URL to `http://localhost:8000`.

API docs available at: http://localhost:8000/docs

### 4. Run Evaluation Harness

```bash
python eval/eval_harness.py
```

---

## 🧪 Modules Covered

| Module | Implementation |
|--------|---------------|
| **Pydantic Structured Outputs** | `instructor` library forces LLM to return valid `IncidentReport` |
| **LangGraph Orchestration** | StateGraph with conditional edge (threat routing) |
| **ChromaDB + Embeddings** | `text-embedding-3-small` indexes SOP manual, semantic search at runtime |
| **FastAPI Deployment** | `/report` and `/chat` endpoints, CORS, startup event |
| **Streamlit UI** | Dark SOC theme, history, sample inputs, JSON export |
| **Evaluation Harness** | 10-case eval: threat accuracy, SOP recall, alert quality |

---

## 📋 SOP Knowledge Base

The `data/sop_manual.txt` contains 10 protocols for AP Securitas (Godrej IT Park):

| Protocol | Scenario | Threat |
|----------|----------|--------|
| 101 | Unattended/Suspicious Object | HIGH |
| 102 | Fire, Smoke, Explosion | HIGH |
| 103 | Medical Emergency | HIGH |
| 104 | Unauthorized Access / Intruder | HIGH/MEDIUM |
| 105 | Robbery / Violent Crime | HIGH |
| 106 | Bomb Threat | HIGH |
| 107 | Suspicious Vehicle | MEDIUM |
| 108 | Suspected Theft | LOW/MEDIUM |
| 109 | Missing Person | MEDIUM |
| 110 | Natural Disaster / Evacuation | HIGH |

---

## 📊 Resume Bullet Points

```
AI-Powered Security Operations Automation Agent (Capstone Project)

• Built an end-to-end Intelligent Incident Logging system automating physical
  security incident triage using FastAPI and LangGraph for a corporate SOC context.

• Implemented Pydantic Structured Outputs (via instructor) to parse unstructured
  guard logs into verified JSON IncidentReport objects, eliminating manual
  transcription errors and enforcing schema validation.

• Integrated a RAG pipeline using ChromaDB and OpenAI text-embedding-3-small to
  semantically match live HIGH-threat incidents with relevant institutional SOPs,
  reducing manual protocol lookup time to near-zero.

• Designed a conditional LangGraph StateGraph that routes HIGH-threat incidents
  through SOP retrieval while skipping unnecessary API calls for LOW/MEDIUM
  incidents, reducing latency and token cost.

• Formulated a 10-prompt live evaluation harness measuring threat classification
  accuracy, SOP retrieval coverage, and alert generation quality across diverse
  multilingual security scenarios.
```

---

## 🔧 Customisation

- **Swap LLM**: Set `LLM_MODEL=gpt-4o` in `.env` for higher accuracy
- **Add SOPs**: Append new protocols to `data/sop_manual.txt`, then delete
  `./chroma_db/` to force re-indexing on next startup
- **Deploy on Streamlit Cloud**: Add `OPENAI_API_KEY` to Streamlit secrets;
  note that ChromaDB will use in-memory mode (not persistent) on Streamlit Cloud

---

## 👤 Author

**Pramod Jadhav** | AI-Augmented SOC Analyst  
AP Securitas Pvt. Ltd. — Godrej IT Park, Thane  
IIT Patna — Applied AI & ML Essentials (2025–26)  
Portfolio: [pramodjadhav.vercel.app](https://pramodjadhav.vercel.app)  
GitHub: [pramodj551-oss](https://github.com/pramodj551-oss)
