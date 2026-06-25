"""
backend/main.py — FastAPI REST API
====================================
Module 1 & 3: API Deployment

Endpoints:
    GET  /health        — Service health check
    POST /report        — Process a guard's incident report (full agent pipeline)
    POST /chat          — Alias for /report (conversational interface)
    GET  /kb/stats      — Knowledge base statistics

Run locally:
    cd incident-logger
    uvicorn backend.main:app --reload --port 8000

API docs auto-generated at: http://localhost:8000/docs
"""

import os
import sys

# Add project root to path when running backend/main.py directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.database import get_db, IncidentLog
from database import engine
from models import Base

Base.metadata.create_all(bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

load_dotenv()

from src.agent import IncidentAgent
from src.rag_pipeline import SOPRetriever


# ── App Initialisation ────────────────────────────────────────────────────────

app = FastAPI(
    title="AP Securitas — Intelligent Incident Logger API",
    description=(
        "AI-Powered Security Incident Triage using LangGraph + RAG. "
        "Extracts structured incident reports from guard natural-language input "
        "and returns SOP-matched action guidance."
    ),
    version="1.0.0",
    contact={
        "name": "Pramod Jadhav | AP Securitas SOC",
        "url":  "https://pramodjadhav.vercel.app"
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Global agent (initialised once at startup) ────────────────────────────────

_agent: Optional[IncidentAgent] = None
_retriever: Optional[SOPRetriever] = None


@app.on_event("startup")
async def startup_event():
    global _agent, _retriever

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    sop_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sop_manual.txt"
    )

    print("[API] Initialising SOPRetriever (ChromaDB + embeddings)...")
    _retriever = SOPRetriever(sop_path, openai_key)

    print("[API] Initialising IncidentAgent (LangGraph)...")
    _agent = IncidentAgent(openai_key, _retriever)

    print("[API] ✅ Agent ready. API is live.")


# ── Request / Response Models ─────────────────────────────────────────────────

class ReportRequest(BaseModel):
    guard_input: str = Field(
        ...,
        description="Guard's free-form incident report in any language (English/Marathi/Hindi/mix).",
        example="Gate 2 par ek suspicious black bag mili hai jo pichhle 20 minute se koi nahi le gaya."
    )
    guard_name: Optional[str] = Field(
        default="Unknown Guard",
        description="Name or ID of the reporting guard.",
        example="Guard Ramesh Kumar (ID: G-047)"
    )


class HealthResponse(BaseModel):
    status: str
    model: str
    kb_chunks: int
    service: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns service health and knowledge base statistics."""
    if not _agent or not _retriever:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not yet initialised. Please retry in a moment."
        )
    kb_stats = _retriever.get_stats()
    return HealthResponse(
        status="healthy",
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        kb_chunks=kb_stats["total_chunks"],
        service="Incident Logger v1.0"
    )


@app.post("/report", tags=["Incident Processing"])
def process_report(req: ReportRequest, current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    if not req.guard_input.strip():
        raise HTTPException(status_code=422, detail="guard_input cannot be empty.")

    try:
        result = _agent.process(guard_input=req.guard_input, guard_name=req.guard_name)
        
        report_data = result["incident_report"]
        db_log = IncidentLog(
            location=report_data.get("location"),
            object_involved=report_data.get("object_involved"),
            threat_level=report_data.get("threat_level"),
            confidence_score=str(report_data.get("confidence_score", 0.0)),
            protocol_number=result.get("protocol_number"),
            requires_immediate_action=result.get("requires_immediate_action", False)
        )
        db.add(db_log)
        db.commit()
        
        return {"status": "success", "data": result}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")
    """
    Main endpoint: run the full LangGraph agent pipeline on a guard report.

    - Extracts structured IncidentReport (Pydantic via instructor)
    - If threat is HIGH, retrieves relevant SOP from ChromaDB
    - Returns structured AgentResponse with action guidance
    """


@app.post("/chat", tags=["Incident Processing"])
def chat(req: ReportRequest):
    """Alias for /report — conversational interface for frontend integration."""
    return process_report(req)


@app.get("/kb/stats", tags=["System"])
def kb_stats():
    """Returns knowledge base (ChromaDB) statistics."""
    if not _retriever:
        raise HTTPException(503, "Retriever not initialised.")
    return _retriever.get_stats()
