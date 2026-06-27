# -*- coding: utf-8 -*-
"""
backend/main.py -- FastAPI REST API
====================================
Provider is resolved entirely from .env via get_provider_from_env().
No provider-specific code lives here.

Dev  (.env):  LLM_PROVIDER=groq   + GROQ_API_KEY
Prod (.env):  LLM_PROVIDER=openai + OPENAI_API_KEY

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.providers import get_provider_from_env, list_providers
from src.agent import IncidentAgent
from src.rag_pipeline import SOPRetriever


# -- App ----------------------------------------------------------------------

app = FastAPI(
    title="AP Securitas -- Intelligent Incident Logger API",
    description=(
        "Multi-provider AI triage engine. "
        "Switch LLM via .env: LLM_PROVIDER=groq|openai|huggingface"
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# -- Optional API key auth (disabled when DISABLE_AUTH_FOR_DEMO=true) ---------

_DEMO_MODE = os.getenv("DISABLE_AUTH_FOR_DEMO", "false").lower() == "true"
_API_KEY   = os.getenv("APP_API_KEY", "")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: Optional[str] = Security(api_key_header)):
    """Pass-through in demo mode; enforce key in production."""
    if _DEMO_MODE:
        return          # no auth required
    if not _API_KEY:
        return          # APP_API_KEY not set -> open access
    if key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key header."
        )


# -- Global singletons --------------------------------------------------------

_agent:     Optional[IncidentAgent] = None
_retriever: Optional[SOPRetriever]  = None


@app.on_event("startup")
async def startup():
    global _agent, _retriever

    embed_mode = os.getenv("EMBEDDING_MODE", "local")
    embed_key  = os.getenv("OPENAI_API_KEY", "") if embed_mode == "openai" else ""

    sop_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sop_manual.txt"
    )

    # Provider resolved from .env -- no provider code here
    provider = get_provider_from_env()

    _retriever = SOPRetriever(sop_path, openai_api_key=embed_key)
    _agent     = IncidentAgent(sop_retriever=_retriever, provider=provider)
    print("[API] Ready.")


# -- Models -------------------------------------------------------------------

class ReportRequest(BaseModel):
    guard_input: str = Field(
        ...,
        min_length=5,
        example="Gate 2 par ek suspicious black bag mili hai jo pichhle 20 minute se koi nahi le gaya."
    )
    guard_name: Optional[str] = Field(default="Unknown Guard")


# -- Endpoints ----------------------------------------------------------------

@app.get("/health")
def health(_=Depends(verify_api_key)):
    if not _agent:
        raise HTTPException(503, "Agent not initialised.")
    kb = _retriever.get_stats()
    return {
        "status":         "healthy",
        "llm_provider":   _agent._provider.provider_name,
        "llm_model":      _agent._model,
        "embedding_mode": os.getenv("EMBEDDING_MODE", "local"),
        "kb_chunks":      kb["total_chunks"],
        "demo_mode":      _DEMO_MODE,
        "available_providers": list_providers(),
    }


@app.post("/report")
def process_report(req: ReportRequest, _=Depends(verify_api_key)):
    """Full triage pipeline: extract -> [retrieve SOP] -> respond."""
    if not _agent:
        raise HTTPException(503, "Agent not initialised.")
    try:
        result = _agent.process(req.guard_input, req.guard_name)
        return {"status": "success", "data": result}
    except Exception as exc:
        raise HTTPException(500, f"Agent error: {exc}")


@app.post("/chat")
def chat(req: ReportRequest, _=Depends(verify_api_key)):
    """Alias for /report."""
    return process_report(req)


@app.get("/providers")
def providers():
    """List all registered provider names."""
    return {"providers": list_providers(), "active": os.getenv("LLM_PROVIDER", "groq")}


@app.get("/kb/stats")
def kb_stats(_=Depends(verify_api_key)):
    if not _retriever:
        raise HTTPException(503, "Retriever not initialised.")
    return _retriever.get_stats()
