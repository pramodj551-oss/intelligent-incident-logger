# -*- coding: utf-8 -*-
"""
backend/main.py -- FastAPI REST API
====================================
Reads LLM_PROVIDER to pick the right API key automatically:
  LLM_PROVIDER=groq   -> reads GROQ_API_KEY   (free, default)
  LLM_PROVIDER=openai -> reads OPENAI_API_KEY

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.agent import IncidentAgent
from src.rag_pipeline import SOPRetriever


app = FastAPI(
    title="AP Securitas -- Intelligent Incident Logger API",
    description=(
        "Powered by Groq (free) or OpenAI. "
        "Set LLM_PROVIDER=groq and GROQ_API_KEY in .env for zero-cost operation."
    ),
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

_agent:    Optional[IncidentAgent] = None
_retriever: Optional[SOPRetriever] = None


@app.on_event("startup")
async def startup_event():
    global _agent, _retriever

    llm_provider   = os.getenv("LLM_PROVIDER", "groq")
    embedding_mode = os.getenv("EMBEDDING_MODE", "openai")

    # -- Pick LLM key based on provider ---------------------------------------
    if llm_provider == "groq":
        llm_key = os.getenv("GROQ_API_KEY", "")
        if not llm_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at console.groq.com, then add it to .env"
            )
    else:
        llm_key = os.getenv("OPENAI_API_KEY", "")
        if not llm_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    # -- Pick embedding key (only needed for openai embedding mode) -----------
    embed_key = os.getenv("OPENAI_API_KEY", "") if embedding_mode == "openai" else ""

    sop_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sop_manual.txt"
    )

    print(f"[API] LLM provider : {llm_provider}")
    print(f"[API] Embedding    : {embedding_mode}")

    _retriever = SOPRetriever(sop_path, openai_api_key=embed_key)
    _agent     = IncidentAgent(api_key=llm_key, sop_retriever=_retriever)
    print("[API] Agent ready.")


class ReportRequest(BaseModel):
    guard_input: str = Field(
        ...,
        example="Gate 2 par ek suspicious black bag mili hai jo pichhle 20 minute se koi nahi le gaya."
    )
    guard_name: Optional[str] = Field(default="Unknown Guard")


@app.get("/health")
def health_check():
    if not _agent:
        raise HTTPException(503, "Agent not initialised.")
    kb = _retriever.get_stats()
    return {
        "status":          "healthy",
        "llm_provider":    os.getenv("LLM_PROVIDER", "groq"),
        "llm_model":       _agent._model,
        "embedding_mode":  os.getenv("EMBEDDING_MODE", "openai"),
        "kb_chunks":       kb["total_chunks"],
    }


@app.post("/report")
def process_report(req: ReportRequest):
    if not _agent:
        raise HTTPException(503, "Agent not initialised.")
    if not req.guard_input.strip():
        raise HTTPException(422, "guard_input cannot be empty.")
    try:
        result = _agent.process(req.guard_input, req.guard_name)
        return {"status": "success", "data": result}
    except Exception as exc:
        raise HTTPException(500, f"Agent error: {exc}")


@app.post("/chat")
def chat(req: ReportRequest):
    return process_report(req)


@app.get("/kb/stats")
def kb_stats():
    if not _retriever:
        raise HTTPException(503, "Retriever not initialised.")
    return _retriever.get_stats()
