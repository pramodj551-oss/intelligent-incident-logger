# -*- coding: utf-8 -*-
"""
agent.py -- LangGraph Agent Orchestration
==========================================
Supports two LLM providers via LLM_PROVIDER env variable:
  - groq   (default, FREE): llama-3.3-70b-versatile via Groq Cloud
  - openai            : gpt-4o-mini via OpenAI
"""

from __future__ import annotations

import os
import re
import uuid
from typing import TypedDict, Optional, Any, Tuple

import instructor
from langgraph.graph import StateGraph, END, START

from src.schemas import IncidentReport, AgentResponse
from src.rag_pipeline import SOPRetriever


# -- Provider defaults --------------------------------------------------------
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "groq":   "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}

# -- System prompts -----------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """You are an AI Security Operations Center (SOC) analyst for AP Securitas Pvt. Ltd.
Your task is to extract structured incident data from free-form security guard reports.
Always estimate a confidence_score between 0.0 and 1.0 based on how clear and specific the guard input is.

Extraction rules:
- Guards may write in English, Marathi, Hindi, or a mix (Hinglish). Extract correctly regardless of language.
- Infer threat level from context:
  * HIGH: explosive/bomb, active fire, armed intruder, medical emergency, active robbery, bomb threat, chemical hazard
  * MEDIUM: suspicious loiterer, unauthorized access attempt, abandoned/suspicious vehicle, minor altercation, suspected theft
  * LOW: lost/found property, visitor management issue, noise, minor policy breach
- If the guard already took an action, capture it in action_taken.
- Be conservative: when uncertain between Medium and High, choose High for safety."""

ALERT_SYSTEM_PROMPT = """You are an AI SOC advisor for AP Securitas Pvt. Ltd.
Generate a clear, direct action directive for the guard on the ground.

Rules:
- Write 2-3 sentences maximum.
- Start with the most urgent action the guard must take RIGHT NOW.
- Use simple, direct language -- the guard is in the field.
- If SOP steps are provided, incorporate the most critical ones.
- For HIGH threat: Begin with safety and containment.
- For MEDIUM: Begin with observation and reporting.
- For LOW: Begin with documentation."""


# -- Provider factory ---------------------------------------------------------

def _build_llm_clients(api_key: str, provider: str) -> Tuple[Any, Any]:
    """
    Returns (raw_client, instructor_client) for the chosen provider.
    Both expose the same .chat.completions.create() interface.

    provider="groq"   -> Groq(api_key) + instructor.from_groq(GROQ_JSON mode)
    provider="openai" -> OpenAI(api_key) + instructor.from_openai()
    """
    if provider == "groq":
        from groq import Groq
        raw        = Groq(api_key=api_key)
        structured = instructor.from_groq(raw, mode=instructor.Mode.GROQ_JSON)
    else:
        from openai import OpenAI
        raw        = OpenAI(api_key=api_key)
        structured = instructor.from_openai(raw)
    return raw, structured


# -- Agent State --------------------------------------------------------------

class AgentState(TypedDict):
    raw_input:       str
    incident_id:     str
    incident_report: Optional[dict]
    sop_content:     Optional[str]
    final_response:  Optional[dict]


# -- Agent Class --------------------------------------------------------------

class IncidentAgent:
    """
    LangGraph-based agent that orchestrates the full incident triage pipeline.
    Works with Groq (free) or OpenAI depending on LLM_PROVIDER env variable.
    """

    def __init__(self, api_key: str, sop_retriever: SOPRetriever):
        # Read provider at init time so Streamlit cache invalidation works correctly
        self._provider  = os.getenv("LLM_PROVIDER", "groq")
        self._model     = os.getenv(
            "LLM_MODEL",
            PROVIDER_DEFAULT_MODELS.get(self._provider, "gpt-4o-mini")
        )
        self._llm, self._instructor = _build_llm_clients(api_key, self._provider)
        self._retriever = sop_retriever
        self._graph     = self._build_graph()
        print(f"[Agent] Provider: {self._provider} | Model: {self._model}")

    # -- Node 1: Extract Incident Data ----------------------------------------

    def _extract_incident_data(self, state: AgentState) -> AgentState:
        """Use instructor + Pydantic to extract a clean IncidentReport."""
        report: IncidentReport = self._instructor.chat.completions.create(
            model=self._model,
            response_model=IncidentReport,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user",   "content": "Guard Report:\n" + state["raw_input"]}
            ],
            max_retries=3
        )
        return {**state, "incident_report": report.model_dump()}

    # -- Conditional Edge: Route by Threat Level ------------------------------

    def _route_by_threat(self, state: AgentState) -> str:
        """Route HIGH and MEDIUM to SOP retrieval; LOW skips to direct response."""
        threat = (state.get("incident_report") or {}).get("threat_level", "Low")
        if threat in ("High", "Medium"):
            return "retrieve_sop"
        return "generate_response"

    # -- Node 2 (HIGH/MEDIUM): Retrieve SOP -----------------------------------

    def _retrieve_sop(self, state: AgentState) -> AgentState:
        """Semantic search against ChromaDB for relevant SOP protocol chunks."""
        report = state["incident_report"]
        query  = (
            str(report.get("object_involved", "")) + " "
            + "at " + str(report.get("location", "")) + ". "
            + str(report.get("summary", ""))
        )
        docs     = self._retriever.retrieve(query, n_results=2)
        sop_text = "\n\n---\n\n".join(docs) if docs else ""
        return {**state, "sop_content": sop_text}

    # -- Node 3: Generate Response --------------------------------------------

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final AgentResponse with actionable alert message."""
        report = state["incident_report"]
        sop    = state.get("sop_content") or ""

        context_parts = ["Incident Report:\n" + str(report)]
        if sop:
            context_parts.append("Relevant SOP:\n" + sop)
        context_parts.append(
            "Generate a 2-3 sentence actionable alert message for the guard."
        )
        context = "\n\n".join(context_parts)

        alert_message = self._llm.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": ALERT_SYSTEM_PROMPT},
                {"role": "user",   "content": context}
            ],
            max_tokens=200
        ).choices[0].message.content.strip()

        protocol_number = None
        if sop:
            match = re.search(r"Protocol\s+(\d+)", sop)
            if match:
                protocol_number = "Protocol " + match.group(1)

        incident_report_obj = IncidentReport(**report)
        final = AgentResponse(
            incident_report=incident_report_obj,
            sop_action=sop if sop else None,
            alert_message=alert_message,
            requires_immediate_action=(report.get("threat_level") == "High"),
            protocol_number=protocol_number
        )
        return {**state, "final_response": final.model_dump()}

    # -- Graph Builder --------------------------------------------------------

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("extract_data",      self._extract_incident_data)
        g.add_node("retrieve_sop",      self._retrieve_sop)
        g.add_node("generate_response", self._generate_response)

        g.add_edge(START, "extract_data")
        g.add_conditional_edges(
            "extract_data",
            self._route_by_threat,
            {
                "retrieve_sop":      "retrieve_sop",
                "generate_response": "generate_response"
            }
        )
        g.add_edge("retrieve_sop",      "generate_response")
        g.add_edge("generate_response", END)
        return g.compile()

    # -- Public API -----------------------------------------------------------

    def process(self, guard_input: str, guard_name: str = "Unknown Guard") -> dict:
        """Process a guard report through the full LangGraph pipeline."""
        initial_state: AgentState = {
            "raw_input":       guard_input,
            "incident_id":     str(uuid.uuid4()),
            "incident_report": None,
            "sop_content":     None,
            "final_response":  None
        }
        result   = self._graph.invoke(initial_state)
        response = result.get("final_response", {})
        if response and response.get("incident_report"):
            response["incident_report"]["reported_by"] = guard_name
            response["incident_id"]  = initial_state["incident_id"]
            response["llm_provider"] = self._provider
            response["llm_model"]    = self._model
        return response
