# -*- coding: utf-8 -*-
"""
agent.py -- LangGraph Agent Orchestration
==========================================
Module 3: LangGraph Single-Agent Controller

State machine flow:
    START
      -> extract_incident_data   (LLM + Pydantic via instructor)
            |
            +--[threat == HIGH or MEDIUM]--> retrieve_sop --> generate_response --> END
            |
            +--[threat == LOW]----------------------------> generate_response --> END
"""

from __future__ import annotations

import os
import re
import uuid
from typing import TypedDict, Optional, Any

import instructor
from openai import OpenAI
from langgraph.graph import StateGraph, END, START

from src.schemas import IncidentReport, AgentResponse
from src.rag_pipeline import SOPRetriever


# -- Configurable model -------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# -- System prompts -----------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """You are an AI Security Operations Center (SOC) analyst for AP Securitas Pvt. Ltd.
Your task is to extract structured incident data from free-form security guard reports.
Always estimate a confidence_score between 0.0 and 1.0 based on how clear and specific the guard's input is.

Extraction rules:
- Guards may write in English, Marathi, Hindi, or a mix (Hinglish). Extract correctly regardless of language.
- Infer threat level from context:
  * HIGH: explosive/bomb, active fire, armed intruder, medical emergency, active robbery, bomb threat, chemical hazard
  * MEDIUM: suspicious loiterer, unauthorized access attempt, abandoned/suspicious vehicle, minor altercation, suspected theft
  * LOW: lost/found property, visitor management issue, noise, minor policy breach
- If the guard already took an action, capture it in action_taken.
- Be conservative: when uncertain between Medium and High, choose High for safety."""

ALERT_SYSTEM_PROMPT = """You are an AI SOC advisor for AP Securitas Pvt. Ltd.
Based on the structured incident report provided, generate a clear, direct action directive
for the guard on the ground.

Rules:
- Write 2-3 sentences maximum.
- Start with the most urgent action the guard must take RIGHT NOW.
- Use simple, direct language -- the guard is in the field.
- If SOP steps are provided, incorporate the most critical ones.
- For HIGH threat: Begin with safety and containment.
- For MEDIUM: Begin with observation and reporting.
- For LOW: Begin with documentation."""


# -- Agent State --------------------------------------------------------------

class AgentState(TypedDict):
    """Shared state flowing through the LangGraph nodes."""
    raw_input: str                   # Guard's original free-form text
    incident_id: str                 # UUID generated per .process() call
    incident_report: Optional[dict]  # Extracted IncidentReport as dict
    sop_content: Optional[str]       # Retrieved SOP chunks (HIGH/MEDIUM only)
    final_response: Optional[dict]   # Final AgentResponse as dict


# -- Agent Class --------------------------------------------------------------

class IncidentAgent:
    """
    LangGraph-based agent that orchestrates the full incident triage pipeline:
    Extract -> [Retrieve SOP if HIGH or MEDIUM] -> Generate Response
    """

    def __init__(self, openai_api_key: str, sop_retriever: SOPRetriever):
        self._instructor = instructor.from_openai(OpenAI(api_key=openai_api_key))
        self._openai     = OpenAI(api_key=openai_api_key)
        self._retriever  = sop_retriever
        self._graph      = self._build_graph()

    # -- Node 1: Extract Incident Data ----------------------------------------

    def _extract_incident_data(self, state: AgentState) -> AgentState:
        """Use instructor + Pydantic to extract a clean IncidentReport."""
        report: IncidentReport = self._instructor.chat.completions.create(
            model=LLM_MODEL,
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
        """
        Route HIGH and MEDIUM threats to SOP retrieval.
        LOW threats skip directly to response generation.
        """
        threat = (state.get("incident_report") or {}).get("threat_level", "Low")
        if threat in ("High", "Medium"):
            return "retrieve_sop"
        return "generate_response"

    # -- Node 2 (HIGH/MEDIUM): Retrieve SOP -----------------------------------

    def _retrieve_sop(self, state: AgentState) -> AgentState:
        """Semantic search against ChromaDB for relevant SOP protocol chunks."""
        report = state["incident_report"]
        query = (
            str(report.get("object_involved", "")) + " "
            + "at " + str(report.get("location", "")) + ". "
            + str(report.get("summary", ""))
        )
        docs = self._retriever.retrieve(query, n_results=2)
        sop_text = "\n\n---\n\n".join(docs) if docs else ""
        return {**state, "sop_content": sop_text}

    # -- Node 3: Generate Response --------------------------------------------

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final AgentResponse with alert message and SOP guidance."""
        report = state["incident_report"]
        sop    = state.get("sop_content") or ""

        context_parts = ["Incident Report:\n" + str(report)]
        if sop:
            context_parts.append("Relevant SOP:\n" + sop)
        context_parts.append(
            "Generate a 2-3 sentence actionable alert message for the guard on the ground."
        )
        context = "\n\n".join(context_parts)

        alert_message = self._openai.chat.completions.create(
            model=LLM_MODEL,
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

        # Optional: DB logging hook -- wire up your SQLAlchemy session here
        # self._log_to_db(state["incident_id"], final)

        return {**state, "final_response": final.model_dump()}

    # -- (Optional) DB Logging stub -------------------------------------------

    def _log_to_db(self, incident_id: str, response: AgentResponse) -> None:
        """
        Stub for database logging. Uncomment and wire up your SQLAlchemy
        session (db) and IncidentLog model here.

        Example:
            from your_app.models import IncidentLog
            from your_app.database import db

            record = IncidentLog(
                incident_id=incident_id,
                location=response.incident_report.location,
                object_involved=response.incident_report.object_involved,
                threat_level=response.incident_report.threat_level,
                protocol_number=response.protocol_number,
                status="OPEN",
                requires_immediate_action=response.requires_immediate_action
            )
            db.add(record)
            db.commit()
        """
        pass

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
        """
        Process a guard's free-form incident report through the full pipeline.
        Generates a fresh incident_id (UUID) per call.
        """
        initial_state: AgentState = {
            "raw_input":       guard_input,
            "incident_id":     str(uuid.uuid4()),   # Fresh UUID per incident
            "incident_report": None,
            "sop_content":     None,
            "final_response":  None
        }

        result   = self._graph.invoke(initial_state)
        response = result.get("final_response", {})

        if response and response.get("incident_report"):
            response["incident_report"]["reported_by"] = guard_name
            response["incident_id"] = initial_state["incident_id"]

        return response
