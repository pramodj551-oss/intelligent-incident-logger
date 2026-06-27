# -*- coding: utf-8 -*-
"""
src/agent.py -- LangGraph Agent Orchestration
=============================================
The agent depends ONLY on BaseLLMProvider -- never on Groq/OpenAI directly.
Switch providers by changing .env; this file never needs to change.

Flow:
    START
      -> extract_incident_data  (instructor + Pydantic)
            |
            +--[High or Medium]--> retrieve_sop --> generate_response --> END
            |
            +--[Low]----------------------------> generate_response --> END
"""

from __future__ import annotations

import os
import re
import uuid
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END, START

from src.schemas import IncidentReport, AgentResponse
from src.rag_pipeline import SOPRetriever
from src.providers import BaseLLMProvider, get_provider, get_provider_from_env


# -- System prompts -----------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """You are an AI Security Operations Center (SOC) analyst for AP Securitas Pvt. Ltd.
Extract structured incident data from free-form security guard reports.
Always estimate a confidence_score between 0.0 and 1.0 based on how clear the guard input is.

Rules:
- Guards may write in English, Marathi, Hindi, or a mix. Extract correctly regardless of language.
- Threat levels:
  * HIGH:   bomb/explosive, active fire, armed intruder, medical emergency, robbery, chemical hazard
  * MEDIUM: suspicious loiterer, unauthorized access, abandoned vehicle, minor altercation, suspected theft
  * LOW:    lost property, visitor issue, noise complaint, minor policy breach
- When uncertain between Medium and High, choose High for safety."""

ALERT_SYSTEM_PROMPT = """You are a SOC advisor for AP Securitas Pvt. Ltd.
Generate a clear, direct action directive for the guard on the ground.
- 2-3 sentences maximum.
- Start with the most urgent action the guard must take RIGHT NOW.
- Simple language -- the guard is in the field.
- HIGH: safety and containment first.
- MEDIUM: observation and reporting first.
- LOW: documentation first."""


# -- LangGraph state ----------------------------------------------------------

class AgentState(TypedDict):
    raw_input:       str
    incident_id:     str
    incident_report: Optional[dict]
    sop_content:     Optional[str]
    final_response:  Optional[dict]


# -- Agent --------------------------------------------------------------------

class IncidentAgent:
    """
    LangGraph agent that orchestrates triage via any BaseLLMProvider.
    Provider is injected -- agent has zero knowledge of Groq vs OpenAI.
    """

    def __init__(
        self,
        sop_retriever: SOPRetriever,
        provider: Optional[BaseLLMProvider] = None,
        api_key: str = "",          # kept for backward compat with Streamlit cache
    ):
        """
        Args:
            sop_retriever: Initialised SOPRetriever
            provider:      BaseLLMProvider instance (preferred)
            api_key:       API key string -- used only when `provider` is None
                           (auto-detects provider from LLM_PROVIDER env var)
        """
        if provider is not None:
            self._provider = provider
        elif api_key:
            name = os.getenv("LLM_PROVIDER", "groq")
            self._provider = get_provider(name, api_key)
        else:
            self._provider = get_provider_from_env()

        self._model                   = self._provider.model
        self._llm, self._instructor   = self._provider.get_clients()
        self._retriever               = sop_retriever
        self._graph                   = self._build_graph()

        print(
            f"[Agent] provider={self._provider.provider_name} "
            f"model={self._model}"
        )

    # -- Node 1 ---------------------------------------------------------------

    def _extract_incident_data(self, state: AgentState) -> AgentState:
        """instructor + Pydantic: free-form text -> validated IncidentReport."""
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

    # -- Conditional edge -----------------------------------------------------

    def _route_by_threat(self, state: AgentState) -> str:
        """HIGH or MEDIUM -> SOP retrieval.  LOW -> direct response."""
        threat = (state.get("incident_report") or {}).get("threat_level", "Low")
        return "retrieve_sop" if threat in ("High", "Medium") else "generate_response"

    # -- Node 2 (HIGH / MEDIUM only) ------------------------------------------

    def _retrieve_sop(self, state: AgentState) -> AgentState:
        """Semantic search against ChromaDB for the most relevant SOP chunks."""
        r     = state["incident_report"]
        query = (
            str(r.get("object_involved", ""))
            + " at " + str(r.get("location", ""))
            + ". " + str(r.get("summary", ""))
        )
        docs     = self._retriever.retrieve(query, n_results=2)
        sop_text = "\n\n---\n\n".join(docs) if docs else ""
        return {**state, "sop_content": sop_text}

    # -- Node 3 ---------------------------------------------------------------

    def _generate_response(self, state: AgentState) -> AgentState:
        """Build final AgentResponse: structured report + alert message."""
        report = state["incident_report"]
        sop    = state.get("sop_content") or ""

        parts = ["Incident Report:\n" + str(report)]
        if sop:
            parts.append("Relevant SOP:\n" + sop)
        parts.append("Write a 2-3 sentence actionable alert message for the guard.")
        context = "\n\n".join(parts)

        alert = self._llm.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": ALERT_SYSTEM_PROMPT},
                {"role": "user",   "content": context}
            ],
            max_tokens=200
        ).choices[0].message.content.strip()

        protocol_number = None
        if sop:
            m = re.search(r"Protocol\s+(\d+)", sop)
            if m:
                protocol_number = "Protocol " + m.group(1)

        final = AgentResponse(
            incident_report=IncidentReport(**report),
            sop_action=sop or None,
            alert_message=alert,
            requires_immediate_action=(report.get("threat_level") == "High"),
            protocol_number=protocol_number
        )
        return {**state, "final_response": final.model_dump()}

    # -- Graph ----------------------------------------------------------------

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("extract_data",      self._extract_incident_data)
        g.add_node("retrieve_sop",      self._retrieve_sop)
        g.add_node("generate_response", self._generate_response)

        g.add_edge(START, "extract_data")
        g.add_conditional_edges(
            "extract_data",
            self._route_by_threat,
            {"retrieve_sop": "retrieve_sop", "generate_response": "generate_response"}
        )
        g.add_edge("retrieve_sop",      "generate_response")
        g.add_edge("generate_response", END)
        return g.compile()

    # -- Public ---------------------------------------------------------------

    def process(self, guard_input: str, guard_name: str = "Unknown Guard") -> dict:
        """Run the full pipeline on a guard's natural-language report."""
        state: AgentState = {
            "raw_input":       guard_input,
            "incident_id":     str(uuid.uuid4()),
            "incident_report": None,
            "sop_content":     None,
            "final_response":  None
        }
        result   = self._graph.invoke(state)
        response = result.get("final_response", {})
        if response and response.get("incident_report"):
            response["incident_report"]["reported_by"] = guard_name
            response["incident_id"]     = state["incident_id"]
            response["llm_provider"]    = self._provider.provider_name
            response["llm_model"]       = self._model
        return response
