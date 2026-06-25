"""
agent.py — LangGraph Agent Orchestration
==========================================
Module 3: LangGraph Single-Agent Controller

State machine flow:
    START
      └─► extract_incident_data   (LLM + Pydantic via instructor)
              │
              ├─► [threat == HIGH] ─► retrieve_sop ─► generate_response ─► END
              │
              └─► [threat == LOW/MEDIUM] ─────────────► generate_response ─► END

The agent:
  1. Extracts structured IncidentReport from guard's raw text
  2. Conditionally triggers RAG retrieval for HIGH-threat incidents
  3. Generates a final actionable response with alert message
"""

import os
import re
from typing import TypedDict, Optional, Any

import instructor
from openai import OpenAI
from langgraph.graph import StateGraph, END, START

from src.schemas import IncidentReport, AgentResponse
from src.rag_pipeline import SOPRetriever


# ── Configurable model ────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ── System prompts ────────────────────────────────────────────────────────────

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
- Write 2–3 sentences maximum.
- Start with the most urgent action the guard must take RIGHT NOW.
- Use simple, direct language — the guard is in the field.
- If SOP steps are provided, incorporate the most critical ones.
- For HIGH threat: Begin with safety and containment.
- For MEDIUM: Begin with observation and reporting.
- For LOW: Begin with documentation."""


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state flowing through the LangGraph nodes."""
    raw_input: str                       # Guard's original free-form text
    incident_report: Optional[dict]      # Extracted IncidentReport as dict
    sop_content: Optional[str]           # Retrieved SOP chunks (HIGH only)
    final_response: Optional[dict]       # Final AgentResponse as dict


# ── Agent Class ───────────────────────────────────────────────────────────────

class IncidentAgent:
    """
    LangGraph-based agent that orchestrates the full incident triage pipeline:
      Extract → [Retrieve SOP if HIGH] → Generate Response
    """

    def __init__(self, openai_api_key: str, sop_retriever: SOPRetriever):
        """
        Args:
            openai_api_key: OpenAI API key
            sop_retriever: Initialised SOPRetriever instance
        """
        # instructor wraps the OpenAI client to enforce Pydantic structured output
        self._instructor = instructor.from_openai(
            OpenAI(api_key=openai_api_key)
        )
        # Plain OpenAI client for non-structured calls (alert message generation)
        self._openai = OpenAI(api_key=openai_api_key)
        self._retriever = sop_retriever

        # Compile the LangGraph state machine once
        self._graph = self._build_graph()

    # ── Node: Extract Incident Data ───────────────────────────────────────────

    def _extract_incident_data(self, state: AgentState) -> AgentState:
        """
        Node 1: Use instructor + Pydantic to extract a clean IncidentReport
        from the guard's raw natural-language input.
        instructor will auto-retry on validation failure (max 3 attempts).
        """
        report: IncidentReport = self._instructor.chat.completions.create(
            model=LLM_MODEL,
            response_model=IncidentReport,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Guard Report:\n{state['raw_input']}"
                }
            ],
            max_retries=3  # instructor retries on Pydantic validation failure
        )
        return {**state, "incident_report": report.model_dump()}

    # ── Conditional Edge: Route by Threat Level ───────────────────────────────

    def _route_by_threat(self, state: AgentState) -> str:
        threat = (state.get("incident_report") or {}).get("threat_level", "Low")
        
        if threat in ["High", "Medium"]:
            return "retrieve_sop"
        return "generate_response"
        """
        Conditional edge function.
        Returns node name based on extracted threat_level.
        HIGH → retrieve SOP before responding.
        LOW/MEDIUM → respond directly.
        """
        threat = (state.get("incident_report") or {}).get("threat_level", "Low")
        return "retrieve_sop" if threat == "High" else "generate_response"

    # ── Node: Retrieve SOP ────────────────────────────────────────────────────

    def _retrieve_sop(self, state: AgentState) -> AgentState:
        """
        Node 2 (HIGH threat only): Semantic search against ChromaDB
        to find the most relevant SOP protocol chunks.
        """
        report = state["incident_report"]
        # Build a focused semantic query from the report fields
        query = (
            f"{report.get('object_involved', '')} "
            f"at {report.get('location', '')}. "
            f"{report.get('summary', '')}"
        )
        docs = self._retriever.retrieve(query, n_results=2)
        sop_text = "\n\n───────────────\n\n".join(docs) if docs else ""
        return {**state, "sop_content": sop_text}

    # ── Node: Generate Response ───────────────────────────────────────────────

    def _generate_response(self, state: AgentState) -> AgentState:
        """
        Node 3 (all paths): Generate the final AgentResponse.
        Combines structured report data + SOP content into a clear directive.
        """
        report = state["incident_report"]
        sop = state.get("sop_content") or ""

        # Build context for the LLM
        context_parts = [f"Incident Report:\n{report}"]
        if sop:
            context_parts.append(f"Relevant SOP (retrieved from AP Securitas manual):\n{sop}")
        context_parts.append(
            "Generate a 2–3 sentence actionable alert message for the guard on the ground."
        )
        context = "\n\n".join(context_parts)

        # Non-structured call — just need a plain text alert message
        alert_message = self._openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": ALERT_SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            max_tokens=200
        ).choices[0].message.content.strip()

        # Extract the protocol number from retrieved SOP text (regex)
        protocol_number = None
        if sop:
            match = re.search(r"Protocol\s+(\d+)", sop)
            if match:
                protocol_number = f"Protocol {match.group(1)}"

        # Reconstruct IncidentReport and assemble AgentResponse manually
        incident_report_obj = IncidentReport(**report)
        final = AgentResponse(
            incident_report=incident_report_obj,
            sop_action=sop if sop else None,
            alert_message=alert_message,
            requires_immediate_action=(report.get("threat_level") == "High"),
            protocol_number=protocol_number
        )
        return {**state, "final_response": final.model_dump()}

    # ── Graph Builder ─────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """
        Assemble and compile the LangGraph StateGraph.
        Returns a compiled graph ready for .invoke().
        """
        g = StateGraph(AgentState)

        # Register nodes
        g.add_node("extract_data",      self._extract_incident_data)
        g.add_node("retrieve_sop",      self._retrieve_sop)
        g.add_node("generate_response", self._generate_response)

        # Edges
        g.add_edge(START, "extract_data")

        # Conditional branching after extraction
        g.add_conditional_edges(
            "extract_data",
            self._route_by_threat,
            {
                "retrieve_sop":      "retrieve_sop",
                "generate_response": "generate_response"
            }
        )

        # SOP retrieval always leads to response generation
        g.add_edge("retrieve_sop",      "generate_response")
        g.add_edge("generate_response", END)

        return g.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, guard_input: str, guard_name: str = "Unknown Guard") -> dict:
        """
        Main entry point. Process a guard's free-form incident report.

        Args:
            guard_input: Raw natural-language report from the guard
            guard_name:  Guard's name/ID to stamp on the report

        Returns:
            AgentResponse as a dict (JSON-serialisable)
        """
        initial_state: AgentState = {
            "raw_input": guard_input,
            "incident_report": None,
            "sop_content": None,
            "final_response": None
        }

        result = self._graph.invoke(initial_state)
        response = result.get("final_response", {})

        # Stamp guard name onto the extracted report
        if response and response.get("incident_report"):
            response["incident_report"]["reported_by"] = guard_name

        return response
