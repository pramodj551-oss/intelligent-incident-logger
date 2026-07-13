"""
eval/eval_harness.py — 10-Prompt Live Evaluation Harness
=========================================================
Module 3: Evaluation

Tests the full agent pipeline against 10 real-world security incident scenarios.
Checks:
  - Threat level classification accuracy
  - SOP retrieval coverage (for HIGH-threat incidents)
  - Response quality (alert message generated)
  - Protocol number extraction

Run:
    cd incident-logger
    python eval/eval_harness.py
"""

import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agent import IncidentAgent
from src.rag_pipeline import SOPRetriever


# ── Test Case Definition ──────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: int
    description: str
    guard_input: str
    expected_threat_level: str        # "Low" | "Medium" | "High"
    expected_sop_triggered: bool      # True = SOP retrieval expected
    expected_protocol_hint: Optional[str] = None  # e.g., "Protocol 101"


# ── 10 Test Cases ─────────────────────────────────────────────────────────────

EVAL_CASES = [
    EvalCase(
        id=1,
        description="Suspicious unattended bag at main gate",
        guard_input=(
            "Gate No. 2 ke paas ek kali bag lagbhag 30 minutes se rakhi hai. "
            "Koi malik nahi aaya. Bag se halki tikhi smell aa rahi hai. "
            "Maine kisi ko paas nahi jaane diya."
        ),
        expected_threat_level="High",
        expected_sop_triggered=True,
        expected_protocol_hint="Protocol 101"
    ),
    EvalCase(
        id=2,
        description="Smoke detected in server room",
        guard_input=(
            "Block B 3rd floor server room mein smoke alarm baj raha hai. "
            "Maine andar dekha toh halka sa dhuan dikh raha hai near one of the racks. "
            "Fire nahi dikh raha abhi. What to do?"
        ),
        expected_threat_level="High",
        expected_sop_triggered=True,
        expected_protocol_hint="Protocol 102"
    ),
    EvalCase(
        id=3,
        description="Employee found unconscious",
        guard_input=(
            "Gents washroom Level 2 mein ek employee floor par pade hain. "
            "Unhe hilaya toh koi response nahi mila. Unki saanson ki awaaz aa rahi hai "
            "lekin eyes closed hain. Please help karo."
        ),
        expected_threat_level="High",
        expected_sop_triggered=True,
        expected_protocol_hint="Protocol 103"
    ),
    EvalCase(
        id=4,
        description="Intruder found in restricted server area",
        guard_input=(
            "Main server room mein ek unknown person mila jo wahan kaam nahi karta. "
            "Usne access card swipe kiya par alarm bajne laga. Wo bhaagne ki koshish kar raha tha. "
            "Maine usse rok liya hai reception mein. Wo zyada cooperative nahi hai."
        ),
        expected_threat_level="High",
        expected_sop_triggered=True,
        expected_protocol_hint="Protocol 104"
    ),
    EvalCase(
        id=5,
        description="Bomb threat phone call received at reception",
        guard_input=(
            "Reception pe ek anonymous call aaya. Caller ne kaha ki building mein "
            "'ek bomb hai aur aaj shaam 5 baje tak sab khatam ho jaega'. "
            "Call 2 minute mein cut ho gayi. Abhi kya karna chahiye?"
        ),
        expected_threat_level="High",
        expected_sop_triggered=True,
        expected_protocol_hint="Protocol 106"
    ),
    EvalCase(
        id=6,
        description="Suspicious vehicle parked at restricted zone for over 2 hours",
        guard_input=(
            "North parking ke restricted zone mein ek white SUV without parking permit "
            "2 ghante se khadi hai. Owner trace nahi ho raha. "
            "Vehicle ki dicky thodi khuli hai."
        ),
        expected_threat_level="Medium",
        expected_sop_triggered=False,
        expected_protocol_hint=None
    ),
    EvalCase(
        id=7,
        description="Unknown male loitering near ATM area",
        guard_input=(
            "Lobby ATM ke paas ek unknown male last 45 minutes se ghoom raha hai. "
            "Usne do employees ke paas jaake kuch poochha jo unkool tha. "
            "Uske paas koi visitor pass nahi hai."
        ),
        expected_threat_level="Medium",
        expected_sop_triggered=False,
        expected_protocol_hint=None
    ),
    EvalCase(
        id=8,
        description="Minor scuffle between two contractors",
        guard_input=(
            "Cafeteria mein do contractor workers ke beech thoda argument hua. "
            "Maine unhe alag kar diya. Koi physical injury nahi hai. "
            "Dono ab shant hain."
        ),
        expected_threat_level="Medium",
        expected_sop_triggered=False,
        expected_protocol_hint=None
    ),
    EvalCase(
        id=9,
        description="Lost laptop bag found near main entrance",
        guard_input=(
            "Main entrance ke paas ek Dell laptop bag mili hai. "
            "Andar visiting card tha — naam: Rajesh Sharma, ABC Tech. "
            "Visitor visit pe the aaj subah. Kya kare?"
        ),
        expected_threat_level="Low",
        expected_sop_triggered=False,
        expected_protocol_hint=None
    ),
    EvalCase(
        id=10,
        description="Visitor overstaying after business hours",
        guard_input=(
            "Ek visitor 6:30 PM tak company mein hai. Office 6 PM pe band ho gayi. "
            "Visitor keh raha hai uski meeting chal rahi hai par koi employee "
            "confirm nahi kar raha. Temporary badge hai uske paas."
        ),
        expected_threat_level="Low",
        expected_sop_triggered=False,
        expected_protocol_hint=None
    ),
]


# ── Evaluation Logic ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: int
    description: str
    expected_threat: str
    actual_threat: str
    threat_correct: bool
    sop_triggered_expected: bool
    sop_triggered_actual: bool
    sop_correct: bool
    protocol_extracted: Optional[str]
    alert_generated: bool
    overall_pass: bool
    error: Optional[str] = None


def run_evaluation(agent: IncidentAgent) -> list[EvalResult]:
    """Run all 10 test cases and return evaluation results."""
    results = []

    for case in EVAL_CASES:
        print(f"\n[EVAL] Case {case.id:02d}: {case.description}")
        print(f"       Input: {case.guard_input[:80]}...")

        try:
            response = agent.process(case.guard_input, f"EvalGuard-{case.id}")

            if not response:
                raise ValueError("Agent returned empty response")

            report       = response.get("incident_report", {})
            actual_threat = report.get("threat_level", "Unknown")
            sop_content  = response.get("sop_action")
            alert_msg    = response.get("alert_message", "")
            protocol_num = response.get("protocol_number")

            sop_triggered = sop_content is not None and len(sop_content.strip()) > 0
            threat_ok     = actual_threat == case.expected_threat_level
            sop_ok        = sop_triggered == case.expected_sop_triggered
            alert_ok      = len(alert_msg.strip()) > 20
            overall_pass  = threat_ok and sop_ok and alert_ok

            icon = "✅" if overall_pass else "❌"
            print(f"       {icon} Threat: {actual_threat} (expected {case.expected_threat_level}) | "
                  f"SOP: {'YES' if sop_triggered else 'NO'} | Protocol: {protocol_num or '—'}")

            results.append(EvalResult(
                case_id=case.id,
                description=case.description,
                expected_threat=case.expected_threat_level,
                actual_threat=actual_threat,
                threat_correct=threat_ok,
                sop_triggered_expected=case.expected_sop_triggered,
                sop_triggered_actual=sop_triggered,
                sop_correct=sop_ok,
                protocol_extracted=protocol_num,
                alert_generated=alert_ok,
                overall_pass=overall_pass
            ))

        except Exception as e:
            print(f"       ❌ ERROR: {e}")
            results.append(EvalResult(
                case_id=case.id,
                description=case.description,
                expected_threat=case.expected_threat_level,
                actual_threat="ERROR",
                threat_correct=False,
                sop_triggered_expected=case.expected_sop_triggered,
                sop_triggered_actual=False,
                sop_correct=False,
                protocol_extracted=None,
                alert_generated=False,
                overall_pass=False,
                error=str(e)
            ))

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print a clean evaluation summary to the console."""
    total        = len(results)
    passed       = sum(1 for r in results if r.overall_pass)
    threat_acc   = sum(1 for r in results if r.threat_correct) / total * 100
    sop_acc      = sum(1 for r in results if r.sop_correct)    / total * 100
    alert_acc    = sum(1 for r in results if r.alert_generated) / total * 100

    print("\n" + "=" * 60)
    print("  AP SECURITAS — EVALUATION HARNESS SUMMARY")
    print("=" * 60)
    print(f"  Total Cases      : {total}")
    print(f"  Overall Pass     : {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"  Threat Level Acc : {threat_acc:.0f}%")
    print(f"  SOP Trigger Acc  : {sop_acc:.0f}%")
    print(f"  Alert Generated  : {alert_acc:.0f}%")
    print("-" * 60)
    print(f"  {'ID':<4} {'Pass':<6} {'Expected':<10} {'Actual':<10} {'Protocol'}")
    print("-" * 60)
    for r in results:
        icon     = "✅" if r.overall_pass else "❌"
        protocol = r.protocol_extracted or "—"
        print(f"  {r.case_id:<4} {icon:<6} {r.expected_threat:<10} {r.actual_threat:<10} {protocol}")
    print("=" * 60)

    # Save to JSON
    output_path = "eval/eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env")
        sys.exit(1)

    print("=" * 60)
    print("  AP SECURITAS — AI INCIDENT LOGGER EVALUATION HARNESS")
    print("  Running 10 incident scenarios...")
    print("=" * 60)

    print("\n[INIT] Loading SOPRetriever (ChromaDB)...")
    retriever = SOPRetriever(openai_api_key=openai_key)  # sop_file_path auto-resolved

    print("[INIT] Loading IncidentAgent (LangGraph)...")
    agent = IncidentAgent(openai_key, retriever)

    results = run_evaluation(agent)
    print_summary(results)
