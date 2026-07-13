# -*- coding: utf-8 -*-
"""
src/rag_pipeline.py -- Knowledge Base & RAG Setup
==================================================
Embedding modes (EMBEDDING_MODE env var):
  local  (default) -- all-MiniLM-L6-v2 via sentence-transformers, free, CPU
  openai           -- text-embedding-3-small, best quality, needs OPENAI_API_KEY

Streamlit Cloud notes:
  - ChromaDB writes to /tmp/ (repo mount is read-only)
  - sop_manual.txt is resolved relative to this file, not the caller
  - Fallback SOP is written to /tmp/ if data/ folder is missing from repo
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import List

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -- Paths resolved relative to THIS file (src/rag_pipeline.py) --------------
# Works regardless of cwd: local, Streamlit Cloud, Docker, etc.
_THIS_DIR    = Path(__file__).resolve().parent          # .../src/
_REPO_ROOT   = _THIS_DIR.parent                         # .../incident-logger/
_DATA_DIR    = _REPO_ROOT / "data"                      # .../incident-logger/data/
_SOP_DEFAULT = _DATA_DIR / "sop_manual.txt"

# ChromaDB: /tmp/ is writable on every platform (Streamlit Cloud, local, Docker)
_CHROMA_PATH = Path(tempfile.gettempdir()) / "ap_securitas_chroma"

# Separate collections per embedding model to avoid vector dimension mismatch
_COLLECTION_NAMES = {
    "openai": "ap_securitas_sop_openai",
    "local":  "ap_securitas_sop_local",
}

# -- Minimal fallback SOP (written to /tmp/ when data/ is missing from repo) --
_FALLBACK_SOP = """
AP SECURITAS PVT. LTD. -- STANDARD OPERATING PROCEDURES (EMBEDDED FALLBACK)

Protocol 101: UNATTENDED OR SUSPICIOUS OBJECT
THREAT LEVEL: HIGH
1. Do NOT touch the object.
2. Clear the area -- maintain a 50-metre safety perimeter.
3. Do NOT use mobile phones or radios within 15 metres.
4. Notify Control Room immediately. State location, description, time first seen.
5. Control Room will contact Bomb Detection Squad (BDS).
6. Keep area clear until BDS gives all-clear.
KEY PHRASE: "Code Silver -- [Location] -- Unattended object -- Perimeter secured"

Protocol 102: FIRE OR SMOKE
THREAT LEVEL: HIGH
1. Activate the nearest fire alarm pull station.
2. Call Control Room: "Code Red -- [Location] -- Fire confirmed/suspected"
3. Initiate building evacuation. Do NOT use elevators.
4. Guide occupants to assembly point. Perform headcount.
5. Call Fire Brigade: 101. Do not re-enter until all-clear.

Protocol 103: MEDICAL EMERGENCY
THREAT LEVEL: HIGH
1. Do NOT move the person unless in immediate danger.
2. Call 108 (EMS). State location, casualties, injury type.
3. Notify Control Room: "Code Blue -- [Location] -- Medical emergency"
4. If trained: begin CPR if unresponsive and not breathing.
5. Do NOT give food, water, or medication.

Protocol 104: UNAUTHORIZED ACCESS / INTRUDER
THREAT LEVEL: HIGH (armed) / MEDIUM (unarmed)
1. Do NOT confront an armed intruder. Maintain safe distance.
2. Notify Control Room: "Code Orange -- [Location] -- Intruder -- [Description]"
3. If unarmed and cooperative: challenge politely, escort to reception.
4. If uncooperative: maintain visual contact, relay movement to Control Room.
5. Lock adjacent server rooms and asset areas immediately.

Protocol 105: ROBBERY OR VIOLENT CRIME
THREAT LEVEL: HIGH
1. Guard safety is top priority. Do not resist if weapon shown.
2. Press silent panic alarm if available.
3. Call Control Room and Police (100) once safe.
4. Secure and preserve crime scene. Do not touch anything.

Protocol 106: BOMB THREAT (phone or written)
THREAT LEVEL: HIGH
1. Keep caller talking. Note exact words, voice, background sounds.
2. Signal colleague to call police on another line. Do NOT hang up.
3. Notify Control Room and Senior Security Manager.
4. Evacuate if ordered. Follow Protocol 102 evacuation procedure.

Protocol 107: SUSPICIOUS VEHICLE
THREAT LEVEL: MEDIUM
1. Note make, model, colour, registration, occupants, time observed.
2. Attempt to locate owner via reception/parking management.
3. If abandoned after 30 minutes: notify Control Room.
4. If unusual contents or wires visible: treat as Protocol 101.

Protocol 108: SUSPECTED THEFT
THREAT LEVEL: LOW/MEDIUM
1. Take detailed statement: what stolen, last location, estimated value.
2. Review CCTV for relevant area and time.
3. Notify Security Supervisor and Client HR within 1 hour.

Protocol 109: MISSING PERSON
THREAT LEVEL: MEDIUM
1. Confirm last-seen location and time.
2. Check access control logs for all exits.
3. Systematic search of common areas with at least two guards.
4. If not found within 20 minutes: call Police (100).

Protocol 110: NATURAL DISASTER / EVACUATION
THREAT LEVEL: HIGH
1. EARTHQUAKE: Drop, Cover, Hold On. Evacuate AFTER shaking stops.
2. FLOOD/STORM: Move occupants to upper floors. Block ground-level access.
3. Account for all persons at assembly point.
4. Notify AP Securitas Emergency Coordinator.
"""


def _resolve_sop_path(sop_file_path: str) -> Path:
    """
    Find the SOP file using a priority list of candidate locations.
    Falls back to writing the embedded SOP to /tmp/ if all locations fail.

    Search order:
      1. Caller-provided path (if given and exists)
      2. Relative to this file: .../incident-logger/data/sop_manual.txt
      3. Current working directory: ./data/sop_manual.txt
      4. Streamlit Cloud absolute: /mount/src/data/sop_manual.txt
      5. Write fallback SOP to /tmp/ and return that path
    """
    candidates = []

    # 1. Caller-provided path
    if sop_file_path:
        candidates.append(Path(sop_file_path).resolve())

    # 2. Relative to this source file (most reliable)
    candidates.append(_SOP_DEFAULT)

    # 3. Relative to current working directory
    candidates.append(Path.cwd() / "data" / "sop_manual.txt")

    # 4. Streamlit Cloud explicit paths
    candidates.append(Path("/mount/src/data/sop_manual.txt"))
    candidates.append(Path("/mount/src/incident-logger/data/sop_manual.txt"))

    for path in candidates:
        if path.exists():
            print(f"[RAG] SOP found: {path}")
            return path

    # 5. All locations failed -- write embedded fallback to /tmp/
    fallback = Path(tempfile.gettempdir()) / "ap_securitas_sop_fallback.txt"
    fallback.write_text(_FALLBACK_SOP.strip(), encoding="utf-8")
    print(
        f"[RAG] WARNING: sop_manual.txt not found in any of:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + f"\n[RAG] Using embedded fallback SOP at {fallback}"
    )
    return fallback


# -- Main class ---------------------------------------------------------------

class SOPRetriever:
    """
    Manages the SOP knowledge base using ChromaDB + configurable embeddings.
    Provides semantic search to match live incidents with the right protocol.

    Fully portable:
      - Path resolution anchored to this file, not the caller
      - ChromaDB writes to /tmp/ (works on Streamlit Cloud, Docker, local)
      - Fallback SOP embedded inline if data/ folder missing from repo
    """

    def __init__(self, sop_file_path: str = "", openai_api_key: str = ""):
        """
        Args:
            sop_file_path:  Optional explicit path to SOP .txt file.
                            Leave empty to auto-resolve.
            openai_api_key: OpenAI key. Required only for EMBEDDING_MODE=openai.
        """
        embedding_mode = os.getenv("EMBEDDING_MODE", "local").lower()

        # -- Embedding function -----------------------------------------------
        if embedding_mode == "local":
            print("[RAG] Embeddings: local (all-MiniLM-L6-v2) -- no API cost")
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction
                )
            except (ImportError, Exception):
                raise ImportError(
                    "\nsentence-transformers not installed.\n"
                    "Fix: pip install sentence-transformers\n"
                    "Or:  set EMBEDDING_MODE=openai in .env (needs OPENAI_API_KEY)"
                )
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            print("[RAG] Embeddings: OpenAI text-embedding-3-small")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY required for EMBEDDING_MODE=openai.\n"
                    "Or switch to EMBEDDING_MODE=local (free, no key needed)."
                )
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            self._embedding_fn = OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )

        collection_name = _COLLECTION_NAMES.get(embedding_mode, "ap_securitas_sop_local")

        # -- ChromaDB: always write to /tmp/ ----------------------------------
        _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self._client     = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[RAG] ChromaDB at: {_CHROMA_PATH}")

        # -- Ingest on first run only -----------------------------------------
        if self._collection.count() == 0:
            sop_path = _resolve_sop_path(sop_file_path)
            print(f"[RAG] Indexing {sop_path} ...")
            self._ingest_sop(sop_path)
            print(f"[RAG] Done. {self._collection.count()} chunks indexed.")
        else:
            print(f"[RAG] Loaded: {self._collection.count()} chunks ({collection_name})")

    # -- Ingestion ------------------------------------------------------------

    def _ingest_sop(self, sop_path: Path) -> None:
        """Chunk the SOP file and store embeddings in ChromaDB."""
        raw_text = sop_path.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\nProtocol", "\n====", "\n\n", "\n", ". "]
        )
        chunks = splitter.split_text(raw_text)
        self._collection.add(
            documents=chunks,
            ids=[f"sop_chunk_{i:04d}" for i in range(len(chunks))],
            metadatas=[
                {
                    "chunk_index": i,
                    "source":      str(sop_path.name),
                    "protocol":    self._extract_protocol_tag(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        )

    @staticmethod
    def _extract_protocol_tag(text: str) -> str:
        m = re.search(r"Protocol\s+(\d+)", text)
        return ("Protocol " + m.group(1)) if m else "General"

    # -- Retrieval ------------------------------------------------------------

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        """Semantic search: return top-N most relevant SOP chunks."""
        safe_n = min(n_results, self._collection.count())
        if safe_n == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=safe_n
        )
        return results["documents"][0] if results["documents"] else []

    def retrieve_with_scores(self, query: str, n_results: int = 3) -> List[dict]:
        """Retrieve with cosine similarity scores. Useful for evaluation."""
        safe_n = min(n_results, self._collection.count())
        if safe_n == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=safe_n,
            include=["documents", "metadatas", "distances"]
        )
        if not results["documents"]:
            return []
        return [
            {
                "content":          doc,
                "protocol":         meta.get("protocol", "General"),
                "similarity_score": round(1 - dist, 4)
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def get_stats(self) -> dict:
        return {
            "total_chunks":    self._collection.count(),
            "collection_name": self._collection.name,
            "db_path":         str(_CHROMA_PATH),
            "embedding_mode":  os.getenv("EMBEDDING_MODE", "local"),
}
