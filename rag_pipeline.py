# -*- coding: utf-8 -*-
"""
rag_pipeline.py -- Knowledge Base & RAG Setup
=============================================
Module 3: Embeddings & Vector Store

Supports two embedding modes via EMBEDDING_MODE env variable:
  - openai (default): text-embedding-3-small (best quality, costs ~$0.0001/1k tokens)
  - local:            all-MiniLM-L6-v2 via sentence-transformers (free, runs on CPU)

On first run, ingests and indexes all SOP chunks.
On subsequent runs, loads from persistent ChromaDB storage (no re-embedding cost).

Usage:
    # OpenAI mode (default)
    retriever = SOPRetriever("data/sop_manual.txt", openai_api_key="sk-...")

    # Local mode (no API key needed for embeddings)
    # Set EMBEDDING_MODE=local in .env
    retriever = SOPRetriever("data/sop_manual.txt", openai_api_key="")
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHROMA_DB_PATH  = "./chroma_db"

# Use different collection names per embedding model to avoid dimension mismatch
# when switching EMBEDDING_MODE between runs.
COLLECTION_NAMES = {
    "openai": "ap_securitas_sop_openai",
    "local":  "ap_securitas_sop_local",
}


class SOPRetriever:
    """
    Manages the SOP knowledge base using ChromaDB + configurable embeddings.
    Provides semantic search to match live incidents with the right protocol.
    """

    def __init__(self, sop_file_path: str, openai_api_key: str = ""):
        """
        Args:
            sop_file_path:  Path to SOP_Manual.txt
            openai_api_key: OpenAI API key. Required only when EMBEDDING_MODE=openai.
        """
        embedding_mode = os.getenv("EMBEDDING_MODE", "openai").lower()

        # -- Choose embedding function based on EMBEDDING_MODE ----------------
        if embedding_mode == "local":
            print("[RAG] Mode: local (all-MiniLM-L6-v2) -- no API cost")
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction
                )
            except (ImportError, Exception):
                raise ImportError(
                    "\n\nsentence-transformers is required for EMBEDDING_MODE=local.\n"
                    "Fix (choose one):\n"
                    "  pip install sentence-transformers\n"
                    "  -- or --\n"
                    "  Set EMBEDDING_MODE=openai in .env (needs OPENAI_API_KEY)\n"
                )
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            print("[RAG] Mode: openai (text-embedding-3-small)")
            if not openai_api_key:
                raise ValueError(
                    "openai_api_key is required when EMBEDDING_MODE=openai. "
                    "Set EMBEDDING_MODE=local in .env to use free local embeddings."
                )
            self._embedding_fn = OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )

        collection_name = COLLECTION_NAMES.get(embedding_mode, COLLECTION_NAMES["openai"])

        # -- Persistent ChromaDB client (stores vectors to disk) --------------
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # -- Ingest SOP on first run only -------------------------------------
        if self._collection.count() == 0:
            print(f"[RAG] First run: ingesting {sop_file_path}...")
            self._ingest_sop(sop_file_path)
            print(f"[RAG] Done. {self._collection.count()} chunks indexed.")
        else:
            print(f"[RAG] Loaded KB: {self._collection.count()} chunks ({collection_name})")

    # -- Ingestion ------------------------------------------------------------

    def _ingest_sop(self, file_path: str) -> None:
        """Chunk the SOP manual and embed + store into ChromaDB."""
        sop_path = Path(file_path)
        if not sop_path.exists():
            raise FileNotFoundError(f"SOP manual not found: {file_path}")

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
                    "source": "SOP_Manual_v2.1",
                    "protocol": self._extract_protocol_tag(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        )

    @staticmethod
    def _extract_protocol_tag(text: str) -> str:
        """Extract protocol number from a text chunk for metadata tagging."""
        match = re.search(r"Protocol\s+(\d+)", text)
        return ("Protocol " + match.group(1)) if match else "General"

    # -- Retrieval ------------------------------------------------------------

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        """
        Semantic search: find the most relevant SOP chunks for a query.

        Args:
            query:     Natural language description of the incident
            n_results: Number of top chunks to return (default 3)

        Returns:
            List of relevant SOP text chunks
        """
        safe_n = min(n_results, self._collection.count())
        if safe_n == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=safe_n
        )
        return results["documents"][0] if results["documents"] else []

    def retrieve_with_scores(self, query: str, n_results: int = 3) -> List[dict]:
        """
        Retrieve with similarity scores and protocol metadata.
        Useful for evaluation and debugging.
        """
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
                "content": doc,
                "protocol": meta.get("protocol", "General"),
                "similarity_score": round(1 - dist, 4)
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def get_stats(self) -> dict:
        """Return KB statistics for diagnostics."""
        return {
            "total_chunks":    self._collection.count(),
            "collection_name": self._collection.name,
            "db_path":         CHROMA_DB_PATH,
            "embedding_mode":  os.getenv("EMBEDDING_MODE", "openai")
        }
