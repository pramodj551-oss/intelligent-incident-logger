"""
rag_pipeline.py — Knowledge Base & RAG Setup
=============================================
Module 3: Embeddings & Vector Store

Loads the SOP_Manual.txt into ChromaDB using OpenAI text-embedding-3-small.
On first run, ingests and indexes all chunks.
On subsequent runs, loads from persistent storage (no re-embedding cost).

Usage:
    retriever = SOPRetriever("data/sop_manual.txt", openai_api_key)
    docs = retriever.retrieve("suspicious unattended bag at gate")
"""

import re
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "ap_securitas_sop_v1"


class SOPRetriever:
    """
    Manages the SOP knowledge base using ChromaDB + OpenAI embeddings.
    Provides semantic search to match live incidents with the right protocol.
    """
    
    embedding_mode = os.getenv("EMBEDDING_MODE", "local").lower()
        
        if embedding_mode == "openai":
            print("[RAG] Using OpenAI Embeddings (text-embedding-3-small)")
            self._embedding_fn = OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
        else:
            print("[RAG] Using Local Embeddings (all-MiniLM-L6-v2)")
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
    """
        Args:
            sop_file_path: Path to SOP_Manual.txt
            openai_api_key: OpenAI API key for text-embedding-3-small
        """
        # OpenAI embedding function (text-embedding-3-small = fast + cost-effective)
        self._embedding_fn = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )

        # Persistent ChromaDB client — stores vectors to disk
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for text
        )

        # Ingest SOP only if the collection is empty (first run)
        if self._collection.count() == 0:
            print(f"[RAG] First run: ingesting {sop_file_path} into ChromaDB...")
            self._ingest_sop(sop_file_path)
            print(f"[RAG] Ingestion complete. {self._collection.count()} chunks indexed.")
        else:
            print(f"[RAG] Loaded existing KB: {self._collection.count()} chunks in {COLLECTION_NAME}.")

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def _ingest_sop(self, file_path: str) -> None:
        """Chunk the SOP manual and embed + store into ChromaDB."""
        sop_path = Path(file_path)
        if not sop_path.exists():
            raise FileNotFoundError(f"SOP manual not found: {file_path}")

        raw_text = sop_path.read_text(encoding="utf-8")

        # Split on protocol boundaries first, then by character count
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=[
                "\nProtocol",   # Split at each new protocol section
                "\n====",       # Split at dividers
                "\n\n",
                "\n",
                ". "
            ]
        )
        chunks = splitter.split_text(raw_text)

        # Add chunks to the collection with metadata
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
        """Extract protocol number from text for metadata tagging."""
        match = re.search(r"Protocol\s+(\d+)", text)
        return f"Protocol {match.group(1)}" if match else "General"

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 3) -> list[str]:
        """
        Semantic search: find the most relevant SOP chunks for an incident query.

        Args:
            query: Natural language description of the incident
            n_results: Number of top chunks to return (default 3)

        Returns:
            List of relevant SOP text chunks
        """
        safe_n = min(n_results, self._collection.count())
        results = self._collection.query(
            query_texts=[query],
            n_results=safe_n
        )
        return results["documents"][0] if results["documents"] else []

    def retrieve_with_scores(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Retrieve with similarity scores and protocol metadata.
        Useful for evaluation and debugging.

        Returns:
            List of dicts: {content, protocol, similarity_score}
        """
        safe_n = min(n_results, self._collection.count())
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
                "similarity_score": round(1 - dist, 4)  # Distance → similarity
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
            "total_chunks": self._collection.count(),
            "collection_name": COLLECTION_NAME,
            "db_path": CHROMA_DB_PATH
        }
