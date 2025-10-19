"""
Unified RAG router for querying a single retrieval approach at a time.

Assumes prior execution of:
    0_ur5e_multiple_pdfs.py
    1_preprocess.py

Usage:
    run_rag_router(
        question="What safety checks should I perform before jogging a UR5e robot?",
        approach="openai_semantic,
        csv_path="results/rag_results.csv",
        append=True
    )
"""

from __future__ import annotations
import html
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.retrievers import BM25Retriever as LC_BM25Retriever
from langchain_openai import OpenAIEmbeddings as LC_OpenAIEmbeddings

from langchain_astradb import AstraDBVectorStore
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager, Mmr


# -------------------------------------------------------------------
# Config (identical to 1_preprocess.py)
# -------------------------------------------------------------------
CONFIG = {
    "model": "gpt-5-nano-2025-08-07",
    "max_tokens": 5000,
    "reasoning_effort": "low",   # options: "low", "medium", "high"
    "embed_model": "text-embedding-3-small",
    "top_k": 10,
    "max_chars_per_content": 25000
}

STORE_DIR = Path("retrieval_store")
BM25_PKL = STORE_DIR / "bm25" / "bm25_retriever.pkl"

# Load environment variables
load_dotenv(override=True)
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
COLLECTION_NAME = "ur5_manual"


# -------------------------------------------------------------------
# Prompt defaults
# -------------------------------------------------------------------
DEFAULT_ANSWER_INSTRUCTIONS = (
    "Answer concisely using only the provided sources. Put safety first. "
    "Begin with a 1–2 sentence safety-forward answer. Then list up to 10 bullets that cite filenames. "
    "Do not invent steps not present in the sources. Do not include your internal reasoning."
)

DEFAULT_FEW_SHOT_PREAMBLE = """
You assist with safe operation of workshop equipment (lathe, mill, CNC, cobots).
Prioritize PPE, lockout/tagout, pinch-point hazards, safe setup and teardown, and verification checks.

Example
Sources: <sources>
  <result filename='lathe_safety.md'><content>Remove the chuck key. Wear eye protection...</content></result>
</sources>
Query: 'How do I safely face a part on the manual lathe?'
Assistant:
Short answer: Verify PPE, secure the work, set a safe speed, and keep clear of the rotating chuck.
• Wear ANSI Z87.1 eye protection and remove the chuck key before start [lathe_safety.md]
• Set spindle speed for diameter, keep hands and rags away from rotating stock [lathe_safety.md]
• Stand out of the line of fire and be ready to hit e-stop [lathe_safety.md]
"""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _format_sources_xml(hits: List[Dict[str, Any]], max_chars_per_content: int) -> str:
    """Format retrieved documents as <sources> XML for model input."""
    parts: List[str] = []
    for h in hits:
        filename = h.get("filename") or ""
        file_id = h.get("file_id") or ""
        score = h.get("score")
        attrs = {"file_id": file_id, "filename": filename}
        if score is not None:
            try:
                attrs["score"] = f"{float(score):.4f}"
            except Exception:
                attrs["score"] = str(score)
        open_tag = "<result " + " ".join(
            f"{k}='{html.escape(str(v), quote=True)}'" for k, v in attrs.items()
        ) + ">"
        body = html.escape((h.get("text") or "")[:max_chars_per_content])
        parts.append(open_tag + f"<content>{body}</content></result>")
    return "<sources>" + "".join(parts) + "</sources>"


def _ask_with_sources(
    client: OpenAI,
    *,
    question: str,
    hits: List[Dict[str, Any]],
    model: str,
    effort: str,
    max_tokens: int,
    answer_instructions: str,
    few_shot_preamble: str,
    max_chars_per_content: int,
) -> Tuple[str, Dict[str, Any]]:
    """Query the model with consistent prompt structure and source context."""
    sources_xml = _format_sources_xml(hits, max_chars_per_content)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": few_shot_preamble.strip()},
            {"role": "user", "content": answer_instructions.strip()},
            {"role": "user", "content": f"Sources: {sources_xml}\n\nQuery: '{question}'"},
        ],
        reasoning={"effort": effort},
        max_output_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    meta = {
        "resp_id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "status": getattr(resp, "status", None),
        "created": getattr(resp, "created_at", None),
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }
    return (getattr(resp, "output_text", "") or ""), meta


# -------------------------------------------------------------------
# AstraDB Loader
# -------------------------------------------------------------------
def _load_astradb_vector_store() -> AstraDBVectorStore:
    """Load the AstraDB vector store."""
    embeddings = LC_OpenAIEmbeddings(model=CONFIG['embed_model'])
    
    vector_store = AstraDBVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
    return vector_store



# -------------------------------------------------------------------
# Retrieval implementations
# -------------------------------------------------------------------
def _retrieve_openai_file_search(
    client: OpenAI,
    *,
    question: str,
    top_k: int,
    rewrite_query: bool,
) -> List[Dict[str, Any]]:
    """Query an OpenAI vector store."""
    if not VECTOR_STORE_ID:
        raise ValueError("OPENAI_VECTOR_STORE_ID is not set in the .env file.")
    res = client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=question,
        rewrite_query=rewrite_query,
        max_num_results=top_k,
    )
    hits: List[Dict[str, Any]] = []
    for r in getattr(res, "data", []) or []:
        texts = []
        for c in getattr(r, "content", []) or []:
            if getattr(c, "type", None) == "text":
                texts.append(getattr(c, "text", "") or "")
        hits.append(
            {
                "filename": getattr(r, "filename", "") or "",
                "file_id": getattr(r, "file_id", None),
                "score": getattr(r, "score", None),
                "text": " ".join(texts),
            }
        )
    return hits


def _retrieve_langchain_bm25(*, question: str, top_k: int) -> List[Dict[str, Any]]:
    """Query the BM25 retriever built by 1_preprocess.py."""
    if not BM25_PKL.exists():
        raise FileNotFoundError(f"BM25 retriever not found at {BM25_PKL}. Run 1_preprocess.py first.")
    with BM25_PKL.open("rb") as f:
        retriever: LC_BM25Retriever = pickle.load(f)
    results = retriever.invoke(question, k=top_k)
    return [
        {
            "filename": (d.metadata or {}).get("source") or (d.metadata or {}).get("filename", ""),
            "file_id": None,
            "score": getattr(d, "score", None),
            "text": d.page_content or "",
        }
        for d in results
    ]


def _retrieve_graph_retriever(*, question: str, top_k: int, strategy: str) -> List[Dict[str, Any]]:
    """Graph RAG over AstraDB using GraphRetriever with Eager or Mmr traversal."""
    vector_store = _load_astradb_vector_store()
    edges = [("source", "source")]
    
    strategy_up = strategy.strip().upper()
    if strategy_up == "EAGER":
        strat = Eager(k=top_k, start_k=1, max_depth=2)
    elif strategy_up == "MMR":
        strat = Mmr(k=top_k, start_k=2, max_depth=2)
    else:
        raise ValueError("strategy must be 'EAGER' or 'MMR'")

    retriever = GraphRetriever(store=vector_store, edges=edges, strategy=strat)
    docs = retriever.invoke(question)

    hits: List[Dict[str, Any]] = []
    for d in docs[:top_k]:
        meta = getattr(d, "metadata", {}) or {}
        hits.append({
            "filename": meta.get("source") or meta.get("filename", "") or "",
            "file_id": None,
            "score": getattr(d, "score", None),
            "text": getattr(d, "page_content", "") or "",
        })
    return hits


def _retrieve_vanilla_astradb(*, question: str, top_k: int) -> List[Dict[str, Any]]:
    """Vanilla similarity search over the AstraDB vector store."""
    vector_store = _load_astradb_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    
    hits: List[Dict[str, Any]] = []
    for d in docs[:top_k]:
        meta = getattr(d, "metadata", {}) or {}
        hits.append({
            "filename": meta.get("source") or meta.get("filename", "") or "",
            "file_id": None,
            "score": getattr(d, "score", None),
            "text": getattr(d, "page_content", "") or "",
        })
    return hits


# -------------------------------------------------------------------
# Unified router
# -------------------------------------------------------------------
def run_rag_router(
    question: str,
    *,
    approach: str,
    csv_path: str = "results/rag_results.csv",
    append: bool = True,
    model: str = CONFIG["model"],
    effort: str = CONFIG["reasoning_effort"],
    max_tokens: int = CONFIG["max_tokens"],
    top_k: int = CONFIG["top_k"],
    max_chars_per_content: int = CONFIG["max_chars_per_content"],
    answer_instructions: str = DEFAULT_ANSWER_INSTRUCTIONS,
    few_shot_preamble: str = DEFAULT_FEW_SHOT_PREAMBLE,
) -> None:
    """
    Run one retrieval method, display the model’s answer,
    and save results to a CSV file (append by default).
    """
    client = OpenAI()
    approach = approach.lower().strip()

    # Retrieve based on approach type
    if approach == "openai_semantic":
        hits = _retrieve_openai_file_search(client, question=question, top_k=top_k, rewrite_query=True)
    elif approach == "openai_keyword":
        hits = _retrieve_openai_file_search(client, question=question, top_k=top_k, rewrite_query=False)
    elif approach == "lc_bm25":
        hits = _retrieve_langchain_bm25(question=question, top_k=top_k)
    elif approach == "graph_eager":
        hits = _retrieve_graph_retriever(question=question, top_k=top_k, strategy="EAGER")
    elif approach == "graph_mmr":
        hits = _retrieve_graph_retriever(question=question, top_k=top_k, strategy="MMR")
    elif approach == "vanilla":
        hits = _retrieve_vanilla_astradb(question=question, top_k=top_k)
    else:
        raise ValueError(f"Unknown approach '{approach}'.")

    # Ask the model
    answer, meta = _ask_with_sources(
        client,
        question=question,
        hits=hits,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        answer_instructions=answer_instructions,
        few_shot_preamble=few_shot_preamble,
        max_chars_per_content=max_chars_per_content,
    )

    # Display answer
    print(f"\n--- {approach.upper()} ANSWER ---\n")
    print(answer.strip())
    print("\n-----------------------------\n")

    # Save CSV
    rows = [
        {
            "time": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "question": question,
            "approach": approach,
            "filename": h.get("filename"),
            "file_id": h.get("file_id"),
            "score": h.get("score"),
            "snippet": (h.get("text") or "")[:200].replace("\n", " "),
            "answer": answer,
            **meta,
        }
        for h in hits
    ]
    df = pd.DataFrame(rows)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if append and out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"✅ Results saved to {out_path.resolve()}")


run_rag_router(
    question="What are the safety functions for a UR5e?",
    # approach can be: "openai_semantic", "openai_keyword", "lc_bm25", "graph_mmr", "graph_eager", "vanilla"
    approach="lc_bm25", 
    csv_path="results/rag_results.csv",
    append=True,
)
