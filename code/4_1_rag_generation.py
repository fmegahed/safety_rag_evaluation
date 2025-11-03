"""
Minimal experiment runner for the UR5e RAG system.

How this file operates
----------------------
• Assumes you already ran file 0 (PDF split/crop with MIN_WORDS_FOR_SUBSPLIT=3000) and file 1 (build BM25 pickle and Astra collection). It does not redo preprocessing.
• Imports a small helper from file 2 (`retrieve_and_answer`) to avoid duplicating logic. That helper returns (answer, hits, meta) for a given question and config.
• Sweeps experimental factors (approach, model, max_tokens, reasoning_effort, top_k, A/B answer instructions, A/B few-shot preambles) and evaluates each run.
• If the user omits B-variants (`--answer_instructions_b` or `--fewshot_b`), only A-variants are run.
• Computes automated similarity metrics using LangFair (Cosine, RougeL, Bleu).
• Uses LangSmith prompt packs (LLM-as-judge) for document relevance, faithfulness, helpfulness, and correctness-vs-reference.
• Lets you pick a separate **judge_model** for LLM-as-judge (default: `gpt-5`) independent of generation models.
• Writes a tidy CSV with one row per (question × configuration × approach) including datetime, all factor values, the generated answer, retrieved filenames, metrics, and judge outputs.

Inputs
------
• `--test_csv` a CSV with columns: question, gold_answer
• `--answer_instructions_a`, `--answer_instructions_b` Either a path to a file OR a literal text string (B is optional)
• `--fewshot_a`, `--fewshot_b` Either a path to a file OR a literal text string (B is optional)

Environment
-----------
Loads `.env` automatically. Expected keys:
• OPENAI_API_KEY
• LANGSMITH_API_KEY
• For AstraDB approaches (graph_eager, graph_mmr, vanilla): ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN
• For OpenAI vector store approaches (openai_semantic, openai_keyword): OPENAI_VECTOR_STORE_ID

Usage
-----
# Import and call run_experiment() directly
# Example minimal call at the bottom of this file.
"""

from __future__ import annotations
import time

import itertools
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

import nltk
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load .env so API keys and endpoints are available everywhere
load_dotenv(override=True)

# Import the function from file 2 (which should expose retrieve_and_answer and be import-safe)
# response = requests.get("https://raw.githubusercontent.com/fmegahed/safety_rag_evaluation/refs/heads/main/code/2_rag.py")
namespace = {}
with open("code/2_rag.py") as f:
    exec(f.read(), namespace)
# exec(response.text, namespace)
retrieve_and_answer = namespace["retrieve_and_answer"]

# Provenance value from file 0
MIN_WORDS_FOR_SUBSPLIT = 3000


def now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")


def _read_text(maybe_path: Optional[str]) -> str:
    if maybe_path is None:
        return ""
    p = Path(maybe_path)
    return p.read_text(encoding="utf-8") if p.exists() else maybe_path


async def run_experiment_async(
    *,
    test_csv: Path,
    num_replicates: int,
    approaches: List[str],
    models: List[str],
    max_tokens_list: List[int],
    efforts: List[str],
    topk_list: List[int],
    ans_instr_A: str,
    ans_instr_B: Optional[str],
    fewshot_A: str,
    fewshot_B: Optional[str],
    out_csv: Path,
    max_concurrent: int = 5,
    max_chars_per_content: int = 25_000,
) -> Path:
    """
    Parallel version of run_experiment with concurrency control.
    Processes up to `max_concurrent` questions/configurations simultaneously.
    """
    if num_replicates < 1:
        raise ValueError("num_replicates must be >= 1")

    df = pd.read_csv(test_csv)
    assert {"question", "gold_answer"}.issubset(df.columns), "CSV must include question and gold_answer."

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()

    ai_ids = ["A", "B"] if (ans_instr_B and ans_instr_B.strip()) else ["A"]
    fs_ids = ["A", "B"] if (fewshot_B and fewshot_B.strip()) else ["A"]

    # Prevents main thread blocking
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    total_loop_count = (
        len(approaches)
        * len(models)
        * len(max_tokens_list)
        * len(efforts)
        * len(topk_list)
        * len(ai_ids)
        * len(fs_ids)
        * len(df)                   
        * int(num_replicates)
    )
    print(f"Total permutations: {total_loop_count}")
    # return
    async def process_one(q: str, gold: str, approach: str, model: str, mtoks: int, effort: str,
                          topk: int, ai_id: str, fs_id: str, rep: int):
        """Run one retrieval + eval combo asynchronously."""
        def sync_task():
            ans = ans_instr_A if ai_id == "A" else (ans_instr_B or "")
            fs = fewshot_A if fs_id == "A" else (fewshot_B or "")
            start = time.time()
            start_et = now_et()
            generated, hits, meta = retrieve_and_answer(
                question=q,
                approach=approach,
                model=model,
                effort=effort,
                max_tokens=mtoks,
                top_k=topk,
                max_chars_per_content=max_chars_per_content,
                answer_instructions=ans,
                few_shot_preamble=fs,
            )
            elapsed_gen = time.time() - start
            end_et = now_et()

            row = {
                "time_started": start_et,
                "time_ended": end_et,
                "total_elapsed_time": f"{elapsed_gen:.2f} Seconds",
                "min_words_for_subsplit": MIN_WORDS_FOR_SUBSPLIT,
                "approach": approach,
                "model": model,
                "max_tokens": mtoks,
                "reasoning_effort": effort,
                "top_k": topk,
                "answer_instructions_id": ai_id,
                "few_shot_id": fs_id,
                "replicate": rep,
                "question": q,
                "gold_answer": gold,
                "generated_answer": generated,
                "retrieved_files": ";".join([h.get("filename") or "" for h in hits]),
                **{f"meta_{k}": v for k, v in (meta or {}).items()},
            }
            return row

        return await loop.run_in_executor(executor, sync_task)

    index = 0
    for approach, model, mtoks, effort, topk, ai_id, fs_id in itertools.product(
        approaches, models, max_tokens_list, efforts, topk_list, ai_ids, fs_ids,
    ):
        tasks = []
        for _, r in df.iterrows():
            q = str(r["question"]) if pd.notna(r["question"]) else ""
            gold = str(r["gold_answer"]) if pd.notna(r["gold_answer"]) else None
            for rep in range(1, int(num_replicates) + 1):
                tasks.append(process_one(q, gold, approach, model, mtoks, effort, topk, ai_id, fs_id, rep))

        # Run in batches of max_concurrent
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            results = await asyncio.gather(*batch)
            pd.DataFrame(results).to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False
            index += len(batch)
            print(f"Completed {index} runs for approach={approach}, model={model}")

    print(f"\nAll results written to {out_csv}")
    return out_csv

asyncio.run(run_experiment_async(
        test_csv=Path("data/sample_test_questions.csv"),
        num_replicates=1,
        approaches=["openai_keyword"],
        models=["gpt-5-mini-2025-08-07"],
        max_tokens_list=[500, 750],
        efforts=["low"],
        topk_list=[3, 5],
        ans_instr_A=_read_text("prompts/ans_instr_A.txt"),
        ans_instr_B=None,
        fewshot_A=_read_text("prompts/fewshot_A.txt"),
        fewshot_B=None,
        out_csv=Path("results/rag_generation.csv"),
        max_concurrent=5,
    ))