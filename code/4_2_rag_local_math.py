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

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Inspired by https://python.langchain.com/docs/integrations/providers/langfair/
# Common metrics reported in either `CounterfactualMetrics` or `AutoEval` 
def langfair_metrics(pred: str, ref: str) -> Dict[str, float | None]:
    """Compute similarity metrics between prediction and reference texts.
       Inspired by the LangFair library but computed by hand as the library
       produced errors.
    """
    
    # BLEU score
    smoothing = SmoothingFunction()
    reference_tokens = nltk.word_tokenize(ref.lower())
    prediction_tokens = nltk.word_tokenize(pred.lower())
    bleu = sentence_bleu(
        [reference_tokens], 
        prediction_tokens,
        smoothing_function=smoothing.method1
    )
    
    # ROUGE-L score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ref, pred)
    rougeL = rouge_scores['rougeL'].fmeasure
    
    # Cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([ref, pred])
    cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return {
        "cosine": float(cosine) if cosine is not None else None,
        "rougeL": float(rougeL) if rougeL is not None else None,
        "bleu": float(bleu) if bleu is not None else None,
    }


import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from datetime import datetime

async def run_local_math(
    *,
    q_a_csv: Path,
    out_csv: Path | None = None,
    max_concurrent: int = 8,
) -> Path:
    """
    Compute LangFair-like metrics (Cosine, ROUGE-L, BLEU) in parallel for each row of a CSV,
    append the metrics as new columns, and write to a new CSV file.

    Args:
        math_csv: Input CSV containing at least 'generated_answer' and 'gold_answer' columns.
        out_csv: Output CSV path. If None, saves as '<input>_with_metrics.csv' next to input.
        max_concurrent: Number of rows to process concurrently.

    Returns:
        Path to the newly written CSV file.
    """
    df = pd.read_csv(q_a_csv)
    if not {"generated_answer", "gold_answer"}.issubset(df.columns):
        raise ValueError("CSV must contain 'generated_answer' and 'gold_answer' columns.")

    if out_csv is None:
        out_csv = q_a_csv.with_name(q_a_csv.stem + "_with_metrics.csv")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def process_row(generated: str, gold: str):
        """Compute metrics for one row asynchronously."""
        def sync_task():
            mets = langfair_metrics(generated or "", gold or "")
            return {
                "cosine": mets.get("cosine"),
                "rougeL": mets.get("rougeL"),
                "bleu": mets.get("bleu"),
                "q": gold,
            }
        return await loop.run_in_executor(executor, sync_task)

    print(f"Total rows to process: {len(df)}")
    results = []
    tasks = []

    for idx, row in df.iterrows():
        tasks.append(process_row(row.get("generated_answer", ""), row.get("gold_answer", "")))

        # Run in batches of max_concurrent
        if len(tasks) >= max_concurrent:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            tasks = []
            print(f"Processed {len(results)}/{len(df)} rows...")

    # Handle remaining tasks
    if tasks:
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    # Merge metrics back into original DataFrame
    metrics_df = pd.DataFrame(results)
    merged_df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

    merged_df.to_csv(out_csv, index=False)
    print(f"All {len(df)} rows processed and saved to {out_csv.resolve()}")
    return out_csv


asyncio.run(run_local_math(
    q_a_csv=Path("results/rag_generation.csv"),
    max_concurrent=999,
))