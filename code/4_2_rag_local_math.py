"""
Calculates the langfair metrics from an existing csv and exports to a new one
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd

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
        q_a_csv: Input CSV containing at least 'generated_answer' and 'gold_answer' columns.
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
    q_a_csv=Path("results/gold_set_part_1.csv"),
    max_concurrent=500,
))