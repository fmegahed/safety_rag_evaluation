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
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from langsmith import traceable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re

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

def extract_boolean_answer(text: str, prefix_word: str) -> str:
    if text is None:
        return None
    match = re.search(rf"((?<={prefix_word}:\s)|(?<={prefix_word}:))(True|False)", text)
    if match is None or match.group(0) is None:
        return None
    return match.group(0)

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


# Prompts are based on https://docs.langchain.com/langsmith/evaluate-rag-tutorial#heres-a-consolidated-script-with-all-the-above-code
# Accessed on Oct 20, 2025
@traceable(name="judge_with_langsmith")
def judge_with_langsmith(
    *,
    question: str,
    answer: str,
    gold: Optional[str],
    contexts: str,
    judge_model: str = "gpt-5",
) -> Dict[str, Any]:
    """Run LLM-as-judge prompts using a specified model.
    Returns a dict of raw model outputs for the four judgments.
    """
    
    llm = ChatOpenAI(model=judge_model, temperature=0)

    # Document relevance
    retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
    
    doc_rel_prompt = ChatPromptTemplate.from_messages([
        ("system", retrieval_relevance_instructions),
        ("user", "FACTS: {contexts}\nQUESTION: {question}")
    ])
    doc_rel_chain = doc_rel_prompt | llm | StrOutputParser()

    # Faithfulness (groundedness/hallucination check)
    grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
    
    faithful_prompt = ChatPromptTemplate.from_messages([
        ("system", grounded_instructions),
        ("user", "FACTS: {contexts}\nSTUDENT ANSWER: {answer}")
    ])
    faithful_chain = faithful_prompt | llm | StrOutputParser()

    # Helpfulness (relevance)
    relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
    
    helpful_prompt = ChatPromptTemplate.from_messages([
        ("system", relevance_instructions),
        ("user", "QUESTION: {question}\nSTUDENT ANSWER: {answer}")
    ])
    helpful_chain = helpful_prompt | llm | StrOutputParser()

    chains: Dict[str, Any] = {
        "doc_relevance": doc_rel_chain,
        "faithfulness": faithful_chain,
        "helpfulness": helpful_chain,
    }

    # Correctness vs reference
    if gold is not None and len(gold.strip()) > 0:
        correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""
        
        correct_prompt = ChatPromptTemplate.from_messages([
            ("system", correctness_instructions),
            ("user", "QUESTION: {question}\nGROUND TRUTH ANSWER: {reference}\nSTUDENT ANSWER: {answer}")
        ])
        chains["correctness_vs_ref"] = correct_prompt | llm | StrOutputParser()

    parallel = RunnableParallel(**chains)
    inputs = {
        "question": question,
        "answer": answer,
        "reference": gold,
        "contexts": contexts,
    }
    results = parallel.invoke(inputs)

    if "correctness_vs_ref" not in results:
        results["correctness_vs_ref"] = None
    
    return results

def run_experiment(
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
    max_chars_per_content: int = 25_000,
    judge_model: str = "gpt-5",
) -> Path:
    """
    Run a full factorial experiment with n replicates and write results to CSV incrementally.

    Each unique configuration in the Cartesian product of:
      - approach
      - model
      - max_tokens
      - reasoning_effort
      - top_k
      - answer_instructions_id in {A, B if provided}
      - few_shot_id in {A, B if provided}
    is evaluated for every row in `test_csv`.

    The argument `num_replicates` controls how many times each configuration is repeated.
    Set this greater than 1 when the LLM or retrieval pipeline is nondeterministic to allow
    averaging and variance estimation across runs. A `replicate` column is included in the
    output to differentiate runs, starting at 1.

    Args:
        test_csv: Path to a CSV with at least columns `question` and `gold_answer`.
        num_replicates: Number of repeated runs for every configuration and question.
            Must be >= 1. The output CSV will contain one row per replicate.
        approaches: Retrieval or orchestration approaches to test.
        models: Generation model identifiers to test.
        max_tokens_list: Max generation tokens per configuration.
        efforts: Reasoning effort settings to try, for example ["minimal", "low", "medium", "high"].
        topk_list: Values for top-k retrieval to try.
        ans_instr_A: Required answer instruction template for variant A.
        ans_instr_B: Optional answer instruction template for variant B. If empty or None, only A is used.
        fewshot_A: Required few-shot preamble for variant A.
        fewshot_B: Optional few-shot preamble for variant B. If empty or None, only A is used.
        out_csv: Destination CSV. File is created if missing and appended to otherwise.
        max_chars_per_content: Truncation limit for retrieved content fed to the model.
        judge_model: LLM used for evaluation of doc relevance, faithfulness, helpfulness, and correctness.
            This is independent from the generation models in `models`.

    Behavior:
        - The function streams results row by row to `out_csv`. Headers are written only once.
        - For each replicate, a fresh call to `retrieve_and_answer` and `judge_with_langsmith` is made.
        - The output CSV includes a `replicate` column to disambiguate repeated runs.

    Returns:
        Path to `out_csv`.

    Output columns (in addition to metadata from `meta_*`):
        datetime, min_words_for_subsplit, approach, model, max_tokens, reasoning_effort,
        top_k, answer_instructions_id, few_shot_id, replicate, question, gold_answer,
        generated_answer, retrieved_files, cosine, rougeL, bleu, judge_doc_relevance,
        judge_faithfulness, judge_helpfulness, judge_correctness_vs_ref.
    """
    if num_replicates < 1:
        raise ValueError("num_replicates must be >= 1")

    df = pd.read_csv(test_csv)
    assert {"question", "gold_answer"}.issubset(df.columns), "CSV must include question and gold_answer."

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Determine which A/B variants to run based on user inputs
    ai_ids = ["A", "B"] if (ans_instr_B and ans_instr_B.strip()) else ["A"]
    fs_ids = ["A", "B"] if (fewshot_B and fewshot_B.strip()) else ["A"]

    # Track if we need to write headers
    write_header = not out_csv.exists()
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
    index = 0
    
    print("Loop dimensions:")
    print(f"approaches      = {len(approaches)}")
    print(f"models          = {len(models)}")
    print(f"max_tokens_list = {len(max_tokens_list)}")
    print(f"efforts         = {len(efforts)}")
    print(f"topk_list       = {len(topk_list)}")
    print(f"ai_ids          = {len(ai_ids)}")
    print(f"fs_ids          = {len(fs_ids)}")
    print(f"df rows         = {len(df)}")
    print(f"replicates      = {int(num_replicates)}")
    
    for approach, model, mtoks, effort, topk, ai_id, fs_id in itertools.product(
        approaches, models, max_tokens_list, efforts, topk_list, ai_ids, fs_ids,
    ):
        ans = ans_instr_A if ai_id == "A" else (ans_instr_B or "")
        fs = fewshot_A if fs_id == "A" else (fewshot_B or "")

        for _, r in df.iterrows():
            q = str(r["question"]) if pd.notna(r["question"]) else ""
            gold = str(r["gold_answer"]) if pd.notna(r["gold_answer"]) else None

            for rep in range(1, int(num_replicates) + 1):
                index += 1
                print(f"On Pass {index} / {total_loop_count}")
                generate_answer_start = time.time()
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
                generate_answer_start_elapsed = time.time() - generate_answer_start
                
                metrics_start = time.time()
                mets = (
                    langfair_metrics(generated, gold or "")
                    if gold is not None
                    else {"cosine": None, "rougeL": None, "bleu": None}
                )
                metrics_elapsed = time.time() - metrics_start

                contexts = "".join(h.get("text", "") for h in hits)
                judge_start = time.time()
                judges = judge_with_langsmith(
                    question=q,
                    answer=generated,
                    gold=gold,
                    contexts=contexts,
                    judge_model=judge_model,
                )
                judge_elapsed = time.time() - judge_start
                total_elapsed = generate_answer_start_elapsed + metrics_elapsed + judge_elapsed
                row = {
                    "datetime": now_et(),
                    "generate_elapsed_time": f"{generate_answer_start_elapsed:.2f} Seconds",
                    "metrics_elapsed_time": f"{metrics_elapsed:.2f} Seconds",
                    "judge_elapsed_time": f"{judge_elapsed:.2f} Seconds",
                    "total_elapsed_time": f"{total_elapsed:.2f} Seconds",
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
                    "cosine": mets.get("cosine"),
                    "rougeL": mets.get("rougeL"),
                    "bleu": mets.get("bleu"),
                    "judge_doc_relevance": judges.get("doc_relevance"),
                    "judge_doc_relevance_answer": extract_boolean_answer(judges.get("doc_relevance"), "Relevance"),
                    "judge_faithfulness": judges.get("faithfulness"),
                    "judge_faithfulness_answer": extract_boolean_answer(judges.get("faithfulness"), "Grounded"),
                    "judge_helpfulness": judges.get("helpfulness"),
                    "judge_helpfulness_answer": extract_boolean_answer(judges.get("helpfulness"), "Relevance"),
                    "judge_correctness_vs_ref": judges.get("correctness_vs_ref"),
                    "judge_correctness_vs_ref_answer": extract_boolean_answer(judges.get("correctness_vs_ref"), "Correctness"),
                    **{f"meta_{k}": v for k, v in (meta or {}).items()},
                }

                (
                    pd.DataFrame([row])
                    .to_csv(
                        out_csv,
                        mode="a",
                        header=write_header,
                        index=False,
                    )
                )
                write_header = False

    print(f"Wrote results to {out_csv}")
    return out_csv


# Example manual call
out = run_experiment(
    test_csv=Path("data/sample_test_questions.csv"),
    num_replicates=3,
    approaches=["openai_keyword"],
    models=["gpt-5-mini-2025-08-07"],
    max_tokens_list=[250],
    efforts=["low"],
    topk_list=[10],
    ans_instr_A=_read_text("prompts/ans_instr_A.txt"),
    ans_instr_B=None,
    fewshot_A=_read_text("prompts/fewshot_A.txt"),
    fewshot_B=None,
    out_csv=Path("results/experiment_results_with_replicates_parrallel.csv"),
    judge_model="gpt-5",
)