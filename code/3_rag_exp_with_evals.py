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

import requests
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from langsmith import traceable

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load .env so API keys and endpoints are available everywhere
load_dotenv(override=True)

# Import the function from file 2 (which should expose retrieve_and_answer and be import-safe)
response = requests.get("https://raw.githubusercontent.com/fmegahed/safety_rag_evaluation/refs/heads/main/code/2_rag.py")
namespace = {}
exec(response.text, namespace)
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
    out: Dict[str, Any] = {}

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
    out["doc_relevance"] = (
        doc_rel_prompt
        .pipe(llm)
        .invoke({"question": question, "contexts": contexts})
        .content
    )

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
    out["faithfulness"] = (
        faithful_prompt
        .pipe(llm)
        .invoke({"answer": answer, "contexts": contexts})
        .content
    )

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
    out["helpfulness"] = (
        helpful_prompt
        .pipe(llm)
        .invoke({"question": question, "answer": answer})
        .content
    )

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
        out["correctness_vs_ref"] = (
            correct_prompt
            .pipe(llm)
            .invoke({"question": question, "answer": answer, "reference": gold})
            .content
        )
    else:
        out["correctness_vs_ref"] = None

    return out


def run_experiment(
    *,
    test_csv: Path,
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
    """Run a full sweep and write results to CSV incrementally.

    `judge_model` controls which LLM evaluates (doc relevance, faithfulness, helpfulness, correctness),
    independent from the generation models in `models`.
    """
    df = pd.read_csv(test_csv)
    assert {"question", "gold_answer"}.issubset(df.columns), "CSV must include question and gold_answer."

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine which A/B variants to run based on user inputs
    ai_ids = ["A", "B"] if (ans_instr_B and ans_instr_B.strip()) else ["A"]
    fs_ids = ["A", "B"] if (fewshot_B and fewshot_B.strip()) else ["A"]

    # Track if we need to write headers
    write_header = not out_csv.exists()

    for approach, model, mtoks, effort, topk, ai_id, fs_id in itertools.product(
        approaches, models, max_tokens_list, efforts, topk_list, ai_ids, fs_ids,
    ):
        ans = ans_instr_A if ai_id == "A" else (ans_instr_B or "")
        fs = fewshot_A if fs_id == "A" else (fewshot_B or "")

        for _, r in df.iterrows():
            q = str(r["question"]) if pd.notna(r["question"]) else ""
            gold = str(r["gold_answer"]) if pd.notna(r["gold_answer"]) else None

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

            mets = langfair_metrics(generated, gold or "") if gold is not None else {"cosine": None, "rougeL": None, "bleu": None}

            contexts = "".join(h.get("text", "") for h in hits)
            judges = judge_with_langsmith(
                question=q,
                answer=generated,
                gold=gold,
                contexts=contexts,
                judge_model=judge_model,
            )

            row = {
                "datetime": now_et(),
                "min_words_for_subsplit": MIN_WORDS_FOR_SUBSPLIT,
                "approach": approach,
                "model": model,
                "max_tokens": mtoks,
                "reasoning_effort": effort,
                "top_k": topk,
                "answer_instructions_id": ai_id,
                "few_shot_id": fs_id,
                "question": q,
                "gold_answer": gold,
                "generated_answer": generated,
                "retrieved_files": ";".join([h.get("filename") or "" for h in hits]),
                "cosine": mets.get("cosine"),
                "rougeL": mets.get("rougeL"),
                "bleu": mets.get("bleu"),
                "judge_doc_relevance": judges.get("doc_relevance"),
                "judge_faithfulness": judges.get("faithfulness"),
                "judge_helpfulness": judges.get("helpfulness"),
                "judge_correctness_vs_ref": judges.get("correctness_vs_ref"),
                **{f"meta_{k}": v for k, v in (meta or {}).items()},
            }

            # Append row to CSV
            (
                pd.DataFrame([row])
                .to_csv(out_csv, mode='a', header=write_header, index=False)
            )
            write_header = False

    print(f"✅ Wrote results to {out_csv}")
    return out_csv


# Example manual call (uncomment to run directly)
out = run_experiment(
    test_csv=Path('data/sample_test_questions.csv'),
    approaches=['lc_bm25', 'graph_eager'],
    models=['gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07'],
    max_tokens_list=[1000, 5000],
    efforts=['minimal', 'high'],
    topk_list=[5, 10],
    ans_instr_A=_read_text('prompts/ans_instr_A.txt'),
    ans_instr_B=None,
    fewshot_A=_read_text('prompts/fewshot_A.txt'),
    fewshot_B=None,
    out_csv=Path('results/experiment_results.csv'),
    judge_model='gpt-5',
)
