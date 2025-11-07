"""
Utilities for building OpenAI Batch requests that score RAG outputs with the
same judge prompts used in ``3_rag_exp_with_evals.py``.

Other parts of the pipeline are responsible for:
1. Generating the question/answer/context records
2. Computing any numeric metrics (cosine, BLEU, etc.)

This module simply converts those records into the POST /v1/responses payloads
needed by the Batch API and optionally submits the batch.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv, set_key
from openai import OpenAI
import requests

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Prompt templates (mirrors 3_rag_exp_with_evals.py)
# ---------------------------------------------------------------------------




HELPFULNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class JudgeInput:
    """
    Minimal record required to build batch judge requests.

    Add any custom fields to ``metadata`` so they travel with every request. Minimise metadata since openAI batch request have size limit of 200mb.
    """

    qa_id: str
    question: str
    generated_answer: str
    contexts: str
    gold_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")


def _response_body(*, system_prompt: str, user_prompt: str, judge_model: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the body for a single POST /v1/responses request.
    """
    return {    
        "model": judge_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "metadata": metadata,
    }


def build_requests(records: Iterable[JudgeInput], judge_model: str = "gpt-5") -> List[Dict[str, Any]]:
    """Convert each JudgeInput into one or more Batch POST payloads."""
    requests: List[Dict[str, Any]] = []
    for record in records:
        permutation_id = record.metadata.get("permutation_id") or record.qa_id

        helpfulness_body = _response_body(
            system_prompt=HELPFULNESS_INSTRUCTIONS,
            user_prompt=f"QUESTION: {record.question}\nSTUDENT ANSWER: {record.generated_answer}",
            judge_model=judge_model,
            metadata={"permutation_id": permutation_id},
        )
        requests.append(
            {
                "custom_id": f"{record.qa_id}__helpfulness",
                "method": "POST",
                "url": "/v1/responses",
                "body": helpfulness_body,
            }
        )

        if record.gold_answer:
            correctness_body = _response_body(
                system_prompt=CORRECTNESS_INSTRUCTIONS,
                user_prompt=(
                    f"QUESTION: {record.question}\n"
                    f"GROUND TRUTH ANSWER: {record.gold_answer}\n"
                    f"STUDENT ANSWER: {record.generated_answer}"
                ),
                judge_model=judge_model,
                metadata={"permutation_id": permutation_id},
            )
            requests.append(
                {
                    "custom_id": f"{record.qa_id}__correctness_vs_ref",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": correctness_body,
                }
            )

    return requests


def write_requests_jsonl(requests: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Persist the prepared Batch payload to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for payload in requests:
            f.write(json.dumps(payload))
            f.write("\n")


def submit_batch(requests: Iterable[Dict[str, Any]], completion_window: str = "24h") -> Dict[str, Any]:
    """
    Upload the payload and create a Batch job.

    Returns file and batch identifiers so a downstream step can poll for results.
    """
    client = OpenAI()
    payload = "\n".join(json.dumps(r) for r in requests)
    buffer = io.BytesIO(payload.encode("utf-8"))
    buffer.seek(0)

    uploaded = client.files.create(
        file=("rag_eval_batch.jsonl", buffer),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata={"app": "safety_rag_eval", "kind": "judge_batch"},
    )
    return {"file_id": uploaded.id, "batch_id": batch.id, "status": batch.status}

def load_judge_inputs_from_csv(csv_path: Path) -> List[JudgeInput]:
    """
    Load ``JudgeInput`` records from the RAG generation CSV produced in step 3.

    The CSV already contains everything needed to construct judge requests, so we
    simply map columns onto the ``JudgeInput`` fields and metadata.
    """
    records: List[JudgeInput] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            permutation_id = row.get("permutation_id", None)
            if not permutation_id:
                raise ValueError(f"Missing permutation_id in row {idx}: {row}")

            records.append(
                JudgeInput(
                    qa_id=permutation_id,
                    question=(row.get("question") or "").strip(),
                    generated_answer=(row.get("generated_answer") or "").strip(),
                    contexts=(row.get("meta_hits_text") or "").strip(),
                    gold_answer=(row.get("gold_answer") or "").strip() or None,
                    metadata={"permutation_id": permutation_id},
                )
            )
    return records


def main() -> None:
    """
    Build batch requests using the CSV from step 3. and sends request 
    """
    import argparse

    parser = argparse.ArgumentParser(description="Prepare judge Batch payloads from an experiment CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/rag_generation.csv"),
        help="Path to the rag_generation.csv produced by step 3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/4_3_batch_requests.jsonl"),
        help="Where to write the Batch-friendly JSONL payload.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5",
        help="Judge model name to use when constructing the requests.",
    )
    args = parser.parse_args()

    records = load_judge_inputs_from_csv(args.csv)
    requests = build_requests(records, judge_model=args.judge_model)
    write_requests_jsonl(requests, args.output)

    print(f"Prepared {len(requests)} requests. JSONL written to {args.output}")
    # submission = submit_batch(requests)
    #print(json.dumps(submission, indent=2))
    #batch_id = submission.id
    #set_key("OPENAI_BATCH_ID", batch_id)

    print(f"Prepared {len(requests)} requests. JSONL written to {args.output}")



# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__ = [
    "JudgeInput",
    "build_requests",
    "write_requests_jsonl",
    "submit_batch",
    "make_permutation_id",
    "load_judge_inputs_from_csv",
]


if __name__ == "__main__":
    main()