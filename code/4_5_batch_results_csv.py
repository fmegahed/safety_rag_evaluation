"""
Download a finished OpenAI Batch run, persist the raw JSONL payload, and export per-record
judging metadata to CSV for downstream analysis.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# Resolve the repo root regardless of where this script is executed from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_PATH = RESULTS_DIR / "4_5_batch_output.jsonl"
JSON_PATH = RESULTS_DIR / "4_5_batch_output.json"
CSV_PATH = RESULTS_DIR / "4_5_batch_output.csv"

# Allow overriding the Batch ID via environment variables.
DEFAULT_BATCH_ID = "batch_690ae971d7048190a982f22185698051"
BATCH_ID = os.environ.get("OPENAI_BATCH_ID") or DEFAULT_BATCH_ID

client = OpenAI()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file from disk."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_record_info(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key fields from a single batch record."""
    info = {
        "custom_id": record.get("custom_id"),
        "text": None,
        "judge_model": None,
        "temperature": None,
        "permutation_id": None,
    }

    response = record.get("response") or {}
    body = response.get("body") or {}

    info["judge_model"] = body.get("model")
    info["temperature"] = body.get("temperature")

    metadata = body.get("metadata") or {}
    info["permutation_id"] = metadata.get("permutation_id")

    text_parts = []
    for item in body.get("output") or []:
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if content.get("type") == "output_text" and content.get("text"):
                    text_parts.append(content["text"].strip())
    info["text"] = "\n".join(text_parts) if text_parts else None

    return info


def extract_boolean_answer(text: str | None, prefix_word: str) -> str | None:
    """Return the boolean value that follows the requested prefix in the judge output."""
    if not text or not prefix_word:
        return None
    pattern = rf"(?<={re.escape(prefix_word)}:)\s*(True|False)"
    match = re.search(pattern, text)
    return match.group(1) if match else None


def extract_judge_type(custom_id: str | None) -> str | None:
    """Return the judge_type part from a custom_id like 'qa123__doc_relevance'."""
    if custom_id and "__" in custom_id:
        _, judge_type = custom_id.split("__", 1)
        return judge_type
    return None


def download_batch_results(batch_id: str) -> List[Dict[str, Any]]:
    """Download the Batch output file and return the parsed JSONL records."""
    batch = client.batches.retrieve(batch_id).model_dump()
    if batch.get("status") != "completed":
        raise SystemExit(f"Batch {batch_id} is not complete yet (status={batch.get('status')!r}).")

    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise SystemExit(f"Batch {batch_id} does not expose an output_file_id.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with client.files.with_streaming_response.content(output_file_id) as stream:
        stream.stream_to_file(RAW_PATH)

    records = _load_jsonl(RAW_PATH)
    JSON_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main() -> None:
    raw_records = download_batch_results(BATCH_ID)

    mapping_judge_type_key = {
        "doc_relevance": "Relevance",
        "correctness_vs_ref": "Correctness",
        "helpfulness": "Relevance",
        "faithfulness": "Grounded",
    }

    rows = []
    for record in raw_records:
        rec = extract_record_info(record)
        custom_id = rec.get("custom_id")
        judge_type = extract_judge_type(custom_id)
        judge_answer = extract_boolean_answer(
            rec.get("text"),
            mapping_judge_type_key.get(judge_type, ""),
        )

        rows.append(
            {
                "custom_id": rec.get("custom_id"),
                "text": rec.get("text"),
                "judge_model": rec.get("judge_model"),
                "temperature": rec.get("temperature"),
                "permutation_id": rec.get("permutation_id"),
                "judge_type": judge_type,
                "judge_answer": judge_answer,
            }
        )

    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote parsed batch results to {CSV_PATH}")


if __name__ == "__main__":
    main()
