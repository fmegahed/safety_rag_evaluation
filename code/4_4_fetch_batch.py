"""
Download a finished OpenAI Batch run and pivot the JSONL responses into one JSON file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# Set these to whatever is convenient before running the script.
BATCH_ID = os.environ.get("OPENAI_BATCH_ID") or "batch_690ae971d7048190a982f22185698051"
RAW_PATH = Path("results/4_5_batch_output.jsonl")
JSON_PATH = Path("results/4_5_batch_output.json")


def _extract_text(body: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for item in body.get("output") or []:
        item_type = item.get("type")
        if item_type == "output_text":
            text = (item.get("text") or "").strip()
            if text:
                pieces.append(text)
        elif item_type == "output_message":
            for part in item.get("content") or []:
                if part.get("type") != "output_text":
                    continue
                text = (part.get("text") or "").strip()
                if text:
                    pieces.append(text)
    return "\n".join(pieces)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



client = OpenAI()

batch = client.batches.retrieve(BATCH_ID).model_dump()
if batch.get("status") != "completed":
    raise SystemExit(f"Batch {BATCH_ID} is not complete yet (status={batch.get('status')!r}).")

output_file_id = batch.get("output_file_id")
if not output_file_id:
    raise SystemExit(f"Batch {BATCH_ID} does not expose an output_file_id.")

RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

with client.files.with_streaming_response.content(output_file_id) as stream:
    stream.stream_to_file(RAW_PATH)

raw_records = _load_jsonl(RAW_PATH)

pivot: Dict[str, Dict[str, Any]] = {}
for record in raw_records:
    custom_id = record.get("custom_id")
    if not custom_id or "__" not in custom_id:
        continue
    qa_id, judge_type = custom_id.split("__", 1)
    slot = pivot.setdefault(qa_id, {"qa_id": qa_id})

    if record.get("error"):
        slot[f"{judge_type}_error"] = record["error"]
        continue

    response = record.get("response") or {}
    body = response.get("body") or {}

    metadata = body.get("metadata") or {}
    if metadata.get("permutation_id") and not slot.get("permutation_id"):
        slot["permutation_id"] = metadata["permutation_id"]

    slot[f"{judge_type}_text"] = _extract_text(body)

JSON_PATH.write_text(json.dumps(list(pivot.values()), indent=2, ensure_ascii=False), encoding="utf-8")

print(
    f"Downloaded batch output to {RAW_PATH}\n"
    f"Wrote widened JSON to {JSON_PATH}\n"
    f"Total groups: {len(pivot)}"
)
