"""
Utilities for retrieving and normalizing OpenAI Batch judge outputs (step 4.4).

Use this after submitting the batch in 4_3 so you can download the response JSONL
and convert it into a lighter-weight structure for analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Core retrieval helpers
# ---------------------------------------------------------------------------
def _ensure_client(client: Optional[OpenAI] = None) -> OpenAI:
    return client or OpenAI()


def retrieve_batch(batch_id: str, *, client: Optional[OpenAI] = None) -> Dict[str, Any]:
    """
    Fetch the latest metadata for a Batch job.

    Only call this once you know the job has finished; otherwise the output file
    will not be available yet.
    """
    client = _ensure_client(client)
    return client.batches.retrieve(batch_id)


def download_output_file(output_file_id: str, destination: Path, *, client: Optional[OpenAI] = None) -> Path:
    """
    Download the batch output JSONL to ``destination``.

    Parameters
    ----------
    output_file_id:
        The ``output_file_id`` returned by ``client.batches.retrieve`` once the job
        has finished processing.
    destination:
        Where to save the JSONL file locally.
    """
    client = _ensure_client(client)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with client.files.with_streaming_response.content(output_file_id) as stream:
        stream.stream_to_file(destination)
    return destination


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_output_text(response_body: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for item in response_body.get("output") or []:
        if item.get("type") == "output_text":
            text = item.get("text", "")
            if text:
                pieces.append(text)
        elif item.get("type") == "output_message":
            for content in item.get("content") or []:
                if content.get("type") == "output_text":
                    text = content.get("text", "")
                    if text:
                        pieces.append(text)
    return "\n".join(piece.strip() for piece in pieces if piece.strip())


def _judge_type_from_custom_id(custom_id: Optional[str]) -> Optional[str]:
    if not custom_id:
        return None
    if "__" not in custom_id:
        return None
    return custom_id.split("__", 1)[1]


def normalize_batch_records(raw_records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw batch response lines into lightweight dictionaries.
    """
    normalized: List[Dict[str, Any]] = []
    for record in raw_records:
        custom_id = record.get("custom_id")
        base: Dict[str, Any] = {
            "custom_id": custom_id,
            "judge_type": _judge_type_from_custom_id(custom_id),
            "permutation_id": None,
            "response_id": None,
            "status_code": None,
            "model": None,
            "usage": None,
            "text": "",
            "error": None,
        }
        if "error" in record:
            base["error"] = record["error"]
            normalized.append(base)
            continue

        response_info = record.get("response") or {}
        body = response_info.get("body") or {}
        metadata = body.get("metadata") or {}

        base["permutation_id"] = metadata.get("permutation_id")
        base["response_id"] = body.get("id")
        base["status_code"] = response_info.get("status_code")
        base["model"] = body.get("model")
        base["usage"] = body.get("usage")
        base["text"] = _extract_output_text(body)

        normalized.append(base)
    return normalized


def write_normalized_records(records: Iterable[Dict[str, Any]], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2, ensure_ascii=False)
    return destination


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def fetch_batch_output(
    *,
    batch_id: Optional[str] = None,
    output_file_id: Optional[str] = None,
    raw_output_path: Path = Path("results/4_4_batch_output.jsonl"),
    normalized_output_path: Path = Path("results/4_4_batch_output.json"),
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Download and normalize the responses for a completed batch job.

    Provide either ``batch_id`` (preferred, so the script can look up the latest
    file id) or ``output_file_id`` if you have already captured it.
    """
    if not batch_id and not output_file_id:
        raise ValueError("Provide either batch_id or output_file_id.")

    client = _ensure_client(client)

    if batch_id:
        batch = retrieve_batch(batch_id, client=client)
        status = batch.get("status")
        if status != "completed":
            raise RuntimeError(f"Batch {batch_id} is not complete yet (status={status!r}).")
        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} does not expose an output_file_id yet.")
    assert output_file_id is not None

    download_output_file(output_file_id, raw_output_path, client=client)
    raw_records = _load_jsonl(raw_output_path)
    normalized_records = normalize_batch_records(raw_records)
    write_normalized_records(normalized_records, normalized_output_path)

    return {
        "raw_path": str(raw_output_path),
        "normalized_path": str(normalized_output_path),
        "records": normalized_records,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Convenience CLI so the user can fetch batch results without writing extra code.

    Run this only after the Batch API reports ``status == 'completed'``.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Download and normalize judge outputs from an OpenAI Batch job.")
    parser.add_argument("--batch-id", help="The Batch job identifier returned when you created the job.")
    parser.add_argument("--output-file-id", help="Optional: download directly if you already have the file id.")
    parser.add_argument(
        "--raw-output-path",
        type=Path,
        default=Path("results/4_4_batch_output.jsonl"),
        help="Where to save the raw JSONL returned by the Batch API.",
    )
    parser.add_argument(
        "--normalized-output-path",
        type=Path,
        default=Path("results/4_4_batch_output.json"),
        help="Where to write the simplified JSON summary.",
    )
    args = parser.parse_args()

    result = fetch_batch_output(
        batch_id=args.batch_id,
        output_file_id=args.output_file_id,
        raw_output_path=args.raw_output_path,
        normalized_output_path=args.normalized_output_path,
    )

    print(
        f"Downloaded {len(result['records'])} records.\n"
        f"Raw JSONL saved to {result['raw_path']}\n"
        f"Normalized JSON saved to {result['normalized_path']}"
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__ = [
    "retrieve_batch",
    "download_output_file",
    "normalize_batch_records",
    "write_normalized_records",
    "fetch_batch_output",
]


if __name__ == "__main__":
    main()
