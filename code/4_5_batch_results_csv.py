"""
Turn the widened JSON from step 4.4 into a CSV with the judge columns from step 3.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict


SOURCE_CSV = Path("results/rag_generation.csv")
BATCH_JSON = Path("results/4_5_batch_output.json")
OUT_CSV = Path("results/4_5_judge_results.csv")


def make_permutation_id(index: int, row: Dict[str, Any]) -> str:
    def abbr(value: str, max_len: int) -> str:
        value = (value or "").strip()
        if not value:
            return "x"
        parts = re.split(r"[_\-\s]+", value)
        parts = [p for p in parts if p]
        if len(parts) == 1:
            token = re.sub(r"[^A-Za-z0-9]", "", parts[0])
            return (token[:max_len] or token[:1]).lower()
        letters = "".join(p[0] for p in parts if p)
        return (letters[:max_len] or letters[:1]).lower()

    approach = abbr(row.get("approach", ""), 3)
    model = abbr(row.get("model", ""), 2)
    effort = abbr(row.get("reasoning_effort", ""), 2)
    ans_id = str(row.get("answer_instructions_id") or "A")[:1].lower()
    fs_id = str(row.get("few_shot_id") or "A")[:1].lower()

    top_k_raw = str(row.get("top_k") or "").strip()
    try:
        top_k_val = int(float(top_k_raw))
        top_k = f"k{top_k_val:02d}"
    except ValueError:
        top_k = abbr(top_k_raw, 2)

    max_tok_raw = str(row.get("max_tokens") or "").strip()
    try:
        max_tok_val = int(float(max_tok_raw))
        max_tok = f"t{max_tok_val // 100:02d}"
    except ValueError:
        max_tok = abbr(max_tok_raw, 2)

    return f"{approach}{model}{ans_id}{fs_id}{effort}{top_k}{max_tok}_{index:04d}"


def extract_flag(text: str | None, keyword: str) -> str:
    if not text:
        return ""
    match = re.search(rf"((?<={keyword}:\s)|(?<={keyword}:))(True|False)", text)
    return match.group(0) if match else ""


with SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise SystemExit(f"{SOURCE_CSV} is missing headers.")
    base_fields = list(reader.fieldnames)
    rows = []
    for idx, row in enumerate(reader, start=1):
        row = dict(row)
        row["permutation_id"] = make_permutation_id(idx, row)
        rows.append(row)

if not rows:
    raise SystemExit("No rows found in rag_generation.csv.")

pivot_data = json.loads(BATCH_JSON.read_text(encoding="utf-8"))
if not isinstance(pivot_data, list):
    raise SystemExit("Batch JSON should be a list of records.")

pivot_map: Dict[str, Dict[str, Any]] = {}
for item in pivot_data:
    perm = item.get("permutation_id") or item.get("qa_id")
    if perm:
        pivot_map[perm] = item

judge_cols = [
    "judge_doc_relevance",
    "judge_doc_relevance_answer",
    "judge_faithfulness",
    "judge_faithfulness_answer",
    "judge_helpfulness",
    "judge_helpfulness_answer",
    "judge_correctness_vs_ref",
    "judge_correctness_vs_ref_answer",
]

header = base_fields + ["permutation_id"]
for col in judge_cols:
    if col not in header:
        header.append(col)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

missing = []
with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()

    for row in rows:
        judge = pivot_map.get(row["permutation_id"])
        doc_relevance = judge.get("doc_relevance_text") if judge else ""
        faithfulness = judge.get("faithfulness_text") if judge else ""
        helpfulness = judge.get("helpfulness_text") if judge else ""
        correctness = judge.get("correctness_vs_ref_text") if judge else ""

        row["judge_doc_relevance"] = doc_relevance
        row["judge_doc_relevance_answer"] = extract_flag(doc_relevance, "Relevance")
        row["judge_faithfulness"] = faithfulness
        row["judge_faithfulness_answer"] = extract_flag(faithfulness, "Grounded")
        row["judge_helpfulness"] = helpfulness
        row["judge_helpfulness_answer"] = extract_flag(helpfulness, "Relevance")
        row["judge_correctness_vs_ref"] = correctness
        row["judge_correctness_vs_ref_answer"] = extract_flag(correctness, "Correctness")

        if judge is None:
            missing.append(row["permutation_id"])

        writer.writerow({key: row.get(key, "") for key in header})

print(
    f"Wrote {len(rows)} rows to {OUT_CSV}\n"
    f"Missing judge groups: {len(missing)}"
)
