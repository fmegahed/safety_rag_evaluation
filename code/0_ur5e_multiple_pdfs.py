"""
Prepare UR5 manual sections for RAG or downstream processing.

Steps:
1) Split the source PDF into separate PDFs using the Table of Contents (TOC) at a chosen level.
2) Crop each generated PDF by a fixed percentage on all sides to remove repeated margins or banners.
3) Count words in each cropped PDF and write a summary CSV (ur5_pdf_word_counts.csv).
4) Automatically sub-split any large cropped PDFs with more than MIN_WORDS_FOR_SUBSPLIT words using a deeper TOC level.
   - Export subchapters from the original manual.
   - Crop the subchapter PDFs into the cropped folder.
   - Delete the original cropped chapter PDFs that were replaced.
   - Recompute and save updated word counts for all cropped PDFs (ur5_pdf_word_counts_after_subsplit.csv).
5) Final clean-up: for any remaining oversized cropped PDFs, split each exactly in half by pages.
   - Save halves as ...__part01__pp... and ...__part02__pp..., inserting the part tag before the __pp segment.
   - Delete the original oversized file after the halves are saved.
   - Recompute and save final word counts for all cropped PDFs (ur5_pdf_word_counts_final.csv).
"""


import re
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Source manual and output folders
INPUT_PDF = "pdfs/UR5e_Universal_Robots User Manual.pdf"
SPLIT_DIR = Path("results/pdfs/ur5_splits")
CROPPED_DIR = Path("results/pdfs/ur5_splits_cropped")

# TOC splitting controls
TOC_LEVEL = 1           # 1 = top level, 2 = second level, 3 = third level
MIN_PAGES = 1           # skip entries with fewer than this many pages

# Cropping controls
CROP_PERCENT = 0.075    # fraction removed from each side (e.g., 0.075 = 7.5%)

# Subchapter split controls (step 4)
SUB_LEVEL = 2                   # deeper TOC level for subchapters (2 or 3 are common)
MIN_WORDS_FOR_SUBSPLIT = 3000   # threshold for auto sub-splitting
SUBCHAPTER_OUT_DIR = SPLIT_DIR / f"by_subchapter_L{SUB_LEVEL}"

# Output CSVs
WORDCOUNT_CSV_INITIAL = "results/csvs/ur5_pdf_word_counts.csv"
WORDCOUNT_CSV_UPDATED = "results/csvs/ur5_pdf_word_counts_after_subsplit.csv"
WORDCOUNT_CSV_FINAL = "results/csvs/ur5_pdf_word_counts_final.csv"


# =============================================================================
# Helpers
# =============================================================================
def safe(name: str, max_len: int = 80) -> str:
    """
    Create a filesystem-safe filename from a title.
    Keeps letters, digits, space, underscore, hyphen, period, and parentheses.
    """
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[^A-Za-z0-9 _\-\.\(\)]", "", name)
    name = name[:max_len].strip().replace(" ", "_")
    return name or "untitled"


def parse_page_range_from_name(name: str):
    """
    Extract 1-based inclusive page range from a filename pattern like:
    "...__pp59-96.pdf" -> (59, 96)
    """
    m = re.search(r"__pp(\d+)-(\d+)\.pdf$", name)
    if not m:
        raise ValueError(f"Could not parse page range from filename: {name}")
    start_1b = int(m.group(1))
    end_1b = int(m.group(2))
    if end_1b < start_1b:
        raise ValueError(f"Invalid page range in filename: {name}")
    return start_1b, end_1b


# =============================================================================
# (1) Split the source PDF by TOC entries at the desired level
# =============================================================================
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

doc = fitz.open(INPUT_PDF)
toc = doc.get_toc(simple=True)  # list of [level, title, page_1based]
n_pages = len(doc)

# Indices in the TOC that match the desired level
level_idxs = [i for i, (lvl, _, _) in enumerate(toc) if lvl == TOC_LEVEL]

splits = []
for _, i in enumerate(level_idxs):
    lvl, title, start_1b = toc[i]

    # Find the next TOC entry whose level is less than or equal to the current level
    next_start_1b = None
    for j in range(i + 1, len(toc)):
        lvl_j, _, next_page_1b = toc[j]
        if lvl_j <= lvl:
            next_start_1b = next_page_1b
            break

    # Compute inclusive 1-based end page
    end_1b = (next_start_1b - 1) if next_start_1b is not None else n_pages

    # Convert to 0-based page indices and clamp to valid range
    start0 = max(0, start_1b - 1)
    end0 = min(n_pages - 1, end_1b - 1)

    # Skip invalid ranges
    if end0 < start0:
        continue

    # Skip very small sections if requested
    pages = end0 - start0 + 1
    if pages < MIN_PAGES:
        continue

    splits.append(
        dict(
            title=title,
            start0=start0,
            end0=end0,
            start_1b=start_1b,
            end_1b=end_1b
        )
    )

# Write each split as its own PDF
for k, s in enumerate(splits, start=1):
    out_name = (
        f"{k:03d}__L{TOC_LEVEL}__{safe(s['title'])}"
        f"__pp{s['start_1b']}-{s['end_1b']}.pdf"
    )
    out_path = SPLIT_DIR / out_name

    part = fitz.open()
    part.insert_pdf(doc, from_page=s["start0"], to_page=s["end0"])
    part.save(out_path, deflate=True, garbage=4)
    part.close()

doc.close()
print(f"Created {len(splits)} PDFs in {SPLIT_DIR.resolve()}")


# =============================================================================
# (2) Crop each generated PDF by a fixed percentage on all sides
# =============================================================================
CROPPED_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in SPLIT_DIR.glob("*.pdf"):
    output_path = CROPPED_DIR / pdf_path.name
    doc = fitz.open(pdf_path)

    for page in doc:
        rect = page.rect
        width, height = rect.width, rect.height

        # Compute margins to trim
        lm = width * CROP_PERCENT
        rm = width * CROP_PERCENT
        tm = height * CROP_PERCENT
        bm = height * CROP_PERCENT

        # New crop rectangle
        new_rect = fitz.Rect(
            rect.x0 + lm,
            rect.y0 + tm,
            rect.x1 - rm,
            rect.y1 - bm
        )

        # Apply crop to the page
        page.set_cropbox(new_rect)

    # Save cropped PDF
    doc.save(output_path, deflate=True, garbage=4)
    doc.close()

    print(f"Cropped {pdf_path.name} -> {output_path.name}")

print(f"All PDFs cropped and saved to: {CROPPED_DIR.resolve()}")


# =============================================================================
# (3) Count words in each cropped PDF and write a CSV
# =============================================================================
rows = []
for pdf in sorted(CROPPED_DIR.glob("*.pdf")):
    d = fitz.open(pdf)
    page_words = [len(p.get_text("text").split()) for p in d]
    rows.append(
        {
            "file": pdf.name,
            "n_pages": len(d),
            "total_words": sum(page_words),
            "max_words_in_a_page": max(page_words) if page_words else 0,
            "mean_words_per_page": (sum(page_words) / len(d)) if len(d) else 0.0,
        }
    )
    d.close()

summary = (
    pd.DataFrame(rows)
    .sort_values("total_words", ascending=False)
    .reset_index(drop=True)
)
print(summary)
summary.to_csv(WORDCOUNT_CSV_INITIAL, index=False)
print(f"Saved word counts to {WORDCOUNT_CSV_INITIAL}")


# =============================================================================
# (4) Auto sub-split any cropped PDFs with > MIN_WORDS_FOR_SUBSPLIT words,
#     crop results, delete originals that were split, then recompute and save updated word counts
# =============================================================================
SUBCHAPTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Determine targets from the just-created summary
targets = (
    summary.loc[summary["total_words"] > MIN_WORDS_FOR_SUBSPLIT, "file"]
    .tolist()
)

if targets:
    manual = fitz.open(INPUT_PDF)
    toc = manual.get_toc(simple=True)  # list of [level, title, page_1based]
    n_pages = len(manual)
    toc_entries = [{"level": lvl, "title": title, "page_1b": p} for (lvl, title, p) in toc]
    all_idxs = list(range(len(toc_entries)))

    deleted_files = []  # track which cropped files we remove after splitting

    for fname in targets:
        # Ensure the original split file exists to read its page range
        split_pdf_path = SPLIT_DIR / fname
        cropped_pdf_path = CROPPED_DIR / fname
        if not split_pdf_path.exists():
            print(f"WARNING: {fname} not found in {SPLIT_DIR}. Skipping.")
            continue

        chap_start_1b, chap_end_1b = parse_page_range_from_name(fname)

        # Collect subchapter starts within chapter bounds at SUB_LEVEL
        sub_starts = []
        for i in all_idxs:
            e = toc_entries[i]
            if e["level"] == SUB_LEVEL and chap_start_1b <= e["page_1b"] <= chap_end_1b:
                sub_starts.append((i, e))

        if not sub_starts:
            print(
                f"No level {SUB_LEVEL} entries found within {fname} "
                f"pages {chap_start_1b}-{chap_end_1b}."
            )
            continue

        for k, (idx, entry) in enumerate(sub_starts, start=1):
            start_1b = entry["page_1b"]

            # Find next boundary with level <= SUB_LEVEL within the chapter
            next_start_1b = None
            for j in range(idx + 1, len(toc_entries)):
                e2 = toc_entries[j]
                if e2["page_1b"] > chap_end_1b:
                    break
                if chap_start_1b <= e2["page_1b"] <= chap_end_1b and e2["level"] <= SUB_LEVEL:
                    next_start_1b = e2["page_1b"]
                    break

            end_1b = (next_start_1b - 1) if next_start_1b is not None else chap_end_1b

            # Clamp and convert to 0-based
            start0 = max(0, start_1b - 1)
            end0 = min(n_pages - 1, end_1b - 1)
            if end0 < start0:
                continue

            base_title = entry["title"]
            out_name = (
                f"{Path(fname).stem}__L{SUB_LEVEL}__{k:02d}__{safe(base_title)}"
                f"__pp{start_1b}-{end_1b}.pdf"
            )
            out_path = SUBCHAPTER_OUT_DIR / out_name

            # Export from the original manual
            part = fitz.open()
            part.insert_pdf(manual, from_page=start0, to_page=end0)
            part.save(out_path, deflate=True, garbage=4)
            part.close()

            # Crop subchapter output into the same cropped folder
            cropped_out = CROPPED_DIR / out_path.name
            with fitz.open(out_path) as d:
                for page in d:
                    rect = page.rect
                    w, h = rect.width, rect.height
                    lm = w * CROP_PERCENT
                    rm = w * CROP_PERCENT
                    tm = h * CROP_PERCENT
                    bm = h * CROP_PERCENT
                    new_rect = fitz.Rect(rect.x0 + lm, rect.y0 + tm, rect.x1 - rm, rect.y1 - bm)
                    page.set_cropbox(new_rect)
                d.save(cropped_out, deflate=True, garbage=4)

            print(f"Created subchapter: {out_path.name}  and cropped ->  {cropped_out.name}")

        # After successful sub-splitting, delete the original cropped file
        if cropped_pdf_path.exists():
            cropped_pdf_path.unlink()
            deleted_files.append(fname)
            print(f"Deleted original cropped file: {fname}")

    manual.close()
    print(f"Deleted {len(deleted_files)} original cropped PDFs after sub-splitting.")
else:
    print(f"No files exceeded {MIN_WORDS_FOR_SUBSPLIT} words. Skipping sub-splitting.")


# Recompute and save updated word counts for all cropped PDFs, including subchapters
rows_updated = []
for pdf in sorted(CROPPED_DIR.glob("*.pdf")):
    d = fitz.open(pdf)
    page_words = [len(p.get_text("text").split()) for p in d]
    rows_updated.append(
        {
            "file": pdf.name,
            "n_pages": len(d),
            "total_words": sum(page_words),
            "max_words_in_a_page": max(page_words) if page_words else 0,
            "mean_words_per_page": (sum(page_words) / len(d)) if len(d) else 0.0,
        }
    )
    d.close()

summary_updated = (
    pd.DataFrame(rows_updated)
    .sort_values("total_words", ascending=False)
    .reset_index(drop=True)
)
print(summary_updated)
summary_updated.to_csv(WORDCOUNT_CSV_UPDATED, index=False)
print(f"Saved updated word counts to {WORDCOUNT_CSV_UPDATED}")


# =============================================================================
# (5) Simple half-split for any remaining large PDFs
# =============================================================================
THRESHOLD = MIN_WORDS_FOR_SUBSPLIT 

large_files = summary_updated.loc[summary_updated["total_words"] > THRESHOLD, "file"].tolist()
if large_files:
    print(f"Half-splitting {len(large_files)} oversized PDFs (> {THRESHOLD} words)...")

    for fname in large_files:
        pdf_path = CROPPED_DIR / fname
        if not pdf_path.exists():
            continue

        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        if n_pages < 2:
            doc.close()
            continue  # can't split one-pagers

        mid_page = n_pages // 2 - 1  # zero-based midpoint
        base_stem = pdf_path.stem

        # First half
        part1_path = CROPPED_DIR / f"{base_stem}_part01.pdf"
        part1 = fitz.open()
        part1.insert_pdf(doc, from_page=0, to_page=mid_page)
        part1.save(part1_path, deflate=True, garbage=4)
        part1.close()

        # Second half
        part2_path = CROPPED_DIR / f"{base_stem}_part02.pdf"
        part2 = fitz.open()
        part2.insert_pdf(doc, from_page=mid_page + 1, to_page=n_pages - 1)
        part2.save(part2_path, deflate=True, garbage=4)
        part2.close()

        doc.close()
        pdf_path.unlink()  # delete original
        print(f"Split {fname} -> {part1_path.name}, {part2_path.name}")

    # Recount words for the updated set
    rows_final = []
    for pdf in sorted(CROPPED_DIR.glob("*.pdf")):
        d = fitz.open(pdf)
        page_words = [len(p.get_text('text').split()) for p in d]
        rows_final.append({
            "file": pdf.name,
            "n_pages": len(d),
            "total_words": sum(page_words),
            "max_words_in_a_page": max(page_words) if page_words else 0,
            "mean_words_per_page": (sum(page_words) / len(d)) if len(d) else 0.0,
        })
        d.close()

    summary_final = (
        pd.DataFrame(rows_final)
        .sort_values("total_words", ascending=False)
        .reset_index(drop=True)
    )
    print(summary_final)
    summary_final.to_csv(WORDCOUNT_CSV_FINAL, index=False)
else:
    print("No oversized PDFs remaining after sub-splitting.")
