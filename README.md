# Safety RAG Evaluation

Code and evaluation data for a retrieval-augmented generation (RAG) safety chatbot for manufacturing equipment. The repository turns equipment manuals into a retrieval corpus, answers benchmark questions with several retrieval approaches and model settings, scores the answers against reference answers, and aggregates the results.

It accompanies the paper:

> Ryan Singh, Austin Hamilton, Amanda White, Michael Wise, Ibrahim Yousif, Arthur Carvalho, Zhe Shan, Reza Abrisham Baf, Mohammad Mayyas, Lora A. Cavuoto, and Fadel M. Megahed (2026). "A Multimodal Manufacturing Safety Chatbot: Knowledge Base Design, Benchmark Development, and Evaluation of Multiple RAG Approaches." *Manufacturing Letters* 49: 30-41. Preprint: [arXiv:2511.11847](https://arxiv.org/abs/2511.11847).

## Overview

The benchmark covers three machines:

| Machine | Manual(s) in `data/input/input_pdfs/` | Question set in `data/QA/` |
|---|---|---|
| Bridgeport Series 1 manual milling machine | `Bridgeport Series 1 Milling manual with schematics.pdf` | `MILL/Mill Feedback Accepted.csv` (51 questions) |
| Haas TL-1 CNC lathe | `HAAS_LATHE_OUTLINED.pdf`, `TL1_lathe operator's-manual.pdf` | `HAAS/Final HAAS Lathe QA.csv` (51 questions) |
| Universal Robots UR5e collaborative robot | `UR5e_Universal_Robots User Manual.pdf` | `COBOT/Final_COBOT_QA.csv` (64 questions) |

The pipeline has four stages:

1. **Preprocess**: split each manual into sections by its table of contents, crop page margins, and build the retrieval assets (BM25 index, Astra DB vector store, graph retrievers).
2. **Retrieve and generate**: for every question, retrieve context with each retrieval approach and generate an answer with an OpenAI model, over a grid of settings.
3. **Evaluate**: compute text-similarity metrics (TF-IDF cosine, ROUGE-L, BLEU) against the reference answer, and grade each answer for correctness and helpfulness with an LLM judge through the OpenAI Batch API.
4. **Analyze**: merge generation, metric, and judge results, then aggregate them by configuration and plot them.

The repository includes the complete outputs of stages 2 to 4 for the mill and the lathe question sets. For each of these two machines, the generation and merged results files have 2,448 rows: 51 questions x 48 configurations (6 retrieval approaches x 2 models x 2 reasoning-effort levels x 2 top-k values). For the UR5e question set, the repository includes only a small long-context test run (4 rows).

## Citation

If you use this code or data, please cite:

```bibtex
@article{singh2026multimodal,
  title         = {A Multimodal Manufacturing Safety Chatbot: Knowledge Base Design, Benchmark Development, and Evaluation of Multiple {RAG} Approaches},
  author        = {Ryan Singh and Austin Hamilton and Amanda White and Michael Wise and Ibrahim Yousif and Arthur Carvalho and Zhe Shan and Reza Abrisham Baf and Mohammad Mayyas and Lora A. Cavuoto and Fadel M. Megahed},
  journal       = {Manufacturing Letters},
  volume        = {49},
  pages         = {30--41},
  year          = {2026},
  eprint        = {2511.11847},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2511.11847}
}
```

## Repository structure

```
safety_rag_evaluation/
├── system/                         # Python package with the pipeline code
│   ├── main.py                     # Orchestration script (run as python -m system.main)
│   ├── main_long_context.py        # Import stub only (no executable steps)
│   ├── main_long_context.ipynb     # Long-context notebook (full manual text as context)
│   ├── README.md                   # Module-level reference for the package
│   ├── preprocess/
│   │   ├── pdf_pipeline.py         # PDFPreprocessor: TOC split, crop, sub-split, word counts
│   │   └── corpus_builder.py       # CorpusBuilder: docs cache, BM25, Astra DB, graph retrievers
│   ├── rag/
│   │   ├── approach_retrievers.py  # ApproachRetrievers: one method per retrieval approach
│   │   ├── rag_utils.py            # Prompt assembly and model calls
│   │   ├── rag_generation.py       # RAGExperimentRunner: runs the configuration grid
│   │   ├── enums/                  # Approaches and LLM flags
│   │   └── math/                   # Cosine, ROUGE-L, and BLEU metrics (LangfairRunner)
│   ├── evaluation/
│   │   ├── judge_batch.py          # Builds and submits LLM-judge Batch requests
│   │   ├── batch_fetcher.py        # Downloads Batch output, pivots it to JSON
│   │   ├── batch_results_parser.py # Batch output -> long-form CSV with True/False grades
│   │   ├── judge_results_merger.py # Long-form judge CSV -> one row per permutation
│   │   └── csv_column_merger.py    # Joins two CSVs on a shared column
│   ├── analysis/
│   │   └── analysis.py             # analyze_csv(): aggregated table, figures, text summary
│   ├── scripts/                    # Command-line entry points (see Usage)
│   └── utils/
│       ├── environment_config.py   # Reads .env settings
│       └── utils.py                # Timestamps, prompt loading, permutation IDs
├── notebooks/
│   ├── analysis haas.ipynb         # Exploratory analysis of data/results/final_merged/haas.csv
│   ├── analysis mill.ipynb         # Exploratory analysis of data/results/final_merged/mill.csv
│   └── synthetic_qa_generation.ipynb  # Generates data/QA/synthetic_qa_cobot_mill.csv
├── data/
│   ├── input/
│   │   ├── input_pdfs/             # Original equipment manuals
│   │   └── cropped_pdfs/           # Whole-manual cropped copies (mill, UR5e)
│   ├── QA/
│   │   ├── COBOT/                  # UR5e question set (+ raw review sheet, test subset)
│   │   ├── HAAS/                   # Haas lathe question set (+ test subset)
│   │   ├── MILL/                   # Bridgeport mill question set (+ test subset)
│   │   └── synthetic_qa_cobot_mill.csv
│   ├── preprocessed/
│   │   ├── pdfs/combined/          # 125 section PDFs from the three manuals
│   │   └── vstore/combined/        # Document cache (docs/docs.jsonl) and BM25 pickle (PKL)
│   ├── prompts/                    # Answer instructions and few-shot preamble
│   └── results/
│       ├── RAG_Output/             # Generation outputs (HAAS/, Mill/, LongContext/)
│       ├── batchprocess/           # LLM-judge requests and outputs
│       ├── csvs/                   # Word counts from PDF preprocessing
│       ├── final_merged/           # Merged results and analysis outputs
│       └── summary/                # Empty placeholder
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

### Source manuals (`data/input/`)

- `input_pdfs/`: the original manuals listed in the overview table. The code uses `HAAS_LATHE_OUTLINED.pdf` for the Haas lathe. No code in the repository reads `TL1_lathe operator's-manual.pdf`.
- `cropped_pdfs/`: margin-cropped copies of the full Bridgeport and UR5e manuals. The notebooks `system/main_long_context.ipynb` and `notebooks/synthetic_qa_generation.ipynb` write these files.

### Benchmarks (`data/QA/`)

| File | Rows | Columns |
|---|---|---|
| `MILL/Mill Feedback Accepted.csv` | 51 | `question`, `gold_answer` |
| `HAAS/Final HAAS Lathe QA.csv` | 51 | `question`, `gold_answer`, `Source Page`, `Difficulty Type`, `Relation` |
| `COBOT/Final_COBOT_QA.csv` | 64 | `question`, `gold_answer` |
| `COBOT/Final COBOT Lathe QA raw .csv` | 64 | Review sheet for the UR5e questions: original and revised wording, comments, answer, revised answer, `Source Page`, `Difficulty Type`, `Relation` |
| `*_test.csv` / `*_Test.csv` | 1-2 | Small subsets of the sets above, for test runs |
| `synthetic_qa_cobot_mill.csv` | 100 | `machine`, `question`, `gold_answer` (50 UR5e and 50 Bridgeport mill pairs generated by `notebooks/synthetic_qa_generation.ipynb`) |

The experiment code needs only the `question` and `gold_answer` columns.

### Preprocessed corpus (`data/preprocessed/`)

- `pdfs/combined/`: the 125 section PDFs that form the retrieval corpus (34 UR5e, 29 Bridgeport mill, and 62 Haas lathe sections). File names encode the table-of-contents level, section title, and page range, for example `003__L1__3._Safety__pp28-32.pdf`. These names are the same as the file lists in the `*_pdf_word_counts_final.csv` files.
- `vstore/combined/docs/docs.jsonl`: one JSON record per section PDF (`text`, plus `metadata.source` and `metadata.n_pages`). `CorpusBuilder` uses it as a cache.
- `vstore/combined/PKL`: a pickled LangChain `BM25Retriever` (the `lc_bm25` approach loads it). Load pickle files only from sources you trust.

### Prompts (`data/prompts/`)

- `ans_instr_A.txt`: answer instructions (answer from the retrieved sources only, cite file names, say when the sources do not contain the answer).
- `fewshot_A.txt`: system preamble with worked examples of the expected answer format.

### Results (`data/results/`)

| Path | Contents | Related code |
|---|---|---|
| `RAG_Output/HAAS/HAAS_RAG_OUTPUT-DONE.csv`, `RAG_Output/Mill/MILL_RAG_OUTPUT.csv` | Generated answers, one row per question and configuration (2,448 rows each) | `RAGExperimentRunner` |
| `RAG_Output/*/..._with_metrics.csv` | Same rows plus `cosine`, `rougeL`, `bleu`, and `q` (a copy of the reference answer) | `LangfairRunner` |
| `RAG_Output/*/Old/`, `RAG_Output/HAAS/HAAS_RAG_OUTPUT-DONE-Batch-Single-Test.csv` | Earlier backups, a small test file, and duplicate-check output | Intermediate files |
| `RAG_Output/LongContext/` | Two small long-context test runs (4 rows each, mill and UR5e) | `system/main_long_context.ipynb` |
| `batchprocess/BatchJson/HAAS_BATCH.jsonl`, `MILL_BATCH.jsonl` | LLM-judge requests (one helpfulness and one correctness request per answer) | `JudgeBatchBuilder` / `scripts/build_judge_batch.py` |
| `batchprocess/BatchJson/*_batch_output.jsonl`, `*_batch_output.json` | Raw Batch output, and the same output grouped per permutation | `scripts/fetch_batch_output.py` |
| `batchprocess/*_batch_output.csv` | One row per judge response (`custom_id`, `text`, `judge_model`, `temperature`, `permutation_id`, `judge_type`, `judge_answer`) | `scripts/batch_results_to_csv.py` |
| `batchprocess/*_batch_output_wide.csv` | One row per permutation with both judge results | `scripts/merge_judge_results.py` |
| `final_merged/haas.csv`, `final_merged/mill.csv` | `_with_metrics` results left-joined with the wide judge results on `permutation_id` (without `q`, `retrieved_files`, `meta_hits_text`) | `CSVColumnMerger` / `scripts/merge_csvs.py` |
| `final_merged/analysis_haas/`, `final_merged/analysis_mill/` | `aggregated_metrics.csv`, `summary.txt`, and four figures | `analyze_csv()` |
| `final_merged/analysis_haas.zip`, `final_merged/analysis_mill.zip` | The matching analysis folder plus the matching notebook from `notebooks/` | Archive copies |
| `csvs/` | Word counts per section PDF after cropping, after sub-splitting, and final | `PDFPreprocessor` |

The judge results in `batchprocess/` were produced with `gpt-5-2025-08-07` (column `judge_model`).

#### Columns of the generation output

| Column | Description |
|---|---|
| `permutation_id` | URL-safe ID that encodes the configuration and a run UUID, with a SHA-256 check suffix. Decode it with `system.utils.parse_permutation_id`. |
| `time_started`, `time_ended`, `total_elapsed_time` | Start and end timestamps (US Eastern time) and the wall-clock time of the retrieval and generation step |
| `min_words_for_subsplit` | Preprocessing setting, recorded for traceability |
| `approach`, `model`, `max_tokens`, `reasoning_effort`, `top_k` | Experimental factors |
| `answer_instructions_id`, `few_shot_id`, `replicate` | Prompt variant IDs (`A` or `B`) and replicate number |
| `question`, `gold_answer`, `generated_answer` | Question, reference answer, and model answer |
| `retrieved_files` | Semicolon-separated file names of the retrieved sections |
| `meta_hits_text` | The XML-formatted sources sent to the model |
| `meta_resp_id`, `meta_model`, `meta_status`, `meta_created`, `meta_input_tokens`, `meta_output_tokens`, `meta_total_tokens`, `meta_reason` | Response metadata returned by the API |

#### Columns added by evaluation

| Column | Description |
|---|---|
| `cosine` | Cosine similarity of TF-IDF vectors of the generated and reference answers |
| `rougeL` | ROUGE-L F-measure (with stemming) |
| `bleu` | Sentence BLEU with smoothing (NLTK `method1`) |
| `judge_answer_correctness_vs_ref` | `True`/`False`: is the answer factually consistent with the reference answer? |
| `judge_answer_helpfulness` | `True`/`False`: is the answer concise, relevant, and helpful for the question? (The judge reports this as `Relevance: True/False`.) |
| `text_correctness_vs_ref`, `text_helpfulness` | Full text of the judge's reasoning |

`aggregated_metrics.csv` groups the merged results by `approach`, `model`, `effort`, and `top_k`. It reports mean tokens, cosine, ROUGE-L, BLEU, estimated price, and latency; counts and percentages of `True` correctness and helpfulness grades; and min-max scaled scores. The price is a rough estimate in the code: total tokens x a fixed rate per million tokens (0.40 USD for models with "nano" in the name, 2.00 USD for "mini", 1.00 USD otherwise). The figures are `heatmap_avg_cosine.png`, `bar_correctness.png`, `bar_helpfulness.png`, and `bubble_correctness_helpfulness.png`.

## Pipeline

### 1. Preprocess (`system/preprocess/`)

`PDFPreprocessor` (`pdf_pipeline.py`) uses PyMuPDF to process one manual:

1. Split the manual into one PDF per table-of-contents entry at `toc_level` (default 1).
2. Crop a fixed fraction of each page margin (`crop_percent`, default 0.075; `system/main.py` uses 0.05 for the mill and lathe manuals).
3. Write word counts per file (`summary_csv_initial`).
4. Split files with more than `min_words_for_subsplit` words (default 3,000) again at TOC level `sub_level` (default 2), and write updated word counts.
5. Cut files that are still above the threshold into two halves (`_part01`, `_part02`), and write the final word counts.

Text extraction uses the PDF text layer. There is no OCR step.

`CorpusBuilder` (`corpus_builder.py`) then:

- loads each section PDF as one document (cached in `docs.jsonl`);
- builds and pickles a LangChain BM25 retriever;
- applies the `ShreddingTransformer` from `langchain-graph-retriever`, embeds the documents with OpenAI `text-embedding-3-small`, and adds them to an Astra DB collection;
- builds graph retrievers (EAGER and MMR strategies, edges on the `source` metadata) and a plain similarity retriever on the same collection.

`system/main.py` has one preprocessing function per manual (`preprocess_robot_arm`, `preprocess_bridgeport_lathe`, `preprocess_haas_lathe`) and `merge_bridge_and_haas_and_arm`, which builds the vector-store retrievers from `data/preprocessed/pdfs/combined/`. (The function names say "lathe" for the Bridgeport manual, but it is the milling machine manual.)

### 2. Retrieval and generation (`system/rag/`)

`ApproachRetrievers` implements the retrieval approaches selected with the `Approaches` flags:

| `approach` value | Retrieval |
|---|---|
| `openai_keyword` | OpenAI vector store search, `rewrite_query=False` |
| `openai_semantic` | OpenAI vector store search, `rewrite_query=True` |
| `lc_bm25` | LangChain BM25 retriever loaded from the pickle |
| `graph_eager` | Graph retriever on Astra DB, EAGER strategy (`start_k=1`, `max_depth=2`) |
| `graph_mmr` | Graph retriever on Astra DB, MMR strategy (`start_k=2`, `max_depth=2`) |
| `vanilla` | Astra DB similarity search |
| `long_context` | No retrieval: the full text of each document set with `set_long_context_texts()` is sent as context |

The repository has no code that creates the OpenAI vector store. The two `openai_*` approaches use the existing store given by `OPENAI_VECTOR_STORE_ID`.

`rag_utils.retrieve_and_answer()` formats the retrieved sections as XML (`<sources><result filename=... score=...><content>...</content></result></sources>`, each section cut to `max_chars_per_content` characters, default 25,000) and calls the OpenAI Responses API. The few-shot preamble is the system message, followed by the answer instructions and the sources plus query.

`RAGExperimentRunner` (`rag_generation.py`) runs every combination of approaches, models, `max_tokens`, reasoning efforts, `top_k` values, prompt variants, questions, and replicates. It appends results to the output CSV in batches of `max_concurrent` requests. The configuration that matches the committed results is in `system/main.py` (commented out): all six retrieval approaches, `gpt-5-nano-2025-08-07` and `gpt-5-mini-2025-08-07`, `max_tokens=5000`, reasoning effort `low` and `minimal`, `top_k` 3 and 7, prompts `A`, one replicate.

### 3. Evaluation (`system/rag/math/`, `system/evaluation/`)

- **Similarity metrics**: `LangfairRunner` with `LangfairMetricsCalculator` adds `cosine`, `rougeL`, and `bleu` to a results CSV. The input must have `generated_answer` and `gold_answer` columns. If no output path is given, it writes `<input>_with_metrics.csv`.
- **LLM judge**: `JudgeBatchBuilder` writes one helpfulness request (question and answer) and, when a reference answer exists, one correctness request (question, reference, and answer) per row to a Batch JSONL file. It can also upload the file and create the Batch job (judge model default: `gpt-5`). The prompt templates are `HELPFULNESS_INSTRUCTIONS` and `CORRECTNESS_INSTRUCTIONS` in `judge_batch.py`.
- **Batch results**: `BatchFetchRunner` downloads completed Batch output. `BatchResultsExporter` extracts the `True`/`False` grade from each judge response. `JudgeResultsMerger` pivots the grades to one row per `permutation_id`.

### 4. Merge and analysis (`system/evaluation/csv_column_merger.py`, `system/analysis/`, `notebooks/`)

- `CSVColumnMerger` joins the metrics file and the wide judge file on `permutation_id`.
- `analyze_csv()` writes `aggregated_metrics.csv`, the four figures, and `summary.txt` to an output folder.
- `notebooks/analysis haas.ipynb` and `notebooks/analysis mill.ipynb` contain the exploratory analysis of the two merged files (tables of the best configurations, heatmaps, bar charts, and bubble charts).

### Other notebooks

- `system/main_long_context.ipynb`: counts the tokens of the original and cropped mill and UR5e manuals (with `tiktoken`), runs the `long_context` approach on test subsets of the question sets, computes metrics, builds judge batches for those runs, and compares the long-context results with the retrieval approaches.
- `notebooks/synthetic_qa_generation.ipynb`: generates `data/QA/synthetic_qa_cobot_mill.csv`. It samples example pairs from the mill and UR5e question sets and text chunks from the cropped manuals, and asks a model for new question-answer pairs grounded in those chunks.

### Input checks

The code checks these conditions and stops with an error when one fails:

- The experiment input CSV has `question` and `gold_answer` columns, and `num_replicates` is at least 1 (`RAGExperimentRunner`).
- The metrics input CSV has `generated_answer` and `gold_answer` columns (`CSVProcessor`).
- Every row of the judge input CSV has a `permutation_id` (`load_judge_inputs_from_csv`).
- The long-form judge CSV has `custom_id`, `permutation_id`, `judge_type`, `judge_answer`, and `text` (`JudgeResultsMerger`).
- A join column is given, and the join type is `inner`, `outer`, `left`, or `right` (`CSVColumnMergeConfig`).
- A Batch job is `completed` and has an output file before its results are downloaded.
- Astra DB credentials are present before vector-store retrievers are built (`scripts/preprocess_documents.py`).
- `parse_permutation_id` verifies the check suffix of each ID.

`scripts/check_dups.py` finds duplicate rows in a results CSV (same `question`, `gold_answer`, `top_k`, `reasoning_effort`, `model`, and `approach`).

## Installation

1. Clone the repository. The download is about 180 MB, and the checked-out files take about 1 GB. Some file paths under `data/preprocessed/pdfs/combined/` are up to 153 characters long, so on Windows clone into a short directory or enable long paths in Git:

   ```bash
   git clone -c core.longpaths=true https://github.com/fmegahed/safety_rag_evaluation.git
   cd safety_rag_evaluation
   ```

2. Install the dependencies (the code uses `match` statements, so it needs Python 3.10 or later):

   ```bash
   pip install -r requirements.txt
   ```

   The notebooks also need Jupyter, and `system/main_long_context.ipynb` needs `tiktoken`. These are not in `requirements.txt`.

3. When `system.rag` is first imported, it downloads the NLTK `punkt` and `punkt_tab` tokenizers if they are not installed.

## Configuration

The code reads settings from a `.env` file with `python-dotenv` (`load_dotenv(override=True)`, so values in `.env` replace values already set in the environment). The `.env` file is not in the repository (`.gitignore` excludes it). Create it in the repository root. Never commit it.

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | Read by the OpenAI client: answer generation, embeddings, OpenAI vector store search, and the Batch API |
| `OPENAI_VECTOR_STORE_ID` | The `openai_keyword` and `openai_semantic` approaches |
| `ASTRA_DB_API_ENDPOINT` | Astra DB vector store (building it, and the `graph_eager`, `graph_mmr`, and `vanilla` approaches) |
| `ASTRA_DB_APPLICATION_TOKEN` | Same as above |
| `OPENAI_BATCH_ID` | Default Batch ID for `fetch_batch_output.py` and `check_batch_status.py`. `build_judge_batch.py --submit` writes it to `.env` unless you add `--skip-env-update`. |
| `ANTHROPIC_API_KEY` | Read by the Anthropic client. Needed only if you select a model whose ID starts with `claude-`. |

Other settings are in the code. `EnvironmentConfig` (`system/utils/environment_config.py`) sets the Astra DB collection name, the BM25 pickle path, the document store folder, and the embedding model (`text-embedding-3-small`). `system/main.py` changes these for the combined corpus: collection `COMBINED_ARM_HAAS_LATHE_MANUALS`, BM25 pickle `data/preprocessed/vstore/combined/PKL`, and document cache `data/preprocessed/vstore/combined/docs/docs.jsonl`.

## Usage

Run all commands from the repository root. The modules use package-relative imports and paths relative to the root, so run them with `python -m`.

### Orchestration script

```bash
python -m system.main
```

As committed, `main()` only runs `analyze_csv()` on `data/results/final_merged/haas.csv` and `mill.csv`. It writes `data/results/final_merged/analysis_haas/` and `analysis_mill/` again, and needs no API keys. The other steps (preprocessing, the experiment grid, metrics, judge batches, and merges) are commented-out calls in `main()`; they include the experiment-grid settings of the committed results. Uncomment the calls you need, and check their paths first: some refer to an older folder layout (for example, `data/QA/Final HAAS Lathe QA.csv` is now `data/QA/HAAS/Final HAAS Lathe QA.csv`).

Note: `RAGExperimentRunner` appends to its output CSV, and the other steps overwrite their outputs. Change the output paths if you want to keep the committed results.

### Command-line scripts (`system/scripts/`)

All scripts except `check_dups.py` accept `--help`. Several defaults point to a `results/` folder that is not in the repository, so give the paths explicitly.

| Command | Purpose | Main options |
|---|---|---|
| `python -m system.scripts.preprocess_documents` | Run the PDF pipeline on one manual and build retrieval assets (default input: the UR5e manual; builds BM25 only unless told otherwise) | `--input-pdf`, `--split-dir`, `--cropped-dir`, `--crop-percent`, `--min-words-for-subsplit`, `--skip-pdf`, `--store-dir`, `--skip-bm25`, `--build-vector-store`, `--build-graph`, `--build-vanilla-retriever`, `--collection-name`, `--top-k` |
| `python -m system.scripts.build_judge_batch` | Write judge requests for a results CSV, and optionally submit them | `--csv`, `--output`, `--judge-model` (default `gpt-5`), `--completion-window` (default `24h`), `--submit`, `--env-file`, `--skip-env-update` |
| `python -m system.scripts.check_batch_status` | Show the status of a Batch job | `--batch-id` |
| `python -m system.scripts.list_batches` | List Batch jobs for the API key | `--page-size`, `--max-batches`, `--show-json` |
| `python -m system.scripts.fetch_batch_output` | Download a completed Batch output (raw JSONL plus grouped JSON) | `--batch-id`, `--raw-output`, `--pivot-json` |
| `python -m system.scripts.batch_results_to_csv` | Convert local Batch output to a long-form judge CSV | `--raw-jsonl`, `--json`, `--csv` |
| `python -m system.scripts.merge_judge_results` | Pivot a long-form judge CSV to one row per permutation | `--input`, `--output` |
| `python -m system.scripts.merge_csvs` | Join two CSVs on a column | `--left`, `--right`, `--on` (required); `--how` (default `inner`), `--exclude` (repeatable), `--output` |
| `python -m system.scripts.check_dups <results.csv>` | Report duplicate configuration rows and write `<results>_duplicates.csv` | (positional CSV path) |

Example (no API calls): convert the committed raw Batch output for the mill into long-form and wide judge CSVs:

```bash
python -m system.scripts.batch_results_to_csv \
  --raw-jsonl data/results/batchprocess/BatchJson/mill_batch_output.jsonl \
  --csv mill_judge_long.csv
python -m system.scripts.merge_judge_results \
  --input mill_judge_long.csv \
  --output mill_judge_wide.csv
```

### Notebooks

Run each notebook with its own folder as the working directory (the Jupyter default). The notebooks use paths relative to that folder (for example `../data/...`). The long-context and synthetic-QA notebooks add the repository root to `sys.path` so that they can import `system`.

## License

MIT License. See [LICENSE](LICENSE).
