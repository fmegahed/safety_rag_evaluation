# Safety RAG Evaluation

A Python-based retrieval-augmented generation (RAG) system for evaluating different retrieval approaches on safety-critical documentation, specifically the UR5e Universal Robots User Manual.

## Overview

This project implements and compares multiple retrieval strategies for answering safety questions about industrial robotics equipment. It processes technical documentation, builds multiple retrieval indexes, and evaluates their effectiveness using various RAG approaches.

## Project Structure

```
safety_rag_evaluation/
├── code/                       # Python scripts for processing and retrieval
│   ├── 0_ur5e_multiple_pdfs.py    # PDF splitting and preprocessing
│   ├── 1_preprocess.py            # Document indexing and retriever setup
│   └── 2_rag.py                   # RAG query router and evaluation
├── pdfs/                       # Source documents
│   └── UR5e_Universal_Robots User Manual.pdf
├── results/                    # Output files
│   ├── csvs/                      # Word count summaries
│   ├── pdfs/                      # Split and cropped PDF sections
│   │   ├── ur5_splits/
│   │   └── ur5_splits_cropped/
│   └── rag_results.csv           # RAG evaluation results
├── retrieval_store/            # Serialized retrievers
│   ├── bm25/                      # BM25 retriever files
│   │   └── bm25_retriever.pkl
│   └── docs.jsonl                # Processed documents
├── figs/                       # Documentation figures
│   └── openai_vectorstore_settings.png
├── .env                        # Environment variables (not in repo)
└── requirements.txt            # Python dependencies
```

## Features

### PDF Processing (`0_ur5e_multiple_pdfs.py`)
- Splits source PDF using table of contents at configurable levels
- Crops pages to remove repeated margins and headers
- Automatically sub-splits large sections based on word count thresholds
- Generates word count statistics for each section
- Creates three CSV reports tracking processing stages

### Document Preprocessing (`1_preprocess.py`)
- Loads and indexes processed PDF sections
- Builds multiple retrieval systems:
  - **BM25 Retriever**: Traditional keyword-based search
  - **AstraDB Vector Store**: Semantic embeddings with OpenAI text-embedding-3-small
  - **Graph RAG**: Document graph with EAGER and MMR traversal strategies
  - **Vanilla RAG**: Standard similarity search
- Saves retrievers for reuse

### RAG Query Router (`2_rag.py`)
- Unified interface for comparing retrieval approaches:
  - `openai_semantic`: OpenAI file search with query rewriting
  - `openai_keyword`: OpenAI file search without rewriting
  - `lc_bm25`: LangChain BM25 retriever
  - `graph_eager`: Graph retrieval with eager traversal
  - `graph_mmr`: Graph retrieval with maximum marginal relevance
  - `vanilla`: Standard vector similarity search
- Safety-focused prompting with few-shot examples
- Logs results with retrieval scores, snippets, and token usage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd safety_rag_evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_VECTOR_STORE_ID=your_vector_store_id
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
```

## Configuration

Key parameters in `CONFIG` (code/1_preprocess.py and code/2_rag.py):
- `model`: OpenAI model (default: gpt-5-nano-2025-08-07)
- `max_tokens`: Maximum output tokens (default: 5000)
- `reasoning_effort`: Reasoning level (low/medium/high)
- `embed_model`: Embedding model (default: text-embedding-3-small)
- `top_k`: Number of documents to retrieve (default: 10)
- `max_chars_per_content`: Character limit per retrieved chunk (default: 25000)

## Usage

### Step 1: Process the PDF
```bash
python code/0_ur5e_multiple_pdfs.py
```

This splits the source manual into sections, crops pages, and generates word count statistics.

### Step 2: Build Retrievers
```bash
python code/1_preprocess.py
```

This creates BM25, vector, and graph-based retrievers from the processed documents.

### Step 3: Query with RAG
```bash
python code/2_rag.py
```

## Output

Results are saved to `results/rag_results.csv` with columns:
- `time`: Timestamp of query
- `question`: User question
- `approach`: Retrieval method used
- `filename`: Source document
- `score`: Retrieval score
- `snippet`: Text preview (first 200 chars)
- `answer`: Generated response
- `resp_id`, `model`, `input_tokens`, `output_tokens`: API metadata



