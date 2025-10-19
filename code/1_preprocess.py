# ---------------------------------------------------------------------------
# Imports and setup
# ---------------------------------------------------------------------------
from pathlib import Path
import os
import json
import pickle

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever

from langchain_openai import OpenAIEmbeddings as LC_OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_graph_retriever.transformers import ShreddingTransformer
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager, Mmr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "model": "gpt-5-nano-2025-08-07",
    "max_tokens": 5000,
    "reasoning_effort": "low",
    "embed_model": "text-embedding-3-small",
    "top_k": 10,
}


load_dotenv(override=True)
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# ---------------------------------------------------------------------------
# (1) Define folders and file paths
# ---------------------------------------------------------------------------
PDF_DIR = Path("results/pdfs/ur5_splits_cropped")

STORE_DIR = Path("retrieval_store")

BM25_DIR = STORE_DIR / "bm25"

DOCS_JSONL = STORE_DIR / "docs.jsonl"
BM25_PKL = BM25_DIR / "bm25_retriever.pkl"

# Ensure all folders exist
BM25_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# (2) Helper functions
# ---------------------------------------------------------------------------
def load_docs_from_jsonl(path: Path) -> list[Document]:
    """Load Document objects from an existing JSONL file."""
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(
                Document(
                    page_content=rec["text"],
                    metadata=rec.get("metadata", {})
                )
            )
    return docs


def save_docs_to_jsonl(docs: list[Document], path: Path) -> None:
    """Save Document objects to JSONL so they can be reused later."""
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {"text": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_pdfs_as_documents(pdf_dir: Path) -> list[Document]:
    """Read each PDF, join all its pages, and create one Document per file."""
    docs = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()
        full_text = "\n\n".join(p.page_content for p in pages).strip()
        if not full_text:
            continue
        docs.append(
            Document(
                page_content=full_text,
                metadata={"source": pdf_path.name, "n_pages": len(pages)}
            )
        )
    return docs


# ---------------------------------------------------------------------------
# (3) Load or create document set
# ---------------------------------------------------------------------------
if DOCS_JSONL.exists():
    print(f"Loading documents from {DOCS_JSONL} ...")
    docs = load_docs_from_jsonl(DOCS_JSONL)
else:
    print(f"Reading PDFs from {PDF_DIR} ...")
    docs = load_pdfs_as_documents(PDF_DIR)
    if not docs:
        raise SystemExit("No documents found. Check the PDF folder path.")
    print(f"Saving {len(docs)} documents to {DOCS_JSONL} ...")
    save_docs_to_jsonl(docs, DOCS_JSONL)

print(f"Total documents ready: {len(docs)}")


# ---------------------------------------------------------------------------
# (4) Build and save BM25 retriever
# ---------------------------------------------------------------------------
print("Building BM25 retriever ...")
bm25 = BM25Retriever.from_documents(docs)

print(f"Saving BM25 retriever to {BM25_PKL} ...")
with BM25_PKL.open("wb") as f:
    pickle.dump(bm25, f)

print("✅ BM25 retriever built and saved successfully.")


# ---------------------------------------------------------------------------
# (5) Load ASTRA DB Vector store
# ---------------------------------------------------------------------------
def build_astradb_vector_store():
    print(f"Loading documents from {DOCS_JSONL} ...")
    docs = load_docs_from_jsonl(DOCS_JSONL)
    embeddings = LC_OpenAIEmbeddings(model=CONFIG['embed_model'])
    
    print("Building AstraDB vector store ...")
    shredded_docs = list(ShreddingTransformer().transform_documents(docs))
    
    vector_store = AstraDBVectorStore(
        collection_name="ur5_manual",
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
    
    # Add documents (skip if already exists)
    try:
        vector_store.add_documents(shredded_docs)
        print(f"Added {len(shredded_docs)} documents to AstraDB")
    except Exception as e:
        print(f"Documents may already exist: {e}")
    
    return vector_store

vector_store = build_astradb_vector_store()
print("✅ AstraDB vector store built and saved successfully.")


# ---------------------------------------------------------------------------
# (6) Graph RAG retrievers
# ---------------------------------------------------------------------------
def build_graph_rag_retrievers(
    vector_store: AstraDBVectorStore,
    k: int = CONFIG['top_k'],
    start_k_eager: int = 1,
    start_k_mmr: int = 2,
    max_depth: int = 2,
    edges: list[tuple[str, str]] | None = None,
):
    if edges is None:
        edges = [("source", "source")]

    eager_retriever = GraphRetriever(
        store=vector_store,
        edges=edges,
        strategy=Eager(k=k, start_k=start_k_eager, max_depth=max_depth),
    )

    mmr_retriever = GraphRetriever(
        store=vector_store,
        edges=edges,
        strategy=Mmr(k=k, start_k=start_k_mmr, max_depth=max_depth),
    )

    print("✅ Graph RAG retrievers ready (EAGER + MMR).")
    return {"EAGER": eager_retriever, "MMR": mmr_retriever}


retrievers = build_graph_rag_retrievers(vector_store)


# ---------------------------------------------------------------------------
# (7) Vanilla RAG
# ---------------------------------------------------------------------------
vanilla_retriever = vector_store.as_retriever(search_kwargs={"k": CONFIG['top_k']})
print("✅ Vanilla RAG retriever ready (standard similarity retriever).")
