"""
ingest.py - Ingestion pipeline for Modern History & Archival Chronicle Assistant
Loads and extracts text page-by-page from data/indian_railways_and_telegraph_history_50p.pdf,
splits into overlapping chunks, generates embeddings locally, and stores them in ChromaDB.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_PDF_PATH = os.path.join("data", "indian_railways_and_telegraph_history_50p.pdf")
DEFAULT_CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "railway_telegraph_chronicle"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embedding_function():
    """Initializes local Chroma default embedding function (all-MiniLM-L6-v2)."""
    return embedding_functions.DefaultEmbeddingFunction()


def extract_pages_from_pdf(pdf_path: str = DEFAULT_PDF_PATH) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from the target PDF using pypdf,
    preserving 1-indexed page numbers.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    pages_data = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"[Ingest] Loaded '{pdf_path}' with {total_pages} pages.")

        for idx, page in enumerate(reader.pages):
            page_number = idx + 1  # 1-indexed page numbering
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                print(f"[Ingest] Warning: Failed to extract page {page_number} via standard method: {e}")
                page_text = ""

            pages_data.append({
                "page_number": page_number,
                "text": page_text,
                "source": os.path.basename(pdf_path),
                "source_path": pdf_path
            })
    except Exception as e:
        print(f"[Ingest] Error opening PDF '{pdf_path}': {e}")
        raise

    return pages_data


def chunk_pages(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Splits page text into overlapping chunks using RecursiveCharacterTextSplitter,
    assigning deterministic IDs like page_<N>_chunk_<M> (1-indexed).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    for page in pages_data:
        page_num = page["page_number"]
        raw_text = page["text"]
        if not raw_text.strip():
            continue

        chunks = splitter.split_text(raw_text)
        for chunk_idx, chunk_content in enumerate(chunks, start=1):
            chunk_id = f"page_{page_num}_chunk_{chunk_idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_content,
                "metadata": {
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "source": page["source_path"],
                    "filename": page["source"]
                }
            })

    print(f"[Ingest] Generated {len(all_chunks)} chunks from {len(pages_data)} pages.")
    return all_chunks


def ingest_pdf(
    pdf_path: str = DEFAULT_PDF_PATH,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    force_reindex: bool = False
) -> chromadb.Collection:
    """
    Loads PDF, chunks text, generates local embeddings, and stores them in
    ChromaDB PersistentClient at ./chroma_db.
    """
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    ef = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "19th-Century Indian Railways and Telegraph History Archival Corpus"}
    )

    existing_count = collection.count()
    if existing_count > 0 and not force_reindex:
        print(f"[Ingest] Vector collection '{COLLECTION_NAME}' already populated with {existing_count} chunks. Skipping re-indexing.")
        return collection

    if force_reindex and existing_count > 0:
        print(f"[Ingest] Force re-index requested. Clearing existing {existing_count} chunks...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"description": "19th-Century Indian Railways and Telegraph History Archival Corpus"}
        )

    # Extract and chunk
    pages_data = extract_pages_from_pdf(pdf_path)
    chunks = chunk_pages(pages_data)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Ingest into Chroma in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"[Ingest] Indexed chunks {i+1} to {end} / {len(chunks)}")

    print(f"[Ingest] Successfully indexed {collection.count()} chunks into persistent ChromaDB at '{chroma_path}'.")
    return collection


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = DEFAULT_PDF_PATH
    for arg in sys.argv[1:]:
        if arg != "--force" and not arg.startswith("-"):
            path = arg
    ingest_pdf(pdf_path=path, force_reindex=force)
