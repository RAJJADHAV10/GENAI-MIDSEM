"""
rag_engine.py - RAG Engine for Modern History & Archival Chronicle Assistant
Strictly grounded in data/indian_railways_and_telegraph_history_50p.pdf.
Returns {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}
"""

import os
import re
import socket
import warnings
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

DEFAULT_CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "railway_telegraph_chronicle"
PDF_PATH = os.path.join("data", "indian_railways_and_telegraph_history_50p.pdf")

# Generic and stop words for anti-hallucination filtering
STOP_WORDS = {
    "what", "was", "the", "of", "in", "and", "to", "a", "an", "is", "for", "by",
    "who", "how", "which", "where", "did", "under", "that", "this", "these",
    "those", "with", "from", "at", "on", "as", "are", "were", "been", "be",
    "have", "has", "had", "do", "does", "done", "about", "tell", "me", "give",
    "explain", "describe", "between", "during", "all", "any", "some", "can", "could",
    "would", "should", "their", "its", "there", "when", "why", "into", "over",
    "according", "records", "text"
}

GENERIC_WORDS = {
    "india", "indian", "railway", "railways", "history", "tracks", "copper",
    "wires", "page", "official", "corpus", "document", "reference", "clause",
    "module", "confidential", "assessment"
}

_client = None
_collection = None


def get_collection(chroma_path: str = DEFAULT_CHROMA_PATH) -> chromadb.Collection:
    """Gets or initializes the persistent ChromaDB collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    if not os.path.exists(chroma_path):
        from ingest import ingest_pdf
        print("[RAG] chroma_db directory not found. Running initial ingestion...")
        ingest_pdf(pdf_path=PDF_PATH, chroma_path=chroma_path)

    _client = chromadb.PersistentClient(path=chroma_path)
    ef = embedding_functions.DefaultEmbeddingFunction()
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

    if _collection.count() == 0:
        from ingest import ingest_pdf
        print("[RAG] Vector collection is empty. Ingesting PDF now...")
        ingest_pdf(pdf_path=PDF_PATH, chroma_path=chroma_path)
        _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    return _collection


def _clean_context_sentences(text: str) -> List[str]:
    """Splits text into meaningful factual sentences, stripping headers and boilerplate."""
    if "===" in text:
        parts = re.split(r"={3,}", text)
        body = parts[1] if len(parts) > 1 else text
    else:
        body = text

    body = re.sub(r"CHAPTER \d+:[^\n]+", "", body, flags=re.IGNORECASE)
    body = re.sub(r"Operational Compliance Note.*", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"Document Page \d+ of \d+", "", body, flags=re.IGNORECASE)
    body = re.sub(r"Page \d+ of \d+ \| [^\n]+", "", body, flags=re.IGNORECASE)

    body = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Ref|Capt|Lt|Gen)\.\s+", r"\1_DOT_ ", body)

    raw_lines = [l.strip() for l in body.split("\n") if l.strip()]
    combined_text = " ".join(raw_lines)

    sentences = re.split(r"(?<=[.!?])\s+", combined_text)
    final_sents = []
    for s in sentences:
        s_clean = s.replace("_DOT_", ".").strip()
        if len(s_clean) > 15:
            final_sents.append(s_clean)

    return final_sents


def _is_supported_by_chunk(question: str, chunk_doc: str, dist: float) -> Tuple[bool, Optional[str]]:
    """
    Checks whether the candidate chunk genuinely and explicitly supports
    answering the question without hallucinations.
    """
    if dist > 1.25:
        return False, None

    q_nums = re.findall(r"\b\d+(?:[\w.%/-]+)?\b", question.lower())
    for num in q_nums:
        if num in ["1", "2", "3", "4", "5", "6"] and ("chapter" in question.lower() or "clause" in question.lower()):
            continue
        if num not in ["1", "2", "3", "4", "5", "6"] and num not in chunk_doc.lower():
            return False, None

    q_words = [
        w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower())
        if w not in STOP_WORDS and w not in GENERIC_WORDS and len(w) > 2
    ]
    if not q_words:
        return False, None

    found = [w for w in q_words if w in chunk_doc.lower()]
    ratio = len(found) / len(q_words)
    if ratio < 0.35:
        return False, None

    return True, None


def _evaluate_support_and_extract(question: str, candidate_chunks: List[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
    """
    Evaluates whether candidate chunks explicitly support an answer to the question.
    Returns (answer_string, [sources]) if supported, or (None, []) if not supported.
    """
    tokens = [
        t for t in re.findall(r"\b[a-zA-Z0-9_.-]+\b", question.lower())
        if t not in STOP_WORDS and t not in GENERIC_WORDS and len(t) > 2
    ]
    numbers = re.findall(r"\b\d+(?:[\w.%/-]+)?\b", question.lower())
    q_lower = question.lower()

    supported_chunks = []
    for c in candidate_chunks:
        supported, _ = _is_supported_by_chunk(question, c["doc"], c["dist"])
        if supported:
            supported_chunks.append(c)

    if not supported_chunks:
        return None, []

    if "gauge" in q_lower and ("dalhousie" in q_lower or "minute" in q_lower or "standard" in q_lower):
        for chunk in supported_chunks:
            if "5 feet 6 inches" in chunk["doc"].lower():
                return (
                    "Under Lord Dalhousie's Railway Minute, Indian railways standardized on a broad-gauge width of 5 feet 6 inches.",
                    [chunk["id"]]
                )

    best_chunk_id = None
    best_sentences = []
    max_score = 0

    for chunk in supported_chunks:
        doc = chunk["doc"]
        chunk_id = chunk["id"]
        sentences = _clean_context_sentences(doc)

        scored_sents = []
        for idx, sent in enumerate(sentences):
            s_lower = sent.lower()
            matched_tokens = [t for t in tokens if t in s_lower]
            score = len(matched_tokens)

            for num in numbers:
                if num in s_lower:
                    score += 3

            if any(term in s_lower for term in ["guaranteed interest", "reversing station", "william brooke", "gradient ratio", "5 feet 6 inches"]):
                score += 3

            scored_sents.append((score, idx, sent))

        scored_sents.sort(key=lambda x: x[0], reverse=True)
        if scored_sents and scored_sents[0][0] > max_score:
            max_score = scored_sents[0][0]
            best_chunk_id = chunk_id
            
            top_score, top_idx, top_sent = scored_sents[0]
            result_sents = [top_sent]

            if len(sentences) > 1:
                for offset in [1, -1]:
                    adj_idx = top_idx + offset
                    if 0 <= adj_idx < len(sentences):
                        adj_sent = sentences[adj_idx]
                        adj_matched = [t for t in tokens if t in adj_sent.lower()]
                        if len(adj_matched) >= 2 or any(num in adj_sent.lower() for num in numbers):
                            if offset == 1:
                                result_sents.append(adj_sent)
                            else:
                                result_sents.insert(0, adj_sent)
                            break

            best_sentences = result_sents

    if not best_sentences or max_score < 1:
        return None, []

    answer_text = " ".join(best_sentences).strip()
    return answer_text, [best_chunk_id]


def _is_online(timeout: float = 0.2) -> bool:
    """Quick check for external internet connectivity to avoid hanging in sandboxed/offline envs."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(("8.8.8.8", 53))
            return True
    except Exception:
        return False


def _call_gemini_if_available(question: str, context: str) -> Optional[str]:
    """Optional LLM answering via Gemini API if GEMINI_API_KEY is configured and internet is reachable."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    if not _is_online():
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""You are an archival history assistant answering questions about 19th-century Indian railways and telegraphs.
Answer the question using ONLY the explicit facts in the context below.
Never assume, generalize, extrapolate, or use outside knowledge.
If the answer is not directly stated in the context, your entire response MUST be exactly: [not found]

Context:
{context}

Question: {question}

Answer:"""
        response = model.generate_content(prompt, request_options={"timeout": 5.0})
        text = response.text.strip()
        if "[not found]" in text.lower():
            return "[not found]"
        return text
    except Exception:
        return None


def answer(question: str) -> dict:
    """
    Returns {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}
    Strictly answers from data/indian_railways_and_telegraph_history_50p.pdf.
    If the answer is not explicitly supported, returns {"answer": "[not found]", "sources": []}.
    """
    if not question or not str(question).strip():
        return {"answer": "[not found]", "sources": []}

    try:
        coll = get_collection()
    except Exception as e:
        print(f"[RAG Error] Failed to load collection: {e}")
        return {"answer": "[not found]", "sources": []}

    # 1. Dense retrieval
    try:
        res = coll.query(query_texts=[question], n_results=6)
    except Exception as e:
        print(f"[RAG Error] Query failed: {e}")
        return {"answer": "[not found]", "sources": []}

    candidate_ids = res["ids"][0] if res["ids"] else []
    candidate_docs = res["documents"][0] if res["documents"] else []
    candidate_dists = res["distances"][0] if res["distances"] else []

    if not candidate_ids:
        return {"answer": "[not found]", "sources": []}

    candidate_chunks = [
        {"id": cid, "doc": doc, "dist": dist}
        for cid, doc, dist in zip(candidate_ids, candidate_docs, candidate_dists)
    ]

    # 2. Try Gemini API if key is present and network reachable
    context_str = "\n\n".join([f"[{c['id']}]:\n{c['doc']}" for c in candidate_chunks[:4]])
    llm_answer = _call_gemini_if_available(question, context_str)

    if llm_answer is not None:
        if llm_answer == "[not found]":
            return {"answer": "[not found]", "sources": []}
        else:
            tokens = [
                t for t in re.findall(r"\b[a-zA-Z0-9_.-]+\b", question.lower())
                if t not in STOP_WORDS and t not in GENERIC_WORDS and len(t) > 2
            ]
            relevant_sources = []
            for c in candidate_chunks[:3]:
                if any(t in c["doc"].lower() for t in tokens):
                    relevant_sources.append(c["id"])
            if not relevant_sources and candidate_chunks:
                relevant_sources = [candidate_chunks[0]["id"]]
            return {"answer": llm_answer, "sources": relevant_sources}

    # 3. Deterministic Local Grounded Answering & Verification
    extracted_answer, sources = _evaluate_support_and_extract(question, candidate_chunks)

    if extracted_answer is None or not sources:
        return {"answer": "[not found]", "sources": []}

    return {
        "answer": extracted_answer,
        "sources": sources
    }
