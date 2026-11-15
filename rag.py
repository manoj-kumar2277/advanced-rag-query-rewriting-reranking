"""Core RAG pipeline: ingestion, basic retrieval, query rewriting, reranking, answering."""
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import ollama
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

import config

MODES = ["basic", "rewrite", "rerank", "both"]
_cache = {}


def _embedder():
    if "emb" not in _cache:
        _cache["emb"] = SentenceTransformer(config.EMBED_MODEL)
    return _cache["emb"]


def _reranker():
    if "rr" not in _cache:
        _cache["rr"] = CrossEncoder(config.RERANK_MODEL)
    return _cache["rr"]


# ---------- 1. Ingestion ----------
NOISE_PATTERNS = [
    r"https?://\S+",
    r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2}[^|]{0,120}\| National Cyber Security Centre",
    r"Was this article helpful\?\s*(Yes\s*)?(No\s*)?",
    r"Share and print this article.{0,80}",
    r"Previous page:.{0,60}?Topics",
    r"Next page:.{0,60}?(?=\s[A-Z])",
    r"Also see\s.*?(?=\d{2}/\d{2}/\d{4}|$)",
    r"Skip to main content",
]


def clean_text(text):
    for p in NOISE_PATTERNS:
        text = re.sub(p, " ", text, flags=re.I | re.S)
    return re.sub(r"\s+", " ", text).strip()


def load_pdfs(folder=config.DOCS_DIR):
    pages = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        for i, page in enumerate(PdfReader(str(pdf)).pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append({"source": pdf.name, "page": i, "text": text})
    return pages


def chunk_pages(pages, size=config.CHUNK_WORDS, overlap=config.CHUNK_OVERLAP):
    chunks = []
    step = size - overlap
    for pg in pages:
        words = pg["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + size]
            chunks.append({"source": pg["source"], "page": pg["page"],
                           "text": " ".join(piece)})
            if start + size >= len(words):
                break
    return chunks


def build_index():
    pages = load_pdfs()
    if not pages:
        raise SystemExit(f"No PDFs found in '{config.DOCS_DIR}/'. Add some first.")
    chunks = chunk_pages(pages)
    vecs = _embedder().encode([c["text"] for c in chunks],
                              normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.asarray(vecs, dtype="float32"))
    out = Path(config.INDEX_DIR)
    out.mkdir(exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    (out / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return len(pages), len(chunks)


def _load_store():
    if "index" not in _cache:
        out = Path(config.INDEX_DIR)
        _cache["index"] = faiss.read_index(str(out / "faiss.index"))
        _cache["chunks"] = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
    return _cache["index"], _cache["chunks"]


# ---------- 2. Retrieval ----------
def vector_search(query, k):
    index, chunks = _load_store()
    q = _embedder().encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [dict(chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


REWRITE_PROMPT = """Rewrite the question as a short search query (max 15 words) for a
cyber security document search.
Rules:
- Keep the user's own key words.
- Add 3-5 helpful keywords from cyber security.
- Do NOT invent a topic the question does not imply.
- Return ONLY the query, no explanation.

Question: {q}
Query:"""


def rewrite_query(query):
    try:
        r = ollama.chat(model=config.LLM_MODEL,
                        messages=[{"role": "user", "content": REWRITE_PROMPT.format(q=query)}],
                        options={"temperature": 0})
        text = r["message"]["content"].strip().strip('"').splitlines()[0]
        return (query+chr(32)+text) if text else query  # v3
    except Exception as e:  # LLM down -> fall back to original query
        print("Rewrite failed:", e)
        return query


def rerank(query, candidates, k=config.TOP_K):
    scores = _reranker().predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:k]
    return [dict(c, score=float(s)) for s, c in ranked]


def retrieve(query, mode="basic"):
    """mode: basic | rewrite | rerank | both. Returns chunks, search query and timings."""
    assert mode in MODES
    use_rewrite = mode in ("rewrite", "both")
    use_rerank = mode in ("rerank", "both")
    t0 = time.perf_counter()

    search_q = rewrite_query(query) if use_rewrite else query
    t_rewrite = time.perf_counter() - t0

    t1 = time.perf_counter()
    if use_rerank:
        chunks = rerank(search_q, vector_search(search_q, config.CANDIDATES))
    else:
        chunks = vector_search(search_q, config.TOP_K)
    t_search = time.perf_counter() - t1

    return {"query": query, "search_query": search_q, "chunks": chunks,
            "t_rewrite": t_rewrite, "t_search": t_search,
            "t_retrieval": time.perf_counter() - t0}


# ---------- 3. Generation ----------
ANSWER_PROMPT = """Answer the question using ONLY the context below.
Cite sources like [1], [2]. If the context does not contain the answer, say
"I could not find this in the documents."

Context:
{ctx}

Question: {q}
Answer:"""


def answer(query, chunks):
    ctx = "\n\n".join(f"[{i}] ({c['source']}, p.{c['page']}) {c['text']}"
                      for i, c in enumerate(chunks, 1))
    t = time.perf_counter()
    r = ollama.chat(model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": ANSWER_PROMPT.format(ctx=ctx, q=query)}],
                    options={"temperature": 0})
    return r["message"]["content"].strip(), time.perf_counter() - t
