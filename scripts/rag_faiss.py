# rag_faiss.py
from __future__ import annotations

import json
import math
import os
import random
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# pip install faiss-cpu pymupdf openai
import faiss  # type: ignore
import fitz  # PyMuPDF
from openai import OpenAI


# ---------------------------
# Config defaults
# ---------------------------
DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # OpenAI embeddings v3 :contentReference[oaicite:2]{index=2}
DEFAULT_CHAT_MODEL = "gpt-5-nano"  # GPT-5 nano :contentReference[oaicite:3]{index=3}


# ---------------------------
# Helpers
# ---------------------------
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _backoff_sleep(attempt: int) -> None:
    # Exponential backoff recommended for rate limits :contentReference[oaicite:4]{index=4}
    time.sleep((2**attempt) + random.random())


def _safe_openai_call(fn, max_retries: int = 6):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            _backoff_sleep(attempt)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_text_by_pages(pdf_path: Path, start_page_1idx: int, end_page_1idx: int) -> List[Dict[str, Any]]:
    """
    Extract text from PDF pages [start_page_1idx, end_page_1idx] inclusive.
    Returns list of dicts: {"page": int (1-indexed), "text": str}
    """
    if start_page_1idx < 1 or end_page_1idx < start_page_1idx:
        raise ValueError("Invalid page range (must be 1-indexed and start<=end).")

    doc = fitz.open(str(pdf_path))
    out = []
    for p1 in range(start_page_1idx, end_page_1idx + 1):
        p0 = p1 - 1  # PyMuPDF is 0-indexed
        if p0 < 0 or p0 >= doc.page_count:
            continue
        page = doc.load_page(p0)
        txt = page.get_text("text") or ""
        txt = txt.strip()
        if txt:
            out.append({"page": p1, "text": txt})
    return out


def embed_texts(client: OpenAI, texts: List[str], embed_model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    # Batch to avoid request size issues
    B = 64
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), B):
        batch = texts[i : i + B]

        def _call():
            return client.embeddings.create(model=embed_model, input=batch)

        resp = _safe_openai_call(_call)
        data = resp.data
        # Keep original order
        batch_vecs = np.array([d.embedding for d in data], dtype=np.float32)
        vecs.append(batch_vecs)

    mat = np.vstack(vecs)
    return mat


@dataclass
class FaissRagStore:
    index: faiss.Index
    chunks: List[Dict[str, Any]]  # {id, page, text}
    embed_model: str
    pages: Tuple[int, int]


def build_store(
    pdf_path: Path,
    start_page_1idx: int,
    end_page_1idx: int,
    client: OpenAI,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> FaissRagStore:
    pages = extract_pdf_text_by_pages(pdf_path, start_page_1idx, end_page_1idx)

    chunks: List[Dict[str, Any]] = []
    for page_obj in pages:
        p = page_obj["page"]
        for c in chunk_text(page_obj["text"], chunk_chars=chunk_chars, overlap=overlap):
            chunks.append({"page": p, "text": c})

    texts = [c["text"] for c in chunks]
    emb = embed_texts(client, texts, embed_model=embed_model)

    # Cosine similarity = inner product after L2 norm
    emb = _l2_normalize(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # assign ids
    for i, c in enumerate(chunks):
        c["id"] = i

    return FaissRagStore(index=index, chunks=chunks, embed_model=embed_model, pages=(start_page_1idx, end_page_1idx))


def _store_cache_key(
    pdf_path: Path,
    pages: Tuple[int, int],
    embed_model: str,
    chunk_chars: int,
    overlap: int,
) -> str:
    s = f"{pdf_path.resolve()}|{pages[0]}-{pages[1]}|{embed_model}|{chunk_chars}|{overlap}"
    return _sha1(s)


def save_store(store: FaissRagStore, cache_dir: Path, key: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = cache_dir / f"{key}.faiss"
    meta_path = cache_dir / f"{key}.json"

    faiss.write_index(store.index, str(faiss_path))
    meta = {
        "embed_model": store.embed_model,
        "pages": list(store.pages),
        "chunks": store.chunks,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_store(cache_dir: Path, key: str) -> FaissRagStore:
    faiss_path = cache_dir / f"{key}.faiss"
    meta_path = cache_dir / f"{key}.json"
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError("FAISS store not found.")

    index = faiss.read_index(str(faiss_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return FaissRagStore(
        index=index,
        chunks=meta["chunks"],
        embed_model=meta["embed_model"],
        pages=(int(meta["pages"][0]), int(meta["pages"][1])),
    )


def get_or_build_store(
    pdf_path: Path,
    start_page_1idx: int,
    end_page_1idx: int,
    client: OpenAI,
    cache_dir: Path,
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_chars: int = 1200,
    overlap: int = 200,
    force_rebuild: bool = False,
) -> FaissRagStore:
    key = _store_cache_key(pdf_path, (start_page_1idx, end_page_1idx), embed_model, chunk_chars, overlap)
    if not force_rebuild:
        try:
            return load_store(cache_dir, key)
        except FileNotFoundError:
            pass

    store = build_store(
        pdf_path=pdf_path,
        start_page_1idx=start_page_1idx,
        end_page_1idx=end_page_1idx,
        client=client,
        embed_model=embed_model,
        chunk_chars=chunk_chars,
        overlap=overlap,
    )
    save_store(store, cache_dir=cache_dir, key=key)
    return store


def retrieve(
    store: FaissRagStore,
    client: OpenAI,
    query: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    def _call():
        return client.embeddings.create(model=store.embed_model, input=[query])

    resp = _safe_openai_call(_call)
    q = np.array([resp.data[0].embedding], dtype=np.float32)
    q = _l2_normalize(q)

    scores, idxs = store.index.search(q, top_k)
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(store.chunks):
            continue
        c = store.chunks[idx]
        results.append({"score": float(score), "page": int(c["page"]), "text": c["text"], "id": int(c["id"])})
    return results


def answer_with_rag_chat_completions(
    client: OpenAI,
    chat_model: str,
    question: str,
    retrieved: List[Dict[str, Any]],
    lang: str = "es",
    reco_context: Optional[Dict[str, Any]] = None,
) -> str:
    # Build compact context with page labels
    ctx_blocks = []
    for r in retrieved:
        ctx_blocks.append(f"[p. {r['page']}] {r['text']}")
    context = "\n\n".join(ctx_blocks)

    reco_note = ""
    if reco_context:
        # Use only as context; must not change rates
        reco_note = (
            "\n\nAPP RECOMMENDATION (do not change these rates):\n"
            f"- N: {reco_context.get('N', 'NA')} kg/ha\n"
            f"- P2O5: {reco_context.get('P2O5', 'NA')} kg/ha\n"
            f"- K2O: {reco_context.get('K2O', 'NA')} kg/ha\n"
        )

    system_es = (
        "Eres un asistente agronómico. Responde usando SOLAMENTE los extractos proporcionados "
        "de la Agenda Técnica (págs. 65–102). Si no hay evidencia en los extractos, di que "
        "no aparece en esas páginas.\n"
        "No inventes dosis nuevas. No modifiques las dosis recomendadas por la app; solo explica manejo práctico "
        "(momento, forma de aplicación, cuidados, riesgos, monitoreo, etc.).\n"
        "Siempre incluye citas de página en el texto, por ejemplo: (p. 72)."
    )
    system_en = (
        "You are an agronomy assistant. Answer using ONLY the provided excerpts "
        "from the Technical Agenda (pp. 65–102). If the excerpts don't contain the answer, say so.\n"
        "Do not invent new fertilizer rates. Do not change the app's recommended rates; only discuss practical management "
        "(timing, placement, precautions, risks, monitoring, etc.).\n"
        "Always include page citations like (p. 72)."
    )
    system = system_es if lang == "es" else system_en

    user = (
        f"QUESTION:\n{question}\n\n"
        f"SOURCE EXCERPTS:\n{context}"
        f"{reco_note}\n\n"
        "Write a practical, step-by-step answer. Keep it concise."
    )

    def _call():
        return client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    resp = _safe_openai_call(_call)
    return resp.choices[0].message.content