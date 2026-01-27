# build_doc_index.py
import os
from pathlib import Path
from openai import OpenAI

from rag_faiss import get_or_build_store, DEFAULT_EMBED_MODEL

PDF_PATH = Path("agenda-tecnica-chiapas.pdf")
CACHE_DIR = Path(".rag_cache")

START_PAGE = 65
END_PAGE = 102

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first.")

    client = OpenAI()
    store = get_or_build_store(
        pdf_path=PDF_PATH,
        start_page_1idx=START_PAGE,
        end_page_1idx=END_PAGE,
        client=client,
        cache_dir=CACHE_DIR,
        embed_model=DEFAULT_EMBED_MODEL,
        force_rebuild=True,
    )
    print("Built store:", store.pages, "chunks:", len(store.chunks), "cache:", CACHE_DIR.resolve())

if __name__ == "__main__":
    main()