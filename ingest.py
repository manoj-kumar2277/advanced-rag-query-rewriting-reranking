"""Run once after adding PDFs to docs/:  python ingest.py"""
import rag

pages, chunks = rag.build_index()
print(f"Indexed {pages} pages into {chunks} chunks.")
