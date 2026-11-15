"""Central settings: change values here, not in the code."""
DOCS_DIR = "docs"
INDEX_DIR = "index_store"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3.2:3b"      # run: ollama pull llama3.2:3b

CHUNK_WORDS = 200              # words per chunk
CHUNK_OVERLAP = 40             # words shared between neighbouring chunks
TOP_K = 5                      # chunks sent to the LLM
CANDIDATES = 20                # chunks fetched before reranking
