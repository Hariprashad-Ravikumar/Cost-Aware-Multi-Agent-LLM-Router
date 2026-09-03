"""Shared MiniLM embedder singleton.

Extracted from app/main.py so app/agents/memory.py can reuse the exact same model
instance for pgvector writes/retrieval instead of loading a second copy - the
calibrator's prompt_embedding feature and the multi-agent memory layer should embed
with the same model, not two independently-loaded ones.
"""
_embedder = None
_embedder_load_attempted = False


def get_embedder():
    global _embedder, _embedder_load_attempted
    if not _embedder_load_attempted:
        _embedder_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedder = None  # graceful degradation - callers handle None embeddings
    return _embedder
