"""pgvector-backed long-term memory for the multi-agent graph.

This is the RAG layer from the design doc (../../MULTI_AGENT.md): a retriever node
(nodes/retriever.py) calls retrieve_memory before planning, and the synthesizer calls
write_memory after a run so future requests can recall it. Scoped deliberately narrow -
long-term recall feeding into the graph, not the live inter-agent transport, which is
the shared AgentState object in state.py instead.
"""
import logging

from sqlalchemy import select

from app import db
from app.embeddings import get_embedder

logger = logging.getLogger("router.agents.memory")


def write_memory(text: str, metadata: dict) -> None:
    embedder = get_embedder()
    if embedder is None:
        logger.warning("embedder unavailable, skipping memory write")
        return
    embedding = embedder.encode(text).tolist()

    session = None
    try:
        session = db.get_session()
        session.add(db.MemoryEntry(text=text, embedding=embedding, entry_metadata=metadata))
        session.commit()
    except Exception as e:
        if session is not None:
            session.rollback()
        logger.error(f"failed to write memory entry (non-fatal): {e}")
    finally:
        if session is not None:
            session.close()


def retrieve_memory(query: str, k: int = 3) -> list[dict]:
    """Top-k memory entries by cosine similarity, formatted as MemoryHit dicts.
    Returns [] on any failure (missing embedder, empty table, DB error) - memory
    recall is an enhancement, never a hard dependency for the graph to run.
    """
    embedder = get_embedder()
    if embedder is None:
        return []
    query_embedding = embedder.encode(query).tolist()

    session = None
    try:
        session = db.get_session()
        # pgvector's cosine_distance operator (<=>) is 1 - cosine_similarity, so
        # similarity = 1 - distance. Ordering by distance ascending == similarity descending.
        distance = db.MemoryEntry.embedding.cosine_distance(query_embedding)
        rows = session.execute(
            select(db.MemoryEntry, distance.label("distance")).order_by(distance).limit(k)
        ).all()
        return [
            {
                "text": entry.text,
                "metadata": entry.entry_metadata,
                "similarity": 1.0 - dist,
            }
            for entry, dist in rows
        ]
    except Exception as e:
        logger.error(f"failed to retrieve memory (non-fatal, empty recall): {e}")
        return []
    finally:
        if session is not None:
            session.close()
