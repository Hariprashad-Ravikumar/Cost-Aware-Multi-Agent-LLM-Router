import os
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

# Matches all-MiniLM-L6-v2's output dimension (app/embeddings.py) - the same embedder
# used for the calibrator's prompt_embedding feature, so a dimension mismatch here
# would mean the two have drifted apart, not an independent constant to tune.
EMBEDDING_DIM = 384


class RoutedRequest(Base):
    __tablename__ = "requests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt = Column(String, nullable=False)
    features = Column(JSON, nullable=False)
    predicted_p_correct = Column(Float, nullable=True)  # None for tiers not separately calibrated (mid/capable escalations)
    error_budget = Column(Float, nullable=False)
    chosen_tier = Column(String, nullable=False)
    escalated = Column(Boolean, default=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    latency_ms = Column(Integer, default=0)
    correct = Column(Boolean, nullable=True)  # only known in eval mode, where ground_truth is supplied
    cache_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MemoryEntry(Base):
    """Long-term memory for the multi-agent graph (app/agents/), retrieved via
    similarity search before planning - see migrations/versions/0002_memory_entries.py
    for the pgvector extension + ivfflat index this table depends on."""

    __tablename__ = "memory_entries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    text = Column(String, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    entry_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


_engine = None
_SessionLocal = None


def get_session():
    global _engine, _SessionLocal
    if _engine is None:
        url = os.environ.get("DATABASE_URL", "postgresql://router:router@localhost:5432/router")
        # pool_pre_ping: Neon (serverless Postgres) closes idle connections after a
        # period of inactivity; without this, SQLAlchemy hands out a stale connection
        # from the pool and the first query on it fails with "SSL connection has been
        # closed unexpectedly" - crashing an otherwise-successful request just because
        # the DB write at the end happened to reuse a dead connection. pool_pre_ping
        # validates the connection before each checkout and transparently reconnects.
        # pool_recycle caps how long a connection can live in the pool regardless, as a
        # second safety net against the same class of staleness.
        _engine = create_engine(url, pool_pre_ping=True, pool_recycle=300)
        # Schema is owned by Alembic (migrations/versions/0001_initial_requests_table.py),
        # not created ad hoc here - run `alembic upgrade head` before starting the service.
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()
