import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


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
