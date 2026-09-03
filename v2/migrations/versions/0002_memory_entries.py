"""memory_entries table for multi-agent long-term memory (pgvector)

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-03
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dim - see app/embeddings.py


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "memory_entries",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("entry_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    # Deliberately no ivfflat/HNSW index: ivfflat is an *approximate* index, and
    # verified directly against this table (not assumed from docs) that with only a
    # handful of rows it can silently return zero matches for a query that should hit
    # the table's only row - the partitioning into `lists` buckets doesn't have enough
    # data to route the query's probe to the right bucket. At this project's realistic
    # scale (a memory table for a portfolio demo, not production traffic), an exact
    # sequential scan is fast, correct, and never drops a real match - a plain
    # `ORDER BY embedding <=> query LIMIT k` on a bare column. Revisit only if this
    # table grows into the tens of thousands of rows and sequential scan cost shows up
    # in practice.


def downgrade():
    op.drop_table("memory_entries")
    op.execute("DROP EXTENSION IF EXISTS vector")
