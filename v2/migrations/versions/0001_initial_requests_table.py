"""initial requests table

Revision ID: 0001
Revises:
Create Date: 2026-08-31
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "requests",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("prompt", sa.String(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("predicted_p_correct", sa.Float(), nullable=True),
        sa.Column("error_budget", sa.Float(), nullable=False),
        sa.Column("chosen_tier", sa.String(), nullable=False),
        sa.Column("escalated", sa.Boolean(), default=False),
        sa.Column("input_tokens", sa.Integer(), default=0),
        sa.Column("output_tokens", sa.Integer(), default=0),
        sa.Column("cost_usd", sa.Float(), default=0.0),
        sa.Column("latency_ms", sa.Integer(), default=0),
        sa.Column("correct", sa.Boolean(), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_requests_created_at", "requests", ["created_at"])
    op.create_index("ix_requests_chosen_tier", "requests", ["chosen_tier"])


def downgrade():
    op.drop_index("ix_requests_chosen_tier", table_name="requests")
    op.drop_index("ix_requests_created_at", table_name="requests")
    op.drop_table("requests")
