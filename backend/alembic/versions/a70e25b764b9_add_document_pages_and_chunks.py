
"""add document pages and chunks

Revision ID: a70e25b764b9
Revises: 400442b8b4c3
Create Date: 2026-11-25 09:10:41.360711

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a70e25b764b9"
down_revision: Union[str, Sequence[str], None] = "400442b8b4c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create document page and chunk tables."""

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("ingestion_run_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column(
            "chunk_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start > 0 AND page_end >= page_start)",
            name="ck_document_chunks_valid_page_range",
        ),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name="ck_document_chunks_nonnegative_chunk_index",
        ),
        sa.CheckConstraint(
            "end_char > start_char",
            name="ck_document_chunks_valid_character_range",
        ),
        sa.CheckConstraint(
            "start_char >= 0",
            name="ck_document_chunks_nonnegative_start_char",
        ),
        sa.CheckConstraint(
            "token_count >= 0",
            name="ck_document_chunks_nonnegative_token_count",
        ),
        sa.ForeignKeyConstraint(
            ["ingestion_run_id"],
            ["ingestion_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "ingestion_run_id",
            "chunk_index",
            name="uq_document_chunks_run_chunk_index",
        ),
    )

    op.create_table(
        "document_pages",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_content_id", sa.UUID(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            "end_char >= start_char",
            name="ck_document_pages_valid_character_range",
        ),
        sa.CheckConstraint(
            "page_number > 0",
            name="ck_document_pages_positive_page_number",
        ),
        sa.CheckConstraint(
            "start_char >= 0",
            name="ck_document_pages_nonnegative_start_char",
        ),
        sa.ForeignKeyConstraint(
            ["document_content_id"],
            ["document_contents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_content_id",
            "page_number",
            name="uq_document_pages_content_page_number",
        ),
    )


def downgrade() -> None:
    """Drop document page and chunk tables."""

    op.drop_table("document_pages")
    op.drop_table("document_chunks")
