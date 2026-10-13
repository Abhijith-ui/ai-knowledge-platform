"""create refresh sessions table

Revision ID: a41f9c72d6b0
Revises: 558d125d3e26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a41f9c72d6b0"
down_revision: Union[str, Sequence[str], None] = "558d125d3e26"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "refresh_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("family_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="ck_refresh_sessions_expiry_after_creation",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["replaced_by_id"],
            ["refresh_sessions.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(
        op.f("ix_refresh_sessions_user_id"),
        "refresh_sessions",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_refresh_sessions_family_id"),
        "refresh_sessions",
        ["family_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_refresh_sessions_family_id"),
        table_name="refresh_sessions",
    )
    op.drop_index(
        op.f("ix_refresh_sessions_user_id"),
        table_name="refresh_sessions",
    )
    op.drop_table("refresh_sessions")
