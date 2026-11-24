"""Add partial ingestion run status.

Revision ID: 230b4a4d1e68
Revises: 4469693a0f44
"""

from typing import Sequence, Union

from alembic import op

revision: str = "230b4a4d1e68"
down_revision: Union[str, Sequence[str], None] = "4469693a0f44"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add PARTIAL to the ingestion run lifecycle."""
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE ingestion_run_status ADD VALUE IF NOT EXISTS 'partial'")


def downgrade() -> None:
    """PostgreSQL does not safely support removing an enum value."""
    pass
