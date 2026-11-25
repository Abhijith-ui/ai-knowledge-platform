from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_knowledge_platform.db.base import Base

if TYPE_CHECKING:
    from ai_knowledge_platform.models.document_content import DocumentContent


class DocumentPage(Base):
    """Store page boundaries within extracted document content."""

    __tablename__ = "document_pages"

    __table_args__ = (
        UniqueConstraint(
            "document_content_id",
            "page_number",
            name="uq_document_pages_content_page_number",
        ),
        CheckConstraint(
            "page_number > 0",
            name="ck_document_pages_positive_page_number",
        ),
        CheckConstraint(
            "start_char >= 0",
            name="ck_document_pages_nonnegative_start_char",
        ),
        CheckConstraint(
            "end_char >= start_char",
            name="ck_document_pages_valid_character_range",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    document_content_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_contents.id", ondelete="CASCADE"),
        nullable=False,
    )

    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    start_char: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    end_char: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    document_content: Mapped["DocumentContent"] = relationship(
        back_populates="pages",
    )
