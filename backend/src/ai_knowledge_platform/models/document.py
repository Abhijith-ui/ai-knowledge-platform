import enum
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    String,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_knowledge_platform.db.base import Base

if TYPE_CHECKING:
    from ai_knowledge_platform.models.ingestion_run import IngestionRun


class DocumentStatus(str, enum.Enum):
    """Represent the lifecycle of a document's ingestion."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    """Store document metadata and processing state."""

    __tablename__ = "documents"

    __table_args__ = (
        ForeignKeyConstraint(
            ["active_ingestion_run_id", "id"],
            ["ingestion_runs.id", "ingestion_runs.document_id"],
            name="fk_documents_active_ingestion_run",
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)

    active_ingestion_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )

    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=DocumentStatus.PENDING,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    ingestion_runs: Mapped[list["IngestionRun"]] = relationship(
        back_populates="document",
        foreign_keys="IngestionRun.document_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    active_ingestion_run: Mapped["IngestionRun | None"] = relationship(
        foreign_keys=[active_ingestion_run_id],
        primaryjoin="Document.active_ingestion_run_id == IngestionRun.id",
        uselist=False,
        post_update=True,
    )
