import enum
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ai_knowledge_platform.db.base import Base


class IngestionRunStatus(str, enum.Enum):
    """Represent the lifecycle of an ingestion run."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngestionStage(str, enum.Enum):
    """Define the stages of document ingestion."""

    EXTRACTION = "extraction"
    STRUCTURE_ANALYSIS = "structure_analysis"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    VECTOR_INDEXING = "vector_indexing"


class IngestionRun(Base):
    """Represent one versioned document ingestion execution."""

    __tablename__ = "ingestion_runs"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "run_number",
            name="uq_ingestion_runs_document_run_number",
        ),
        CheckConstraint(
            "run_number > 0",
            name="ck_ingestion_runs_positive_run_number",
        ),
        CheckConstraint(
            "embedding_dimensions IS NULL OR embedding_dimensions > 0",
            name="ck_ingestion_runs_positive_embedding_dimensions",
        ),
        Index(
            "ix_ingestion_runs_status_created_at",
            "status",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    run_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[IngestionRunStatus] = mapped_column(
        Enum(
            IngestionRunStatus,
            name="ingestion_run_status",
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=IngestionRunStatus.QUEUED,
    )

    current_stage: Mapped[IngestionStage | None] = mapped_column(
        Enum(
            IngestionStage,
            name="ingestion_stage",
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=True,
    )

    trigger_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="upload",
        server_default="upload",
    )

    config_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    embedding_provider: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_revision: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    embedding_dimensions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    vector_collection: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # Worker ownership and lease management.
    lease_token: Mapped[UUID | None] = mapped_column(
        Uuid,
        nullable=True,
    )

    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    worker_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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
