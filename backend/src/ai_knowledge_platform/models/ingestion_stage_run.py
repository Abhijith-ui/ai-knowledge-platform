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
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ai_knowledge_platform.db.base import Base
from ai_knowledge_platform.models.ingestion_run import IngestionStage


class IngestionStageStatus(str, enum.Enum):
    """Represent the lifecycle of an individual ingestion stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class IngestionStageRun(Base):
    """Track execution, retries, and metrics for an ingestion stage."""

    __tablename__ = "ingestion_stage_runs"

    __table_args__ = (
        UniqueConstraint(
            "ingestion_run_id",
            "stage",
            name="uq_ingestion_stage_runs_run_stage",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_ingestion_stage_runs_nonnegative_attempt_count",
        ),
        Index(
            "ix_ingestion_stage_runs_status_retry",
            "status",
            "next_retry_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
    )

    ingestion_run_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    stage: Mapped[IngestionStage] = mapped_column(
        Enum(
            IngestionStage,
            name="ingestion_stage",
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
    )

    status: Mapped[IngestionStageStatus] = mapped_column(
        Enum(
            IngestionStageStatus,
            name="ingestion_stage_run_status",
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=IngestionStageStatus.PENDING,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    metrics: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
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
