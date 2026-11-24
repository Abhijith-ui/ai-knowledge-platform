"""Database operations for versioned document ingestion runs."""

from datetime import datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.document import Document, DocumentStatus
from ai_knowledge_platform.models.ingestion_run import (
    IngestionRun,
    IngestionRunStatus,
    IngestionStage,
)
from ai_knowledge_platform.models.ingestion_stage_run import (
    IngestionStageRun,
    IngestionStageStatus,
)

_RESERVED_TRIGGER_TYPES = {"migration"}
_RESERVED_CONFIG_KEYS = {"migration_revision"}

_TERMINAL_RUN_STATUSES = {
    IngestionRunStatus.SUCCEEDED,
    IngestionRunStatus.PARTIAL,
    IngestionRunStatus.FAILED,
    IngestionRunStatus.CANCELLED,
}


class IngestionRunRepository:
    """Persist ingestion runs and coordinate worker ownership."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def _database_now(self) -> datetime:
        """Return the database clock timestamp."""
        now = self.db.scalar(select(func.clock_timestamp()))

        if now is None:
            raise RuntimeError("Unable to retrieve database timestamp.")

        return cast(datetime, now)

    def create_run(
        self,
        *,
        document_id: UUID,
        trigger_type: str = "upload",
        config_snapshot: dict | None = None,
    ) -> IngestionRun:
        """Create an ingestion run and initialize all ingestion stages."""

        if not trigger_type.strip():
            raise ValueError("trigger_type must not be empty.")

        if len(trigger_type) > 32:
            raise ValueError("trigger_type must not exceed 32 characters.")

        if trigger_type in _RESERVED_TRIGGER_TYPES:
            raise ValueError(f"trigger_type '{trigger_type}' is reserved.")

        if config_snapshot is not None:
            if _RESERVED_CONFIG_KEYS.intersection(config_snapshot):
                raise ValueError(
                    "The migration_revision configuration key is reserved."
                )

        # Serialize run-number allocation for this document.
        document = self.db.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )

        if document is None:
            raise ValueError(f"Document {document_id} does not exist.")

        latest_run_number = self.db.scalar(
            select(func.max(IngestionRun.run_number)).where(
                IngestionRun.document_id == document_id
            )
        )

        run = IngestionRun(
            document_id=document_id,
            run_number=(latest_run_number or 0) + 1,
            status=IngestionRunStatus.QUEUED,
            trigger_type=trigger_type,
            config_snapshot=dict(config_snapshot or {}),
        )

        self.db.add(run)
        self.db.flush()

        stage_runs = [
            IngestionStageRun(
                ingestion_run_id=run.id,
                stage=stage,
                status=IngestionStageStatus.PENDING,
            )
            for stage in IngestionStage
        ]

        self.db.add_all(stage_runs)
        self.db.flush()

        return run

    def claim_next_run(
        self,
        *,
        worker_id: str,
        lease_duration: timedelta,
    ) -> IngestionRun | None:
        """
        Atomically claim the oldest eligible ingestion run.

        A run is eligible when it is queued or when its previous worker
        lease has expired. Reclaiming an expired run assigns a new token,
        fencing off the previous worker.
        """

        normalized_worker_id = worker_id.strip()

        if not normalized_worker_id:
            raise ValueError("worker_id must not be empty.")

        if len(normalized_worker_id) > 128:
            raise ValueError("worker_id must not exceed 128 characters.")

        if lease_duration.total_seconds() <= 0:
            raise ValueError("lease_duration must be positive.")

        now = self._database_now()

        eligible_run = or_(
            IngestionRun.status == IngestionRunStatus.QUEUED,
            and_(
                IngestionRun.status == IngestionRunStatus.RUNNING,
                IngestionRun.lease_expires_at.is_not(None),
                IngestionRun.lease_expires_at <= now,
            ),
        )

        statement = (
            select(IngestionRun)
            .where(eligible_run)
            .order_by(IngestionRun.created_at, IngestionRun.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        run = self.db.scalar(statement)

        if run is None:
            return None

        # Generate a fresh fencing token for every claim.
        run.status = IngestionRunStatus.RUNNING
        run.lease_token = uuid4()
        run.lease_expires_at = now + lease_duration
        run.heartbeat_at = now
        run.worker_id = normalized_worker_id

        if run.started_at is None:
            run.started_at = now

        # Keep the document summary synchronized only when this is
        # its active ingestion run.
        document = self.db.scalar(
            select(Document)
            .where(
                Document.id == run.document_id,
                Document.active_ingestion_run_id == run.id,
            )
            .with_for_update()
        )

        if document is not None:
            document.status = DocumentStatus.PROCESSING

        self.db.flush()

        return run

    def renew_lease(
        self,
        *,
        run_id: UUID,
        lease_token: UUID,
        lease_duration: timedelta,
    ) -> bool:
        """
        Extend a valid worker lease.

        An expired lease cannot be renewed, even if no other worker has
        reclaimed the run yet.
        """

        if lease_duration.total_seconds() <= 0:
            raise ValueError("lease_duration must be positive.")

        now = self._database_now()

        statement = (
            update(IngestionRun)
            .where(
                IngestionRun.id == run_id,
                IngestionRun.status == IngestionRunStatus.RUNNING,
                IngestionRun.lease_token == lease_token,
                IngestionRun.lease_expires_at > now,
            )
            .values(
                lease_expires_at=now + lease_duration,
                heartbeat_at=now,
            )
            .returning(IngestionRun.id)
        )

        renewed_run_id = self.db.scalar(statement)
        self.db.flush()

        return renewed_run_id is not None

    def complete_run(
        self,
        *,
        run_id: UUID,
        lease_token: UUID,
        status: IngestionRunStatus,
    ) -> bool:
        """
        Complete a run if the worker still owns a valid lease.

        SUCCEEDED requires all stages to be terminal and none to have failed.
        PARTIAL permits failed or skipped stages but requires no pending or
        running stages.
        """

        if status not in {
            IngestionRunStatus.SUCCEEDED,
            IngestionRunStatus.PARTIAL,
        }:
            raise ValueError("Run completion status must be SUCCEEDED or PARTIAL.")

        run = self._get_owned_run(
            run_id=run_id,
            lease_token=lease_token,
        )

        if run is None:
            return False

        unfinished_count = self.db.scalar(
            select(func.count())
            .select_from(IngestionStageRun)
            .where(
                IngestionStageRun.ingestion_run_id == run_id,
                IngestionStageRun.status.in_(
                    {
                        IngestionStageStatus.PENDING,
                        IngestionStageStatus.RUNNING,
                    }
                ),
            )
        )

        if unfinished_count:
            raise ValueError(
                "Cannot complete an ingestion run while stages remain unfinished."
            )

        failed_count = self.db.scalar(
            select(func.count())
            .select_from(IngestionStageRun)
            .where(
                IngestionStageRun.ingestion_run_id == run_id,
                IngestionStageRun.status == IngestionStageStatus.FAILED,
            )
        )

        if status == IngestionRunStatus.SUCCEEDED and failed_count:
            raise ValueError("A run with failed stages cannot be marked SUCCEEDED.")

        now = self._database_now()

        run.status = status
        run.completed_at = now
        run.lease_token = None
        run.lease_expires_at = None
        run.heartbeat_at = None
        run.worker_id = None

        self._synchronize_document_status(
            run=run,
            status=DocumentStatus.READY,
        )

        self.db.flush()

        return True

    def fail_run(
        self,
        *,
        run_id: UUID,
        lease_token: UUID,
    ) -> bool:
        """Mark a run failed if the worker still owns a valid lease."""

        run = self._get_owned_run(
            run_id=run_id,
            lease_token=lease_token,
        )

        if run is None:
            return False

        now = self._database_now()

        run.status = IngestionRunStatus.FAILED
        run.completed_at = now
        run.lease_token = None
        run.lease_expires_at = None
        run.heartbeat_at = None
        run.worker_id = None

        self._synchronize_document_status(
            run=run,
            status=DocumentStatus.FAILED,
        )

        self.db.flush()

        return True

    def _get_owned_run(
        self,
        *,
        run_id: UUID,
        lease_token: UUID,
    ) -> IngestionRun | None:
        """Lock and retrieve a run only when the lease is still valid."""

        now = self._database_now()

        statement = (
            select(IngestionRun)
            .where(
                IngestionRun.id == run_id,
                IngestionRun.status == IngestionRunStatus.RUNNING,
                IngestionRun.lease_token == lease_token,
                IngestionRun.lease_expires_at > now,
            )
            .with_for_update()
        )

        return self.db.scalar(statement)

    def _synchronize_document_status(
        self,
        *,
        run: IngestionRun,
        status: DocumentStatus,
    ) -> None:
        """Update the document summary only for its active ingestion run."""

        document = self.db.scalar(
            select(Document)
            .where(
                Document.id == run.document_id,
                Document.active_ingestion_run_id == run.id,
            )
            .with_for_update()
        )

        if document is not None:
            document.status = status
