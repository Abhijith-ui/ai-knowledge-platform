"""Database operations for individual ingestion stage executions."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.ingestion_run import (
    IngestionRun,
    IngestionRunStatus,
    IngestionStage,
)
from ai_knowledge_platform.models.ingestion_stage_run import (
    IngestionStageRun,
    IngestionStageStatus,
)


class IngestionStageRunRepository:
    """Persist ingestion stage state and enforce worker ownership."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_stage(
        self,
        *,
        ingestion_run_id: UUID,
        stage: IngestionStage,
    ) -> IngestionStageRun | None:
        """Retrieve a stage belonging to an ingestion run."""

        return self.db.scalar(
            select(IngestionStageRun).where(
                IngestionStageRun.ingestion_run_id == ingestion_run_id,
                IngestionStageRun.stage == stage,
            )
        )

    def _get_owned_run(
        self,
        *,
        ingestion_run_id: UUID,
        lease_token: UUID,
    ) -> tuple[IngestionRun, datetime] | None:
        """
        Lock a run and verify that the caller owns its unexpired lease.

        The database clock is queried after acquiring the row lock so that
        time spent waiting for the lock cannot make the timestamp stale.
        """

        run = self.db.scalar(
            select(IngestionRun)
            .where(IngestionRun.id == ingestion_run_id)
            .with_for_update()
        )

        if run is None:
            return None

        now = self.db.scalar(select(func.clock_timestamp()))

        if now is None:
            raise RuntimeError("Unable to retrieve database timestamp.")

        if (
            run.status != IngestionRunStatus.RUNNING
            or run.lease_token != lease_token
            or run.lease_expires_at is None
            or run.lease_expires_at <= now
        ):
            return None

        return run, now

    def _get_locked_stage(
        self,
        *,
        ingestion_run_id: UUID,
        stage: IngestionStage,
    ) -> IngestionStageRun | None:
        """Lock and retrieve a stage after its parent run has been locked."""

        return self.db.scalar(
            select(IngestionStageRun)
            .where(
                IngestionStageRun.ingestion_run_id == ingestion_run_id,
                IngestionStageRun.stage == stage,
            )
            .with_for_update()
        )

    def start_stage(
        self,
        *,
        ingestion_run_id: UUID,
        lease_token: UUID,
        stage: IngestionStage,
    ) -> bool:
        """
        Start a pending stage or a failed stage whose retry time has arrived.

        A successful transition increments the execution attempt count.
        """

        owned = self._get_owned_run(
            ingestion_run_id=ingestion_run_id,
            lease_token=lease_token,
        )

        if owned is None:
            return False

        _, now = owned

        stage_run = self._get_locked_stage(
            ingestion_run_id=ingestion_run_id,
            stage=stage,
        )

        if stage_run is None:
            return False

        is_pending = stage_run.status == IngestionStageStatus.PENDING

        is_retryable = (
            stage_run.status == IngestionStageStatus.FAILED
            and stage_run.next_retry_at is not None
            and stage_run.next_retry_at <= now
        )

        if not (is_pending or is_retryable):
            return False

        stage_run.status = IngestionStageStatus.RUNNING
        stage_run.attempt_count += 1
        stage_run.started_at = now
        stage_run.completed_at = None
        stage_run.next_retry_at = None
        stage_run.last_error = None

        self.db.flush()

        return True

    def complete_stage(
        self,
        *,
        ingestion_run_id: UUID,
        lease_token: UUID,
        stage: IngestionStage,
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        """Mark a running stage as successful and persist its metrics."""

        owned = self._get_owned_run(
            ingestion_run_id=ingestion_run_id,
            lease_token=lease_token,
        )

        if owned is None:
            return False

        _, now = owned

        stage_run = self._get_locked_stage(
            ingestion_run_id=ingestion_run_id,
            stage=stage,
        )

        if stage_run is None:
            return False

        if stage_run.status != IngestionStageStatus.RUNNING:
            return False

        stage_run.status = IngestionStageStatus.SUCCEEDED
        stage_run.completed_at = now
        stage_run.next_retry_at = None
        stage_run.last_error = None
        stage_run.metrics = dict(metrics) if metrics is not None else {}

        self.db.flush()

        return True

    def fail_stage(
        self,
        *,
        ingestion_run_id: UUID,
        lease_token: UUID,
        stage: IngestionStage,
        error_message: str,
        next_retry_at: datetime | None = None,
    ) -> bool:
        """
        Mark a running stage as failed.

        A retry timestamp makes the failure eligible for a future retry.
        None represents a terminal failure.
        """

        if not error_message.strip():
            raise ValueError("error_message must not be empty.")

        if next_retry_at is not None and next_retry_at.tzinfo is None:
            raise ValueError("next_retry_at must be timezone-aware.")

        owned = self._get_owned_run(
            ingestion_run_id=ingestion_run_id,
            lease_token=lease_token,
        )

        if owned is None:
            return False

        _, now = owned

        stage_run = self._get_locked_stage(
            ingestion_run_id=ingestion_run_id,
            stage=stage,
        )

        if stage_run is None:
            return False

        if stage_run.status != IngestionStageStatus.RUNNING:
            return False

        stage_run.status = IngestionStageStatus.FAILED
        stage_run.completed_at = now
        stage_run.next_retry_at = next_retry_at
        stage_run.last_error = error_message

        self.db.flush()

        return True

    def skip_stage(
        self,
        *,
        ingestion_run_id: UUID,
        lease_token: UUID,
        stage: IngestionStage,
    ) -> bool:
        """Mark a pending stage as skipped."""

        owned = self._get_owned_run(
            ingestion_run_id=ingestion_run_id,
            lease_token=lease_token,
        )

        if owned is None:
            return False

        _, now = owned

        stage_run = self._get_locked_stage(
            ingestion_run_id=ingestion_run_id,
            stage=stage,
        )

        if stage_run is None:
            return False

        if stage_run.status != IngestionStageStatus.PENDING:
            return False

        stage_run.status = IngestionStageStatus.SKIPPED
        stage_run.completed_at = now
        stage_run.next_retry_at = None

        self.db.flush()

        return True

    def recover_interrupted_stage(
        self,
        *,
        ingestion_run_id: UUID,
        lease_token: UUID,
        stage: IngestionStage,
        next_retry_at: datetime | None = None,
    ) -> bool:
        """
        Recover a stage left running by a worker that lost its lease.

        The new worker must own the current valid run lease. Recovery does
        not increment attempt_count because it does not execute the stage.
        """

        if next_retry_at is not None and next_retry_at.tzinfo is None:
            raise ValueError("next_retry_at must be timezone-aware.")

        owned = self._get_owned_run(
            ingestion_run_id=ingestion_run_id,
            lease_token=lease_token,
        )

        if owned is None:
            return False

        _, now = owned

        stage_run = self._get_locked_stage(
            ingestion_run_id=ingestion_run_id,
            stage=stage,
        )

        if stage_run is None:
            return False

        if stage_run.status != IngestionStageStatus.RUNNING:
            return False

        stage_run.status = IngestionStageStatus.FAILED
        stage_run.completed_at = now
        stage_run.next_retry_at = next_retry_at
        stage_run.last_error = (
            "Stage execution was interrupted because the worker lost its lease."
        )

        self.db.flush()

        return True
