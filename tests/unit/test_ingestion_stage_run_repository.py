from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.ingestion_run import (
    IngestionRun,
    IngestionStage,
)
from ai_knowledge_platform.models.ingestion_stage_run import (
    IngestionStageRun,
    IngestionStageStatus,
)
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.ingestion_run import (
    IngestionRunRepository,
)
from ai_knowledge_platform.repositories.ingestion_stage_run import (
    IngestionStageRunRepository,
)
from ai_knowledge_platform.repositories.user import UserRepository


def create_document(db: Session) -> Document:
    """Create a document for ingestion stage repository tests."""
    user = UserRepository(db).create(f"stage-owner-{uuid4()}@example.com")

    return DocumentRepository(db).create(
        user_id=user.id,
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user.id}/documents/{uuid4()}",
    )


def create_claimed_run(
    db: Session,
) -> tuple[IngestionRun, UUID]:
    """Create an ingestion run and claim it for a worker."""
    document = create_document(db)

    run_repository = IngestionRunRepository(db)

    run = run_repository.create_run(
        document_id=document.id,
    )

    claimed_run = run_repository.claim_next_run(
        worker_id="stage-test-worker",
        lease_duration=timedelta(minutes=5),
    )

    assert claimed_run is not None
    assert claimed_run.id == run.id
    assert claimed_run.lease_token is not None

    return claimed_run, claimed_run.lease_token


def get_stage(
    db: Session,
    ingestion_run_id: UUID,
    stage: IngestionStage,
) -> IngestionStageRun:
    """Retrieve a stage and fail clearly if it does not exist."""
    stage_run = IngestionStageRunRepository(db).get_stage(
        ingestion_run_id=ingestion_run_id,
        stage=stage,
    )

    assert stage_run is not None
    return stage_run


def test_get_stage_returns_stage_for_run(db: Session) -> None:
    run, _ = create_claimed_run(db)

    stage_run = IngestionStageRunRepository(db).get_stage(
        ingestion_run_id=run.id,
        stage=IngestionStage.EXTRACTION,
    )

    assert stage_run is not None
    assert stage_run.ingestion_run_id == run.id
    assert stage_run.stage == IngestionStage.EXTRACTION


def test_get_stage_returns_none_when_stage_does_not_exist(
    db: Session,
) -> None:
    repository = IngestionStageRunRepository(db)

    assert (
        repository.get_stage(
            ingestion_run_id=uuid4(),
            stage=IngestionStage.EXTRACTION,
        )
        is None
    )


def test_start_stage_increments_attempt_count(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    started = repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert started is True
    assert stage_run.status == IngestionStageStatus.RUNNING
    assert stage_run.attempt_count == 1
    assert stage_run.started_at is not None
    assert stage_run.completed_at is None


def test_start_stage_rejects_already_running_stage(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    assert repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    assert not repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert stage_run.attempt_count == 1


def test_complete_stage_persists_metrics(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    metrics = {
        "pages_extracted": 12,
        "characters_extracted": 15000,
    }

    completed = repository.complete_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
        metrics=metrics,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert completed is True
    assert stage_run.status == IngestionStageStatus.SUCCEEDED
    assert stage_run.metrics == metrics
    assert stage_run.completed_at is not None
    assert stage_run.next_retry_at is None
    assert stage_run.last_error is None


def test_complete_stage_defaults_metrics_to_empty_dictionary(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    completed = repository.complete_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert completed is True
    assert stage_run.metrics == {}


def test_complete_stage_rejects_non_running_stage(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    completed = repository.complete_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    assert completed is False


def test_fail_stage_records_retryable_failure(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    retry_at = datetime.now(UTC) + timedelta(minutes=5)

    failed = repository.fail_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
        error_message="Temporary extraction failure.",
        next_retry_at=retry_at,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert failed is True
    assert stage_run.status == IngestionStageStatus.FAILED
    assert stage_run.last_error == "Temporary extraction failure."
    assert stage_run.next_retry_at == retry_at
    assert stage_run.completed_at is not None


def test_fail_stage_records_terminal_failure(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    failed = repository.fail_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
        error_message="Unsupported document format.",
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert failed is True
    assert stage_run.status == IngestionStageStatus.FAILED
    assert stage_run.next_retry_at is None


def test_failed_stage_cannot_retry_before_retry_time(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    repository.fail_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
        error_message="Temporary failure.",
        next_retry_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    started = repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert started is False
    assert stage_run.status == IngestionStageStatus.FAILED
    assert stage_run.attempt_count == 1


def test_failed_stage_can_retry_after_retry_time(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    repository.fail_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
        error_message="Temporary failure.",
        next_retry_at=datetime.now(UTC) - timedelta(seconds=5),
    )

    started = repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert started is True
    assert stage_run.status == IngestionStageStatus.RUNNING
    assert stage_run.attempt_count == 2
    assert stage_run.next_retry_at is None


def test_terminal_failure_cannot_be_retried(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    repository.fail_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
        error_message="Permanent failure.",
    )

    started = repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    assert started is False


def test_skip_stage_marks_pending_stage_as_skipped(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    skipped = repository.skip_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EMBEDDING,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EMBEDDING,
    )

    assert skipped is True
    assert stage_run.status == IngestionStageStatus.SKIPPED
    assert stage_run.completed_at is not None
    assert stage_run.attempt_count == 0


def test_skip_stage_rejects_running_stage(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    skipped = repository.skip_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    assert skipped is False


def test_recover_interrupted_stage_preserves_attempt_count(
    db: Session,
) -> None:
    run, old_lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=old_lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert stage_run.attempt_count == 1

    # Simulate the previous worker's lease expiring.
    run.lease_expires_at = datetime.now(UTC) - timedelta(seconds=5)
    db.flush()

    # A new worker claims the expired run and receives a new token.
    new_claim = IngestionRunRepository(db).claim_next_run(
        worker_id="replacement-worker",
        lease_duration=timedelta(minutes=5),
    )

    assert new_claim is not None
    assert new_claim.id == run.id
    assert new_claim.lease_token is not None
    assert new_claim.lease_token != old_lease_token

    recovered = repository.recover_interrupted_stage(
        ingestion_run_id=run.id,
        lease_token=new_claim.lease_token,
        stage=IngestionStage.EXTRACTION,
        next_retry_at=datetime.now(UTC) + timedelta(seconds=10),
    )

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert recovered is True
    assert stage_run.status == IngestionStageStatus.FAILED
    assert stage_run.attempt_count == 1
    assert stage_run.completed_at is not None
    assert stage_run.next_retry_at is not None
    assert "interrupted" in stage_run.last_error.lower()


def test_old_worker_cannot_recover_stage_after_reclaim(
    db: Session,
) -> None:
    run, old_lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=old_lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    run.lease_expires_at = datetime.now(UTC) - timedelta(seconds=5)
    db.flush()

    new_claim = IngestionRunRepository(db).claim_next_run(
        worker_id="replacement-worker",
        lease_duration=timedelta(minutes=5),
    )

    assert new_claim is not None

    recovered = repository.recover_interrupted_stage(
        ingestion_run_id=run.id,
        lease_token=old_lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    assert recovered is False

    stage_run = get_stage(
        db,
        run.id,
        IngestionStage.EXTRACTION,
    )

    assert stage_run.status == IngestionStageStatus.RUNNING


def test_stage_operation_rejects_invalid_lease_token(
    db: Session,
) -> None:
    run, _ = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    started = repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=uuid4(),
        stage=IngestionStage.EXTRACTION,
    )

    assert started is False


def test_stage_operation_rejects_expired_lease(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    run.lease_expires_at = datetime.now(UTC) - timedelta(seconds=5)
    db.flush()

    started = IngestionStageRunRepository(db).start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    assert started is False


def test_fail_stage_rejects_empty_error_message(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        repository.fail_stage(
            ingestion_run_id=run.id,
            lease_token=lease_token,
            stage=IngestionStage.EXTRACTION,
            error_message="   ",
        )


def test_fail_stage_rejects_naive_retry_timestamp(
    db: Session,
) -> None:
    run, lease_token = create_claimed_run(db)
    repository = IngestionStageRunRepository(db)

    repository.start_stage(
        ingestion_run_id=run.id,
        lease_token=lease_token,
        stage=IngestionStage.EXTRACTION,
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        repository.fail_stage(
            ingestion_run_id=run.id,
            lease_token=lease_token,
            stage=IngestionStage.EXTRACTION,
            error_message="Temporary failure.",
            next_retry_at=datetime.now(),
        )
