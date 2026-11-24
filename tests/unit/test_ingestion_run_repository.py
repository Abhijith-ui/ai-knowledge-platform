
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.ingestion_run import (
    IngestionRun,
    IngestionRunStatus,
    IngestionStage,
)
from ai_knowledge_platform.models.ingestion_stage_run import (
    IngestionStageStatus,
)
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.ingestion_run import (
    IngestionRunRepository,
)
from ai_knowledge_platform.repositories.user import UserRepository


MIGRATION_MARKER_KEY = "migration_revision"


def create_document(db: Session) -> Document:
    """Create a document for ingestion repository tests."""
    user = UserRepository(db).create("ingestion-owner@example.com")

    return DocumentRepository(db).create(
        user_id=user.id,
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user.id}/documents/{uuid4()}",
    )


def test_create_run_initializes_all_stages(db: Session) -> None:
    document = create_document(db)

    run = IngestionRunRepository(db).create_run(
        document_id=document.id,
    )

    assert run.document_id == document.id
    assert run.run_number == 1
    assert run.status == IngestionRunStatus.QUEUED
    assert run.trigger_type == "upload"
    assert run.config_snapshot == {}

    assert len(run.stage_runs) == len(IngestionStage)

    stage_statuses = {
        stage_run.stage: stage_run.status
        for stage_run in run.stage_runs
    }

    assert stage_statuses == {
        stage: IngestionStageStatus.PENDING
        for stage in IngestionStage
    }


def test_create_run_assigns_sequential_run_numbers(
    db: Session,
) -> None:
    document = create_document(db)
    repository = IngestionRunRepository(db)

    first_run = repository.create_run(document_id=document.id)
    second_run = repository.create_run(document_id=document.id)

    assert first_run.run_number == 1
    assert second_run.run_number == 2


def test_create_run_preserves_configuration_snapshot(
    db: Session,
) -> None:
    document = create_document(db)

    config_snapshot = {
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
        },
        "embedding": {
            "provider": "huggingface",
            "model": "test-model",
        },
    }

    run = IngestionRunRepository(db).create_run(
        document_id=document.id,
        trigger_type="retry",
        config_snapshot=config_snapshot,
    )

    assert run.trigger_type == "retry"
    assert run.config_snapshot == config_snapshot


def test_create_run_rejects_migration_trigger_type(
    db: Session,
) -> None:
    document = create_document(db)

    with pytest.raises(ValueError, match="reserved"):
        IngestionRunRepository(db).create_run(
            document_id=document.id,
            trigger_type="migration",
        )


def test_create_run_rejects_reserved_migration_metadata(
    db: Session,
) -> None:
    document = create_document(db)

    with pytest.raises(ValueError, match="reserved"):
        IngestionRunRepository(db).create_run(
            document_id=document.id,
            config_snapshot={
                MIGRATION_MARKER_KEY: "400442b8b4c3",
            },
        )


def test_create_run_rejects_nonexistent_document(
    db: Session,
) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        IngestionRunRepository(db).create_run(
            document_id=uuid4(),
        )
