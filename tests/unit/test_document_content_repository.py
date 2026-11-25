from uuid import UUID

import pytest
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.document_content import (
    DocumentContentRepository,
)
from ai_knowledge_platform.repositories.ingestion_run import (
    IngestionRunRepository,
)
from ai_knowledge_platform.repositories.user import UserRepository
from sqlalchemy.orm import Session


def create_ingestion_run(db: Session) -> UUID:
    """Create an ingestion run for repository tests."""

    user: User = UserRepository(db).create("content@example.com")

    document: Document = DocumentRepository(db).create(
        user_id=user.id,
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user.id}/documents/test.pdf",
    )

    run = IngestionRunRepository(db).create_run(
        document_id=document.id,
    )

    return run.id


def test_create_document_content(db: Session) -> None:
    ingestion_run_id = create_ingestion_run(db)
    extracted_text = "This is extracted document text."

    repository = DocumentContentRepository(db)

    content = repository.create(
        ingestion_run_id=ingestion_run_id,
        extracted_text=extracted_text,
        character_count=len(extracted_text),
        page_count=2,
    )

    assert content.id is not None
    assert content.ingestion_run_id == ingestion_run_id
    assert content.extracted_text == extracted_text
    assert content.character_count == len(extracted_text)
    assert content.page_count == 2


def test_get_document_content_by_ingestion_run_id(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)
    extracted_text = "Extracted content."

    repository = DocumentContentRepository(db)

    created = repository.create(
        ingestion_run_id=ingestion_run_id,
        extracted_text=extracted_text,
        character_count=len(extracted_text),
    )

    retrieved = repository.get_by_ingestion_run_id(ingestion_run_id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.extracted_text == extracted_text
    assert retrieved.page_count is None


def test_get_document_content_returns_none_when_missing(
    db: Session,
) -> None:
    repository = DocumentContentRepository(db)

    result = repository.get_by_ingestion_run_id(
        UUID("00000000-0000-0000-0000-000000000001")
    )

    assert result is None


def test_create_rejects_empty_extracted_text(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)

    with pytest.raises(ValueError, match="must not be empty"):
        DocumentContentRepository(db).create(
            ingestion_run_id=ingestion_run_id,
            extracted_text="",
            character_count=0,
        )


def test_create_rejects_incorrect_character_count(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)

    with pytest.raises(ValueError, match="text length"):
        DocumentContentRepository(db).create(
            ingestion_run_id=ingestion_run_id,
            extracted_text="Hello",
            character_count=10,
        )


def test_create_rejects_non_positive_page_count(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)

    with pytest.raises(ValueError, match="page_count must be positive"):
        DocumentContentRepository(db).create(
            ingestion_run_id=ingestion_run_id,
            extracted_text="Hello",
            character_count=5,
            page_count=0,
        )
