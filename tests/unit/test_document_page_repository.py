from uuid import UUID

import pytest
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.document_content import (
    DocumentContentRepository,
)
from ai_knowledge_platform.repositories.document_page import (
    DocumentPageRepository,
)
from ai_knowledge_platform.repositories.ingestion_run import (
    IngestionRunRepository,
)
from ai_knowledge_platform.repositories.user import UserRepository
from sqlalchemy.orm import Session


def create_document_content(db: Session) -> UUID:
    """Create document content for repository tests."""
    user: User = UserRepository(db).create("pages@example.com")

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

    content = DocumentContentRepository(db).create(
        ingestion_run_id=run.id,
        extracted_text="Page one content.\nPage two content.",
        character_count=len("Page one content.\nPage two content."),
        page_count=2,
    )

    return content.id


def test_replace_page_boundaries(db: Session) -> None:
    content_id = create_document_content(db)
    repository = DocumentPageRepository(db)

    pages = [
        (1, 0, 18),
        (2, 19, 37),
    ]

    created = repository.replace_for_content(
        document_content_id=content_id,
        pages=pages,
    )

    assert len(created) == 2
    assert created[0].page_number == 1
    assert created[1].page_number == 2
    assert all(page.document_content_id == content_id for page in created)


def test_list_pages_returns_ordered_results(db: Session) -> None:
    content_id = create_document_content(db)
    repository = DocumentPageRepository(db)

    repository.replace_for_content(
        document_content_id=content_id,
        pages=[
            (3, 38, 50),
            (1, 0, 18),
            (2, 19, 37),
        ],
    )

    pages = repository.list_by_content_id(content_id)

    assert [page.page_number for page in pages] == [1, 2, 3]


def test_replace_pages_removes_previous_records(db: Session) -> None:
    content_id = create_document_content(db)
    repository = DocumentPageRepository(db)

    repository.replace_for_content(
        document_content_id=content_id,
        pages=[
            (1, 0, 18),
            (2, 19, 37),
        ],
    )

    repository.replace_for_content(
        document_content_id=content_id,
        pages=[
            (1, 0, 10),
        ],
    )

    pages = repository.list_by_content_id(content_id)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].start_char == 0
    assert pages[0].end_char == 10


def test_replace_with_empty_pages_clears_existing_records(
    db: Session,
) -> None:
    content_id = create_document_content(db)
    repository = DocumentPageRepository(db)

    repository.replace_for_content(
        document_content_id=content_id,
        pages=[(1, 0, 18)],
    )

    repository.replace_for_content(
        document_content_id=content_id,
        pages=[],
    )

    assert repository.list_by_content_id(content_id) == []


def test_replace_allows_empty_page_boundaries(db: Session) -> None:
    content_id = create_document_content(db)
    repository = DocumentPageRepository(db)

    created = repository.replace_for_content(
        document_content_id=content_id,
        pages=[(1, 0, 0)],
    )

    assert len(created) == 1
    assert created[0].start_char == 0
    assert created[0].end_char == 0


@pytest.mark.parametrize(
    ("pages", "error_message"),
    [
        ([(0, 0, 10)], "Page number must be positive"),
        ([(-1, 0, 10)], "Page number must be positive"),
        (
            [(1, 0, 10), (1, 10, 20)],
            "Duplicate page numbers",
        ),
        ([(-1, 0, 10)], "Page number must be positive"),
        ([(1, -1, 10)], "start offset cannot be negative"),
        ([(1, 10, 5)], "end offset cannot precede start offset"),
    ],
)
def test_replace_rejects_invalid_page_boundaries(
    db: Session,
    pages: list[tuple[int, int, int]],
    error_message: str,
) -> None:
    content_id = create_document_content(db)
    repository = DocumentPageRepository(db)

    with pytest.raises(ValueError, match=error_message):
        repository.replace_for_content(
            document_content_id=content_id,
            pages=pages,
        )


def test_invalid_replacement_preserves_existing_pages(
    db: Session,
) -> None:
    content_id = create_document_content(db)
    repository = DocumentPageRepository(db)

    repository.replace_for_content(
        document_content_id=content_id,
        pages=[(1, 0, 18)],
    )

    with pytest.raises(ValueError):
        repository.replace_for_content(
            document_content_id=content_id,
            pages=[(0, 0, 10)],
        )

    pages = repository.list_by_content_id(content_id)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].end_char == 18


def test_list_pages_returns_empty_for_missing_content(
    db: Session,
) -> None:
    repository = DocumentPageRepository(db)

    pages = repository.list_by_content_id(UUID("00000000-0000-0000-0000-000000000001"))

    assert pages == []
