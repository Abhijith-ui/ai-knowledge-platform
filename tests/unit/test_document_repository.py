from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from ai_knowledge_platform.models.document import Document, DocumentStatus
from ai_knowledge_platform.models.document_content import DocumentContent
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.user import UserRepository
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tests.support.database import get_test_database_url


def create_user(db: Session, email: str) -> User:
    """Create a user for repository tests."""
    return UserRepository(db).create(email)


def create_document(
    db: Session,
    user_id: UUID,
    filename: str = "report.pdf",
) -> Document:
    """Create document metadata for repository tests."""
    return DocumentRepository(db).create(
        user_id=user_id,
        filename=filename,
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user_id}/documents/{uuid4()}",
    )


def test_create_document(db: Session) -> None:
    user = create_user(db, "owner@example.com")

    document = create_document(db, user.id)
    db.commit()

    persisted = db.get(Document, document.id)

    assert persisted is not None
    assert persisted.user_id == user.id
    assert persisted.filename == "report.pdf"
    assert persisted.size_bytes == 1024


def test_get_document_for_owner(db: Session) -> None:
    user = create_user(db, "owner@example.com")
    document = create_document(db, user.id)

    result = DocumentRepository(db).get_by_id_for_owner(
        document.id,
        user.id,
    )

    assert result is not None
    assert result.id == document.id


def test_cannot_get_another_users_document(db: Session) -> None:
    owner = create_user(db, "owner@example.com")
    another_user = create_user(db, "another@example.com")
    document = create_document(db, owner.id)

    result = DocumentRepository(db).get_by_id_for_owner(
        document.id,
        another_user.id,
    )

    assert result is None


def test_list_documents_only_returns_owners_documents(
    db: Session,
) -> None:
    owner = create_user(db, "owner@example.com")
    another_user = create_user(db, "another@example.com")

    first = create_document(db, owner.id, "first.pdf")
    second = create_document(db, owner.id, "second.pdf")
    create_document(db, another_user.id, "private.pdf")

    documents = DocumentRepository(db).list_by_owner(owner.id)

    assert len(documents) == 2
    assert {document.id for document in documents} == {
        first.id,
        second.id,
    }
    assert all(document.user_id == owner.id for document in documents)


def test_list_documents_respects_pagination(db: Session) -> None:
    user = create_user(db, "owner@example.com")

    for index in range(3):
        create_document(db, user.id, f"document-{index}.pdf")

    repository = DocumentRepository(db)

    first_page = repository.list_by_owner(user.id, limit=2, offset=0)
    second_page = repository.list_by_owner(user.id, limit=2, offset=2)

    assert len(first_page) == 2
    assert len(second_page) == 1
    assert {document.id for document in first_page}.isdisjoint(
        {document.id for document in second_page}
    )


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(0, 0), (101, 0), (10, -1)],
)
def test_list_documents_rejects_invalid_pagination(
    db: Session,
    limit: int,
    offset: int,
) -> None:
    repository = DocumentRepository(db)

    with pytest.raises(ValueError):
        repository.list_by_owner(
            uuid4(),
            limit=limit,
            offset=offset,
        )


def test_claim_next_pending_document(db: Session) -> None:
    user = create_user(db, "owner@example.com")
    document = create_document(db, user.id)

    claimed = DocumentRepository(db).claim_next_pending()

    assert claimed is not None
    assert claimed.id == document.id
    assert claimed.status == DocumentStatus.PROCESSING
    assert claimed.processing_attempts == 1
    assert claimed.processing_started_at is not None
    assert claimed.processing_token is not None
    assert claimed.next_attempt_at is None

    db.commit()

    persisted = db.get(Document, document.id)
    assert persisted is not None
    assert persisted.status == DocumentStatus.PROCESSING


def test_claim_returns_none_when_no_document_is_eligible(
    db: Session,
) -> None:
    repository = DocumentRepository(db)

    assert repository.claim_next_pending() is None


def test_claim_skips_document_with_future_retry_time(
    db: Session,
) -> None:
    user = create_user(db, "owner@example.com")
    document = create_document(db, user.id)
    document.next_attempt_at = datetime.now(UTC) + timedelta(minutes=5)
    db.flush()

    claimed = DocumentRepository(db).claim_next_pending()

    assert claimed is None
    assert document.status == DocumentStatus.PENDING
    assert document.processing_attempts == 0


def test_claim_allows_document_whose_retry_time_has_arrived(
    db: Session,
) -> None:
    user = create_user(db, "owner@example.com")
    document = create_document(db, user.id)
    document.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
    db.flush()

    claimed = DocumentRepository(db).claim_next_pending()

    assert claimed is not None
    assert claimed.id == document.id
    assert claimed.status == DocumentStatus.PROCESSING
    assert claimed.processing_attempts == 1


def test_concurrent_claim_skips_locked_document(db: Session) -> None:
    user = create_user(db, "concurrency@example.com")
    document = create_document(db, user.id)
    db.commit()

    engine = create_engine(get_test_database_url(), pool_pre_ping=True)

    try:
        with Session(engine) as first_session, Session(engine) as second_session:
            first_claim = DocumentRepository(first_session).claim_next_pending()

            assert first_claim is not None
            assert first_claim.id == document.id

            # The first transaction still holds the row lock.
            second_claim = DocumentRepository(second_session).claim_next_pending()

            assert second_claim is None

            first_session.commit()

            # The document is now PROCESSING, so it is no longer eligible.
            second_session.rollback()
            second_claim_after_commit = DocumentRepository(
                second_session
            ).claim_next_pending()
            assert second_claim_after_commit is None
    finally:
        engine.dispose()


def test_fail_processing_schedules_retry(db: Session) -> None:
    user = create_user(db, "retry@example.com")
    document = create_document(db, user.id)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None
    token = claimed.processing_token
    assert token is not None
    db.commit()

    before = datetime.now(UTC)

    result = repository.fail_processing(
        document_id=document.id,
        processing_token=token,
        error_message="Temporary storage failure.",
        retryable=True,
        max_attempts=3,
    )

    assert result is True
    assert document.status == DocumentStatus.PENDING
    assert document.processing_attempts == 1
    assert document.processing_token is None
    assert document.processing_started_at is None
    assert document.next_attempt_at is not None
    assert document.next_attempt_at >= before + timedelta(seconds=29)
    assert document.next_attempt_at <= datetime.now(UTC) + timedelta(seconds=31)
    assert document.last_processing_error == "Temporary storage failure."
    db.commit()


def test_fail_processing_uses_longer_backoff_for_second_attempt(
    db: Session,
) -> None:
    user = create_user(db, "backoff@example.com")
    document = create_document(db, user.id)
    repository = DocumentRepository(db)

    first_claim = repository.claim_next_pending()
    assert first_claim is not None
    first_token = first_claim.processing_token
    assert first_token is not None
    db.commit()

    assert repository.fail_processing(
        document_id=document.id,
        processing_token=first_token,
        error_message="First temporary failure.",
        retryable=True,
    )
    db.commit()

    document.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()

    second_claim = repository.claim_next_pending()
    assert second_claim is not None
    assert second_claim.processing_attempts == 2
    second_token = second_claim.processing_token
    assert second_token is not None
    db.commit()

    before = datetime.now(UTC)

    assert repository.fail_processing(
        document_id=document.id,
        processing_token=second_token,
        error_message="Second temporary failure.",
        retryable=True,
    )

    assert document.status == DocumentStatus.PENDING
    assert document.next_attempt_at is not None
    assert document.next_attempt_at >= before + timedelta(seconds=59)
    assert document.next_attempt_at <= datetime.now(UTC) + timedelta(seconds=61)


def test_fail_processing_marks_document_failed_after_max_attempts(
    db: Session,
) -> None:
    user = create_user(db, "exhausted@example.com")
    document = create_document(db, user.id)
    document.processing_attempts = 3
    document.status = DocumentStatus.PROCESSING
    document.processing_token = uuid4()
    document.processing_started_at = datetime.now(UTC)
    db.flush()

    token = document.processing_token
    assert token is not None

    result = DocumentRepository(db).fail_processing(
        document_id=document.id,
        processing_token=token,
        error_message="Temporary storage failure.",
        retryable=True,
        max_attempts=3,
    )

    assert result is True
    assert document.status == DocumentStatus.FAILED
    assert document.next_attempt_at is None
    assert document.processing_token is None
    assert document.processing_started_at is None
    assert document.last_processing_error == "Temporary storage failure."


def test_fail_processing_marks_permanent_failure_immediately(
    db: Session,
) -> None:
    user = create_user(db, "permanent@example.com")
    document = create_document(db, user.id)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None
    token = claimed.processing_token
    assert token is not None
    db.commit()

    result = repository.fail_processing(
        document_id=document.id,
        processing_token=token,
        error_message="Invalid document.",
        retryable=False,
    )

    assert result is True
    assert document.status == DocumentStatus.FAILED
    assert document.next_attempt_at is None
    assert document.processing_token is None
    assert document.processing_started_at is None
    db.commit()


def test_fail_processing_rejects_stale_token(db: Session) -> None:
    user = create_user(db, "stale@example.com")
    document = create_document(db, user.id)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None
    actual_token = claimed.processing_token
    assert actual_token is not None
    db.commit()

    result = repository.fail_processing(
        document_id=document.id,
        processing_token=uuid4(),
        error_message="Stale worker failure.",
        retryable=True,
    )

    assert result is False
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_token == actual_token
    assert document.last_processing_error is None
    assert db.get(DocumentContent, document.id) is None


@pytest.mark.parametrize(
    ("error_message", "retryable", "max_attempts"),
    [
        ("", True, 3),
        ("   ", True, 3),
        ("Some failure.", True, 0),
        ("Some failure.", True, -1),
    ],
)
def test_fail_processing_rejects_invalid_arguments(
    db: Session,
    error_message: str,
    retryable: bool,
    max_attempts: int,
) -> None:
    repository = DocumentRepository(db)

    with pytest.raises(ValueError):
        repository.fail_processing(
            document_id=uuid4(),
            processing_token=uuid4(),
            error_message=error_message,
            retryable=retryable,
            max_attempts=max_attempts,
        )


def test_complete_processing_persists_content_and_marks_ready(
    db: Session,
) -> None:
    user = create_user(db, "complete@example.com")
    document = create_document(db, user.id)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None
    token = claimed.processing_token
    assert token is not None
    db.commit()

    extracted_text = "This is extracted document text."

    result = repository.complete_processing(
        document_id=document.id,
        processing_token=token,
        extracted_text=extracted_text,
        character_count=len(extracted_text),
        page_count=2,
    )

    assert result is True
    assert document.status == DocumentStatus.READY
    assert document.processed_at is not None
    assert document.processing_token is None
    assert document.processing_started_at is None
    assert document.next_attempt_at is None
    assert document.last_processing_error is None

    content = db.get(DocumentContent, document.id)
    assert content is not None
    assert content.extracted_text == extracted_text
    assert content.character_count == len(extracted_text)
    assert content.page_count == 2

    db.commit()


def test_complete_processing_rejects_stale_token(db: Session) -> None:
    user = create_user(db, "stale-complete@example.com")
    document = create_document(db, user.id)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None
    actual_token = claimed.processing_token
    assert actual_token is not None
    db.commit()

    result = repository.complete_processing(
        document_id=document.id,
        processing_token=uuid4(),
        extracted_text="Must not be saved.",
        character_count=len("Must not be saved."),
        page_count=None,
    )

    assert result is False
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_token == actual_token
    assert db.get(DocumentContent, document.id) is None


def test_complete_processing_updates_existing_content(db: Session) -> None:
    user = create_user(db, "update-content@example.com")
    document = create_document(db, user.id)

    original_content = DocumentContent(
        document_id=document.id,
        extracted_text="Old text",
        character_count=8,
        page_count=1,
    )
    db.add(original_content)
    db.commit()

    repository = DocumentRepository(db)
    claimed = repository.claim_next_pending()
    assert claimed is not None
    token = claimed.processing_token
    assert token is not None
    db.commit()

    original_created_at = original_content.created_at
    updated_text = "Updated extracted text"

    result = repository.complete_processing(
        document_id=document.id,
        processing_token=token,
        extracted_text=updated_text,
        character_count=len(updated_text),
        page_count=3,
    )

    assert result is True
    assert original_content.extracted_text == updated_text
    assert original_content.character_count == len(updated_text)
    assert original_content.page_count == 3
    assert original_content.created_at == original_created_at
    assert original_content.updated_at >= original_created_at

    db.commit()

    contents = db.query(DocumentContent).filter_by(document_id=document.id).all()
    assert len(contents) == 1


def test_complete_processing_rollback_preserves_persisted_state(
    db: Session,
) -> None:
    user = create_user(db, "rollback-complete@example.com")
    document = create_document(db, user.id)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None
    token = claimed.processing_token
    assert token is not None
    db.commit()

    result = repository.complete_processing(
        document_id=document.id,
        processing_token=token,
        extracted_text="Text that should be rolled back.",
        character_count=len("Text that should be rolled back."),
        page_count=1,
    )
    assert result is True

    db.rollback()

    db.refresh(document)
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_token == token
    assert db.get(DocumentContent, document.id) is None
