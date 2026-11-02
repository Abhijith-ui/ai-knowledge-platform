from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from ai_knowledge_platform.models.document import Document, DocumentStatus
from ai_knowledge_platform.models.document_content import DocumentContent
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.user import UserRepository
from ai_knowledge_platform.services.document_extraction import (
    DocumentExtractionError,
    DocumentTextExtractor,
    ExtractionResult,
)
from ai_knowledge_platform.services.document_processing import (
    DocumentProcessingService,
)
from ai_knowledge_platform.storage.base import StorageError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker


class FakeExtractor:
    """Return a configured extraction result or raise a configured error."""

    def __init__(
        self,
        result: ExtractionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error

    def extract(
        self,
        *,
        storage_key: str,
        content_type: str,
    ) -> ExtractionResult:
        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError("No extraction result configured.")

        return self.result


@pytest.fixture
def session_factory(db: Session) -> Callable[[], Session]:
    """Create independent sessions using the dedicated test database."""
    factory = sessionmaker(
        bind=db.get_bind(),
        autoflush=False,
        autocommit=False,
    )
    return factory


def create_pending_document(db: Session) -> Document:
    user = UserRepository(db).create(f"processing-test-{uuid4()}@example.com")

    document = DocumentRepository(db).create(
        user_id=user.id,
        filename="report.txt",
        content_type="text/plain",
        size_bytes=100,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user.id}/documents/report.txt",
    )
    db.commit()
    return document


def load_document(
    session_factory: Callable[[], Session],
    document_id: UUID,
) -> Document:
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document is not None
        return document


def test_returns_idle_when_no_document_is_pending(
    session_factory: Callable[[], Session],
) -> None:
    service = DocumentProcessingService(
        session_factory=session_factory,
        extractor=cast(DocumentTextExtractor, FakeExtractor()),
    )

    outcome = service.process_next_document()

    assert outcome.status == "idle"
    assert outcome.document_id is None


def test_successful_processing_persists_extracted_content(
    db: Session,
    session_factory: Callable[[], Session],
) -> None:
    document = create_pending_document(db)
    expected_text = "Document processing works."

    extractor = FakeExtractor(
        result=ExtractionResult(
            text=expected_text,
            character_count=len(expected_text),
            page_count=None,
        )
    )
    service = DocumentProcessingService(
        session_factory=session_factory,
        extractor=cast(DocumentTextExtractor, extractor),
    )

    outcome = service.process_next_document()

    assert outcome.status == "ready"
    assert outcome.document_id == document.id

    persisted_document = load_document(session_factory, document.id)
    assert persisted_document.status == DocumentStatus.READY
    assert persisted_document.processing_token is None
    assert persisted_document.processing_started_at is None
    assert persisted_document.processed_at is not None

    with session_factory() as session:
        content = session.get(DocumentContent, document.id)
        assert content is not None
        assert content.extracted_text == expected_text
        assert content.character_count == len(expected_text)


def test_permanent_extraction_failure_marks_document_failed(
    db: Session,
    session_factory: Callable[[], Session],
) -> None:
    document = create_pending_document(db)

    service = DocumentProcessingService(
        session_factory=session_factory,
        extractor=cast(
            DocumentTextExtractor,
            FakeExtractor(error=DocumentExtractionError("Invalid document content.")),
        ),
    )

    outcome = service.process_next_document()

    assert outcome.status == "failure_recorded"

    persisted_document = load_document(session_factory, document.id)
    assert persisted_document.status == DocumentStatus.FAILED
    assert persisted_document.processing_attempts == 1
    assert persisted_document.processing_token is None
    assert persisted_document.last_processing_error is not None
    assert "Invalid document content" in persisted_document.last_processing_error


def test_retryable_storage_failure_schedules_retry(
    db: Session,
    session_factory: Callable[[], Session],
) -> None:
    document = create_pending_document(db)

    service = DocumentProcessingService(
        session_factory=session_factory,
        extractor=cast(
            DocumentTextExtractor,
            FakeExtractor(error=StorageError("Temporary storage failure.")),
        ),
        max_attempts=3,
    )

    outcome = service.process_next_document()

    assert outcome.status == "failure_recorded"

    persisted_document = load_document(session_factory, document.id)
    assert persisted_document.status == DocumentStatus.PENDING
    assert persisted_document.processing_attempts == 1
    assert persisted_document.next_attempt_at is not None
    assert persisted_document.processing_token is None
    assert persisted_document.last_processing_error is not None
    assert "Temporary storage failure" in persisted_document.last_processing_error


def test_stale_claim_cannot_overwrite_newer_worker_result(
    db: Session,
    session_factory: Callable[[], Session],
) -> None:
    """A worker with an obsolete token cannot complete processing."""
    create_pending_document(db)

    repository = DocumentRepository(db)
    claimed_document = repository.claim_next_pending()

    assert claimed_document is not None
    document_id = claimed_document.id
    stale_token = claimed_document.processing_token

    assert stale_token is not None
    db.commit()

    # Simulate another worker taking ownership of the document.
    newer_token = uuid4()
    claimed_document = db.get(Document, document_id)
    assert claimed_document is not None

    claimed_document.processing_token = newer_token
    db.commit()

    with session_factory() as session:
        repository = DocumentRepository(session)

        completed = repository.complete_processing(
            document_id=document_id,
            processing_token=stale_token,
            extracted_text="Stale worker content",
            character_count=len("Stale worker content"),
            page_count=1,
        )

        assert completed is False
        session.rollback()

    with session_factory() as session:
        persisted_document = session.get(Document, document_id)

        assert persisted_document is not None
        assert persisted_document.processing_token == newer_token
        assert persisted_document.status == DocumentStatus.PROCESSING

        content = session.get(DocumentContent, document_id)
        assert content is None


def test_retry_exhaustion_marks_document_failed(
    db: Session,
    session_factory: Callable[[], Session],
) -> None:
    """A transient failure on the final attempt marks the document FAILED."""
    document = create_pending_document(db)
    document.processing_attempts = 2
    db.commit()

    service = DocumentProcessingService(
        session_factory=session_factory,
        extractor=cast(
            DocumentTextExtractor,
            FakeExtractor(error=StorageError("Temporary storage failure")),
        ),
        max_attempts=3,
    )

    outcome = service.process_next_document()

    assert outcome.status == "failure_recorded"
    assert outcome.document_id == document.id

    persisted_document = load_document(session_factory, document.id)

    assert persisted_document.status == DocumentStatus.FAILED
    assert persisted_document.processing_attempts == 3
    assert persisted_document.next_attempt_at is None
    assert persisted_document.processing_started_at is None
    assert persisted_document.processing_token is None
    assert persisted_document.last_processing_error is not None
    assert "Temporary storage failure" in persisted_document.last_processing_error


def test_recovers_stale_claim_and_schedules_retry(
    db: Session,
    session_factory: Callable[[], Session],
) -> None:
    """An abandoned claim returns to PENDING when retries remain."""
    document = create_pending_document(db)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None
    assert claimed.id == document.id
    assert claimed.processing_token is not None

    old_token = claimed.processing_token
    claimed.processing_started_at = datetime.now(UTC) - timedelta(minutes=10)
    db.commit()

    recovered_count = repository.recover_stale_processing(
        stale_before=datetime.now(UTC) - timedelta(minutes=5),
        max_attempts=3,
    )
    db.commit()

    assert recovered_count == 1

    persisted_document = load_document(session_factory, document.id)
    assert persisted_document.status == DocumentStatus.PENDING
    assert persisted_document.processing_token is None
    assert persisted_document.processing_started_at is None
    assert persisted_document.next_attempt_at is not None
    assert persisted_document.last_processing_error is not None

    # The abandoned worker's token must no longer authorize completion.
    with session_factory() as session:
        completed = DocumentRepository(session).complete_processing(
            document_id=document.id,
            processing_token=old_token,
            extracted_text="Stale worker content",
            character_count=len("Stale worker content"),
            page_count=1,
        )
        assert completed is False
        session.rollback()


def test_marks_stale_claim_failed_after_attempts_exhausted(
    db: Session,
    session_factory: Callable[[], Session],
) -> None:
    """An abandoned final attempt becomes permanently FAILED."""
    create_pending_document(db)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None

    document_id = claimed.id
    claimed.processing_started_at = datetime.now(UTC) - timedelta(minutes=10)
    claimed.processing_attempts = 3
    db.commit()

    recovered_count = repository.recover_stale_processing(
        stale_before=datetime.now(UTC) - timedelta(minutes=5),
        max_attempts=3,
    )
    db.commit()

    assert recovered_count == 1

    persisted_document = load_document(session_factory, document_id)
    assert persisted_document.status == DocumentStatus.FAILED
    assert persisted_document.processing_token is None
    assert persisted_document.processing_started_at is None
    assert persisted_document.next_attempt_at is None


def test_does_not_recover_recent_processing_claim(
    db: Session,
) -> None:
    """A claim newer than the cutoff remains untouched."""
    create_pending_document(db)
    repository = DocumentRepository(db)

    claimed = repository.claim_next_pending()
    assert claimed is not None

    original_token = claimed.processing_token
    original_started_at = claimed.processing_started_at
    db.commit()

    recovered_count = repository.recover_stale_processing(
        stale_before=datetime.now(UTC) - timedelta(minutes=5),
    )
    db.commit()
    db.refresh(claimed)

    assert recovered_count == 0
    assert claimed.status == DocumentStatus.PROCESSING
    assert claimed.processing_token == original_token
    assert claimed.processing_started_at == original_started_at


def test_recovery_respects_batch_limit(
    db: Session,
) -> None:
    """Recovery processes no more than the configured batch size."""
    create_pending_document(db)
    create_pending_document(db)
    repository = DocumentRepository(db)

    claimed_ids: set[UUID] = set()

    for _ in range(2):
        claimed = repository.claim_next_pending()
        assert claimed is not None
        claimed.processing_started_at = datetime.now(UTC) - timedelta(minutes=10)
        claimed_ids.add(claimed.id)

    db.commit()

    recovered_count = repository.recover_stale_processing(
        stale_before=datetime.now(UTC) - timedelta(minutes=5),
        limit=1,
    )
    db.commit()

    assert recovered_count == 1

    statuses = list(
        db.scalars(select(Document.status).where(Document.id.in_(claimed_ids))).all()
    )
    assert statuses.count(DocumentStatus.PENDING) == 1
    assert statuses.count(DocumentStatus.PROCESSING) == 1


def test_service_recovers_stale_documents_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock(spec=Session)
    repository_instance = Mock()
    repository_instance.recover_stale_processing.return_value = 2

    class FakeRepository:
        def __init__(self, db: Session) -> None:
            assert db is session

        def recover_stale_processing(
            self,
            *,
            stale_before: datetime,
            max_attempts: int,
            limit: int,
        ) -> int:
            assert stale_before.tzinfo is not None
            assert max_attempts == 3
            assert limit == 100

            repository_instance.recover_stale_processing(
                stale_before=stale_before,
                max_attempts=max_attempts,
                limit=limit,
            )
            return 2

    monkeypatch.setattr(
        "ai_knowledge_platform.services.document_processing.DocumentRepository",
        FakeRepository,
    )

    service = DocumentProcessingService(
        session_factory=lambda: session,
        extractor=cast(DocumentTextExtractor, FakeExtractor()),
    )

    recovered_count = service.recover_stale_documents()

    assert recovered_count == 2
    repository_instance.recover_stale_processing.assert_called_once()
    session.commit.assert_called_once()
    session.rollback.assert_not_called()
    session.close.assert_called_once()


def test_service_returns_zero_when_no_stale_documents_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock(spec=Session)

    class FakeRepository:
        def __init__(self, db: Session) -> None:
            assert db is session

        def recover_stale_processing(
            self,
            *,
            stale_before: datetime,
            max_attempts: int,
            limit: int,
        ) -> int:
            return 0

    monkeypatch.setattr(
        "ai_knowledge_platform.services.document_processing.DocumentRepository",
        FakeRepository,
    )

    service = DocumentProcessingService(
        session_factory=lambda: session,
        extractor=cast(DocumentTextExtractor, FakeExtractor()),
    )

    assert service.recover_stale_documents() == 0
    session.commit.assert_called_once()
    session.close.assert_called_once()


@pytest.mark.parametrize(
    ("stale_timeout", "limit"),
    [
        (timedelta(0), 100),
        (timedelta(seconds=-1), 100),
        (timedelta(minutes=15), 0),
        (timedelta(minutes=15), -1),
    ],
)
def test_service_rejects_invalid_recovery_arguments(
    stale_timeout: timedelta,
    limit: int,
) -> None:
    session_factory = Mock()

    service = DocumentProcessingService(
        session_factory=session_factory,
        extractor=cast(DocumentTextExtractor, FakeExtractor()),
    )

    with pytest.raises(ValueError):
        service.recover_stale_documents(
            stale_timeout=stale_timeout,
            limit=limit,
        )

    session_factory.assert_not_called()


def test_service_rolls_back_and_closes_session_on_recovery_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock(spec=Session)

    class FakeRepository:
        def __init__(self, db: Session) -> None:
            assert db is session

        def recover_stale_processing(
            self,
            *,
            stale_before: datetime,
            max_attempts: int,
            limit: int,
        ) -> int:
            raise RuntimeError("Database recovery failed.")

    monkeypatch.setattr(
        "ai_knowledge_platform.services.document_processing.DocumentRepository",
        FakeRepository,
    )

    service = DocumentProcessingService(
        session_factory=lambda: session,
        extractor=cast(DocumentTextExtractor, FakeExtractor()),
    )

    with pytest.raises(RuntimeError, match="Database recovery failed"):
        service.recover_stale_documents()

    session.commit.assert_not_called()
    session.rollback.assert_called_once()
    session.close.assert_called_once()
