from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.document import Document, DocumentStatus
from ai_knowledge_platform.models.document_content import DocumentContent


class DocumentRepository:
    """Provide database operations for user-owned documents."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        user_id: UUID,
        filename: str,
        content_type: str,
        size_bytes: int,
        checksum_sha256: str,
        storage_key: str,
    ) -> Document:
        """Create document metadata within the current transaction."""
        document = Document(
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
            storage_key=storage_key,
        )
        self.db.add(document)
        self.db.flush()
        return document

    def get_by_id_for_owner(
        self,
        document_id: UUID,
        user_id: UUID,
    ) -> Document | None:
        """Retrieve a document only if it belongs to the specified user."""
        statement = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )
        return self.db.scalar(statement)

    def list_by_owner(
        self,
        user_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        """Return a user's documents in deterministic, newest-first order."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        statement = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(statement).all())

    def claim_next_pending(self) -> Document | None:
        """Claim one eligible document for processing."""
        now = datetime.now(UTC)

        statement = (
            select(Document)
            .where(
                Document.status == DocumentStatus.PENDING,
                or_(
                    Document.next_attempt_at.is_(None),
                    Document.next_attempt_at <= now,
                ),
            )
            .order_by(Document.created_at.asc(), Document.id.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        document = self.db.scalar(statement)

        if document is None:
            return None

        document.status = DocumentStatus.PROCESSING
        document.processing_attempts += 1
        document.processing_started_at = now
        document.processing_token = uuid4()
        document.next_attempt_at = None
        document.last_processing_error = None

        self.db.flush()
        return document

    def complete_processing(
        self,
        *,
        document_id: UUID,
        processing_token: UUID,
        extracted_text: str,
        character_count: int,
        page_count: int | None,
    ) -> bool:
        """Persist extracted content if the worker still owns the claim."""
        now = datetime.now(UTC)

        statement = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_token == processing_token,
            )
            .with_for_update()
        )

        document = self.db.scalar(statement)

        if document is None:
            return False

        content = self.db.get(DocumentContent, document_id)

        if content is None:
            content = DocumentContent(
                document_id=document_id,
                extracted_text=extracted_text,
                character_count=character_count,
                page_count=page_count,
            )
            self.db.add(content)
        else:
            content.extracted_text = extracted_text
            content.character_count = character_count
            content.page_count = page_count
            content.updated_at = now

        document.status = DocumentStatus.READY
        document.processed_at = now
        document.processing_started_at = None
        document.processing_token = None
        document.next_attempt_at = None
        document.last_processing_error = None

        self.db.flush()
        return True

    def fail_processing(
        self,
        *,
        document_id: UUID,
        processing_token: UUID,
        error_message: str,
        retryable: bool,
        max_attempts: int = 3,
    ) -> bool:
        """Record a processing failure if the worker still owns the claim."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        if not error_message.strip():
            raise ValueError("error_message must not be empty")

        now = datetime.now(UTC)

        statement = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_token == processing_token,
            )
            .with_for_update()
        )

        document = self.db.scalar(statement)

        if document is None:
            return False

        document.processing_started_at = None
        document.processing_token = None
        document.last_processing_error = error_message[:2000]

        if retryable and document.processing_attempts < max_attempts:
            delay_seconds = min(
                30 * (2 ** (document.processing_attempts - 1)),
                3600,
            )
            document.status = DocumentStatus.PENDING
            document.next_attempt_at = now + timedelta(seconds=delay_seconds)
        else:
            document.status = DocumentStatus.FAILED
            document.next_attempt_at = None

        self.db.flush()
        return True

    def recover_stale_processing(
        self,
        *,
        stale_before: datetime,
        max_attempts: int = 3,
        limit: int = 100,
    ) -> int:
        """Recover abandoned processing claims in a bounded batch."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        statement = (
            select(Document)
            .where(
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_started_at.is_not(None),
                Document.processing_started_at < stale_before,
            )
            .order_by(
                Document.processing_started_at.asc(),
                Document.id.asc(),
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )

        documents = list(self.db.scalars(statement).all())
        now = datetime.now(UTC)

        for document in documents:
            document.processing_started_at = None
            document.processing_token = None
            document.last_processing_error = (
                "Processing claim expired before completion."
            )

            if document.processing_attempts < max_attempts:
                delay_seconds = min(
                    30 * (2 ** max(document.processing_attempts - 1, 0)),
                    3600,
                )
                document.status = DocumentStatus.PENDING
                document.next_attempt_at = now + timedelta(seconds=delay_seconds)
            else:
                document.status = DocumentStatus.FAILED
                document.next_attempt_at = None

        self.db.flush()
        return len(documents)
