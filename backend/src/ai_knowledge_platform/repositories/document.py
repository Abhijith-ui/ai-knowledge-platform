from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.document import Document


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
