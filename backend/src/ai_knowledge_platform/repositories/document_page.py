from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.document_page import DocumentPage

PageBoundary = tuple[int, int, int]


class DocumentPageRepository:
    """Persist page boundaries for extracted document content."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def replace_for_content(
        self,
        *,
        document_content_id: UUID,
        pages: Sequence[PageBoundary],
    ) -> list[DocumentPage]:
        """Atomically replace page boundaries within the caller's transaction."""
        self._validate_pages(pages)

        self.db.execute(
            delete(DocumentPage).where(
                DocumentPage.document_content_id == document_content_id,
            )
        )

        page_records = [
            DocumentPage(
                document_content_id=document_content_id,
                page_number=page_number,
                start_char=start_char,
                end_char=end_char,
            )
            for page_number, start_char, end_char in sorted(pages)
        ]

        self.db.add_all(page_records)
        self.db.flush()

        return page_records

    def list_by_content_id(
        self,
        document_content_id: UUID,
    ) -> list[DocumentPage]:
        """Return page boundaries ordered by page number."""
        statement = (
            select(DocumentPage)
            .where(DocumentPage.document_content_id == document_content_id)
            .order_by(DocumentPage.page_number)
        )

        return list(self.db.scalars(statement).all())

    @staticmethod
    def _validate_pages(pages: Sequence[PageBoundary]) -> None:
        """Validate page numbers and character boundaries."""
        seen_page_numbers: set[int] = set()

        for page_number, start_char, end_char in pages:
            if page_number <= 0:
                raise ValueError("Page number must be positive.")

            if page_number in seen_page_numbers:
                raise ValueError("Duplicate page numbers are not allowed.")

            if start_char < 0:
                raise ValueError("Page start offset cannot be negative.")

            if end_char < start_char:
                raise ValueError("Page end offset cannot precede start offset.")

            seen_page_numbers.add(page_number)
