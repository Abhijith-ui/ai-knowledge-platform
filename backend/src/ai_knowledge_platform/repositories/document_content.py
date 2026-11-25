from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_knowledge_platform.models.document_content import DocumentContent


class DocumentContentRepository:
    """Provide database operations for extracted document content."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        ingestion_run_id: UUID,
        extracted_text: str,
        character_count: int,
        page_count: int | None = None,
    ) -> DocumentContent:
        """Create extracted content within the current transaction."""

        if not extracted_text:
            raise ValueError("extracted_text must not be empty.")

        if character_count != len(extracted_text):
            raise ValueError("character_count must match the extracted text length.")

        if page_count is not None and page_count <= 0:
            raise ValueError("page_count must be positive when provided.")

        content = DocumentContent(
            ingestion_run_id=ingestion_run_id,
            extracted_text=extracted_text,
            character_count=character_count,
            page_count=page_count,
        )

        self.db.add(content)
        self.db.flush()

        return content

    def get_by_ingestion_run_id(
        self,
        ingestion_run_id: UUID,
    ) -> DocumentContent | None:
        """Retrieve extracted content associated with an ingestion run."""

        statement = select(DocumentContent).where(
            DocumentContent.ingestion_run_id == ingestion_run_id,
        )

        return self.db.scalar(statement)
