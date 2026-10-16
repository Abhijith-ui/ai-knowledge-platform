from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from ai_knowledge_platform.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """Expose safe document metadata to API clients."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    content_type: str
    size_bytes: int
    checksum_sha256: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Represent a paginated list of documents."""

    items: list[DocumentResponse]
    limit: int
    offset: int
