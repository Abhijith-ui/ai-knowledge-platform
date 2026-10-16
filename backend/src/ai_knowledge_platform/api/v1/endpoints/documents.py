from collections.abc import Iterator
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from ai_knowledge_platform.api.dependencies import get_current_user
from ai_knowledge_platform.db.session import get_db
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
)
from ai_knowledge_platform.services.document import (
    DocumentService,
    DocumentUploadError,
    UnsupportedDocumentTypeError,
)
from ai_knowledge_platform.storage.base import FileTooLargeError, StorageError

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_CHUNK_SIZE = 64 * 1024


def iter_upload_chunks(upload: UploadFile) -> Iterator[bytes]:
    """Read an uploaded file incrementally."""
    while chunk := upload.file.read(UPLOAD_CHUNK_SIZE):
        yield chunk


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Document:
    """Upload a document for the authenticated user."""
    content_type = file.content_type or "application/octet-stream"

    try:
        return DocumentService(db).upload(
            user_id=user.id,
            filename=file.filename or "",
            content_type=content_type,
            chunks=iter_upload_chunks(file),
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the maximum allowed size.",
        ) from exc
    except DocumentUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document storage is temporarily unavailable.",
        ) from exc


@router.get("", response_model=DocumentListResponse)
def list_documents(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """List documents belonging to the authenticated user."""
    documents = DocumentRepository(db).list_by_owner(
        user.id,
        limit=limit,
        offset=offset,
    )

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(document) for document in documents],
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Document:
    """Retrieve document metadata if it belongs to the authenticated user."""
    document = DocumentRepository(db).get_by_id_for_owner(
        document_id,
        user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document
