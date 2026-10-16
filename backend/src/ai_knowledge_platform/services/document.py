import codecs
import logging
from collections.abc import Iterable, Iterator
from pathlib import PurePath
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.storage.base import FileStorage
from ai_knowledge_platform.storage.local import LocalFileStorage

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class DocumentUploadError(Exception):
    """Raised when a document upload cannot be completed."""


class UnsupportedDocumentTypeError(DocumentUploadError):
    """Raised when the uploaded document type is unsupported."""


def validate_document_chunks(
    chunks: Iterable[bytes],
    *,
    content_type: str,
) -> Iterator[bytes]:
    """Validate basic document content while preserving streaming."""
    text_decoder = None

    if content_type in {"text/plain", "text/markdown"}:
        text_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")

    pdf_header = bytearray()
    pdf_header_checked = content_type != "application/pdf"
    pdf_tail = bytearray()
    is_empty = True

    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise DocumentUploadError("File chunks must be bytes.")

        if not chunk:
            continue

        is_empty = False

        if text_decoder is not None:
            try:
                text_decoder.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                raise DocumentUploadError(
                    "Text documents must contain valid UTF-8."
                ) from exc

        if content_type == "application/pdf":
            # Validate the first five bytes, even if they span multiple chunks.
            if not pdf_header_checked:
                required_bytes = 5 - len(pdf_header)
                pdf_header.extend(chunk[:required_bytes])

                if len(pdf_header) >= 5:
                    if not pdf_header.startswith(b"%PDF-"):
                        raise DocumentUploadError(
                            "File content does not match the PDF format."
                        )
                    pdf_header_checked = True

            # Track the actual trailing bytes for the EOF marker check.
            pdf_tail.extend(chunk)
            if len(pdf_tail) > 1024:
                del pdf_tail[:-1024]

        # Never skip or modify a non-empty input chunk.
        yield chunk

    if is_empty:
        raise DocumentUploadError("The uploaded file is empty.")

    if text_decoder is not None:
        try:
            text_decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise DocumentUploadError(
                "Text documents must contain valid UTF-8."
            ) from exc

    if content_type == "application/pdf":
        if not pdf_header_checked or b"%%EOF" not in pdf_tail:
            raise DocumentUploadError("PDF file is missing its expected end marker.")


class DocumentService:
    """Coordinate document validation, file storage, and database persistence."""

    def __init__(
        self,
        db: Session,
        storage: FileStorage | None = None,
    ) -> None:
        self.db = db
        self.repository = DocumentRepository(db)
        self.settings = get_settings()
        self.storage = (
            storage
            if storage is not None
            else LocalFileStorage(self.settings.document_storage_dir)
        )

    def upload(
        self,
        *,
        user_id: UUID,
        filename: str,
        content_type: str,
        chunks: Iterable[bytes],
    ) -> Document:
        """Validate and persist a document owned by the specified user."""
        safe_filename = PurePath(filename.replace("\\", "/")).name.strip()

        if not safe_filename or safe_filename in {".", ".."}:
            raise DocumentUploadError("A valid filename is required.")

        if len(safe_filename) > 255:
            raise DocumentUploadError("Filename exceeds 255 characters.")

        extension = PurePath(safe_filename).suffix.lower()
        expected_content_type = SUPPORTED_CONTENT_TYPES.get(extension)

        if expected_content_type is None:
            raise UnsupportedDocumentTypeError(
                "Only PDF, plain text, and Markdown files are supported."
            )

        if content_type != expected_content_type:
            raise UnsupportedDocumentTypeError(
                "Filename extension and content type do not match."
            )

        document_id = uuid4()
        storage_key = f"users/{user_id}/documents/{document_id}"

        validated_chunks = validate_document_chunks(
            chunks,
            content_type=expected_content_type,
        )

        stored_object = self.storage.save(
            storage_key,
            validated_chunks,
            max_bytes=self.settings.max_document_upload_bytes,
        )

        try:
            document = self.repository.create(
                user_id=user_id,
                filename=safe_filename,
                content_type=expected_content_type,
                size_bytes=stored_object.size_bytes,
                checksum_sha256=stored_object.checksum_sha256,
                storage_key=storage_key,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()

            try:
                self.storage.delete(storage_key)
            except Exception:
                # Preserve the original database error even if cleanup fails.
                logger.exception(
                    "Failed to clean up stored document after database failure."
                )

            raise

        return document
