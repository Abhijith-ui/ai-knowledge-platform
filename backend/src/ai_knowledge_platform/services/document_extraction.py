import logging
from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.storage.base import FileStorage, StorageError

logger = logging.getLogger(__name__)


class DocumentExtractionError(Exception):
    """Base exception for document extraction failures."""


class UnsupportedExtractionTypeError(DocumentExtractionError):
    """Raised when the content type is unsupported."""


class EmptyExtractedTextError(DocumentExtractionError):
    """Raised when a document contains no extractable text."""


class ExtractedTextTooLargeError(DocumentExtractionError):
    """Raised when extracted text exceeds the configured limit."""


class InvalidDocumentError(DocumentExtractionError):
    """Raised when a document cannot be decoded or parsed."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Represent text extracted from a single PDF page."""

    page_number: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Represent extracted text and its metadata."""

    text: str
    character_count: int
    page_count: int | None = None
    pages: tuple[ExtractedPage, ...] = ()


class DocumentTextExtractor:
    """Extract text from stored PDF, TXT, and Markdown documents."""

    SUPPORTED_CONTENT_TYPES = {
        "application/pdf",
        "text/plain",
        "text/markdown",
    }

    def __init__(
        self,
        storage: FileStorage,
        max_chars: int | None = None,
    ) -> None:
        self.storage = storage
        self.max_chars = (
            max_chars
            if max_chars is not None
            else get_settings().max_extracted_text_chars
        )

        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive.")

    def extract(
        self,
        *,
        storage_key: str,
        content_type: str,
    ) -> ExtractionResult:
        """Extract text from a stored document."""
        if content_type not in self.SUPPORTED_CONTENT_TYPES:
            raise UnsupportedExtractionTypeError(
                "Only PDF, plain text, and Markdown are supported."
            )

        try:
            with self.storage.open_read(storage_key) as source:
                if content_type == "application/pdf":
                    return self._extract_pdf(source)

                return self._extract_text(source)

        except (StorageError, DocumentExtractionError):
            raise
        except (OSError, PdfReadError, ValueError, EOFError) as exc:
            logger.warning(
                "Document extraction failed for storage key %s.",
                storage_key,
                exc_info=True,
            )
            raise InvalidDocumentError(
                "The document could not be read or parsed."
            ) from exc

    def _extract_text(self, source: BinaryIO) -> ExtractionResult:
        """Decode UTF-8 text while limiting input size."""
        max_input_bytes = self.max_chars * 4
        raw = source.read(max_input_bytes + 1)

        if len(raw) > max_input_bytes:
            raise ExtractedTextTooLargeError(
                "Document exceeds the maximum extraction input size."
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                "Text documents must contain valid UTF-8."
            ) from exc

        self._check_text_size(text)

        if not text.strip():
            raise EmptyExtractedTextError("The document contains no extractable text.")

        return ExtractionResult(
            text=text,
            character_count=len(text),
        )

    def _extract_pdf(self, source: BinaryIO) -> ExtractionResult:
        """Extract text while preserving PDF page boundaries."""
        reader = PdfReader(source, strict=True)

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except (PdfReadError, ValueError) as exc:
                raise InvalidDocumentError("Encrypted PDFs are not supported.") from exc

            if not decrypted:
                raise InvalidDocumentError("Encrypted PDFs are not supported.")

        parts: list[str] = []
        pages: list[ExtractedPage] = []
        character_count = 0
        page_count = len(reader.pages)

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            # Account for the newline inserted between PDF pages.
            if parts:
                character_count += 1

            start_char = character_count
            character_count += len(page_text)
            end_char = character_count

            if character_count > self.max_chars:
                raise ExtractedTextTooLargeError(
                    "Extracted text exceeds the configured character limit."
                )

            parts.append(page_text)

            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text=page_text,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        text = "\n".join(parts)

        if not text.strip():
            raise EmptyExtractedTextError("The document contains no extractable text.")

        return ExtractionResult(
            text=text,
            character_count=len(text),
            page_count=page_count,
            pages=tuple(pages),
        )

    def _check_text_size(self, text: str) -> None:
        """Reject extracted text that exceeds the configured character limit."""
        if len(text) > self.max_chars:
            raise ExtractedTextTooLargeError(
                "Extracted text exceeds the configured character limit."
            )
