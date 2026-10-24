from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pytest
from ai_knowledge_platform.services.document_extraction import (
    DocumentTextExtractor,
    EmptyExtractedTextError,
    ExtractedTextTooLargeError,
    InvalidDocumentError,
    UnsupportedExtractionTypeError,
)
from ai_knowledge_platform.storage.base import StorageError
from ai_knowledge_platform.storage.local import LocalFileStorage
from pypdf import PdfWriter


class TrackingStorage:
    """Minimal storage fake for testing resource lifecycle."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.handle: BytesIO | None = None

    @contextmanager
    def open_read(self, key: str) -> Iterator[BytesIO]:
        self.handle = BytesIO(self.content)
        try:
            yield self.handle
        finally:
            self.handle.close()

    def save(self, key, chunks, *, max_bytes):
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


def test_file_handle_closes_after_success() -> None:
    storage = TrackingStorage(b"valid text")
    extractor = DocumentTextExtractor(storage)

    result = extractor.extract(
        storage_key="documents/sample.txt",
        content_type="text/plain",
    )

    assert result.text == "valid text"
    assert storage.handle is not None
    assert storage.handle.closed


def test_file_handle_closes_after_extraction_failure() -> None:
    storage = TrackingStorage(b"\xff")
    extractor = DocumentTextExtractor(storage)

    with pytest.raises(InvalidDocumentError):
        extractor.extract(
            storage_key="documents/invalid.txt",
            content_type="text/plain",
        )

    assert storage.handle is not None
    assert storage.handle.closed


def test_storage_errors_are_preserved() -> None:
    class FailingStorage(TrackingStorage):
        @contextmanager
        def open_read(self, key: str) -> Iterator[BytesIO]:
            raise StorageError("Simulated storage failure")
            yield BytesIO()

    extractor = DocumentTextExtractor(FailingStorage(b"text"))

    with pytest.raises(StorageError, match="Simulated storage failure"):
        extractor.extract(
            storage_key="documents/sample.txt",
            content_type="text/plain",
        )


@pytest.fixture
def storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(tmp_path / "storage")


def store_file(
    storage: LocalFileStorage,
    key: str,
    content: bytes,
) -> None:
    storage.save(key, [content], max_bytes=1024 * 1024)


def test_extracts_plain_text(storage: LocalFileStorage) -> None:
    content = "Hello, document!".encode()
    store_file(storage, "documents/sample.txt", content)

    result = DocumentTextExtractor(storage).extract(
        storage_key="documents/sample.txt",
        content_type="text/plain",
    )

    assert result.text == "Hello, document!"
    assert result.character_count == len(result.text)
    assert result.page_count is None


def test_extracts_markdown(storage: LocalFileStorage) -> None:
    content = "# Heading\n\nDocument body.".encode()
    store_file(storage, "documents/sample.md", content)

    result = DocumentTextExtractor(storage).extract(
        storage_key="documents/sample.md",
        content_type="text/markdown",
    )

    assert result.text == "# Heading\n\nDocument body."


def test_rejects_invalid_utf8(storage: LocalFileStorage) -> None:
    store_file(storage, "documents/invalid.txt", b"\xff\xfe")

    with pytest.raises(InvalidDocumentError, match="valid UTF-8"):
        DocumentTextExtractor(storage).extract(
            storage_key="documents/invalid.txt",
            content_type="text/plain",
        )


def test_rejects_empty_text(storage: LocalFileStorage) -> None:
    store_file(storage, "documents/empty.txt", b"  \n ")

    with pytest.raises(EmptyExtractedTextError):
        DocumentTextExtractor(storage).extract(
            storage_key="documents/empty.txt",
            content_type="text/plain",
        )


def test_enforces_text_size_limit(storage: LocalFileStorage) -> None:
    store_file(storage, "documents/large.txt", b"abcdefgh")

    with pytest.raises(ExtractedTextTooLargeError):
        DocumentTextExtractor(storage, max_chars=4).extract(
            storage_key="documents/large.txt",
            content_type="text/plain",
        )


def test_rejects_unsupported_content_type(
    storage: LocalFileStorage,
) -> None:
    with pytest.raises(UnsupportedExtractionTypeError):
        DocumentTextExtractor(storage).extract(
            storage_key="documents/sample.docx",
            content_type=(
                "application/"
                "vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )


def test_rejects_malformed_pdf(storage: LocalFileStorage) -> None:
    store_file(storage, "documents/broken.pdf", b"not a PDF")

    with pytest.raises(InvalidDocumentError):
        DocumentTextExtractor(storage).extract(
            storage_key="documents/broken.pdf",
            content_type="application/pdf",
        )


def test_rejects_pdf_without_extractable_text(
    storage: LocalFileStorage,
) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)

    store_file(
        storage,
        "documents/blank.pdf",
        pdf_buffer.getvalue(),
    )

    with pytest.raises(EmptyExtractedTextError):
        DocumentTextExtractor(storage).extract(
            storage_key="documents/blank.pdf",
            content_type="application/pdf",
        )
