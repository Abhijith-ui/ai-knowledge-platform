from collections.abc import Iterable
from pathlib import Path

import pytest
from ai_knowledge_platform.core.config import Settings
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.repositories.user import UserRepository
from ai_knowledge_platform.services.document import (
    DocumentService,
    DocumentUploadError,
    UnsupportedDocumentTypeError,
    validate_document_chunks,
)
from ai_knowledge_platform.storage.base import (
    FileTooLargeError,
    StorageError,
    StoredObject,
)
from ai_knowledge_platform.storage.local import LocalFileStorage
from sqlalchemy.orm import Session


class FailingStorage:
    """Simulate a storage backend failure."""

    def save(
        self,
        key: str,
        chunks: Iterable[bytes],
        *,
        max_bytes: int,
    ) -> StoredObject:
        raise StorageError("Simulated storage failure.")

    def delete(self, key: str) -> None:
        return None


class TrackingStorage(LocalFileStorage):
    """Track stored keys to verify cleanup behavior."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.deleted_keys: list[str] = []

    def delete(self, key: str) -> None:
        self.deleted_keys.append(key)
        super().delete(key)


def configure_document_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_upload_bytes: int = 1024,
) -> None:
    """Configure isolated document storage for a test."""
    settings = Settings(
        _env_file=None,
        document_storage_dir=tmp_path / "documents",
        max_document_upload_bytes=max_upload_bytes,
    )
    monkeypatch.setattr(
        "ai_knowledge_platform.services.document.get_settings",
        lambda: settings,
    )


def create_test_user(db: Session, email: str):
    """Create and persist a user for document-service tests."""
    user = UserRepository(db).create(email)
    db.commit()
    return user


def test_upload_persists_file_and_metadata(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "upload-test@example.com")

    storage = TrackingStorage(tmp_path / "documents")
    service = DocumentService(db, storage)

    document = service.upload(
        user_id=user.id,
        filename="report.txt",
        content_type="text/plain",
        chunks=[b"Hello ", b"document"],
    )

    assert document.filename == "report.txt"
    assert document.size_bytes == len(b"Hello document")
    assert len(document.checksum_sha256) == 64
    assert document.storage_key

    stored_path = tmp_path / "documents" / document.storage_key
    assert stored_path.read_bytes() == b"Hello document"
    assert storage.deleted_keys == []


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("report.exe", "application/octet-stream"),
        ("report.pdf", "text/plain"),
        ("report.txt", "application/pdf"),
    ],
)
def test_upload_rejects_unsupported_or_mismatched_types(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content_type: str,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "invalid-type@example.com")

    service = DocumentService(
        db,
        LocalFileStorage(tmp_path / "documents"),
    )

    with pytest.raises(UnsupportedDocumentTypeError):
        service.upload(
            user_id=user.id,
            filename=filename,
            content_type=content_type,
            chunks=[b"content"],
        )

    assert db.query(Document).count() == 0


def test_upload_rejects_invalid_filename(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "invalid-name@example.com")

    service = DocumentService(
        db,
        LocalFileStorage(tmp_path / "documents"),
    )

    with pytest.raises(DocumentUploadError, match="filename"):
        service.upload(
            user_id=user.id,
            filename="   ",
            content_type="text/plain",
            chunks=[b"content"],
        )

    assert db.query(Document).count() == 0


def test_upload_propagates_storage_failure(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "storage-failure@example.com")

    service = DocumentService(db, FailingStorage())

    with pytest.raises(StorageError, match="Simulated storage failure"):
        service.upload(
            user_id=user.id,
            filename="report.txt",
            content_type="text/plain",
            chunks=[b"content"],
        )

    assert db.query(Document).count() == 0


def test_upload_rejects_oversized_file(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(
        tmp_path,
        monkeypatch,
        max_upload_bytes=4,
    )
    user = create_test_user(db, "oversized-upload@example.com")

    service = DocumentService(
        db,
        LocalFileStorage(tmp_path / "documents"),
    )

    with pytest.raises(FileTooLargeError, match="maximum allowed size"):
        service.upload(
            user_id=user.id,
            filename="report.txt",
            content_type="text/plain",
            chunks=[b"12345"],
        )

    assert db.query(Document).count() == 0
    assert not list((tmp_path / "documents").rglob("*.tmp"))


def test_upload_rejects_invalid_pdf_content(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "invalid-pdf@example.com")

    storage = TrackingStorage(tmp_path / "documents")
    service = DocumentService(db, storage)

    with pytest.raises(
        DocumentUploadError,
        match="content does not match",
    ):
        service.upload(
            user_id=user.id,
            filename="report.pdf",
            content_type="application/pdf",
            chunks=[b"This is not a PDF."],
        )

    assert not list((tmp_path / "documents").rglob("*.tmp"))
    assert db.query(Document).count() == 0
    assert storage.deleted_keys == []


def test_pdf_validation_preserves_chunks_when_header_is_split() -> None:
    """Ensure PDF validation preserves bytes when the header spans chunks."""
    chunks = [
        b"%PD",
        b"F-1.7\nPDF body\n",
        b"%%EOF",
    ]
    expected = b"".join(chunks)

    validated = b"".join(
        validate_document_chunks(
            chunks,
            content_type="application/pdf",
        )
    )

    assert validated == expected


def test_upload_rejects_invalid_utf8(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "invalid-utf8@example.com")

    storage = TrackingStorage(tmp_path / "documents")
    service = DocumentService(db, storage)

    with pytest.raises(
        DocumentUploadError,
        match="valid UTF-8",
    ):
        service.upload(
            user_id=user.id,
            filename="report.txt",
            content_type="text/plain",
            chunks=[b"\xff\xfe"],
        )

    assert not list((tmp_path / "documents").rglob("*.tmp"))
    assert db.query(Document).count() == 0
    assert storage.deleted_keys == []


def test_upload_rejects_empty_file(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "empty-upload@example.com")

    storage = TrackingStorage(tmp_path / "documents")
    service = DocumentService(db, storage)

    with pytest.raises(DocumentUploadError, match="empty"):
        service.upload(
            user_id=user.id,
            filename="empty.txt",
            content_type="text/plain",
            chunks=[],
        )

    assert not list((tmp_path / "documents").rglob("*.tmp"))
    assert db.query(Document).count() == 0
    assert storage.deleted_keys == []


def test_upload_cleans_up_file_when_database_commit_fails(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "commit-failure@example.com")

    storage = TrackingStorage(tmp_path / "documents")
    service = DocumentService(db, storage)

    rollback_called = False
    original_rollback = db.rollback

    def track_rollback() -> None:
        nonlocal rollback_called
        rollback_called = True
        original_rollback()

    def fail_commit() -> None:
        raise RuntimeError("Simulated database commit failure.")

    monkeypatch.setattr(db, "rollback", track_rollback)
    monkeypatch.setattr(db, "commit", fail_commit)

    with pytest.raises(
        RuntimeError,
        match="Simulated database commit failure",
    ):
        service.upload(
            user_id=user.id,
            filename="report.txt",
            content_type="text/plain",
            chunks=[b"Valid document"],
        )

    assert rollback_called
    assert len(storage.deleted_keys) == 1

    stored_path = tmp_path / "documents" / storage.deleted_keys[0]
    assert not stored_path.exists()
    assert not list((tmp_path / "documents").rglob("*.tmp"))


def test_upload_preserves_database_error_when_cleanup_fails(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_document_settings(tmp_path, monkeypatch)
    user = create_test_user(db, "cleanup-failure@example.com")

    storage = TrackingStorage(tmp_path / "documents")
    service = DocumentService(db, storage)

    rollback_called = False
    original_rollback = db.rollback

    def track_rollback() -> None:
        nonlocal rollback_called
        rollback_called = True
        original_rollback()

    def fail_commit() -> None:
        raise RuntimeError("Original database failure.")

    def fail_delete(key: str) -> None:
        storage.deleted_keys.append(key)
        raise OSError("Simulated storage cleanup failure.")

    monkeypatch.setattr(db, "rollback", track_rollback)
    monkeypatch.setattr(db, "commit", fail_commit)
    monkeypatch.setattr(storage, "delete", fail_delete)

    with pytest.raises(RuntimeError, match="Original database failure"):
        service.upload(
            user_id=user.id,
            filename="report.txt",
            content_type="text/plain",
            chunks=[b"Valid document"],
        )

    assert rollback_called
    assert len(storage.deleted_keys) == 1

    # Cleanup failed, so the file may remain. The original database
    # exception must still be preserved.
    stored_path = tmp_path / "documents" / storage.deleted_keys[0]
    assert stored_path.exists()
