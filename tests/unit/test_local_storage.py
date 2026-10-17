import hashlib
from pathlib import Path

import pytest
from ai_knowledge_platform.storage.base import (
    FileTooLargeError,
    InvalidStorageKeyError,
    StorageError,
)
from ai_knowledge_platform.storage.local import LocalFileStorage


def test_save_persists_content_and_returns_metadata(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "storage")
    content = b"Document content"
    key = "users/user-1/documents/document-1.txt"

    result = storage.save(key, [b"Document ", b"content"], max_bytes=1024)

    assert (tmp_path / "storage" / key).read_bytes() == content
    assert result.size_bytes == len(content)
    assert result.checksum_sha256 == hashlib.sha256(content).hexdigest()


def test_save_rejects_empty_content(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "storage")

    with pytest.raises(ValueError, match="must not be empty"):
        storage.save("documents/empty.txt", [], max_bytes=1024)

    assert not (tmp_path / "storage" / "documents" / "empty.txt").exists()


def test_save_rejects_oversized_content_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage"
    storage = LocalFileStorage(storage_root)

    with pytest.raises(ValueError, match="maximum allowed size"):
        storage.save("documents/large.txt", [b"1234", b"5678"], max_bytes=5)

    assert not (storage_root / "documents" / "large.txt").exists()
    assert not list((storage_root / "documents").glob(".upload-*.tmp"))


@pytest.mark.parametrize(
    "key",
    [
        "../outside.txt",
        "/absolute/path.txt",
        "users/../../outside.txt",
        r"users\user-1\file.txt",
        "",
    ],
)
def test_save_rejects_unsafe_keys(tmp_path: Path, key: str) -> None:
    storage = LocalFileStorage(tmp_path / "storage")

    with pytest.raises(InvalidStorageKeyError):
        storage.save(key, [b"content"], max_bytes=1024)


def test_delete_removes_file(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "storage")
    key = "documents/report.txt"
    storage.save(key, [b"report"], max_bytes=1024)

    storage.delete(key)

    assert not (tmp_path / "storage" / key).exists()


def test_delete_is_idempotent_for_missing_file(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "storage")

    storage.delete("documents/missing.txt")


def test_save_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "storage")
    key = "documents/report.txt"
    storage.save(key, [b"original"], max_bytes=1024)

    with pytest.raises(StorageError, match="already exists"):
        storage.save(key, [b"replacement"], max_bytes=1024)

    assert (tmp_path / "storage" / key).read_bytes() == b"original"


def test_save_cleans_temporary_file_when_stream_fails(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage = LocalFileStorage(storage_root)

    def failing_chunks():
        yield b"partial content"
        raise RuntimeError("Simulated stream failure")

    with pytest.raises(RuntimeError, match="Simulated stream failure"):
        storage.save(
            "documents/report.txt",
            failing_chunks(),
            max_bytes=1024,
        )

    assert not (storage_root / "documents" / "report.txt").exists()
    assert not list((storage_root / "documents").glob(".upload-*.tmp"))


def test_save_raises_specific_error_for_oversized_content(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(tmp_path / "storage")

    with pytest.raises(FileTooLargeError, match="maximum allowed size"):
        storage.save(
            "documents/large.txt",
            [b"1234", b"5678"],
            max_bytes=5,
        )


def test_storage_initialization_failure_raises_storage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_mkdir(*args, **kwargs) -> None:
        raise OSError("Simulated directory creation failure")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(StorageError, match="initialize document storage"):
        LocalFileStorage(tmp_path / "storage")


def test_save_preserves_original_exception_if_temp_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    storage = LocalFileStorage(storage_root)

    original_unlink = Path.unlink

    def fail_temp_cleanup(
        path: Path,
        *args,
        **kwargs,
    ) -> None:
        if path.name.startswith(".upload-") and path.suffix == ".tmp":
            raise OSError("Simulated temporary-file cleanup failure")

        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_cleanup)

    def failing_chunks():
        yield b"partial content"
        raise RuntimeError("Original stream failure")

    with pytest.raises(RuntimeError, match="Original stream failure"):
        storage.save(
            "documents/report.txt",
            failing_chunks(),
            max_bytes=1024,
        )


def test_save_removes_final_file_if_temp_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove the final file if temporary-file cleanup breaks finalization."""
    storage_root = tmp_path / "storage"
    storage = LocalFileStorage(storage_root)
    key = "documents/report.txt"
    original_unlink = Path.unlink
    failed_once = False

    def fail_first_temp_unlink(
        path: Path,
        *args,
        **kwargs,
    ) -> None:
        nonlocal failed_once

        if (
            path.name.startswith(".upload-")
            and path.suffix == ".tmp"
            and not failed_once
        ):
            failed_once = True
            raise OSError("Simulated temporary-file cleanup failure")

        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_temp_unlink)

    with pytest.raises(StorageError):
        storage.save(key, [b"document content"], max_bytes=1024)

    assert failed_once
    assert not (storage_root / key).exists()
    assert not list((storage_root / "documents").glob(".upload-*.tmp"))


def test_save_removes_final_file_if_result_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean up the final file if result creation fails after finalization."""
    storage_root = tmp_path / "storage"
    storage = LocalFileStorage(storage_root)
    key = "documents/report.txt"

    def fail_stored_object(**kwargs: object) -> None:
        raise RuntimeError("Simulated result creation failure")

    monkeypatch.setattr(
        "ai_knowledge_platform.storage.local.StoredObject",
        fail_stored_object,
    )

    with pytest.raises(RuntimeError, match="Simulated result creation failure"):
        storage.save(key, [b"document content"], max_bytes=1024)

    assert not (storage_root / key).exists()
    assert not list((storage_root / "documents").glob(".upload-*.tmp"))
