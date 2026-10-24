from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO, Protocol


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Describe a file successfully persisted by a storage backend."""

    size_bytes: int
    checksum_sha256: str


class StorageError(Exception):
    """Raised when a storage operation fails."""


class InvalidStorageKeyError(StorageError):
    """Raised when a storage key is unsafe or invalid."""


class FileTooLargeError(ValueError):
    """Raised when an uploaded file exceeds its configured size limit."""


class FileStorage(Protocol):
    """Define the operations required from a document storage backend."""

    def save(
        self,
        key: str,
        chunks: Iterable[bytes],
        *,
        max_bytes: int,
    ) -> StoredObject:
        """Persist streamed file chunks and return their verified metadata."""
        ...

    def open_read(self, key: str) -> BinaryIO:
        """Open a stored file for binary reading; the caller must close it."""
        ...

    def delete(self, key: str) -> None:
        """Delete a stored file, treating an absent file as already deleted."""
        ...
