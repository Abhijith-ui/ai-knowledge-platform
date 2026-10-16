import hashlib
import logging
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from ai_knowledge_platform.storage.base import (
    FileTooLargeError,
    InvalidStorageKeyError,
    StorageError,
    StoredObject,
)

logger = logging.getLogger(__name__)


class LocalFileStorage:
    """Persist files beneath a configured local storage directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError("Unable to initialize document storage.") from exc

    def _resolve_key(self, key: str) -> Path:
        """Resolve a relative POSIX storage key without allowing traversal."""
        relative_path = PurePosixPath(key)

        if (
            not key
            or "\\" in key
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise InvalidStorageKeyError("Invalid storage key.")

        target = self.root.joinpath(*relative_path.parts).resolve()

        if not target.is_relative_to(self.root):
            raise InvalidStorageKeyError("Storage key escapes the storage root.")

        return target

    def save(
        self,
        key: str,
        chunks: Iterable[bytes],
        *,
        max_bytes: int,
    ) -> StoredObject:
        """Stream chunks to a temporary file and atomically finalize the file."""
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")

        target = self._resolve_key(key)
        temporary_path: Path | None = None

        try:
            target.parent.mkdir(parents=True, exist_ok=True)

            # Revalidate after directory creation to detect existing symlinks.
            target = self._resolve_key(key)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=".upload-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                digest = hashlib.sha256()
                size_bytes = 0

                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("File chunks must be bytes.")

                    if not chunk:
                        continue

                    new_size = size_bytes + len(chunk)

                    if new_size > max_bytes:
                        raise FileTooLargeError(
                            "File exceeds the maximum allowed size."
                        )

                    temporary_file.write(chunk)
                    digest.update(chunk)
                    size_bytes = new_size

                if size_bytes == 0:
                    raise ValueError("File must not be empty.")

                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            # os.link creates the final name atomically and fails if it exists.
            # The temporary file is in the same directory, so both names are
            # on the same filesystem.
            try:
                os.link(temporary_path, target)
            except FileExistsError as exc:
                raise StorageError("Storage key already exists.") from exc

            temporary_path.unlink()
            temporary_path = None

            return StoredObject(
                size_bytes=size_bytes,
                checksum_sha256=digest.hexdigest(),
            )

        except (
            InvalidStorageKeyError,
            StorageError,
            FileTooLargeError,
            ValueError,
            TypeError,
        ):
            raise
        except OSError as exc:
            raise StorageError("Unable to persist the uploaded file.") from exc
        finally:
            if temporary_path is not None:
                # Preserve an exception already being raised if cleanup fails.
                active_exception = sys.exc_info()[0] is not None

                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    if active_exception:
                        logger.exception("Failed to clean up a temporary upload file.")
                    else:
                        raise StorageError(
                            "Unable to clean up a temporary upload file."
                        ) from sys.exc_info()[1]

    def delete(self, key: str) -> None:
        """Delete a stored file without failing if it is already absent."""
        target = self._resolve_key(key)

        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError("Unable to delete the stored file.") from exc
