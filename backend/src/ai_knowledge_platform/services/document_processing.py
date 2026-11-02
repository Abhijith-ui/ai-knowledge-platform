import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.db.session import SessionLocal
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.services.document_extraction import (
    DocumentExtractionError,
    DocumentTextExtractor,
)
from ai_knowledge_platform.storage.base import (
    FileStorage,
    InvalidStorageKeyError,
    StorageError,
)
from ai_knowledge_platform.storage.local import LocalFileStorage

logger = logging.getLogger(__name__)

ProcessingStatus = Literal[
    "idle",
    "ready",
    "failure_recorded",
    "stale_claim",
]


@dataclass(frozen=True, slots=True)
class ProcessingOutcome:
    """Describe the result of one document-processing attempt."""

    status: ProcessingStatus
    document_id: UUID | None = None


class DocumentProcessingService:
    """Coordinate document extraction and persistence."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        extractor: DocumentTextExtractor | None = None,
        storage: FileStorage | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self.session_factory = session_factory
        self.max_attempts = max_attempts

        if extractor is not None:
            self.extractor = extractor
        else:
            if storage is None:
                settings = get_settings()
                storage = LocalFileStorage(settings.document_storage_dir)

            self.extractor = DocumentTextExtractor(storage=storage)

    def recover_stale_documents(
        self,
        *,
        stale_timeout: timedelta = timedelta(minutes=15),
        limit: int = 100,
    ) -> int:
        """Recover abandoned document-processing claims."""
        if stale_timeout <= timedelta(0):
            raise ValueError("stale_timeout must be greater than zero.")

        if limit < 1:
            raise ValueError("limit must be at least 1.")

        stale_before = datetime.now(UTC) - stale_timeout
        recovery_session = self.session_factory()

        try:
            repository = DocumentRepository(recovery_session)
            recovered_count = repository.recover_stale_processing(
                stale_before=stale_before,
                max_attempts=self.max_attempts,
                limit=limit,
            )

            recovery_session.commit()

            if recovered_count:
                logger.info(
                    "Recovered stale document-processing claims.",
                    extra={"recovered_count": recovered_count},
                )

            return recovered_count

        except Exception:
            recovery_session.rollback()
            raise
        finally:
            recovery_session.close()

    def process_next_document(self) -> ProcessingOutcome:
        """Claim and process one pending document, if one is available."""
        claim_session = self.session_factory()

        try:
            repository = DocumentRepository(claim_session)
            document = repository.claim_next_pending()

            if document is None:
                claim_session.rollback()
                return ProcessingOutcome(status="idle")

            # Capture values before commit: the session may expire ORM objects.
            document_id = document.id
            storage_key = document.storage_key
            content_type = document.content_type
            processing_token = document.processing_token

            if processing_token is None:
                raise RuntimeError("Claimed document has no processing token.")

            # Persist the claim before performing potentially slow I/O.
            claim_session.commit()

        except Exception:
            claim_session.rollback()
            raise
        finally:
            claim_session.close()

        try:
            extraction = self.extractor.extract(
                storage_key=storage_key,
                content_type=content_type,
            )
        except InvalidStorageKeyError as exc:
            return self._record_failure(
                document_id=document_id,
                processing_token=processing_token,
                error=exc,
                retryable=False,
            )
        except DocumentExtractionError as exc:
            return self._record_failure(
                document_id=document_id,
                processing_token=processing_token,
                error=exc,
                retryable=False,
            )
        except StorageError as exc:
            return self._record_failure(
                document_id=document_id,
                processing_token=processing_token,
                error=exc,
                retryable=True,
            )

        result_session = self.session_factory()

        try:
            repository = DocumentRepository(result_session)
            completed = repository.complete_processing(
                document_id=document_id,
                processing_token=processing_token,
                extracted_text=extraction.text,
                character_count=extraction.character_count,
                page_count=extraction.page_count,
            )

            if not completed:
                result_session.rollback()
                logger.info(
                    "Document processing claim is stale; "
                    "discarding extracted result.",
                    extra={"document_id": str(document_id)},
                )
                return ProcessingOutcome(
                    status="stale_claim",
                    document_id=document_id,
                )

            result_session.commit()

            logger.info(
                "Document extraction completed.",
                extra={"document_id": str(document_id)},
            )

            return ProcessingOutcome(
                status="ready",
                document_id=document_id,
            )

        except Exception:
            result_session.rollback()
            raise
        finally:
            result_session.close()

    def _record_failure(
        self,
        *,
        document_id: UUID,
        processing_token: UUID,
        error: Exception,
        retryable: bool,
    ) -> ProcessingOutcome:
        """Persist an extraction/storage failure using a fresh session."""
        error_message = f"{type(error).__name__}: {error}"[:2000]

        failure_session = self.session_factory()

        try:
            repository = DocumentRepository(failure_session)
            recorded = repository.fail_processing(
                document_id=document_id,
                processing_token=processing_token,
                error_message=error_message,
                retryable=retryable,
                max_attempts=self.max_attempts,
            )

            if not recorded:
                failure_session.rollback()
                logger.info(
                    "Document failure belongs to a stale claim.",
                    extra={"document_id": str(document_id)},
                )
                return ProcessingOutcome(
                    status="stale_claim",
                    document_id=document_id,
                )

            failure_session.commit()

            logger.warning(
                "Document processing failed.",
                extra={
                    "document_id": str(document_id),
                    "error_type": type(error).__name__,
                    "retryable": retryable,
                },
            )

            return ProcessingOutcome(
                status="failure_recorded",
                document_id=document_id,
            )

        except Exception:
            failure_session.rollback()
            raise
        finally:
            failure_session.close()
