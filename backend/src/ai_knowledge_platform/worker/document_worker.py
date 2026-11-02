"""Background worker for document extraction and stale-claim recovery."""

import logging
import signal
import threading
import time
from datetime import timedelta

from ai_knowledge_platform.core.config import get_settings
from ai_knowledge_platform.core.logging import configure_logging
from ai_knowledge_platform.services.document_processing import (
    DocumentProcessingService,
)

logger = logging.getLogger(__name__)


def run_worker(
    *,
    service: DocumentProcessingService,
    stop_event: threading.Event,
    poll_interval_seconds: float,
    recovery_interval_seconds: float,
    stale_timeout_seconds: float,
) -> None:
    """Poll for document work until a shutdown signal is received."""
    if poll_interval_seconds <= 0:
        raise ValueError("Polling interval must be greater than zero.")

    if recovery_interval_seconds <= 0:
        raise ValueError("Recovery interval must be greater than zero.")

    if stale_timeout_seconds <= 0:
        raise ValueError("Stale timeout must be greater than zero.")

    recovery_interval = recovery_interval_seconds
    stale_timeout = timedelta(seconds=stale_timeout_seconds)

    # Recover abandoned claims immediately on startup.
    next_recovery_at = 0.0

    logger.info("Document worker started.")

    while not stop_event.is_set():
        now = time.monotonic()

        if now >= next_recovery_at:
            try:
                recovered_count = service.recover_stale_documents(
                    stale_timeout=stale_timeout,
                )

                logger.info(
                    "Stale-claim recovery completed.",
                    extra={"recovered_count": recovered_count},
                )
            except Exception:
                logger.exception("Stale-claim recovery failed.")

            # Avoid repeatedly querying PostgreSQL when recovery fails.
            next_recovery_at = time.monotonic() + recovery_interval

        if stop_event.is_set():
            break

        try:
            outcome = service.process_next_document()

            if outcome.status == "idle":
                # Interruptible wait: shutdown does not have to wait for sleep.
                stop_event.wait(timeout=poll_interval_seconds)

        except Exception:
            logger.exception("Unexpected document-processing error.")
            stop_event.wait(timeout=poll_interval_seconds)

    logger.info("Document worker stopped.")


def main() -> None:
    """Configure and start the document worker."""
    configure_logging()

    settings = get_settings()
    stop_event = threading.Event()

    def request_shutdown(signum: int, frame: object) -> None:
        logger.info(
            "Shutdown signal received.",
            extra={"signal_number": signum},
        )
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    service = DocumentProcessingService()

    run_worker(
        service=service,
        stop_event=stop_event,
        poll_interval_seconds=settings.worker_poll_interval_seconds,
        recovery_interval_seconds=settings.worker_recovery_interval_seconds,
        stale_timeout_seconds=settings.worker_stale_timeout_seconds,
    )


if __name__ == "__main__":
    main()
