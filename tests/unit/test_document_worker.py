
import threading
from datetime import timedelta
from unittest.mock import Mock, call

import pytest

from ai_knowledge_platform.services.document_processing import (
    ProcessingOutcome,
)
from ai_knowledge_platform.worker.document_worker import run_worker


def test_worker_recovers_stale_claims_on_startup() -> None:
    service = Mock()
    service.recover_stale_documents.return_value = 0
    service.process_next_document.return_value = ProcessingOutcome(
        status="idle"
    )

    stop_event = threading.Event()

    def stop_after_first_wait(timeout: float | None = None) -> bool:
        stop_event.set()
        return True

    stop_event.wait = Mock(side_effect=stop_after_first_wait)  # type: ignore[method-assign]

    run_worker(
        service=service,
        stop_event=stop_event,
        poll_interval_seconds=2.0,
        recovery_interval_seconds=60.0,
        stale_timeout_seconds=900.0,
    )

    service.recover_stale_documents.assert_called_once_with(
        stale_timeout=timedelta(seconds=900.0),
    )
    service.process_next_document.assert_called_once()


def test_worker_continues_after_processing_error() -> None:
    service = Mock()
    service.recover_stale_documents.return_value = 0
    service.process_next_document.side_effect = [
        RuntimeError("temporary failure"),
        ProcessingOutcome(status="idle"),
    ]

    stop_event = threading.Event()
    wait_calls = 0

    def stop_after_second_wait(timeout: float | None = None) -> bool:
        nonlocal wait_calls
        wait_calls += 1

        if wait_calls == 2:
            stop_event.set()

        return stop_event.is_set()

    stop_event.wait = Mock(side_effect=stop_after_second_wait)  # type: ignore[method-assign]

    run_worker(
        service=service,
        stop_event=stop_event,
        poll_interval_seconds=2.0,
        recovery_interval_seconds=60.0,
        stale_timeout_seconds=900.0,
    )

    assert service.process_next_document.call_count == 2


@pytest.mark.parametrize(
    (
        "poll_interval",
        "recovery_interval",
        "stale_timeout",
    ),
    [
        (0, 60, 900),
        (2, 0, 900),
        (2, 60, 0),
        (-1, 60, 900),
    ],
)
def test_worker_rejects_invalid_intervals(
    poll_interval: float,
    recovery_interval: float,
    stale_timeout: float,
) -> None:
    with pytest.raises(ValueError):
        run_worker(
            service=Mock(),
            stop_event=threading.Event(),
            poll_interval_seconds=poll_interval,
            recovery_interval_seconds=recovery_interval,
            stale_timeout_seconds=stale_timeout,
        )
