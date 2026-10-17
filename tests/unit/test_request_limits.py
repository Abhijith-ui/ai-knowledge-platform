import asyncio

import pytest
from ai_knowledge_platform.core.config import Settings
from ai_knowledge_platform.core.request_limits import (
    DocumentRequestLimitMiddleware,
)
from starlette.types import Message, Receive, Scope, Send


def make_scope(
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Scope:
    """Build a minimal ASGI scope for a document-upload request."""
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/documents",
        "raw_path": b"/api/v1/documents",
        "query_string": b"",
        "root_path": "",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


def test_rejects_excessive_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an oversized request before invoking the application."""
    settings = Settings(
        _env_file=None,
        max_document_request_bytes=5,
    )
    monkeypatch.setattr(
        "ai_knowledge_platform.core.config.get_settings",
        lambda: settings,
    )

    app_called = False

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        nonlocal app_called
        app_called = True

    middleware = DocumentRequestLimitMiddleware(app)
    messages: list[Message] = []

    async def receive() -> Message:
        return {
            "type": "http.request",
            "body": b"",
            "more_body": False,
        }

    async def send(message: Message) -> None:
        messages.append(message)

    asyncio.run(
        middleware(
            make_scope([(b"content-length", b"6")]),
            receive,
            send,
        )
    )

    response_start = next(
        message for message in messages if message["type"] == "http.response.start"
    )

    assert response_start["status"] == 413
    assert app_called is False


def test_rejects_oversized_stream_without_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce the limit using actual bytes when Content-Length is absent."""
    settings = Settings(
        _env_file=None,
        max_document_request_bytes=5,
    )
    monkeypatch.setattr(
        "ai_knowledge_platform.core.config.get_settings",
        lambda: settings,
    )

    received_by_app: list[bytes] = []

    async def app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue

            received_by_app.append(message.get("body", b""))

            if not message.get("more_body", False):
                break

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )

    middleware = DocumentRequestLimitMiddleware(app)
    messages: list[Message] = []
    incoming = iter(
        [
            {
                "type": "http.request",
                "body": b"123",
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": b"456",
                "more_body": False,
            },
        ]
    )

    async def receive() -> Message:
        return next(incoming)

    async def send(message: Message) -> None:
        messages.append(message)

    asyncio.run(
        middleware(
            make_scope(),
            receive,
            send,
        )
    )

    response_start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    response_body = next(
        message for message in messages if message["type"] == "http.response.body"
    )

    assert response_start["status"] == 413
    assert b"maximum allowed size" in response_body["body"]
    assert received_by_app == [b"123"]


def test_does_not_send_second_response_if_response_already_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate the size error if the downstream response has started."""
    from ai_knowledge_platform.core import config
    from ai_knowledge_platform.core.request_limits import (
        RequestBodyTooLargeError,
    )

    settings = Settings(_env_file=None, max_document_request_bytes=5)
    monkeypatch.setattr(config, "get_settings", lambda: settings)

    incoming: list[Message] = [
        {
            "type": "http.request",
            "body": b"123456",
            "more_body": False,
        }
    ]
    sent_messages: list[Message] = []

    async def receive() -> Message:
        return incoming.pop(0)

    async def send(message: Message) -> None:
        sent_messages.append(message)

    async def downstream_app(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await receive()

    middleware = DocumentRequestLimitMiddleware(downstream_app)

    with pytest.raises(RequestBodyTooLargeError):
        asyncio.run(middleware(make_scope(), receive, send))

    response_starts = [
        message for message in sent_messages if message["type"] == "http.response.start"
    ]

    assert len(response_starts) == 1
    assert response_starts[0]["status"] == 200


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", b"5"), (b"content-length", b"6")],
        [(b"content-length", b"invalid")],
        [(b"content-length", b"-1")],
    ],
)
def test_rejects_invalid_content_length(
    monkeypatch: pytest.MonkeyPatch,
    headers: list[tuple[bytes, bytes]],
) -> None:
    """Reject duplicate, malformed, or negative Content-Length values."""
    settings = Settings(_env_file=None, max_document_request_bytes=5)
    monkeypatch.setattr(
        "ai_knowledge_platform.core.config.get_settings",
        lambda: settings,
    )

    app_called = False
    messages: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal app_called
        app_called = True

    async def receive() -> Message:
        return {
            "type": "http.request",
            "body": b"",
            "more_body": False,
        }

    async def send(message: Message) -> None:
        messages.append(message)

    asyncio.run(
        DocumentRequestLimitMiddleware(app)(
            make_scope(headers),
            receive,
            send,
        )
    )

    response_start = next(
        message for message in messages if message["type"] == "http.response.start"
    )

    assert response_start["status"] == 400
    assert app_called is False
