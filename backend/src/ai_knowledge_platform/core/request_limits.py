from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ai_knowledge_platform.core import config


class RequestBodyTooLargeError(Exception):
    """Raised when an HTTP request body exceeds its configured limit."""


class DocumentRequestLimitMiddleware:
    """Limit the body size of document-upload requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/api/v1/documents"
        ):
            await self.app(scope, receive, send)
            return

        max_bytes = config.get_settings().max_document_request_bytes

        content_lengths = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]

        # Reject ambiguous or malformed Content-Length headers.
        if len(content_lengths) > 1:
            await self._reject(
                scope,
                receive,
                send,
                status_code=400,
                detail="Invalid Content-Length header.",
            )
            return

        if content_lengths:
            try:
                declared_length = int(content_lengths[0])
            except ValueError:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status_code=400,
                    detail="Invalid Content-Length header.",
                )
                return

            if declared_length < 0:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status_code=400,
                    detail="Invalid Content-Length header.",
                )
                return

            if declared_length > max_bytes:
                await self._reject(scope, receive, send)
                return

        total_bytes = 0
        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started

            if message["type"] == "http.response.start":
                response_started = True

            await send(message)

        async def limited_receive() -> Message:
            nonlocal total_bytes

            message = await receive()

            if message["type"] == "http.request":
                body = message.get("body", b"")

                if total_bytes + len(body) > max_bytes:
                    raise RequestBodyTooLargeError

                total_bytes += len(body)

            return message

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLargeError:
            if response_started:
                raise

            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int = 413,
        detail: str = ("Document upload request exceeds the maximum allowed size."),
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={"detail": detail},
            headers={"Connection": "close"},
        )

        await response(scope, receive, send)
