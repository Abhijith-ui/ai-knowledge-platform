from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest
from ai_knowledge_platform.services import auth
from ai_knowledge_platform.services.auth import (
    AuthenticationError,
    AuthService,
)


def test_refresh_success_rotates_token() -> None:
    service, db, jwt_service = make_service()

    user_id = uuid4()
    family_id = uuid4()

    session = Mock()
    session.id = uuid4()
    session.user_id = user_id
    session.family_id = family_id
    session.revoked_at = None
    session.replaced_by_id = None
    session.expires_at = datetime.now(UTC) + timedelta(hours=1)

    successor = Mock()
    successor.id = uuid4()

    jwt_service.decode_token.return_value = {"sub": str(user_id)}
    jwt_service.create_refresh_token.return_value = "new-refresh-token"
    jwt_service.create_access_token.return_value = "new-access-token"

    service.refresh_sessions.get_by_token_hash.return_value = session
    service.refresh_sessions.create.return_value = successor
    service.refresh_sessions.mark_rotated.return_value = True

    result = service.refresh(refresh_token="old-refresh-token")

    assert result["access_token"] == "new-access-token"
    assert result["refresh_token"] == "new-refresh-token"

    service.refresh_sessions.mark_rotated.assert_called_once()
    assert (
        service.refresh_sessions.mark_rotated.call_args.kwargs["session_id"]
        == session.id
    )
    assert (
        service.refresh_sessions.mark_rotated.call_args.kwargs["successor_id"]
        == successor.id
    )
    db.commit.assert_called_once()


def test_refresh_reuse_revokes_token_family() -> None:
    service, db, jwt_service = make_service()

    user_id = uuid4()
    family_id = uuid4()

    session = Mock()
    session.id = uuid4()
    session.user_id = user_id
    session.family_id = family_id
    session.revoked_at = datetime.now(UTC)
    session.replaced_by_id = uuid4()

    jwt_service.decode_token.return_value = {"sub": str(user_id)}
    service.refresh_sessions.get_by_token_hash.return_value = session

    with pytest.raises(AuthenticationError, match="Invalid refresh token"):
        service.refresh(refresh_token="previously-used-token")

    service.refresh_sessions.revoke_family.assert_called_once()
    assert (
        service.refresh_sessions.revoke_family.call_args.kwargs["family_id"]
        == family_id
    )
    db.commit.assert_called_once()


def test_refresh_rejects_expired_session() -> None:
    service, db, jwt_service = make_service()

    user_id = uuid4()

    session = Mock()
    session.id = uuid4()
    session.user_id = user_id
    session.family_id = uuid4()
    session.revoked_at = None
    session.replaced_by_id = None
    session.expires_at = datetime.now(UTC) - timedelta(minutes=1)

    jwt_service.decode_token.return_value = {"sub": str(user_id)}
    service.refresh_sessions.get_by_token_hash.return_value = session

    with pytest.raises(AuthenticationError, match="Invalid refresh token"):
        service.refresh(refresh_token="expired-refresh-token")

    service.refresh_sessions.create.assert_not_called()
    service.refresh_sessions.mark_rotated.assert_not_called()
    db.commit.assert_not_called()


def make_service() -> tuple[AuthService, Mock, Mock]:
    db = Mock()
    jwt_service = Mock()
    service = AuthService(db, jwt_service)

    service.users = Mock()
    service.refresh_sessions = Mock()

    return service, db, jwt_service


def test_login_rejects_unknown_user() -> None:
    service, _, jwt_service = make_service()
    service.users.get_by_email.return_value = None

    with pytest.raises(
        AuthenticationError,
        match="Invalid email or password",
    ):
        service.login(
            email="unknown@example.com",
            password="a-long-password",
        )

    jwt_service.create_access_token.assert_not_called()
    jwt_service.create_refresh_token.assert_not_called()


def test_login_rejects_user_without_credentials() -> None:
    service, _, jwt_service = make_service()

    user = Mock()
    user.id = uuid4()
    user.credential = None
    service.users.get_by_email.return_value = user

    with pytest.raises(AuthenticationError):
        service.login(
            email="user@example.com",
            password="a-long-password",
        )

    jwt_service.create_access_token.assert_not_called()
    jwt_service.create_refresh_token.assert_not_called()


def test_login_rejects_incorrect_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, jwt_service = make_service()

    user = Mock()
    user.id = uuid4()
    user.credential.password_hash = "stored-hash"
    service.users.get_by_email.return_value = user

    monkeypatch.setattr(auth, "verify_password", lambda *_: False)

    with pytest.raises(AuthenticationError):
        service.login(
            email="user@example.com",
            password="wrong-password",
        )

    jwt_service.create_access_token.assert_not_called()
    jwt_service.create_refresh_token.assert_not_called()
    service.refresh_sessions.create.assert_not_called()


def test_login_success_creates_refresh_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, jwt_service = make_service()

    user = Mock()
    user.id = uuid4()
    user.credential.password_hash = "stored-hash"
    service.users.get_by_email.return_value = user

    monkeypatch.setattr(auth, "verify_password", lambda *_: True)
    jwt_service.create_access_token.return_value = "access-token"
    jwt_service.create_refresh_token.return_value = "refresh-token"

    result = service.login(
        email="user@example.com",
        password="correct-password",
    )

    assert result["access_token"] == "access-token"
    assert result["refresh_token"] == "refresh-token"
    assert result["token_type"] == "bearer"
    assert isinstance(result["expires_in"], int)

    service.refresh_sessions.create.assert_called_once()

    call_kwargs = service.refresh_sessions.create.call_args.kwargs
    assert call_kwargs["user_id"] == user.id
    assert call_kwargs["family_id"] is not None
    assert len(call_kwargs["token_hash"]) == 64


def test_logout_revokes_matching_session() -> None:
    service, _, jwt_service = make_service()

    user_id = uuid4()
    session = Mock()
    session.user_id = user_id
    session.id = uuid4()

    jwt_service.decode_token.return_value = {"sub": str(user_id)}
    service.refresh_sessions.get_by_token_hash.return_value = session

    service.logout(refresh_token="valid-refresh-token")

    service.refresh_sessions.revoke_session.assert_called_once()
    assert (
        service.refresh_sessions.revoke_session.call_args.kwargs["session_id"]
        == session.id
    )


def test_logout_all_revokes_user_sessions() -> None:
    service, _, _ = make_service()
    user_id = uuid4()
    service.refresh_sessions.revoke_all_for_user.return_value = 3

    count = service.logout_all(user_id=user_id)

    assert count == 3
    service.refresh_sessions.revoke_all_for_user.assert_called_once()

    call_kwargs = service.refresh_sessions.revoke_all_for_user.call_args.kwargs
    assert call_kwargs["user_id"] == user_id
    assert call_kwargs["revoked_at"] is not None
