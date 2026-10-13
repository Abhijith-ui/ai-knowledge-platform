from fastapi.testclient import TestClient


def register_and_login(
    client: TestClient,
    email: str = "auth-api@example.com",
) -> dict[str, str | int]:
    """Create a test account and obtain its tokens."""
    registration = client.post(
        "/api/v1/users/register",
        json={
            "email": email,
            "password": "secure-password-123",
        },
    )
    assert registration.status_code == 201

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": "secure-password-123",
        },
    )
    assert response.status_code == 200

    return response.json()


def test_login_success(client: TestClient) -> None:
    tokens = register_and_login(client)

    assert tokens["access_token"]
    assert tokens["refresh_token"]
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] > 0


def test_login_rejects_incorrect_password(client: TestClient) -> None:
    client.post(
        "/api/v1/users/register",
        json={
            "email": "wrong-password@example.com",
            "password": "secure-password-123",
        },
    )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "wrong-password@example.com",
            "password": "incorrect-password",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_login_rejects_unknown_user(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "unknown-user@example.com",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 401


def test_refresh_rotates_tokens(client: TestClient) -> None:
    tokens = register_and_login(client, "refresh-api@example.com")

    response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert response.status_code == 200

    refreshed = response.json()
    assert refreshed["access_token"]
    assert refreshed["refresh_token"]
    assert refreshed["refresh_token"] != tokens["refresh_token"]


def test_refresh_reuse_revokes_token_family(client: TestClient) -> None:
    tokens = register_and_login(client, "reuse-api@example.com")

    first_refresh = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert first_refresh.status_code == 200

    successor_token = first_refresh.json()["refresh_token"]

    # Reusing the consumed token must trigger refresh-family revocation.
    reused = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert reused.status_code == 401

    # The previously issued successor must now also be invalid.
    successor_attempt = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": successor_token},
    )
    assert successor_attempt.status_code == 401


def test_logout_revokes_refresh_session(client: TestClient) -> None:
    tokens = register_and_login(client, "logout-api@example.com")

    logout_response = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert logout_response.status_code == 200
    assert logout_response.json()["message"] == "Logged out successfully."

    refresh_response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert refresh_response.status_code == 401


def test_login_rejects_invalid_request_body(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "invalid-email"},
    )

    assert response.status_code == 422
