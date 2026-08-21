from fastapi.testclient import TestClient


def test_create_user(client: TestClient) -> None:
    response = client.post(
        "/api/v1/users/",
        json={"email": "api-test-002@example.com"},
    )

    assert response.status_code == 201

    data = response.json()

    assert data["email"] == "api-test-002@example.com"
    assert "id" in data


def test_create_user_invalid_email(client: TestClient) -> None:
    response = client.post(
        "/api/v1/users/",
        json={"email": "not-an-email"},
    )

    assert response.status_code == 422


def test_create_duplicate_user(client: TestClient) -> None:
    email = "api-duplicate-002@example.com"

    first_response = client.post(
        "/api/v1/users/",
        json={"email": email},
    )

    assert first_response.status_code == 201

    second_response = client.post(
        "/api/v1/users/",
        json={"email": email},
    )

    assert second_response.status_code == 409


def test_register_user(client: TestClient) -> None:
    response = client.post(
        "/api/v1/users/register",
        json={
            "email": "register-api@example.com",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["email"] == "register-api@example.com"
    assert "id" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_register_user_invalid_email(client: TestClient) -> None:
    response = client.post(
        "/api/v1/users/register",
        json={
            "email": "not-an-email",
            "password": "secure-password-123",
        },
    )

    assert response.status_code == 422


def test_register_user_short_password(client: TestClient) -> None:
    response = client.post(
        "/api/v1/users/register",
        json={
            "email": "short-password@example.com",
            "password": "short",
        },
    )

    assert response.status_code == 422


def test_register_user_duplicate_email(client: TestClient) -> None:
    payload = {
        "email": "register-duplicate@example.com",
        "password": "secure-password-123",
    }

    first_response = client.post(
        "/api/v1/users/register",
        json=payload,
    )

    assert first_response.status_code == 201

    second_response = client.post(
        "/api/v1/users/register",
        json=payload,
    )

    assert second_response.status_code == 409
