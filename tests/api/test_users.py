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
