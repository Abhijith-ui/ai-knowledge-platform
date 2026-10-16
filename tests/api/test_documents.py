from uuid import UUID, uuid4

from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.user import UserRepository
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def register_and_login(
    client: TestClient,
    email: str,
) -> dict[str, str | int]:
    """Register a user and obtain authentication tokens."""
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


def create_document(
    db: Session,
    user: User,
    filename: str = "report.pdf",
) -> UUID:
    """Create document metadata for API tests."""
    document = DocumentRepository(db).create(
        user_id=user.id,
        filename=filename,
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user.id}/documents/{uuid4()}",
    )
    db.commit()
    return document.id


def test_list_documents_requires_authentication(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/documents")

    assert response.status_code == 401


def test_get_document_requires_authentication(
    client: TestClient,
) -> None:
    response = client.get(f"/api/v1/documents/{uuid4()}")

    assert response.status_code == 401


def test_list_documents_returns_only_current_users_documents(
    client: TestClient,
    db: Session,
) -> None:
    owner_tokens = register_and_login(client, "document-owner@example.com")
    register_and_login(client, "document-other@example.com")

    owner = UserRepository(db).get_by_email("document-owner@example.com")
    other_user = UserRepository(db).get_by_email("document-other@example.com")

    assert owner is not None
    assert other_user is not None

    owned_document_id = create_document(db, owner, "owned.pdf")
    create_document(db, other_user, "private.pdf")

    response = client.get(
        "/api/v1/documents",
        headers={
            "Authorization": f"Bearer {owner_tokens['access_token']}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == str(owned_document_id)
    assert body["items"][0]["filename"] == "owned.pdf"
    assert "storage_key" not in body["items"][0]


def test_get_document_returns_document_for_owner(
    client: TestClient,
    db: Session,
) -> None:
    tokens = register_and_login(client, "document-get@example.com")
    user = UserRepository(db).get_by_email("document-get@example.com")

    assert user is not None
    document_id = create_document(db, user)

    response = client.get(
        f"/api/v1/documents/{document_id}",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(document_id)
    assert response.json()["filename"] == "report.pdf"
    assert "storage_key" not in response.json()


def test_get_document_rejects_another_users_document(
    client: TestClient,
    db: Session,
) -> None:
    register_and_login(client, "document-private-owner@example.com")
    other_tokens = register_and_login(client, "document-private-other@example.com")

    owner = UserRepository(db).get_by_email("document-private-owner@example.com")

    assert owner is not None
    document_id = create_document(db, owner)

    response = client.get(
        f"/api/v1/documents/{document_id}",
        headers={"Authorization": f"Bearer {other_tokens['access_token']}"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_get_document_returns_404_for_unknown_document(
    client: TestClient,
) -> None:
    tokens = register_and_login(client, "document-missing@example.com")

    response = client.get(
        f"/api/v1/documents/{uuid4()}",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )

    assert response.status_code == 404


def test_list_documents_validates_pagination(
    client: TestClient,
) -> None:
    tokens = register_and_login(client, "document-pagination@example.com")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    invalid_limit = client.get(
        "/api/v1/documents?limit=0",
        headers=headers,
    )
    invalid_offset = client.get(
        "/api/v1/documents?offset=-1",
        headers=headers,
    )
    excessive_limit = client.get(
        "/api/v1/documents?limit=101",
        headers=headers,
    )

    assert invalid_limit.status_code == 422
    assert invalid_offset.status_code == 422
    assert excessive_limit.status_code == 422
