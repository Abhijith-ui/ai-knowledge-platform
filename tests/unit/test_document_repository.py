from uuid import UUID, uuid4

import pytest
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.user import UserRepository
from sqlalchemy.orm import Session


def create_user(db: Session, email: str) -> User:
    """Create a user for repository tests."""
    return UserRepository(db).create(email)


def create_document(
    db: Session,
    user_id: UUID,
    filename: str = "report.pdf",
) -> Document:
    """Create document metadata for repository tests."""
    return DocumentRepository(db).create(
        user_id=user_id,
        filename=filename,
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user_id}/documents/{uuid4()}",
    )


def test_create_document(db: Session) -> None:
    user = create_user(db, "owner@example.com")

    document = create_document(db, user.id)
    db.commit()

    persisted = db.get(Document, document.id)

    assert persisted is not None
    assert persisted.user_id == user.id
    assert persisted.filename == "report.pdf"
    assert persisted.size_bytes == 1024


def test_get_document_for_owner(db: Session) -> None:
    user = create_user(db, "owner@example.com")
    document = create_document(db, user.id)

    result = DocumentRepository(db).get_by_id_for_owner(
        document.id,
        user.id,
    )

    assert result is not None
    assert result.id == document.id


def test_cannot_get_another_users_document(db: Session) -> None:
    owner = create_user(db, "owner@example.com")
    another_user = create_user(db, "another@example.com")
    document = create_document(db, owner.id)

    result = DocumentRepository(db).get_by_id_for_owner(
        document.id,
        another_user.id,
    )

    assert result is None


def test_list_documents_only_returns_owners_documents(
    db: Session,
) -> None:
    owner = create_user(db, "owner@example.com")
    another_user = create_user(db, "another@example.com")

    first = create_document(db, owner.id, "first.pdf")
    second = create_document(db, owner.id, "second.pdf")
    create_document(db, another_user.id, "private.pdf")

    documents = DocumentRepository(db).list_by_owner(owner.id)

    assert len(documents) == 2
    assert {document.id for document in documents} == {
        first.id,
        second.id,
    }
    assert all(document.user_id == owner.id for document in documents)


def test_list_documents_respects_pagination(db: Session) -> None:
    user = create_user(db, "owner@example.com")

    for index in range(3):
        create_document(db, user.id, f"document-{index}.pdf")

    repository = DocumentRepository(db)

    first_page = repository.list_by_owner(user.id, limit=2, offset=0)
    second_page = repository.list_by_owner(user.id, limit=2, offset=2)

    assert len(first_page) == 2
    assert len(second_page) == 1
    assert {document.id for document in first_page}.isdisjoint(
        {document.id for document in second_page}
    )


@pytest.mark.parametrize(
    ("limit", "offset"),
    [(0, 0), (101, 0), (10, -1)],
)
def test_list_documents_rejects_invalid_pagination(
    db: Session,
    limit: int,
    offset: int,
) -> None:
    repository = DocumentRepository(db)

    with pytest.raises(ValueError):
        repository.list_by_owner(
            uuid4(),
            limit=limit,
            offset=offset,
        )
