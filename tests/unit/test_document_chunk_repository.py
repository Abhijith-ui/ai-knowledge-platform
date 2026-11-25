from uuid import UUID

import pytest
from ai_knowledge_platform.models.document import Document
from ai_knowledge_platform.models.user import User
from ai_knowledge_platform.repositories.document import DocumentRepository
from ai_knowledge_platform.repositories.document_chunk import (
    DocumentChunkRepository,
)
from ai_knowledge_platform.repositories.ingestion_run import (
    IngestionRunRepository,
)
from ai_knowledge_platform.repositories.user import UserRepository
from ai_knowledge_platform.services.chunking.types import (
    DocumentChunk as ChunkData,
)
from sqlalchemy.orm import Session


def create_ingestion_run(db: Session) -> UUID:
    """Create an ingestion run for repository tests."""
    user: User = UserRepository(db).create("chunks@example.com")

    document: Document = DocumentRepository(db).create(
        user_id=user.id,
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        checksum_sha256="a" * 64,
        storage_key=f"users/{user.id}/documents/test.pdf",
    )

    run = IngestionRunRepository(db).create_run(
        document_id=document.id,
    )

    return run.id


def create_chunks() -> list[ChunkData]:
    """Create valid in-memory chunk DTOs."""
    return [
        ChunkData(
            chunk_index=0,
            content="First document section.",
            token_count=5,
            start_char=0,
            end_char=23,
            page_start=1,
            page_end=1,
        ),
        ChunkData(
            chunk_index=1,
            content="Second document section.",
            token_count=6,
            start_char=20,
            end_char=44,
            page_start=1,
            page_end=2,
        ),
    ]


def test_replace_chunks_persists_chunk_data(db: Session) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    chunks = create_chunks()

    created = repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=chunks,
    )

    assert len(created) == 2

    assert created[0].ingestion_run_id == ingestion_run_id
    assert created[0].chunk_index == 0
    assert created[0].content == chunks[0].content
    assert created[0].token_count == chunks[0].token_count
    assert created[0].start_char == chunks[0].start_char
    assert created[0].end_char == chunks[0].end_char
    assert created[0].page_start == chunks[0].page_start
    assert created[0].page_end == chunks[0].page_end
    assert created[0].chunk_metadata == {}


def test_list_chunks_returns_ordered_results(db: Session) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=create_chunks(),
    )

    chunks = repository.list_by_run_id(ingestion_run_id)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].content == "First document section."
    assert chunks[1].content == "Second document section."


def test_replace_chunks_removes_previous_records(db: Session) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=create_chunks(),
    )

    replacement = [
        ChunkData(
            chunk_index=0,
            content="Replacement content.",
            token_count=4,
            start_char=0,
            end_char=20,
        )
    ]

    repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=replacement,
    )

    chunks = repository.list_by_run_id(ingestion_run_id)

    assert len(chunks) == 1
    assert chunks[0].content == "Replacement content."
    assert chunks[0].page_start is None
    assert chunks[0].page_end is None


def test_replace_with_empty_chunks_clears_existing_records(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=create_chunks(),
    )

    repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=[],
    )

    assert repository.list_by_run_id(ingestion_run_id) == []


def test_replace_rejects_non_sequential_chunk_indexes(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    invalid_chunks = [
        ChunkData(
            chunk_index=1,
            content="Invalid first chunk.",
            token_count=4,
            start_char=0,
            end_char=20,
        )
    ]

    with pytest.raises(
        ValueError,
        match="sequential and start at zero",
    ):
        repository.replace_for_run(
            ingestion_run_id=ingestion_run_id,
            chunks=invalid_chunks,
        )


def test_invalid_replacement_preserves_existing_chunks(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=create_chunks(),
    )

    invalid_chunks = [
        ChunkData(
            chunk_index=1,
            content="Invalid replacement.",
            token_count=4,
            start_char=0,
            end_char=20,
        )
    ]

    with pytest.raises(ValueError):
        repository.replace_for_run(
            ingestion_run_id=ingestion_run_id,
            chunks=invalid_chunks,
        )

    chunks = repository.list_by_run_id(ingestion_run_id)

    assert len(chunks) == 2
    assert chunks[0].content == "First document section."
    assert chunks[1].content == "Second document section."


def test_list_chunks_returns_empty_when_run_has_no_chunks(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    assert repository.list_by_run_id(ingestion_run_id) == []


def test_chunks_without_page_metadata_are_supported(
    db: Session,
) -> None:
    ingestion_run_id = create_ingestion_run(db)
    repository = DocumentChunkRepository(db)

    chunk = ChunkData(
        chunk_index=0,
        content="Plain text document.",
        token_count=4,
        start_char=0,
        end_char=20,
    )

    created = repository.replace_for_run(
        ingestion_run_id=ingestion_run_id,
        chunks=[chunk],
    )

    assert created[0].page_start is None
    assert created[0].page_end is None
