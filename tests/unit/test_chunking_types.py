import pytest
from ai_knowledge_platform.services.chunking.types import (
    ChunkingConfig,
    DocumentChunk,
)


def test_default_chunking_config() -> None:
    config = ChunkingConfig()

    assert config.target_tokens == 448
    assert config.overlap_tokens == 64
    assert config.minimum_tokens == 50


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"target_tokens": 0}, "target_tokens"),
        ({"overlap_tokens": -1}, "overlap_tokens"),
        ({"target_tokens": 100, "overlap_tokens": 100}, "overlap_tokens"),
        ({"minimum_tokens": 0}, "minimum_tokens"),
        ({"target_tokens": 100, "minimum_tokens": 101}, "minimum_tokens"),
    ],
)
def test_invalid_chunking_config(
    kwargs: dict[str, int],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        ChunkingConfig(**kwargs)


def test_create_document_chunk() -> None:
    content = "Machine learning is interesting."

    chunk = DocumentChunk(
        chunk_index=0,
        content=content,
        token_count=5,
        start_char=0,
        end_char=len(content),
        page_start=1,
        page_end=1,
    )

    assert chunk.chunk_index == 0
    assert chunk.content == content
    assert chunk.start_char == 0
    assert chunk.end_char == len(content)
    assert chunk.page_start == 1
    assert chunk.page_end == 1


def test_document_chunk_without_page_metadata() -> None:
    chunk = DocumentChunk(
        chunk_index=0,
        content="Hello world",
        token_count=2,
        start_char=0,
        end_char=11,
    )

    assert chunk.page_start is None
    assert chunk.page_end is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_index": -1},
        {"content": ""},
        {"token_count": 0},
        {"start_char": -1},
        {"end_char": 0},
        {"end_char": 12},
        {"page_start": 1, "page_end": None},
        {"page_start": 0, "page_end": 1},
        {"page_start": 2, "page_end": 1},
    ],
)
def test_invalid_document_chunk(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "chunk_index": 0,
        "content": "Hello world",
        "token_count": 2,
        "start_char": 0,
        "end_char": 11,
        "page_start": None,
        "page_end": None,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        DocumentChunk(**values)
