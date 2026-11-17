import pytest
from ai_knowledge_platform.services.chunking.boundaries import TextSpan
from ai_knowledge_platform.services.chunking.service import (
    DocumentChunkingService,
)
from ai_knowledge_platform.services.chunking.types import ChunkingConfig
from ai_knowledge_platform.services.tokenization.huggingface import (
    HuggingFaceTokenizer,
)


@pytest.fixture(scope="module")
def tokenizer() -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(
        model_name="intfloat/multilingual-e5-base",
        max_input_tokens=512,
    )


def test_effective_token_budget(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=448,
            overlap_tokens=64,
        ),
    )

    expected = min(
        448,
        tokenizer.max_input_tokens - tokenizer.special_tokens_count,
    )

    assert service.token_budget == expected


def test_split_oversized_span(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=10,
            overlap_tokens=2,
            minimum_tokens=1,
        ),
    )

    text = "Machine learning is useful for solving complex problems. " * 20

    spans = service._split_span_by_tokens(
        source_text=text,
        span=TextSpan(start=0, end=len(text)),
    )

    assert len(spans) > 1

    for span in spans:
        content = text[span.start : span.end]

        assert content
        assert tokenizer.count_tokens(content) <= service.token_budget


def test_split_preserves_source_offsets(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=10,
            overlap_tokens=2,
            minimum_tokens=1,
        ),
    )

    text = "First paragraph contains useful information. " * 15

    spans = service._split_span_by_tokens(
        source_text=text,
        span=TextSpan(start=0, end=len(text)),
    )

    for span in spans:
        assert text[span.start : span.end]
        assert span.start >= 0
        assert span.end <= len(text)


def test_atomic_spans_do_not_overlap(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=10,
            overlap_tokens=2,
            minimum_tokens=1,
        ),
    )

    text = "Machine learning is useful for solving complex problems. " * 20

    spans = service._split_span_by_tokens(
        source_text=text,
        span=TextSpan(start=0, end=len(text)),
    )

    assert len(spans) > 1

    for previous, current in zip(spans, spans[1:], strict=False):
        assert current.start >= previous.end


def test_chunk_text_returns_single_chunk_for_short_text(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(tokenizer=tokenizer)

    text = "Machine learning is a field of artificial intelligence."

    chunks = service.chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


def test_chunk_text_splits_oversized_document(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=4,
            minimum_tokens=1,
        ),
    )

    text = (
        "Machine learning is useful for solving complex problems. "
        "Deep learning is a subset of machine learning.\n\n"
    ) * 20

    chunks = service.chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.token_count <= service.token_budget
        assert text[chunk.start_char : chunk.end_char] == chunk.content


def test_chunk_text_preserves_paragraph_separators(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=4,
            minimum_tokens=1,
        ),
    )

    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains more useful information."
    )

    chunks = service.chunk_text(text)

    combined_content = "".join(chunk.content for chunk in chunks)

    assert combined_content == text


def test_chunk_text_handles_empty_text(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(tokenizer=tokenizer)

    assert service.chunk_text("") == []
    assert service.chunk_text("   \n\n") == []


def test_chunk_text_assigns_sequential_indexes(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=4,
            minimum_tokens=1,
        ),
    )

    text = "Machine learning solves problems. " * 30

    chunks = service.chunk_text(text)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_text_preserves_complete_source_coverage(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=4,
            minimum_tokens=1,
        ),
    )

    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains more useful information.\n\n"
        "Third paragraph contains additional information."
    )

    chunks = service.chunk_text(text)

    assert chunks
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_char <= previous.end_char
        assert current.start_char > previous.start_char

    for chunk in chunks:
        assert chunk.content == text[chunk.start_char : chunk.end_char]


def test_chunk_text_applies_token_overlap(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    overlap_tokens = 4

    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=overlap_tokens,
            minimum_tokens=1,
        ),
    )

    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains more useful information.\n\n"
        "Third paragraph contains additional information.\n\n"
        "Fourth paragraph contains important information."
    )

    chunks = service.chunk_text(text)

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_char < previous.end_char

        overlap_text = text[current.start_char : previous.end_char]

        overlap_count = tokenizer.count_tokens(overlap_text)

        assert overlap_count == overlap_tokens


def test_chunk_text_handles_multilingual_content(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=4,
            minimum_tokens=1,
        ),
    )

    text = (
        "Machine learning is useful. "
        "मशीन लर्निंग उपयोगी है। "
        "機械学習は便利です。 "
        "L'apprentissage automatique est utile. "
    ) * 10

    chunks = service.chunk_text(text)

    assert chunks

    for chunk in chunks:
        assert chunk.token_count <= service.token_budget
        assert chunk.content == text[chunk.start_char : chunk.end_char]


def test_chunk_text_preserves_every_character_without_overlap(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=0,
            minimum_tokens=1,
        ),
    )

    text = (
        "  First paragraph contains useful information.\n\n"
        "Second paragraph contains more useful information.\n\n"
        "Third paragraph contains additional information.  "
    )

    chunks = service.chunk_text(text)

    assert chunks
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_char == previous.end_char

    reconstructed = "".join(chunk.content for chunk in chunks)
    assert reconstructed == text


def test_every_source_character_is_covered(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=4,
            minimum_tokens=1,
        ),
    )

    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains more useful information.\n\n"
        "Third paragraph contains additional information."
    )

    chunks = service.chunk_text(text)

    assert chunks
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_char <= previous.end_char

    covered = [False] * len(text)

    for chunk in chunks:
        for position in range(chunk.start_char, chunk.end_char):
            covered[position] = True

    assert all(covered)


def test_all_generated_chunks_respect_token_budget(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=20,
            overlap_tokens=4,
            minimum_tokens=1,
        ),
    )

    text = (
        "Machine learning is transforming software engineering. "
        "Neural networks learn complex representations from data. "
    ) * 30

    chunks = service.chunk_text(text)

    assert chunks

    for chunk in chunks:
        actual_token_count = tokenizer.count_tokens(chunk.content)

        assert actual_token_count <= service.token_budget
        assert actual_token_count == chunk.token_count


def test_chunk_text_prefers_paragraph_boundaries(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=15,
            overlap_tokens=0,
            minimum_tokens=1,
        ),
    )

    text = (
        "First paragraph contains useful information.\n\n"
        "Second paragraph contains important information.\n\n"
        "Third paragraph contains additional information."
    )

    chunks = service.chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks[:-1]:
        assert chunk.content.rstrip().endswith("information.")


def test_chunk_text_prefers_sentence_boundaries(
    tokenizer: HuggingFaceTokenizer,
) -> None:
    service = DocumentChunkingService(
        tokenizer=tokenizer,
        config=ChunkingConfig(
            target_tokens=15,
            overlap_tokens=0,
            minimum_tokens=1,
        ),
    )

    text = (
        "Machine learning helps computers learn from data. "
        "Deep learning uses neural networks to learn representations. "
        "Natural language processing enables computers to understand text. "
        "Computer vision enables machines to interpret images."
    )

    chunks = service.chunk_text(text)

    assert len(chunks) > 1

    for chunk in chunks[:-1]:
        assert chunk.content.rstrip().endswith((".", "!", "?"))
