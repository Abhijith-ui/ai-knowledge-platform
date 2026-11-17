from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration controlling document chunk generation."""

    target_tokens: int = 448
    overlap_tokens: int = 64
    minimum_tokens: int = 50

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive.")

        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative.")

        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens.")

        if self.minimum_tokens <= 0:
            raise ValueError("minimum_tokens must be positive.")

        if self.minimum_tokens > self.target_tokens:
            raise ValueError("minimum_tokens cannot exceed target_tokens.")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A text chunk linked to its original document position."""

    chunk_index: int
    content: str
    token_count: int
    start_char: int
    end_char: int
    page_start: int | None = None
    page_end: int | None = None

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")

        if not self.content:
            raise ValueError("Chunk content cannot be empty.")

        if self.token_count <= 0:
            raise ValueError("token_count must be positive.")

        if self.start_char < 0:
            raise ValueError("start_char cannot be negative.")

        if self.end_char <= self.start_char:
            raise ValueError("end_char must exceed start_char.")

        if self.end_char - self.start_char != len(self.content):
            raise ValueError("Character offsets must match the content length.")

        if (self.page_start is None) != (self.page_end is None):
            raise ValueError(
                "page_start and page_end must both be set or both be None."
            )

        if self.page_start is not None and self.page_end is not None:
            if self.page_start <= 0 or self.page_end <= 0:
                raise ValueError("Page numbers must be positive.")

            if self.page_end < self.page_start:
                raise ValueError("page_end cannot be smaller than page_start.")
