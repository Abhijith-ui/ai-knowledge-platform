from bisect import bisect_left, bisect_right

from ai_knowledge_platform.services.chunking.boundaries import (
    TextBoundaryDetector,
    TextSpan,
)
from ai_knowledge_platform.services.chunking.types import (
    ChunkingConfig,
    DocumentChunk,
)
from ai_knowledge_platform.services.document_extraction import ExtractedPage
from ai_knowledge_platform.services.tokenization.base import Tokenizer
from ai_knowledge_platform.services.tokenization.types import TokenOffset


class DocumentChunkingService:
    """Split extracted document text into overlapping, token-bounded chunks."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: ChunkingConfig | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._config = config or ChunkingConfig()
        self._boundary_detector = TextBoundaryDetector()

        available_tokens = (
            self._tokenizer.max_input_tokens - self._tokenizer.special_tokens_count
        )

        self._token_budget = min(
            self._config.target_tokens,
            available_tokens,
        )

        if self._token_budget <= 0:
            raise ValueError("No content token budget remains after special tokens.")

        if self._config.overlap_tokens >= self._token_budget:
            raise ValueError("Overlap must be smaller than the effective token budget.")

    @property
    def token_budget(self) -> int:
        """Return the effective content token budget."""
        return self._token_budget

    def _get_atomic_spans(self, source_text: str) -> list[TextSpan]:
        """Identify paragraph and sentence boundaries in the source text."""

        atomic_spans: list[TextSpan] = []

        paragraphs = self._boundary_detector.split_paragraphs(source_text)

        for paragraph in paragraphs:
            paragraph_text = source_text[paragraph.start : paragraph.end]

            if self._tokenizer.count_tokens(paragraph_text) <= self._token_budget:
                atomic_spans.append(paragraph)
                continue

            sentences = self._boundary_detector.split_sentences(
                paragraph_text,
                start_offset=paragraph.start,
            )

            for sentence in sentences:
                sentence_text = source_text[sentence.start : sentence.end]

                if self._tokenizer.count_tokens(sentence_text) <= self._token_budget:
                    atomic_spans.append(sentence)
                    continue

                atomic_spans.extend(self._split_span_by_tokens(source_text, sentence))

        return atomic_spans

    def _split_span_by_tokens(
        self,
        source_text: str,
        span: TextSpan,
    ) -> list[TextSpan]:
        """Split an oversized span into non-overlapping token-bounded spans."""

        source_segment = source_text[span.start : span.end]
        tokens = self._tokenizer.encode_with_offsets(source_segment)

        if not tokens:
            return []

        spans: list[TextSpan] = []
        token_start = 0

        while token_start < len(tokens):
            token_end = min(
                token_start + self._token_budget,
                len(tokens),
            )

            while token_end > token_start:
                local_start = tokens[token_start].start_char
                local_end = tokens[token_end - 1].end_char

                candidate = source_segment[local_start:local_end]

                if self._tokenizer.count_tokens(candidate) <= self._token_budget:
                    break

                token_end -= 1

            if token_end <= token_start:
                raise ValueError("Unable to split source text within the token budget.")

            spans.append(
                TextSpan(
                    start=span.start + local_start,
                    end=span.start + local_end,
                )
            )

            token_start = token_end

        return spans

    def _select_chunk_end(
        self,
        source_text: str,
        chunk_start: int,
        token_offsets: list[TokenOffset],
        token_start: int,
        token_end: int,
        atomic_spans: list[TextSpan],
        atomic_span_ends: list[int],
        token_start_chars: list[int],
    ) -> tuple[int, int]:
        """Prefer a semantic boundary without exceeding the token budget."""

        maximum_end = (
            len(source_text)
            if token_end == len(token_offsets)
            else token_offsets[token_end].start_char
        )

        minimum_preferred_tokens = min(
            self._config.minimum_tokens,
            self._token_budget,
        )

        preferred_boundary: int | None = None
        preferred_token_end: int | None = None

        first_span = bisect_right(atomic_span_ends, chunk_start)
        last_span = bisect_right(atomic_span_ends, maximum_end)

        for span_index in range(first_span, last_span):
            span = atomic_spans[span_index]

            candidate = source_text[chunk_start : span.end]
            candidate_tokens = self._tokenizer.count_tokens(candidate)

            if minimum_preferred_tokens <= candidate_tokens <= self._token_budget:
                candidate_token_end = bisect_left(
                    token_start_chars,
                    span.end,
                    lo=token_start + 1,
                    hi=min(token_end + 1, len(token_start_chars)),
                )

                if candidate_token_end <= token_start:
                    continue

                preferred_token_end = candidate_token_end

                preferred_boundary = (
                    len(source_text)
                    if candidate_token_end >= len(token_offsets)
                    else token_start_chars[candidate_token_end]
                )

        if (
            preferred_boundary is not None
            and preferred_token_end is not None
            and preferred_token_end > token_start
        ):
            return preferred_boundary, preferred_token_end

        return maximum_end, token_end

    def _pack_spans(
        self,
        source_text: str,
        spans: list[TextSpan],
    ) -> list[TextSpan]:
        """Create token-bounded chunks with configured token overlap."""

        if not source_text:
            return []

        token_offsets = self._tokenizer.encode_with_offsets(source_text)
        atomic_span_ends = [span.end for span in spans]

        token_start_chars = [offset.start_char for offset in token_offsets]

        if not token_offsets:
            return []

        packed: list[TextSpan] = []
        token_start = 0

        while token_start < len(token_offsets):
            chunk_start = (
                0 if token_start == 0 else token_offsets[token_start].start_char
            )

            token_end = min(
                token_start + self._token_budget,
                len(token_offsets),
            )

            chunk_end, selected_token_end = self._select_chunk_end(
                source_text=source_text,
                chunk_start=chunk_start,
                token_offsets=token_offsets,
                token_start=token_start,
                token_end=token_end,
                atomic_spans=spans,
                atomic_span_ends=atomic_span_ends,
                token_start_chars=token_start_chars,
            )

            while (
                chunk_end > chunk_start
                and self._tokenizer.count_tokens(source_text[chunk_start:chunk_end])
                > self._token_budget
            ):
                selected_token_end -= 1

                if selected_token_end <= token_start:
                    raise ValueError(
                        "Unable to create a chunk within the token budget."
                    )

                chunk_end = (
                    len(source_text)
                    if selected_token_end == len(token_offsets)
                    else token_offsets[selected_token_end].start_char
                )

            if chunk_end <= chunk_start:
                raise ValueError("Chunking failed to make forward progress.")

            packed.append(
                TextSpan(
                    start=chunk_start,
                    end=chunk_end,
                )
            )

            if selected_token_end >= len(token_offsets):
                if packed[-1].end < len(source_text):
                    packed[-1] = TextSpan(
                        start=packed[-1].start,
                        end=len(source_text),
                    )
                break

            next_token_start = max(
                token_start + 1,
                selected_token_end - self._config.overlap_tokens,
            )

            if next_token_start <= token_start:
                raise ValueError("Chunking failed to advance the token position.")

            token_start = next_token_start

        return packed

    @staticmethod
    def _map_pages(
        span: TextSpan,
        pages: tuple[ExtractedPage, ...],
    ) -> tuple[int | None, int | None]:
        """Find the first and last PDF pages overlapping a source span."""

        overlapping_pages = [
            page.page_number
            for page in pages
            if span.start < page.end_char and span.end > page.start_char
        ]

        if not overlapping_pages:
            return None, None

        return overlapping_pages[0], overlapping_pages[-1]

    def chunk_text(
        self,
        source_text: str,
        pages: tuple[ExtractedPage, ...] = (),
    ) -> list[DocumentChunk]:
        """Generate token-bounded chunks from extracted document text."""

        if not isinstance(source_text, str):
            raise TypeError("source_text must be a string.")

        if not source_text.strip():
            return []

        atomic_spans = self._get_atomic_spans(source_text)

        packed_spans = self._pack_spans(
            source_text=source_text,
            spans=atomic_spans,
        )

        chunks: list[DocumentChunk] = []

        for span in packed_spans:
            content = source_text[span.start : span.end]

            if not content.strip():
                continue

            token_count = self._tokenizer.count_tokens(content)

            if token_count > self._token_budget:
                raise ValueError("Generated chunk exceeds the effective token budget.")

            page_start, page_end = self._map_pages(span, pages)

            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    content=content,
                    token_count=token_count,
                    start_char=span.start,
                    end_char=span.end,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

        return chunks
