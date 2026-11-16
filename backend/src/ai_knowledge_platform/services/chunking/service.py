from ai_knowledge_platform.services.chunking.boundaries import (
    TextBoundaryDetector,
    TextSpan,
)
from ai_knowledge_platform.services.chunking.types import (
    ChunkingConfig,
    DocumentChunk,
)
from ai_knowledge_platform.services.document_extraction import (
    ExtractedPage,
)
from ai_knowledge_platform.services.tokenization.base import Tokenizer


class DocumentChunkingService:
    """Split extracted document text into token-bounded source spans."""

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

    def _split_span_by_tokens(
        self,
        source_text: str,
        span: TextSpan,
    ) -> list[TextSpan]:
        """Split an oversized source span using tokenizer offsets."""

        source_segment = source_text[span.start : span.end]
        tokens = self._tokenizer.encode_with_offsets(source_segment)

        if not tokens:
            return []

        chunks: list[TextSpan] = []
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
                raise ValueError("Unable to create a chunk within the token budget.")

            chunks.append(
                TextSpan(
                    start=span.start + local_start,
                    end=span.start + local_end,
                )
            )

            if token_end == len(tokens):
                break

            next_start = token_end - self._config.overlap_tokens
            token_start = max(token_start + 1, next_start)

        return chunks

    def _get_atomic_spans(
        self,
        source_text: str,
    ) -> list[TextSpan]:
        """Split text into units that fit the effective token budget."""

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

    def _pack_spans(
        self,
        source_text: str,
        spans: list[TextSpan],
    ) -> list[TextSpan]:
        """Pack consecutive non-overlapping spans within the token budget."""

        if not spans:
            return []

        packed: list[TextSpan] = []

        chunk_start = spans[0].start
        chunk_end = spans[0].end

        for span in spans[1:]:
            if span.start < chunk_end:
                packed.append(TextSpan(start=chunk_start, end=chunk_end))
                chunk_start = span.start
                chunk_end = span.end
                continue

            candidate = source_text[chunk_start : span.end]

            if self._tokenizer.count_tokens(candidate) <= self._token_budget:
                chunk_end = span.end
                continue

            packed.append(TextSpan(start=chunk_start, end=chunk_end))

            chunk_start = chunk_end
            chunk_end = span.end

        packed.append(TextSpan(start=chunk_start, end=chunk_end))

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
            if (span.start < page.end_char and span.end > page.start_char)
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
        packed_spans = self._pack_spans(source_text, atomic_spans)

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
