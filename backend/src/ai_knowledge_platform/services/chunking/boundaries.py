import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A character range in the original document."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start cannot be negative.")

        if self.end <= self.start:
            raise ValueError("end must be greater than start.")


class TextBoundaryDetector:
    """Detect paragraph and sentence boundaries without rewriting text."""

    _PARAGRAPH_SEPARATOR = re.compile(r"\n[ \t]*\n+")

    _SENTENCE_END = re.compile(r"""[.!?…।]["'”’)\]]*(?=\s+)""")

    _ABBREVIATIONS = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "fig.",
        "no.",
        "inc.",
        "ltd.",
    }

    def split_paragraphs(self, text: str) -> list[TextSpan]:
        spans: list[TextSpan] = []
        cursor = 0

        for separator in self._PARAGRAPH_SEPARATOR.finditer(text):
            self._append_trimmed_span(
                text=text,
                start=cursor,
                end=separator.start(),
                spans=spans,
            )
            cursor = separator.end()

        self._append_trimmed_span(
            text=text,
            start=cursor,
            end=len(text),
            spans=spans,
        )

        return spans

    def split_sentences(
        self,
        text: str,
        *,
        start_offset: int = 0,
    ) -> list[TextSpan]:
        spans: list[TextSpan] = []
        cursor = 0

        for boundary in self._SENTENCE_END.finditer(text):
            boundary_end = boundary.end()
            preceding_token = self._preceding_token(
                text,
                boundary.start(),
            )

            if self._is_abbreviation(preceding_token):
                continue

            self._append_trimmed_span(
                text=text,
                start=cursor,
                end=boundary_end,
                spans=spans,
                offset=start_offset,
            )

            cursor = boundary_end

        self._append_trimmed_span(
            text=text,
            start=cursor,
            end=len(text),
            spans=spans,
            offset=start_offset,
        )

        return spans

    @staticmethod
    def _append_trimmed_span(
        *,
        text: str,
        start: int,
        end: int,
        spans: list[TextSpan],
        offset: int = 0,
    ) -> None:
        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        if start < end:
            spans.append(
                TextSpan(
                    start=start + offset,
                    end=end + offset,
                )
            )

    @staticmethod
    def _preceding_token(text: str, position: int) -> str:
        prefix = text[: position + 1].rstrip()
        match = re.search(r"\S+$", prefix)

        return match.group(0).lower() if match else ""

    def _is_abbreviation(self, token: str) -> bool:
        return token in self._ABBREVIATIONS
