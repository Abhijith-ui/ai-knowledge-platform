import pytest
from ai_knowledge_platform.services.chunking.boundaries import (
    TextBoundaryDetector,
    TextSpan,
)


@pytest.fixture
def detector() -> TextBoundaryDetector:
    return TextBoundaryDetector()


def test_text_span_validation() -> None:
    with pytest.raises(ValueError):
        TextSpan(start=-1, end=5)

    with pytest.raises(ValueError):
        TextSpan(start=5, end=5)


def test_split_paragraphs(detector: TextBoundaryDetector) -> None:
    text = "First paragraph.\n\nSecond paragraph."

    spans = detector.split_paragraphs(text)

    assert [text[span.start : span.end] for span in spans] == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_split_paragraphs_preserves_source_offsets(
    detector: TextBoundaryDetector,
) -> None:
    text = "  First paragraph.  \n\n  Second paragraph.  "

    spans = detector.split_paragraphs(text)

    for span in spans:
        assert text[span.start : span.end].strip() == text[span.start : span.end]


def test_split_empty_text(detector: TextBoundaryDetector) -> None:
    assert detector.split_paragraphs("") == []
    assert detector.split_sentences("") == []


def test_split_sentences(detector: TextBoundaryDetector) -> None:
    text = "First sentence. Second sentence! Third sentence?"

    spans = detector.split_sentences(text)

    assert [text[span.start : span.end] for span in spans] == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]


def test_sentence_offsets_with_start_offset(
    detector: TextBoundaryDetector,
) -> None:
    source = "Introduction. First sentence. Second sentence."
    paragraph_start = source.index("First sentence.")
    paragraph = source[paragraph_start:]

    spans = detector.split_sentences(
        paragraph,
        start_offset=paragraph_start,
    )

    assert source[spans[0].start : spans[0].end] == "First sentence."
    assert source[spans[1].start : spans[1].end] == "Second sentence."


def test_common_abbreviations_do_not_create_boundaries(
    detector: TextBoundaryDetector,
) -> None:
    text = "Dr. Smith works here. Mr. John works there."

    spans = detector.split_sentences(text)

    assert [text[span.start : span.end] for span in spans] == [
        "Dr. Smith works here.",
        "Mr. John works there.",
    ]


def test_sentence_spans_preserve_unicode(
    detector: TextBoundaryDetector,
) -> None:
    text = "Machine learning is useful. नमस्ते दुनिया।"

    spans = detector.split_sentences(text)

    assert text[spans[0].start : spans[0].end] == ("Machine learning is useful.")
    assert text[spans[1].start : spans[1].end] == "नमस्ते दुनिया।"


def test_abbreviations_with_multiple_periods(
    detector: TextBoundaryDetector,
) -> None:
    text = "We use e.g. neural networks. They learn representations."

    spans = detector.split_sentences(text)

    assert [text[span.start : span.end] for span in spans] == [
        "We use e.g. neural networks.",
        "They learn representations.",
    ]


def test_decimal_numbers_do_not_create_sentence_boundaries(
    detector: TextBoundaryDetector,
) -> None:
    text = "The model achieved an accuracy of 3.14. It performed well."

    spans = detector.split_sentences(text)

    assert [text[span.start : span.end] for span in spans] == [
        "The model achieved an accuracy of 3.14.",
        "It performed well.",
    ]


def test_sentence_boundary_after_etc(
    detector: TextBoundaryDetector,
) -> None:
    text = "We studied several techniques, etc. This was useful."

    spans = detector.split_sentences(text)

    assert [text[span.start : span.end] for span in spans] == [
        "We studied several techniques, etc.",
        "This was useful.",
    ]
