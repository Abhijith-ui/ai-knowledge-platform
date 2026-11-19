"""Unit tests for Markdown heading-context evaluation."""

from ai_knowledge_platform.services.chunking.types import DocumentChunk

from benchmarks.chunking.scripts.heading_context.analyzer import (
    analyze_markdown_headings,
)
from benchmarks.chunking.scripts.heading_context.metrics import (
    evaluate_chunk_section_relationships,
    evaluate_heading_context,
)


def make_chunk(
    source: str,
    chunk_index: int,
    start: int,
    end: int,
) -> DocumentChunk:
    """Create a chunk covering a source-text interval."""

    return DocumentChunk(
        chunk_index=chunk_index,
        content=source[start:end],
        start_char=start,
        end_char=end,
        token_count=1,
        page_start=None,
        page_end=None,
    )


def test_full_heading_path_is_present_when_all_headings_are_in_chunk() -> None:
    source = "# Root\n" "Root content.\n" "## Child\n" "Child content.\n"
    sections = analyze_markdown_headings(source)
    chunks = [make_chunk(source, 0, 0, len(source))]

    result = evaluate_heading_context(sections, chunks)

    assert result.total_section_occurrences == 2
    assert result.section_heading_coverage == 100.0
    assert result.full_heading_path_coverage == 100.0


def test_missing_parent_heading_reduces_full_path_coverage() -> None:
    source = "# Root\n" "Root content.\n" "## Child\n" "Child content.\n"
    sections = analyze_markdown_headings(source)

    child = sections[1]
    chunks = [
        make_chunk(
            source,
            0,
            child.heading_start,
            len(source),
        )
    ]

    result = evaluate_heading_context(sections, chunks)

    assert result.total_section_occurrences == 2
    assert result.section_heading_matches == 1
    assert result.full_path_matches == 0
    assert result.section_heading_coverage == 50.0
    assert result.full_heading_path_coverage == 0.0


def test_section_split_across_chunks_is_evaluated_per_chunk() -> None:
    source = (
        "# Root\n"
        "## Child\n"
        "First part of the section.\n"
        "Second part of the section.\n"
    )
    sections = analyze_markdown_headings(source)

    split_position = source.index("Second part")
    chunks = [
        make_chunk(source, 0, 0, split_position),
        make_chunk(source, 1, split_position, len(source)),
    ]

    result = evaluate_heading_context(sections, chunks)

    assert result.total_section_occurrences == 4
    assert result.section_heading_coverage == 50.0
    assert result.full_heading_path_coverage == 50.0


def test_document_without_headings_has_no_coverage() -> None:
    source = "Ordinary text without Markdown headings."
    sections = analyze_markdown_headings(source)
    chunks = [make_chunk(source, 0, 0, len(source))]

    result = evaluate_heading_context(sections, chunks)

    assert result.total_section_occurrences == 0
    assert result.section_heading_coverage is None
    assert result.full_heading_path_coverage is None


def test_duplicate_headings_use_the_correct_ancestor() -> None:
    source = (
        "# Root\n"
        "## Overview\n"
        "First overview content.\n"
        "### Details\n"
        "First details content.\n"
        "## Overview\n"
        "Second overview content.\n"
        "### Details\n"
        "Second details content.\n"
    )

    sections = analyze_markdown_headings(source)

    second_details = sections[-1]

    # Include the second Details heading and its content,
    # but exclude both ancestor headings.
    chunks = [
        make_chunk(
            source,
            0,
            second_details.heading_start,
            len(source),
        )
    ]

    result = evaluate_heading_context(sections, chunks)

    details_result = next(
        item
        for item in result.details
        if item.section_path
        == (
            "Root",
            "Overview",
            "Details",
        )
        and item.chunk_index == 0
        and item.section_heading_present
    )

    assert details_result.section_heading_present is True
    assert details_result.full_heading_path_present is False


def test_chunk_spanning_multiple_sections_creates_multiple_relationships() -> None:
    source = (
        "# Root\n"
        "Root content.\n"
        "## First\n"
        "First section content.\n"
        "## Second\n"
        "Second section content.\n"
    )
    sections = analyze_markdown_headings(source)
    chunks = [make_chunk(source, 0, 0, len(source))]

    result = evaluate_chunk_section_relationships(sections, chunks)

    # Root overlaps both child sections.
    assert result.total_relationships == 3
    assert result.per_chunk[0].section_count == 3
    assert result.chunks_with_multiple_sections == 1
    assert result.multi_section_chunk_percentage == 100.0
    assert result.average_sections_per_chunk == 3.0
    assert result.heading_presence_percentage == 100.0


def test_section_spanning_multiple_chunks_creates_multiple_relationships() -> None:
    source = (
        "# Root\n"
        "First part of the section.\n"
        "Second part of the section.\n"
    )
    sections = analyze_markdown_headings(source)

    split_position = source.index("Second part")
    chunks = [
        make_chunk(source, 0, 0, split_position),
        make_chunk(source, 1, split_position, len(source)),
    ]

    result = evaluate_chunk_section_relationships(sections, chunks)

    assert result.total_relationships == 2
    assert result.per_section[0].chunk_count == 2
    assert result.per_chunk[0].section_count == 1
    assert result.per_chunk[1].section_count == 1
    assert result.average_sections_per_chunk == 1.0
    assert result.heading_presence_percentage == 50.0


def test_duplicate_heading_titles_keep_distinct_section_ids() -> None:
    source = (
        "# Root\n"
        "## Overview\n"
        "First overview content.\n"
        "## Overview\n"
        "Second overview content.\n"
    )
    sections = analyze_markdown_headings(source)
    chunks = [make_chunk(source, 0, 0, len(source))]

    result = evaluate_chunk_section_relationships(sections, chunks)

    overview_sections = [
        item for item in result.per_section
        if item.section_path == ("Root", "Overview")
    ]

    assert len(overview_sections) == 2
    assert overview_sections[0].section_id != overview_sections[1].section_id
    assert overview_sections[0].chunk_count == 1
    assert overview_sections[1].chunk_count == 1


def test_sections_without_content_overlap_have_zero_associations() -> None:
    source = (
        "Text before the heading.\n"
        "# Heading\n"
        "Section content.\n"
    )
    sections = analyze_markdown_headings(source)

    heading_position = source.index("# Heading")
    chunks = [
        make_chunk(source, 0, 0, heading_position),
    ]

    result = evaluate_chunk_section_relationships(sections, chunks)

    assert result.total_relationships == 0
    assert result.per_chunk[0].section_count == 0
    assert result.per_section[0].chunk_count == 0
    assert result.average_sections_per_chunk == 0.0
    assert result.heading_presence_percentage is None


def test_empty_chunks_return_empty_relationship_metrics() -> None:
    source = "# Root\nRoot content.\n"
    sections = analyze_markdown_headings(source)

    result = evaluate_chunk_section_relationships(sections, [])

    assert result.total_relationships == 0
    assert result.chunks_with_multiple_sections == 0
    assert result.multi_section_chunk_percentage is None
    assert result.average_sections_per_chunk is None
    assert result.heading_presence_percentage is None
    assert result.per_chunk == ()
    assert len(result.per_section) == 1
    assert result.per_section[0].chunk_count == 0