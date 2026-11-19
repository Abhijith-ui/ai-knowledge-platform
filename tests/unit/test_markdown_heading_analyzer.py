"""Unit tests for the Markdown heading reference analyzer."""

import pytest

from benchmarks.chunking.scripts.heading_context.analyzer import (
    analyze_markdown_headings,
)


def test_extracts_nested_heading_hierarchy() -> None:
    source = (
        "# Root\n"
        "Root content.\n"
        "## Child\n"
        "Child content.\n"
        "### Grandchild\n"
        "Grandchild content.\n"
    )

    sections = analyze_markdown_headings(source)

    assert len(sections) == 3
    assert sections[0].path == ("Root",)
    assert sections[1].path == ("Root", "Child")
    assert sections[2].path == ("Root", "Child", "Grandchild")


def test_preserves_duplicate_heading_titles_as_distinct_sections() -> None:
    source = (
        "# Root\n"
        "## Overview\n"
        "First overview content.\n"
        "## Overview\n"
        "Second overview content.\n"
    )

    sections = analyze_markdown_headings(source)

    assert len(sections) == 3
    assert sections[1].title == sections[2].title
    assert sections[1].heading_start != sections[2].heading_start
    assert sections[1].content_start < sections[2].content_start


def test_ignores_headings_inside_fenced_code_blocks() -> None:
    source = (
        "# Actual heading\n"
        "Content.\n"
        "```markdown\n"
        "# Not a heading\n"
        "## Neither is this\n"
        "```\n"
        "## Another actual heading\n"
        "More content.\n"
    )

    sections = analyze_markdown_headings(source)

    assert [section.title for section in sections] == [
        "Actual heading",
        "Another actual heading",
    ]


def test_supports_tilde_fenced_code_blocks() -> None:
    source = (
        "# Actual heading\n"
        "~~~markdown\n"
        "## Not a heading\n"
        "~~~\n"
        "## Another actual heading\n"
    )

    sections = analyze_markdown_headings(source)

    assert [section.title for section in sections] == [
        "Actual heading",
        "Another actual heading",
    ]


def test_content_before_first_heading_is_not_a_section() -> None:
    source = "Preamble content.\n\n" "# First heading\n" "Section content.\n"

    sections = analyze_markdown_headings(source)

    assert len(sections) == 1
    assert source[sections[0].content_start : sections[0].content_end] == (
        "Section content.\n"
    )


def test_returns_empty_list_when_document_has_no_headings() -> None:
    source = "Just ordinary text.\n\nAnother paragraph.\n"

    sections = analyze_markdown_headings(source)

    assert sections == []


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="source_text must be a string"):
        analyze_markdown_headings(None)  # type: ignore[arg-type]


def test_duplicate_headings_have_distinct_parent_relationships() -> None:
    source = "# Root\n" "## Overview\n" "### Details\n" "## Overview\n" "### Details\n"

    sections = analyze_markdown_headings(source)

    assert len(sections) == 5

    assert sections[0].section_id == 0
    assert sections[0].parent_id is None

    assert sections[1].section_id == 1
    assert sections[1].parent_id == 0

    assert sections[2].section_id == 2
    assert sections[2].parent_id == 1

    assert sections[3].section_id == 3
    assert sections[3].parent_id == 0

    assert sections[4].section_id == 4
    assert sections[4].parent_id == 3
