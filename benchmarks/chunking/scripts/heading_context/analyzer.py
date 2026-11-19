"""Independent Markdown heading and section reference analyzer."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*?)\s*$")
_FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """A Markdown section with source positions and hierarchical context."""

    section_id: int
    parent_id: int | None
    level: int
    title: str
    heading_start: int
    heading_end: int
    content_start: int
    content_end: int
    path: tuple[str, ...]


def analyze_markdown_headings(source_text: str) -> list[MarkdownSection]:
    """Extract ATX headings and their section content ranges.

    Headings inside fenced code blocks are ignored.
    A section ends at the next heading of the same or higher level.
    """

    if not isinstance(source_text, str):
        raise TypeError("source_text must be a string.")

    headings: list[tuple[int, int, int, str, int]] = []

    position = 0
    fence_char: str | None = None
    fence_length = 0

    for line in source_text.splitlines(keepends=True):
        line_content = line.rstrip("\r\n")

        if fence_char is not None:
            closing_fence = re.match(
                rf"^ {{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*$",
                line_content,
            )
            if closing_fence:
                fence_char = None
                fence_length = 0

            position += len(line)
            continue

        fence_match = _FENCE_START.match(line_content)
        if fence_match:
            fence = fence_match.group(1)
            fence_char = fence[0]
            fence_length = len(fence)
            position += len(line)
            continue

        heading_match = _ATX_HEADING.match(line_content)
        if heading_match:
            hashes, raw_title = heading_match.groups()

            title = re.sub(r"[ \t]+#+[ \t]*$", "", raw_title).strip()

            if title:
                heading_start = position
                heading_end = position + len(line_content)
                content_start = position + len(line)

                headings.append(
                    (
                        len(hashes),
                        heading_start,
                        heading_end,
                        title,
                        content_start,
                    )
                )

        position += len(line)

    sections: list[MarkdownSection] = []
    hierarchy: list[tuple[int, int, str]] = []

    for index, heading in enumerate(headings):
        level, heading_start, heading_end, title, content_start = heading

        while hierarchy and hierarchy[-1][0] >= level:
            hierarchy.pop()

        parent_id = hierarchy[-1][1] if hierarchy else None
        section_id = index

        hierarchy.append((level, section_id, title))
        path = tuple(item[2] for item in hierarchy)

        content_end = len(source_text)

        for next_heading in headings[index + 1 :]:
            if next_heading[0] <= level:
                content_end = next_heading[1]
                break

        sections.append(
            MarkdownSection(
                section_id=section_id,
                parent_id=parent_id,
                level=level,
                title=title,
                heading_start=heading_start,
                heading_end=heading_end,
                content_start=content_start,
                content_end=content_end,
                path=path,
            )
        )

    return sections
