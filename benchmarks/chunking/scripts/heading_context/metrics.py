
"""Metrics for evaluating Markdown heading context in generated chunks."""

from __future__ import annotations

from dataclasses import dataclass

from ai_knowledge_platform.services.chunking.types import DocumentChunk

from benchmarks.chunking.scripts.heading_context.analyzer import (
    MarkdownSection,
)


@dataclass(frozen=True, slots=True)
class SectionContextResult:
    """Heading-context evaluation for one chunk and one section."""

    chunk_index: int
    section_path: tuple[str, ...]
    section_heading_present: bool
    full_heading_path_present: bool


@dataclass(frozen=True, slots=True)
class HeadingContextMetrics:
    """Aggregate heading-context evaluation results."""

    total_section_occurrences: int
    section_heading_matches: int
    full_path_matches: int
    section_heading_coverage: float | None
    full_heading_path_coverage: float | None
    details: tuple[SectionContextResult, ...]


@dataclass(frozen=True, slots=True)
class ChunkSectionCount:
    """Number of distinct sections associated with one chunk."""

    chunk_index: int
    section_count: int


@dataclass(frozen=True, slots=True)
class SectionChunkCount:
    """Number of chunks associated with one section."""

    section_id: int
    section_path: tuple[str, ...]
    chunk_count: int


@dataclass(frozen=True, slots=True)
class ChunkSectionRelationshipMetrics:
    """Hierarchical and direct chunk-section relationship measurements."""

    # Hierarchical relationship metrics.
    total_relationships: int
    heading_matches: int
    chunks_with_multiple_sections: int
    multi_section_chunk_percentage: float | None
    average_sections_per_chunk: float | None
    heading_presence_percentage: float | None
    per_chunk: tuple[ChunkSectionCount, ...]
    per_section: tuple[SectionChunkCount, ...]

    # Direct relationship metrics.
    direct_total_relationships: int
    direct_heading_matches: int
    direct_full_path_matches: int
    direct_chunks_with_multiple_sections: int
    direct_multi_section_chunk_percentage: float | None
    direct_average_sections_per_chunk: float | None
    direct_heading_presence_percentage: float | None
    direct_full_path_presence_percentage: float | None
    direct_per_chunk: tuple[ChunkSectionCount, ...]
    direct_per_section: tuple[SectionChunkCount, ...]


def _heading_is_inside_chunk(
    section: MarkdownSection,
    chunk: DocumentChunk,
) -> bool:
    """Check whether a complete heading line is inside a chunk."""

    return (
        chunk.start_char <= section.heading_start
        and chunk.end_char >= section.heading_end
    )


def _get_sections_by_id(
    sections: list[MarkdownSection],
) -> dict[int, MarkdownSection]:
    """Create a section lookup and validate section identifiers."""

    sections_by_id = {
        section.section_id: section for section in sections
    }

    if len(sections_by_id) != len(sections):
        raise ValueError("Section IDs must be unique.")

    return sections_by_id


def _get_ancestor_sections(
    section: MarkdownSection,
    sections_by_id: dict[int, MarkdownSection],
) -> list[MarkdownSection]:
    """Find ancestors by following explicit parent section IDs."""

    ancestors: list[MarkdownSection] = []
    visited_ids = {section.section_id}
    parent_id = section.parent_id

    while parent_id is not None:
        if parent_id in visited_ids:
            raise ValueError("Circular section hierarchy detected.")

        parent = sections_by_id.get(parent_id)

        if parent is None:
            raise ValueError(
                f"Parent section ID {parent_id} does not exist."
            )

        visited_ids.add(parent_id)
        ancestors.append(parent)
        parent_id = parent.parent_id

    return ancestors


def _get_direct_content_intervals(
    section: MarkdownSection,
    sections: list[MarkdownSection],
) -> tuple[tuple[int, int], ...]:
    """Return section content intervals excluding descendant sections.

    Immediate child intervals are removed from the parent's content range.
    This preserves text owned directly by the parent before, between, or
    after its child sections.
    """

    children = [
        candidate
        for candidate in sections
        if candidate.parent_id == section.section_id
    ]

    children.sort(key=lambda child: child.heading_start)

    intervals: list[tuple[int, int]] = []
    cursor = section.content_start

    for child in children:
        child_start = max(
            section.content_start,
            child.heading_start,
        )
        child_end = min(
            section.content_end,
            child.content_end,
        )

        if child_start > cursor:
            intervals.append((cursor, child_start))

        cursor = max(cursor, child_end)

    if cursor < section.content_end:
        intervals.append((cursor, section.content_end))

    return tuple(
        (start, end)
        for start, end in intervals
        if start < end
    )


def _chunk_overlaps_content(
    chunk: DocumentChunk,
    intervals: tuple[tuple[int, int], ...],
) -> bool:
    """Check whether a chunk overlaps non-whitespace content."""

    for interval_start, interval_end in intervals:
        overlap_start = max(chunk.start_char, interval_start)
        overlap_end = min(chunk.end_char, interval_end)

        if overlap_start >= overlap_end:
            continue

        overlapping_text = chunk.content[
            overlap_start - chunk.start_char :
            overlap_end - chunk.start_char
        ]

        if overlapping_text.strip():
            return True

    return False


def _build_relationship_counts(
    chunks: list[DocumentChunk],
    sections: list[MarkdownSection],
    chunk_section_ids: dict[int, set[int]],
    section_chunk_ids: dict[int, set[int]],
) -> tuple[
    tuple[ChunkSectionCount, ...],
    tuple[SectionChunkCount, ...],
]:
    """Build per-chunk and per-section relationship counts."""

    per_chunk = tuple(
        ChunkSectionCount(
            chunk_index=chunk.chunk_index,
            section_count=len(chunk_section_ids[chunk.chunk_index]),
        )
        for chunk in chunks
    )

    per_section = tuple(
        SectionChunkCount(
            section_id=section.section_id,
            section_path=section.path,
            chunk_count=len(section_chunk_ids[section.section_id]),
        )
        for section in sections
    )

    return per_chunk, per_section


def _calculate_relationship_summary(
    per_chunk: tuple[ChunkSectionCount, ...],
    total_relationships: int,
    heading_matches: int,
) -> tuple[int, float | None, float | None, float | None]:
    """Calculate multi-section, average, and heading-presence metrics."""

    multiple_section_chunks = sum(
        item.section_count > 1 for item in per_chunk
    )

    total_chunks = len(per_chunk)

    multi_section_percentage = (
        round(multiple_section_chunks / total_chunks * 100, 2)
        if total_chunks
        else None
    )

    average_sections = (
        round(total_relationships / total_chunks, 2)
        if total_chunks
        else None
    )

    heading_presence = (
        round(heading_matches / total_relationships * 100, 2)
        if total_relationships
        else None
    )

    return (
        multiple_section_chunks,
        multi_section_percentage,
        average_sections,
        heading_presence,
    )


def evaluate_heading_context(
    sections: list[MarkdownSection],
    chunks: list[DocumentChunk],
) -> HeadingContextMetrics:
    """Evaluate heading co-location and complete heading-path coverage.

    Each occurrence represents one section whose non-whitespace content
    overlaps a generated chunk. Parent sections include descendant content.
    """

    sections_by_id = _get_sections_by_id(sections)
    details: list[SectionContextResult] = []

    for chunk in chunks:
        for section in sections:
            intervals = ((section.content_start, section.content_end),)

            if not _chunk_overlaps_content(chunk, intervals):
                continue

            section_heading_present = _heading_is_inside_chunk(
                section,
                chunk,
            )

            ancestors = _get_ancestor_sections(
                section,
                sections_by_id,
            )

            full_heading_path_present = section_heading_present and all(
                _heading_is_inside_chunk(ancestor, chunk)
                for ancestor in ancestors
            )

            details.append(
                SectionContextResult(
                    chunk_index=chunk.chunk_index,
                    section_path=section.path,
                    section_heading_present=section_heading_present,
                    full_heading_path_present=full_heading_path_present,
                )
            )

    total = len(details)

    section_matches = sum(
        result.section_heading_present for result in details
    )

    full_path_matches = sum(
        result.full_heading_path_present for result in details
    )

    section_coverage = (
        round(section_matches / total * 100, 2)
        if total
        else None
    )

    full_path_coverage = (
        round(full_path_matches / total * 100, 2)
        if total
        else None
    )

    return HeadingContextMetrics(
        total_section_occurrences=total,
        section_heading_matches=section_matches,
        full_path_matches=full_path_matches,
        section_heading_coverage=section_coverage,
        full_heading_path_coverage=full_path_coverage,
        details=tuple(details),
    )


def evaluate_chunk_section_relationships(
    sections: list[MarkdownSection],
    chunks: list[DocumentChunk],
) -> ChunkSectionRelationshipMetrics:
    """Evaluate hierarchical and direct chunk-section relationships.

    Hierarchical relationships include parent sections whose content ranges
    overlap descendant content.

    Direct relationships exclude descendant section intervals from parent
    content ranges.

    Each chunk-section pair is counted at most once in each relationship type.
    """

    sections_by_id = _get_sections_by_id(sections)

    hierarchical_chunk_ids: dict[int, set[int]] = {
        chunk.chunk_index: set() for chunk in chunks
    }

    hierarchical_section_ids: dict[int, set[int]] = {
        section.section_id: set() for section in sections
    }

    direct_chunk_ids: dict[int, set[int]] = {
        chunk.chunk_index: set() for chunk in chunks
    }

    direct_section_ids: dict[int, set[int]] = {
        section.section_id: set() for section in sections
    }

    direct_intervals = {
        section.section_id: _get_direct_content_intervals(
            section,
            sections,
        )
        for section in sections
    }

    total_relationships = 0
    heading_matches = 0

    direct_total_relationships = 0
    direct_heading_matches = 0
    direct_full_path_matches = 0

    for chunk in chunks:
        for section in sections:
            # Hierarchical relationship.
            hierarchical_intervals = (
                (section.content_start, section.content_end),
            )

            if _chunk_overlaps_content(
                chunk,
                hierarchical_intervals,
            ):
                hierarchical_chunk_ids[chunk.chunk_index].add(
                    section.section_id
                )
                hierarchical_section_ids[section.section_id].add(
                    chunk.chunk_index
                )

                total_relationships += 1

                if _heading_is_inside_chunk(section, chunk):
                    heading_matches += 1

            # Direct relationship.
            if not _chunk_overlaps_content(
                chunk,
                direct_intervals[section.section_id],
            ):
                continue

            direct_chunk_ids[chunk.chunk_index].add(
                section.section_id
            )
            direct_section_ids[section.section_id].add(
                chunk.chunk_index
            )

            direct_total_relationships += 1

            heading_present = _heading_is_inside_chunk(
                section,
                chunk,
            )

            if heading_present:
                direct_heading_matches += 1

            ancestors = _get_ancestor_sections(
                section,
                sections_by_id,
            )

            if heading_present and all(
                _heading_is_inside_chunk(ancestor, chunk)
                for ancestor in ancestors
            ):
                direct_full_path_matches += 1

    hierarchical_per_chunk, hierarchical_per_section = (
        _build_relationship_counts(
            chunks,
            sections,
            hierarchical_chunk_ids,
            hierarchical_section_ids,
        )
    )

    direct_per_chunk, direct_per_section = _build_relationship_counts(
        chunks,
        sections,
        direct_chunk_ids,
        direct_section_ids,
    )

    (
        chunks_with_multiple_sections,
        multi_section_percentage,
        average_sections,
        heading_presence,
    ) = _calculate_relationship_summary(
        hierarchical_per_chunk,
        total_relationships,
        heading_matches,
    )

    (
        direct_chunks_with_multiple_sections,
        direct_multi_section_percentage,
        direct_average_sections,
        direct_heading_presence,
    ) = _calculate_relationship_summary(
        direct_per_chunk,
        direct_total_relationships,
        direct_heading_matches,
    )

    direct_full_path_presence = (
        round(
            direct_full_path_matches / direct_total_relationships * 100,
            2,
        )
        if direct_total_relationships
        else None
    )

    return ChunkSectionRelationshipMetrics(
        total_relationships=total_relationships,
        heading_matches=heading_matches,
        chunks_with_multiple_sections=chunks_with_multiple_sections,
        multi_section_chunk_percentage=multi_section_percentage,
        average_sections_per_chunk=average_sections,
        heading_presence_percentage=heading_presence,
        per_chunk=hierarchical_per_chunk,
        per_section=hierarchical_per_section,
        direct_total_relationships=direct_total_relationships,
        direct_heading_matches=direct_heading_matches,
        direct_full_path_matches=direct_full_path_matches,
        direct_chunks_with_multiple_sections=(
            direct_chunks_with_multiple_sections
        ),
        direct_multi_section_chunk_percentage=(
            direct_multi_section_percentage
        ),
        direct_average_sections_per_chunk=direct_average_sections,
        direct_heading_presence_percentage=direct_heading_presence,
        direct_full_path_presence_percentage=direct_full_path_presence,
        direct_per_chunk=direct_per_chunk,
        direct_per_section=direct_per_section,
    )
