"""Run heading-context evaluation against the production chunking service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_knowledge_platform.services.chunking.service import DocumentChunkingService
from ai_knowledge_platform.services.chunking.types import ChunkingConfig
from ai_knowledge_platform.services.tokenization.huggingface import (
    HuggingFaceTokenizer,
)

from benchmarks.chunking.scripts.heading_context.analyzer import (
    analyze_markdown_headings,
)
from benchmarks.chunking.scripts.heading_context.metrics import (
    evaluate_chunk_section_relationships,
    evaluate_heading_context,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPOSITORY_ROOT / "benchmarks" / "chunking" / "datasets"
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks" / "chunking" / "results"

TOKENIZER_MODEL = "intfloat/multilingual-e5-base"
MAX_INPUT_TOKENS = 512

CHUNKING_CONFIG = ChunkingConfig(
    target_tokens=448,
    overlap_tokens=64,
    minimum_tokens=50,
)


def serialize_chunk_counts(items: Any) -> list[dict[str, int]]:
    """Serialize per-chunk section relationship counts."""
    return [
        {
            "chunk_index": item.chunk_index,
            "section_count": item.section_count,
        }
        for item in items
    ]


def serialize_section_counts(items: Any) -> list[dict[str, Any]]:
    """Serialize per-section chunk relationship counts."""
    return [
        {
            "section_id": item.section_id,
            "section_path": list(item.section_path),
            "chunk_count": item.chunk_count,
        }
        for item in items
    ]


def evaluate_document(
    file_path: Path,
    chunker: DocumentChunkingService,
) -> dict[str, Any] | None:
    """Evaluate heading context and hierarchical/direct relationships."""

    source_text = file_path.read_text(encoding="utf-8")
    sections = analyze_markdown_headings(source_text)

    if not sections:
        return None

    chunks = chunker.chunk_text(source_text)

    metrics = evaluate_heading_context(
        sections=sections,
        chunks=chunks,
    )

    relationships = evaluate_chunk_section_relationships(
        sections=sections,
        chunks=chunks,
    )

    return {
        "document": file_path.name,
        "source_characters": len(source_text),
        "heading_count": len(sections),
        "chunk_count": len(chunks),
        "total_section_occurrences": metrics.total_section_occurrences,
        "section_heading_matches": metrics.section_heading_matches,
        "full_path_matches": metrics.full_path_matches,
        "section_heading_coverage": metrics.section_heading_coverage,
        "full_heading_path_coverage": metrics.full_heading_path_coverage,
        "relationships": {
            "total_relationships": relationships.total_relationships,
            "heading_matches": relationships.heading_matches,
            "chunks_with_multiple_sections": (
                relationships.chunks_with_multiple_sections
            ),
            "multi_section_chunk_percentage": (
                relationships.multi_section_chunk_percentage
            ),
            "average_sections_per_chunk": (relationships.average_sections_per_chunk),
            "heading_presence_percentage": (relationships.heading_presence_percentage),
            "per_chunk": serialize_chunk_counts(relationships.per_chunk),
            "per_section": serialize_section_counts(relationships.per_section),
        },
        "direct_relationships": {
            "total_relationships": relationships.direct_total_relationships,
            "heading_matches": relationships.direct_heading_matches,
            "full_path_matches": relationships.direct_full_path_matches,
            "chunks_with_multiple_sections": (
                relationships.direct_chunks_with_multiple_sections
            ),
            "multi_section_chunk_percentage": (
                relationships.direct_multi_section_chunk_percentage
            ),
            "average_sections_per_chunk": (
                relationships.direct_average_sections_per_chunk
            ),
            "heading_presence_percentage": (
                relationships.direct_heading_presence_percentage
            ),
            "full_heading_path_presence_percentage": (
                relationships.direct_full_path_presence_percentage
            ),
            "per_chunk": serialize_chunk_counts(relationships.direct_per_chunk),
            "per_section": serialize_section_counts(relationships.direct_per_section),
        },
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in chunks
        ],
        "details": [
            {
                "chunk_index": detail.chunk_index,
                "section_path": list(detail.section_path),
                "section_heading_present": detail.section_heading_present,
                "full_heading_path_present": (detail.full_heading_path_present),
            }
            for detail in metrics.details
        ],
    }


def calculate_percentage(
    numerator: int,
    denominator: int,
) -> float | None:
    """Calculate a percentage safely."""
    if denominator == 0:
        return None

    return round(numerator / denominator * 100, 2)


def calculate_average(
    numerator: int,
    denominator: int,
) -> float | None:
    """Calculate an average safely."""
    if denominator == 0:
        return None

    return round(numerator / denominator, 2)


def main() -> None:
    """Run the evaluation and save a JSON report."""

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    dataset_files = sorted(DATASET_DIR.glob("*.txt"))

    if not dataset_files:
        raise FileNotFoundError(f"No .txt benchmark datasets found in {DATASET_DIR}")

    print("Initializing tokenizer...")

    tokenizer = HuggingFaceTokenizer(
        model_name=TOKENIZER_MODEL,
        max_input_tokens=MAX_INPUT_TOKENS,
    )

    chunker = DocumentChunkingService(
        tokenizer=tokenizer,
        config=CHUNKING_CONFIG,
    )

    document_results: list[dict[str, Any]] = []
    skipped_documents: list[str] = []

    for file_path in dataset_files:
        result = evaluate_document(file_path, chunker)

        if result is None:
            skipped_documents.append(file_path.name)
            continue

        document_results.append(result)

        hierarchical = result["relationships"]
        direct = result["direct_relationships"]

        print(
            f"{file_path.name}: "
            f"{result['heading_count']} headings, "
            f"{result['chunk_count']} chunks, "
            f"section coverage={result['section_heading_coverage']}%, "
            f"full-path coverage={result['full_heading_path_coverage']}%, "
            f"hierarchical relationships="
            f"{hierarchical['total_relationships']}, "
            f"direct relationships="
            f"{direct['total_relationships']}"
        )

    # ---------------------------------------------------------
    # Heading-context aggregates
    # ---------------------------------------------------------

    total_occurrences = sum(
        result["total_section_occurrences"] for result in document_results
    )

    total_section_matches = sum(
        result["section_heading_matches"] for result in document_results
    )

    total_full_path_matches = sum(
        result["full_path_matches"] for result in document_results
    )

    # ---------------------------------------------------------
    # Hierarchical relationship aggregates
    # ---------------------------------------------------------

    total_relationships = sum(
        result["relationships"]["total_relationships"] for result in document_results
    )

    total_heading_matches = sum(
        result["relationships"]["heading_matches"] for result in document_results
    )

    total_multi_section_chunks = sum(
        result["relationships"]["chunks_with_multiple_sections"]
        for result in document_results
    )

    # ---------------------------------------------------------
    # Direct relationship aggregates
    # ---------------------------------------------------------

    total_direct_relationships = sum(
        result["direct_relationships"]["total_relationships"]
        for result in document_results
    )

    total_direct_heading_matches = sum(
        result["direct_relationships"]["heading_matches"] for result in document_results
    )

    total_direct_full_path_matches = sum(
        result["direct_relationships"]["full_path_matches"]
        for result in document_results
    )

    total_direct_multi_section_chunks = sum(
        result["direct_relationships"]["chunks_with_multiple_sections"]
        for result in document_results
    )

    total_chunks = sum(result["chunk_count"] for result in document_results)

    total_headings = sum(result["heading_count"] for result in document_results)

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    report: dict[str, Any] = {
        "configuration": {
            "tokenizer_model": TOKENIZER_MODEL,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "target_tokens": CHUNKING_CONFIG.target_tokens,
            "overlap_tokens": CHUNKING_CONFIG.overlap_tokens,
            "minimum_tokens": CHUNKING_CONFIG.minimum_tokens,
            "effective_token_budget": chunker.token_budget,
        },
        "summary": {
            "dataset_files_found": len(dataset_files),
            "documents_evaluated": len(document_results),
            "documents_skipped_no_headings": len(skipped_documents),
            "total_chunks": total_chunks,
            "total_headings": total_headings,
            "total_section_occurrences": total_occurrences,
            "section_heading_coverage": calculate_percentage(
                total_section_matches,
                total_occurrences,
            ),
            "full_heading_path_coverage": calculate_percentage(
                total_full_path_matches,
                total_occurrences,
            ),
            "hierarchical_relationships": {
                "total_relationships": total_relationships,
                "heading_matches": total_heading_matches,
                "heading_presence_percentage": calculate_percentage(
                    total_heading_matches,
                    total_relationships,
                ),
                "chunks_spanning_multiple_sections": (total_multi_section_chunks),
                "multi_section_chunk_percentage": calculate_percentage(
                    total_multi_section_chunks,
                    total_chunks,
                ),
                "average_sections_per_chunk": calculate_average(
                    total_relationships,
                    total_chunks,
                ),
            },
            "direct_relationships": {
                "total_relationships": total_direct_relationships,
                "heading_matches": total_direct_heading_matches,
                "full_path_matches": total_direct_full_path_matches,
                "heading_presence_percentage": calculate_percentage(
                    total_direct_heading_matches,
                    total_direct_relationships,
                ),
                "full_heading_path_presence_percentage": calculate_percentage(
                    total_direct_full_path_matches,
                    total_direct_relationships,
                ),
                "chunks_spanning_multiple_sections": (
                    total_direct_multi_section_chunks
                ),
                "multi_section_chunk_percentage": calculate_percentage(
                    total_direct_multi_section_chunks,
                    total_chunks,
                ),
                "average_sections_per_chunk": calculate_average(
                    total_direct_relationships,
                    total_chunks,
                ),
            },
        },
        "skipped_documents": skipped_documents,
        "documents": document_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "heading_context_evaluation.json"

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = report["summary"]
    hierarchical = summary["hierarchical_relationships"]
    direct = summary["direct_relationships"]

    print("\nEvaluation complete.")
    print(f"Documents evaluated: {len(document_results)}")
    print(f"Documents skipped: {len(skipped_documents)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total headings: {total_headings}")

    print("\nHeading context:")
    print(f"Section heading coverage: {summary['section_heading_coverage']}%")
    print(f"Full heading path coverage: " f"{summary['full_heading_path_coverage']}%")

    print("\nHierarchical relationships:")
    print(f"Total relationships: {hierarchical['total_relationships']}")
    print(f"Heading presence: " f"{hierarchical['heading_presence_percentage']}%")
    print(
        f"Multi-section chunks: " f"{hierarchical['multi_section_chunk_percentage']}%"
    )
    print(
        f"Average sections per chunk: " f"{hierarchical['average_sections_per_chunk']}"
    )

    print("\nDirect relationships:")
    print(f"Total relationships: {direct['total_relationships']}")
    print(f"Heading presence: {direct['heading_presence_percentage']}%")
    print(
        f"Full heading path presence: "
        f"{direct['full_heading_path_presence_percentage']}%"
    )
    print(f"Multi-section chunks: " f"{direct['multi_section_chunk_percentage']}%")
    print(f"Average sections per chunk: " f"{direct['average_sections_per_chunk']}")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
