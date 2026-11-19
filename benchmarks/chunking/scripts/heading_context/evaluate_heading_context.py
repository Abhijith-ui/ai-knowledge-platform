
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


def evaluate_document(
    file_path: Path,
    chunker: DocumentChunkingService,
) -> dict[str, Any] | None:
    """Evaluate heading context and chunk-section relationships."""

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

    heading_matches = (
        round(
            relationships.total_relationships
            * relationships.heading_presence_percentage
            / 100,
            2,
        )
        if relationships.heading_presence_percentage is not None
        else 0
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
            "heading_matches": heading_matches,
            "chunks_with_multiple_sections": (
                relationships.chunks_with_multiple_sections
            ),
            "multi_section_chunk_percentage": (
                relationships.multi_section_chunk_percentage
            ),
            "average_sections_per_chunk": (
                relationships.average_sections_per_chunk
            ),
            "heading_presence_percentage": (
                relationships.heading_presence_percentage
            ),
            "per_chunk": [
                {
                    "chunk_index": item.chunk_index,
                    "section_count": item.section_count,
                }
                for item in relationships.per_chunk
            ],
            "per_section": [
                {
                    "section_id": item.section_id,
                    "section_path": list(item.section_path),
                    "chunk_count": item.chunk_count,
                }
                for item in relationships.per_section
            ],
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
                "full_heading_path_present": detail.full_heading_path_present,
            }
            for detail in metrics.details
        ],
    }


def main() -> None:
    """Run the evaluation and save a JSON report."""

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    dataset_files = sorted(DATASET_DIR.glob("*.txt"))

    if not dataset_files:
        raise FileNotFoundError(
            f"No .txt benchmark datasets found in {DATASET_DIR}"
        )

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

        relationship_metrics = result["relationships"]

        print(
            f"{file_path.name}: "
            f"{result['heading_count']} headings, "
            f"{result['chunk_count']} chunks, "
            f"section coverage={result['section_heading_coverage']}%, "
            f"full-path coverage={result['full_heading_path_coverage']}%, "
            f"avg sections/chunk="
            f"{relationship_metrics['average_sections_per_chunk']}, "
            f"multi-section chunks="
            f"{relationship_metrics['multi_section_chunk_percentage']}%"
        )

    # Existing heading-context aggregates.
    total_occurrences = sum(
        result["total_section_occurrences"]
        for result in document_results
    )

    total_section_matches = sum(
        result["section_heading_matches"]
        for result in document_results
    )

    total_full_path_matches = sum(
        result["full_path_matches"]
        for result in document_results
    )

    # Chunk-section relationship aggregates.
    total_relationships = sum(
        result["relationships"]["total_relationships"]
        for result in document_results
    )

    total_heading_matches = sum(
        result["relationships"]["heading_matches"]
        for result in document_results
    )

    total_multi_section_chunks = sum(
        result["relationships"]["chunks_with_multiple_sections"]
        for result in document_results
    )

    total_chunks = sum(
        result["chunk_count"]
        for result in document_results
    )

    report = {
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
            "total_headings": sum(
                result["heading_count"]
                for result in document_results
            ),
            "total_section_occurrences": total_occurrences,
            "section_heading_coverage": (
                round(total_section_matches / total_occurrences * 100, 2)
                if total_occurrences
                else None
            ),
            "full_heading_path_coverage": (
                round(total_full_path_matches / total_occurrences * 100, 2)
                if total_occurrences
                else None
            ),
            "total_chunk_section_relationships": total_relationships,
            "chunks_spanning_multiple_sections": total_multi_section_chunks,
            "multi_section_chunk_percentage": (
                round(total_multi_section_chunks / total_chunks * 100, 2)
                if total_chunks
                else None
            ),
            "average_sections_per_chunk": (
                round(total_relationships / total_chunks, 2)
                if total_chunks
                else None
            ),
            "heading_presence_percentage": (
                round(total_heading_matches / total_relationships * 100, 2)
                if total_relationships
                else None
            ),
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

    print("\nEvaluation complete.")
    print(f"Documents evaluated: {len(document_results)}")
    print(f"Documents skipped: {len(skipped_documents)}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(
        f"Section heading coverage: "
        f"{summary['section_heading_coverage']}%"
    )
    print(
        f"Full heading path coverage: "
        f"{summary['full_heading_path_coverage']}%"
    )
    print(
        f"Total chunk-section relationships: "
        f"{summary['total_chunk_section_relationships']}"
    )
    print(
        f"Chunks spanning multiple sections: "
        f"{summary['multi_section_chunk_percentage']}%"
    )
    print(
        f"Average sections per chunk: "
        f"{summary['average_sections_per_chunk']}"
    )
    print(
        f"Heading presence: "
        f"{summary['heading_presence_percentage']}%"
    )
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
