"""Standalone evaluation runner for the document chunking service."""

import json
import logging
import time
from pathlib import Path
from statistics import mean
from typing import Any

from ai_knowledge_platform.services.chunking.boundaries import (
    TextBoundaryDetector,
)
from ai_knowledge_platform.services.chunking.service import (
    DocumentChunkingService,
)
from ai_knowledge_platform.services.chunking.types import DocumentChunk
from ai_knowledge_platform.services.tokenization.huggingface import (
    HuggingFaceTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "benchmarks" / "chunking" / "datasets"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "chunking" / "results"

MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_INPUT_TOKENS = 512

TARGET_TOKENS = 448
OVERLAP_TOKENS = 64
MINIMUM_TOKENS = 50


def measure_adjacent_overlaps(
    source_text: str,
    chunks: list[DocumentChunk],
    tokenizer: HuggingFaceTokenizer,
) -> list[dict[str, int]]:
    """Measure actual token overlap between adjacent chunks."""

    overlaps: list[dict[str, int]] = []

    for previous, current in zip(chunks, chunks[1:], strict=False):
        overlap_start = max(
            previous.start_char,
            current.start_char,
        )
        overlap_end = min(
            previous.end_char,
            current.end_char,
        )

        overlap_characters = max(0, overlap_end - overlap_start)

        if overlap_characters > 0:
            overlap_text = source_text[overlap_start:overlap_end]
            overlap_tokens = tokenizer.count_tokens(overlap_text)
        else:
            overlap_tokens = 0

        overlaps.append(
            {
                "previous_chunk": previous.chunk_index,
                "current_chunk": current.chunk_index,
                "overlap_tokens": overlap_tokens,
                "overlap_characters": overlap_characters,
            }
        )

    return overlaps


def measure_semantic_boundaries(
    source_text: str,
    chunks: list[DocumentChunk],
) -> dict[str, Any]:
    """Measure how frequently chunk endings preserve semantic boundaries."""

    detector = TextBoundaryDetector()

    paragraphs = detector.split_paragraphs(source_text)

    paragraph_endings = {paragraph.end for paragraph in paragraphs}

    sentence_endings: set[int] = set()

    for paragraph in paragraphs:
        paragraph_text = source_text[paragraph.start : paragraph.end]

        sentences = detector.split_sentences(
            paragraph_text,
            start_offset=paragraph.start,
        )

        sentence_endings.update(sentence.end for sentence in sentences)

    paragraph_count = 0
    sentence_count = 0
    mid_sentence_count = 0

    # Exclude the final chunk because it naturally ends at the
    # end of the document.
    for chunk in chunks[:-1]:
        # Ignore trailing whitespace when identifying the actual
        # semantic endpoint of a chunk.
        boundary_position = len(source_text[: chunk.end_char].rstrip())

        if boundary_position in paragraph_endings:
            paragraph_count += 1
        elif boundary_position in sentence_endings:
            sentence_count += 1
        else:
            mid_sentence_count += 1

    total_boundaries = paragraph_count + sentence_count + mid_sentence_count

    preserved_boundaries = paragraph_count + sentence_count

    preservation_rate = (
        round(preserved_boundaries / total_boundaries * 100, 2)
        if total_boundaries > 0
        else None
    )

    return {
        "total_chunk_boundaries": total_boundaries,
        "paragraph_boundary_endings": paragraph_count,
        "sentence_boundary_endings": sentence_count,
        "mid_sentence_endings": mid_sentence_count,
        "boundary_preservation_rate": preservation_rate,
    }


def evaluate_document(
    path: Path,
    chunker: DocumentChunkingService,
    tokenizer: HuggingFaceTokenizer,
) -> dict[str, Any]:
    """Evaluate correctness and basic statistics for one document."""

    source_text = path.read_text(encoding="utf-8")

    start_time = time.perf_counter()
    chunks = chunker.chunk_text(source_text)
    elapsed_seconds = time.perf_counter() - start_time

    token_counts: list[int] = []
    violations: list[str] = []

    # Measure actual overlap between adjacent chunks.
    adjacent_overlaps = measure_adjacent_overlaps(
        source_text=source_text,
        chunks=chunks,
        tokenizer=tokenizer,
    )

    overlap_token_counts = [overlap["overlap_tokens"] for overlap in adjacent_overlaps]

    # Measure semantic boundary preservation.
    semantic_boundaries = measure_semantic_boundaries(
        source_text=source_text,
        chunks=chunks,
    )

    # Validate individual chunks.
    for expected_index, chunk in enumerate(chunks):
        actual_tokens = tokenizer.count_tokens(chunk.content)
        token_counts.append(actual_tokens)

        if chunk.chunk_index != expected_index:
            violations.append(f"Unexpected chunk index: {chunk.chunk_index}")

        if chunk.content != source_text[chunk.start_char : chunk.end_char]:
            violations.append(
                f"Chunk {chunk.chunk_index}: content does not match source offsets"
            )

        if actual_tokens != chunk.token_count:
            violations.append(f"Chunk {chunk.chunk_index}: token count mismatch")

        if actual_tokens > chunker.token_budget:
            violations.append(f"Chunk {chunk.chunk_index}: exceeds token budget")

        if not (0 <= chunk.start_char < chunk.end_char <= len(source_text)):
            violations.append(f"Chunk {chunk.chunk_index}: invalid character offsets")

    # Check whether the chunks collectively cover the complete source.
    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: chunk.start_char,
    )

    coverage_complete = bool(ordered_chunks)

    if coverage_complete:
        coverage_complete = ordered_chunks[0].start_char == 0

        coverage_complete &= max(chunk.end_char for chunk in ordered_chunks) == len(
            source_text
        )

        for previous, current in zip(
            ordered_chunks,
            ordered_chunks[1:],
            strict=False,
        ):
            if current.start_char > previous.end_char:
                coverage_complete = False
                violations.append("Missing source text between chunks")

    if not coverage_complete:
        violations.append("Complete source coverage check failed")

    return {
        "document": path.name,
        "source_characters": len(source_text),
        "source_tokens": tokenizer.count_tokens(source_text),
        "chunk_count": len(chunks),
        "average_chunk_tokens": (round(mean(token_counts), 2) if token_counts else 0),
        "maximum_chunk_tokens": max(token_counts, default=0),
        "minimum_chunk_tokens": min(token_counts, default=0),
        "average_overlap_tokens": (
            round(mean(overlap_token_counts), 2) if overlap_token_counts else 0
        ),
        "maximum_overlap_tokens": max(
            overlap_token_counts,
            default=0,
        ),
        "minimum_overlap_tokens": min(
            overlap_token_counts,
            default=0,
        ),
        "zero_overlap_pairs": sum(count == 0 for count in overlap_token_counts),
        "adjacent_overlaps": adjacent_overlaps,
        "semantic_boundaries": semantic_boundaries,
        "token_budget": chunker.token_budget,
        "execution_seconds": round(elapsed_seconds, 4),
        "coverage_complete": coverage_complete,
        "passed": not violations,
        "violations": violations,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    dataset_paths = sorted(DATASET_DIR.glob("*.txt"))

    if not dataset_paths:
        raise FileNotFoundError("No benchmark datasets found.")

    tokenizer = HuggingFaceTokenizer(
        model_name=MODEL_NAME,
        max_input_tokens=MAX_INPUT_TOKENS,
    )

    chunker = DocumentChunkingService(
        tokenizer=tokenizer,
    )

    results = [evaluate_document(path, chunker, tokenizer) for path in dataset_paths]

    report = {
        "tokenizer": MODEL_NAME,
        "model_max_input_tokens": tokenizer.max_input_tokens,
        "special_tokens": tokenizer.special_tokens_count,
        "effective_chunk_budget": chunker.token_budget,
        "configuration": {
            "target_tokens": TARGET_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "minimum_tokens": MINIMUM_TOKENS,
        },
        "total_documents": len(results),
        "passed_documents": sum(result["passed"] for result in results),
        "failed_documents": sum(not result["passed"] for result in results),
        "documents": results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIR / "chunking_evaluation.json"

    output_path.write_text(
        json.dumps(
            report,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nCHUNKING EVALUATION")
    print("=" * 90)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"

        semantic = result["semantic_boundaries"]

        print(
            f"{status:<6} "
            f"{result['document']:<38} "
            f"chunks={result['chunk_count']:<5} "
            f"max_tokens={result['maximum_chunk_tokens']:<5} "
            f"avg_overlap={result['average_overlap_tokens']:<7} "
            f"boundary_rate={str(semantic['boundary_preservation_rate']):<7} "
            f"time={result['execution_seconds']:.4f}s"
        )

        for violation in result["violations"]:
            print(f"       ERROR: {violation}")

    print("=" * 90)

    print(
        f"Documents: {report['passed_documents']}/"
        f"{report['total_documents']} passed"
    )

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
