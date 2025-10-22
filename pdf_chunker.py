#!/usr/bin/env python3
"""
PDF Chunker - Extract and chunk content from PDFs and plain text files.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from chunker import split_text, token_count
from embeddings import embed_texts
from pdf_extractor import (
    CLEANUP_MODEL,
    ExtractionResult,
    PageContent,
    apply_ai_cleanup_if_configured,
    extract_text_from_pdf,
)


EMBEDDING_MODEL_ID = "mlx-community/all-MiniLM-L6-v2-4bit"
TOKENIZER_ID = "mlx-miniLM"
DIRECTORY_EXTENSIONS = {".pdf", ".md"}
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _collect_input_paths(inputs: Iterable[str]) -> List[Path]:
    """Expand files and directories into a flat list of supported files (no recursion)."""
    resolved: List[Path] = []
    seen = set()

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in DIRECTORY_EXTENSIONS
            )
            if not files:
                print(f"Warning: No supported files found in directory {path}")
            for child in files:
                try:
                    key = child.resolve(strict=True)
                except FileNotFoundError:
                    print(f"Skipping missing file: {child}")
                    continue
                if key in seen:
                    continue
                resolved.append(child)
                seen.add(key)
            continue

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file type: {path}")
            continue

        try:
            key = path.resolve(strict=True)
        except FileNotFoundError:
            print(f"Skipping missing file: {path}")
            continue
        if key in seen:
            continue
        resolved.append(path)
        seen.add(key)

    return resolved


def extract_and_chunk(
    input_files: List[str],
    output_prefix: str,
    chunk_size: int = 256,
    chunk_overlap: Optional[int] = None,
    strategy: str = "sentence",
    use_ai_cleanup: bool = False,
    save_extracted_text: bool = False,
) -> None:
    """Extract and chunk one or more files, emitting JSONL metadata and embeddings."""

    if not input_files:
        raise ValueError("At least one input file must be provided")
    resolved_inputs = _collect_input_paths(input_files)
    if not resolved_inputs:
        raise ValueError("No supported input files were found.")
    print(f"Discovered {len(resolved_inputs)} input file(s).")

    output_base = Path(output_prefix)
    output_dir = output_base.parent if output_base.parent != Path("") else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = output_dir / f"{output_base.name}_docs.jsonl"
    chunks_path = output_dir / f"{output_base.name}_chunks.jsonl"
    embeddings_path = output_dir / f"{output_base.name}_embeddings.npy"

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    effective_overlap = 0 if strategy == "sentence" else max(0, min((chunk_overlap if chunk_overlap is not None else 30), max(0, chunk_size - 1)))

    all_embeddings: List[np.ndarray] = []
    chunk_records: List[dict] = []
    doc_records: List[dict] = []
    current_chunk_id = 0

    for idx, input_path in enumerate(resolved_inputs):

        print(f"\nExtracting content from: {input_path}")
        extraction = _extract_document(input_path, use_ai_cleanup=use_ai_cleanup)
        final_text = extraction.final_text

        if not final_text:
            print(f"Warning: No text content extracted from {input_path}")
            continue

        print(f"Extracted {len(final_text)} characters")
        print(f"Estimated tokens: {token_count(final_text)}")

        extracted_file: Optional[Path] = None
        if save_extracted_text:
            clean_filename = (
                f"{output_base.name}_{idx:04d}_{input_path.stem}_extracted.txt"
            )
            extracted_file = output_dir / clean_filename
            extracted_file.write_text(final_text, encoding="utf-8")
            print(f"✓ Wrote cleaned text to {extracted_file}")

        print(
            f"Chunking with size={chunk_size}, overlap={effective_overlap}, strategy={strategy}"
        )
        doc_chunks = split_text(
            final_text,
            chunk_size=chunk_size,
            chunk_overlap=effective_overlap if strategy != "sentence" else 0,
            strategy=strategy,
        )
        print(f"Created {len(doc_chunks)} chunks")

        if doc_chunks:
            print(f"Generating embeddings for {len(doc_chunks)} chunks")
            doc_embeddings = embed_texts(doc_chunks)
            print(f"Embeddings shape: {doc_embeddings.shape}")
        else:
            doc_embeddings = np.zeros((0, 0), dtype=np.float32)
            print("No chunks produced; skipping embedding generation")

        embedding_dim = int(doc_embeddings.shape[1]) if doc_embeddings.size else 0

        doc_id = _compute_doc_id(input_path)
        pipeline_version = _build_pipeline_version(
            extraction=extraction,
            chunk_size=chunk_size,
            chunk_overlap=effective_overlap,
            strategy=strategy,
            embedding_dim=embedding_dim,
        )

        final_text_hash = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
        clean_text_hash = hashlib.sha256(extraction.clean_text.encode("utf-8")).hexdigest()
        doc_title = _infer_doc_title(input_path, extraction)

        doc_record = _build_doc_record(
            input_path=input_path,
            extraction=extraction,
            chunk_count=len(doc_chunks),
            chunk_size=chunk_size,
            chunk_overlap=effective_overlap,
            strategy=strategy,
            embeddings_dim=embedding_dim,
            doc_id=doc_id,
            timestamp=timestamp,
            extracted_path=extracted_file,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            chunk_id_start=current_chunk_id,
            pipeline_version=pipeline_version,
            doc_index=idx,
            final_text_hash=final_text_hash,
            clean_text_hash=clean_text_hash,
            doc_title=doc_title,
            source_directory=str(input_path.parent.resolve()),
        )
        doc_records.append(doc_record)

        new_chunk_records = _build_chunk_records(
            chunks=doc_chunks,
            embeddings=doc_embeddings,
            extraction=extraction,
            doc_id=doc_id,
            timestamp=timestamp,
            effective_overlap=effective_overlap,
            chunk_id_start=current_chunk_id,
            pipeline_version=pipeline_version,
        )
        chunk_records.extend(new_chunk_records)
        if doc_embeddings.size:
            all_embeddings.append(doc_embeddings)

        chunk_sizes = [token_count(c) for c in doc_chunks]
        if chunk_sizes:
            print(f"Chunk statistics for {input_path.name}:")
            print(f"  Min tokens: {min(chunk_sizes)}")
            print(f"  Max tokens: {max(chunk_sizes)}")
            print(f"  Avg tokens: {sum(chunk_sizes) / len(chunk_sizes):.1f}")

        current_chunk_id += len(doc_chunks)

        if (idx + 1) % 10 == 0:
            cpu_percent = _current_cpu_percent()
            if cpu_percent is not None:
                print(f"\nSystem CPU usage after {idx + 1} files: {cpu_percent:.1f}%")
            else:
                print(f"\nSystem CPU usage after {idx + 1} files: unavailable")
            try:
                input("Press Enter to continue...")
            except EOFError:
                pass

    if not doc_records:
        print("No documents were successfully processed.")
        return

    with docs_path.open("w", encoding="utf-8") as fh:
        for record in doc_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    with chunks_path.open("w", encoding="utf-8") as fh:
        for record in chunk_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if all_embeddings:
        embedding_matrix = np.concatenate(all_embeddings, axis=0)
    else:
        embedding_matrix = np.zeros((0, 0), dtype=np.float32)
    np.save(embeddings_path, embedding_matrix)

    print(
        f"\nProcessed {len(doc_records)} document(s); total chunks: {len(chunk_records)}"
    )

    print("\n📄 Output files:")
    print(f"  Document metadata: {docs_path}")
    print(f"  Chunks metadata:  {chunks_path}")
    print(f"  Embeddings:       {embeddings_path}")


def _extract_document(input_path: Path, use_ai_cleanup: bool) -> ExtractionResult:
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(str(input_path), use_ai_cleanup=use_ai_cleanup)
    if suffix in {".txt", ".md"}:
        raw_text = input_path.read_text(encoding="utf-8")
        if use_ai_cleanup and raw_text.strip():
            final_text, ai_used = apply_ai_cleanup_if_configured(raw_text)
        else:
            final_text, ai_used = raw_text, False
        page = PageContent(number=1, raw=raw_text, clean=raw_text)
        return ExtractionResult(
            final_text=final_text,
            final_pages=[final_text],
            final_page_offsets=[0],
            clean_text=raw_text,
            pages=[page],
            clean_page_offsets=[0],
            ai_used=ai_used,
            extractor="plaintext@1.0",
        )
    raise ValueError(
        f"Unsupported file extension '{suffix}'. Only PDF, TXT, or Markdown are supported."
    )


def _compute_doc_id(input_path: Path) -> str:
    data = input_path.read_bytes()
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _infer_doc_title(input_path: Path, extraction: ExtractionResult) -> Optional[str]:
    suffix = input_path.suffix.lower()
    if suffix == ".md":
        for line in extraction.final_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
    stem = input_path.stem.replace("_", " ").strip()
    return stem or None


def _build_doc_record(
    *,
    input_path: Path,
    extraction: ExtractionResult,
    chunk_count: int,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    embeddings_dim: int,
    doc_id: str,
    timestamp: str,
    extracted_path: Optional[Path],
    chunks_path: Path,
    embeddings_path: Path,
    chunk_id_start: int,
    pipeline_version: str,
    doc_index: int,
    final_text_hash: str,
    clean_text_hash: str,
    doc_title: Optional[str] = None,
    source_directory: Optional[str] = None,
) -> dict:
    file_stat = input_path.stat()
    chunk_id_end = chunk_id_start + chunk_count - 1 if chunk_count else None
    suffix = input_path.suffix.lower()
    source_type = suffix.lstrip(".") or "unknown"

    record = {
        "schema_version": "doc.v1",
        "doc_index": doc_index,
        "doc_id": doc_id,
        "file_name": input_path.name,
        "source_path": str(input_path.resolve()),
        "filesize": file_stat.st_size,
        "file_sha256": doc_id,
        "pdf_sha256": doc_id if suffix == ".pdf" else None,
        "source_type": source_type,
        "pages": len(extraction.pages),
        "extracted_at": timestamp,
        "extractor": extraction.extractor,
        "ocr": {"used": False},
        "pipeline_version": pipeline_version,
        "chunk_count": chunk_count,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_strategy": strategy,
        "embedding_model": EMBEDDING_MODEL_ID,
        "embedding_dim": embeddings_dim,
        "normalized_embeddings": bool(embeddings_dim),
        "tokenizer_id": TOKENIZER_ID,
        "ai_cleanup_used": extraction.ai_used,
        "ai_cleanup_model": CLEANUP_MODEL if extraction.ai_used else None,
        "metric": "ip_cosine",
        "chunks_path": str(chunks_path),
        "extracted_text_path": str(extracted_path) if extracted_path else None,
        "embeddings_path": str(embeddings_path),
        "chunk_id_start": chunk_id_start,
        "chunk_id_end": chunk_id_end,
        "chunk_id_range": [chunk_id_start, chunk_id_end] if chunk_count else [],
        "span_scope": "doc_final",
        "final_text_sha256": final_text_hash,
        "clean_text_sha256": clean_text_hash,
    }
    if doc_title:
        record["title"] = doc_title
    if source_directory:
        record["source_directory"] = source_directory
    return record


def _build_pipeline_version(
    *,
    extraction: ExtractionResult,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    embedding_dim: int,
) -> str:
    clean_component = "python-clean-v1"
    if extraction.ai_used:
        clean_component += f"+{CLEANUP_MODEL}"
    pipeline_version = (
        f"extract={extraction.extractor};"
        f"clean={clean_component};"
        f"chunk={strategy}-v1({chunk_size}/{chunk_overlap});"
        f"embed={EMBEDDING_MODEL_ID}@{embedding_dim or 0}"
    )
    return pipeline_version


def _build_chunk_records(
    *,
    chunks: List[str],
    embeddings: np.ndarray,
    extraction: ExtractionResult,
    doc_id: str,
    timestamp: str,
    effective_overlap: int,
    chunk_id_start: int,
    pipeline_version: str,
) -> List[dict]:
    records: List[dict] = []
    final_text = extraction.final_text
    clean_text = extraction.clean_text
    total_length = len(final_text)
    final_offsets = extraction.final_page_offsets or [0]
    final_page_texts = extraction.final_pages or [final_text]
    page_numbers = [page.number for page in extraction.pages] or [1]
    final_page_lengths = [len(text) for text in final_page_texts]
    search_pos = 0
    clean_search_pos = 0

    for idx, chunk_text in enumerate(chunks):
        global_chunk_id = chunk_id_start + idx
        start = final_text.find(chunk_text, search_pos)
        if start == -1:
            snippet = chunk_text.strip()
            if snippet:
                start = final_text.find(snippet, search_pos)
        if start == -1:
            start = search_pos
        end = start + len(chunk_text)
        search_pos = end

        clean_pos = -1
        if clean_text:
            clean_pos = clean_text.find(chunk_text, clean_search_pos)
            if clean_pos == -1:
                snippet = chunk_text.strip()
                if snippet:
                    clean_pos = clean_text.find(snippet, clean_search_pos)
            if clean_pos != -1:
                clean_search_pos = clean_pos + len(chunk_text)

        page_list = _pages_for_span(
            char_start=start,
            char_end=end,
            offsets=final_offsets,
            lengths=final_page_lengths,
            page_numbers=page_numbers,
        )
        page_number = page_list[0] if page_list else (page_numbers[0] if page_numbers else 1)

        emb_dim = embeddings.shape[1] if embeddings.ndim == 2 else 0

        record = {
            "schema_version": "chunk.v1",
            "id": global_chunk_id,
            "doc_id": doc_id,
            "doc_chunk_index": idx,
            "text": chunk_text,
            "page": page_number,
            "page_end": page_list[-1] if page_list else page_number,
            "pages": page_list,
            "span_scope": "doc_final",
            "span": {"char_start": start, "char_end": end},
            "model": EMBEDDING_MODEL_ID,
            "dim": int(emb_dim),
            "normalized": bool(emb_dim),
            "tokenizer_id": TOKENIZER_ID,
            "token_count": token_count(chunk_text),
            "hash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            "created_at": timestamp,
            "overlap_prev_tokens": effective_overlap if idx > 0 else 0,
            "overlap_next_tokens": effective_overlap if idx < len(chunks) - 1 else 0,
            "metric": "ip_cosine",
            "pipeline_version": pipeline_version,
        }
        if clean_pos != -1:
            record["clean_span"] = {
                "char_start": clean_pos,
                "char_end": clean_pos + len(chunk_text),
            }
            record["clean_span_scope"] = "doc_clean"
        records.append(record)
    return records


def _pages_for_span(
    *,
    char_start: int,
    char_end: int,
    offsets: List[int],
    lengths: List[int],
    page_numbers: List[int],
) -> List[int]:
    pages: List[int] = []
    span_end = max(char_end, char_start)
    for idx, offset in enumerate(offsets):
        length = lengths[idx] if idx < len(lengths) else 0
        page_start = offset
        page_end = offset + length
        if length == 0:
            continue
        if span_end <= page_start:
            break
        if char_start < page_end and span_end > page_start:
            pages.append(page_numbers[idx] if idx < len(page_numbers) else idx + 1)
    if not pages and page_numbers:
        # Fallback: assign to the last page if span starts beyond known offsets
        if char_start >= (offsets[-1] + lengths[-1] if lengths else offsets[-1]):
            pages.append(page_numbers[-1])
        else:
            pages.append(page_numbers[0])
    return pages


def _estimate_page(
    *,
    char_start: int,
    clean_pos: int,
    extraction: ExtractionResult,
    total_pages: int,
    total_length: int,
) -> int:
    if total_pages == 0:
        return 1
    if clean_pos is not None and clean_pos >= 0:
        for idx, offset in enumerate(extraction.clean_page_offsets):
            page_len = len(extraction.pages[idx].clean)
            if clean_pos < offset + page_len:
                return extraction.pages[idx].number
    if total_length == 0:
        return extraction.pages[0].number
    ratio = char_start / total_length if total_length else 0.0
    approx_index = min(total_pages - 1, max(0, int(ratio * total_pages)))
    return extraction.pages[approx_index].number


def _current_cpu_percent() -> Optional[float]:
    """Return a best-effort estimate of the current system CPU utilisation percentage."""
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=0.1))
    except ModuleNotFoundError:
        pass

    if hasattr(os, "getloadavg"):
        try:
            load1, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return float(min(100.0, (load1 / cpu_count) * 100.0))
        except OSError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract and chunk content from PDFs and plain text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s output/run document.pdf
  %(prog)s output/run doc1.pdf doc2.pdf --chunk-size 1000
  %(prog)s output/run doc.pdf --chunk-size 800 --chunk-overlap 100
        """
    )
    
    parser.add_argument(
        "output_prefix",
        help="Output file prefix (base path without extension)"
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input files or directories (PDF, TXT, or Markdown; directories scan only PDFs and Markdown)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Maximum tokens per chunk (default: 256)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=30,
        help="Token overlap between chunks (default: 30)"
    )

    parser.add_argument(
        "--strategy",
        choices=["smart", "sentence", "llama", "langchain"],
        default="sentence",
        help=(
            "Chunking strategy to apply (default: sentence; "
            "use 'llama' for LlamaIndex SentenceSplitter or 'langchain' for RecursiveCharacterTextSplitter)"
        )
    )
    parser.add_argument(
        "--ai-clean",
        action="store_true",
        help="Use OpenAI cleanup (gpt-4o-mini) when extracting PDFs or text",
    )
    parser.add_argument(
        "--save-extracted-text",
        action="store_true",
        help="Write cleaned extracted text alongside other outputs",
    )
    
    args = parser.parse_args()
    
    try:
        extract_and_chunk(
            args.input_files,
            args.output_prefix,
            args.chunk_size,
            args.chunk_overlap,
            args.strategy,
            use_ai_cleanup=args.ai_clean,
            save_extracted_text=args.save_extracted_text,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
