#!/usr/bin/env python3
"""
PDF Chunker - Extract and chunk content from PDFs and plain text files.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from chunker import split_text, token_count
from embeddings import embed_texts
from pdf_extractor import ExtractionResult, PageContent, extract_text_from_pdf


EMBEDDING_MODEL_ID = "mlx-community/all-MiniLM-L6-v2-4bit"
TOKENIZER_ID = "mlx-miniLM"


def extract_and_chunk(
    input_file: str,
    output_file: str,
    chunk_size: int = 256,
    chunk_overlap: Optional[int] = None,
    strategy: str = "sentence",
) -> None:
    """
    Extract content from a file and split it into chunks, emitting JSONL metadata.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extraction = _extract_document(input_path)
    final_text = extraction.final_text

    if not final_text:
        print("Warning: No text content extracted from file")
        return

    print(f"Extracted {len(final_text)} characters")
    print(f"Estimated tokens: {token_count(final_text)}")

    extracted_file = output_path.parent / f"{output_path.stem}_extracted.txt"
    chunks_json = output_path.parent / f"{output_path.stem}_chunks.jsonl"
    docs_json = output_path.parent / f"{output_path.stem}_docs.jsonl"

    extracted_file.write_text(final_text, encoding="utf-8")
    print(f"\n✓ Wrote cleaned text to {extracted_file}")

    if strategy == "sentence":
        effective_overlap = 0
    else:
        overlap_input = chunk_overlap if chunk_overlap is not None else 30
        effective_overlap = max(0, min(overlap_input, max(0, chunk_size - 1)))

    print(
        f"\nChunking with size={chunk_size}, overlap={effective_overlap}, strategy={strategy}"
    )
    chunks = split_text(
        final_text,
        chunk_size=chunk_size,
        chunk_overlap=effective_overlap if strategy != "sentence" else 0,
        strategy=strategy,
    )
    print(f"Created {len(chunks)} chunks")

    print(f"\nGenerating embeddings for {len(chunks)} chunks")
    embeddings = embed_texts(chunks)
    if embeddings.size == 0:
        print("Warning: No embeddings generated")
    else:
        print(f"Embeddings shape: {embeddings.shape}")

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    doc_id = _compute_doc_id(input_path)
    doc_record = _build_doc_record(
        input_path=input_path,
        extraction=extraction,
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=effective_overlap,
        strategy=strategy,
        embeddings=embeddings,
        doc_id=doc_id,
        timestamp=timestamp,
        extracted_path=extracted_file,
        chunks_path=chunks_json,
    )

    chunk_records = _build_chunk_records(
        chunks=chunks,
        embeddings=embeddings,
        extraction=extraction,
        doc_id=doc_id,
        timestamp=timestamp,
        effective_overlap=effective_overlap,
    )

    docs_json.write_text(json.dumps(doc_record, ensure_ascii=False) + "\n", encoding="utf-8")
    with chunks_json.open("w", encoding="utf-8") as fh:
        for record in chunk_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    chunk_sizes = [token_count(c) for c in chunks]
    if chunk_sizes:
        print("\nChunk statistics:")
        print(f"  Min tokens: {min(chunk_sizes)}")
        print(f"  Max tokens: {max(chunk_sizes)}")
        print(f"  Avg tokens: {sum(chunk_sizes) / len(chunk_sizes):.1f}")

    print("\n📄 Output files:")
    print(f"  Extracted:  {extracted_file}")
    print(f"  Document metadata: {docs_json}")
    print(f"  Chunks & embeddings: {chunks_json}")


def _extract_document(input_path: Path) -> ExtractionResult:
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(str(input_path))
    if suffix in {".txt", ".md"}:
        raw_text = input_path.read_text(encoding="utf-8")
        page = PageContent(number=1, raw=raw_text, clean=raw_text)
        return ExtractionResult(
            final_text=raw_text,
            clean_text=raw_text,
            pages=[page],
            clean_page_offsets=[0],
            ai_used=False,
            extractor="plaintext@1.0",
        )
    raise ValueError(
        f"Unsupported file extension '{suffix}'. Only PDF, TXT, or Markdown are supported."
    )


def _compute_doc_id(input_path: Path) -> str:
    data = input_path.read_bytes()
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _build_doc_record(
    *,
    input_path: Path,
    extraction: ExtractionResult,
    chunk_count: int,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    embeddings: np.ndarray,
    doc_id: str,
    timestamp: str,
    extracted_path: Path,
    chunks_path: Path,
) -> dict:
    file_stat = input_path.stat()
    embedding_dim = int(embeddings.shape[1]) if embeddings.size else 0
    clean_component = "python-clean-v1"
    if extraction.ai_used:
        clean_component += "+gpt-5"
    pipeline_version = (
        f"extract={extraction.extractor};"
        f"clean={clean_component};"
        f"chunk={strategy}-v1({chunk_size}/{chunk_overlap});"
        f"embed={EMBEDDING_MODEL_ID}@{embedding_dim}"
    )

    record = {
        "doc_id": doc_id,
        "file_name": input_path.name,
        "source_path": str(input_path.resolve()),
        "filesize": file_stat.st_size,
        "pdf_sha256": doc_id,
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
        "embedding_dim": embedding_dim,
        "normalized_embeddings": bool(embeddings.size),
        "tokenizer_id": TOKENIZER_ID,
        "ai_cleanup_used": extraction.ai_used,
        "chunks_path": str(chunks_path),
        "extracted_text_path": str(extracted_path),
    }
    return record


def _build_chunk_records(
    *,
    chunks: List[str],
    embeddings: np.ndarray,
    extraction: ExtractionResult,
    doc_id: str,
    timestamp: str,
    effective_overlap: int,
) -> List[dict]:
    records: List[dict] = []
    final_text = extraction.final_text
    clean_text = extraction.clean_text
    total_pages = len(extraction.pages)
    total_length = len(final_text)
    search_pos = 0
    clean_search_pos = 0

    for idx, chunk_text in enumerate(chunks):
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

        page_number = _estimate_page(
            char_start=start,
            clean_pos=clean_pos,
            extraction=extraction,
            total_pages=total_pages,
            total_length=total_length,
        )

        vector = embeddings[idx] if embeddings.size else np.array([], dtype=np.float32)
        record = {
            "id": idx,
            "doc_id": doc_id,
            "text": chunk_text,
            "page": page_number,
            "span": {"char_start": start, "char_end": end},
            "model": EMBEDDING_MODEL_ID,
            "dim": int(vector.shape[0]) if vector.size else (int(embeddings.shape[1]) if embeddings.size else 0),
            "normalized": bool(vector.size),
            "tokenizer_id": TOKENIZER_ID,
            "token_count": token_count(chunk_text),
            "hash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            "created_at": timestamp,
            "embedding": vector.tolist(),
            "overlap_prev_tokens": effective_overlap if idx > 0 else 0,
            "overlap_next_tokens": effective_overlap if idx < len(chunks) - 1 else 0,
        }
        if clean_pos != -1:
            record["clean_span"] = {
                "char_start": clean_pos,
                "char_end": clean_pos + len(chunk_text),
            }
        records.append(record)
    return records


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


def main():
    parser = argparse.ArgumentParser(
        description="Extract and chunk content from PDFs and plain text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf output.txt
  %(prog)s document.pdf output.txt --chunk-size 1000
  %(prog)s document.pdf output.txt --chunk-size 800 --chunk-overlap 100
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input file path (PDF, TXT, or Markdown)"
    )
    
    parser.add_argument(
        "output_file",
        help="Output file path (one chunk per line)"
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
    
    args = parser.parse_args()
    
    try:
        extract_and_chunk(
            args.input_file,
            args.output_file,
            args.chunk_size,
            args.chunk_overlap,
            args.strategy,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
