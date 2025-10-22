#!/usr/bin/env python3
"""
Lightweight FAISS-backed search over chunk embeddings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from rag_module import (
    load_chunk_records,
    load_embeddings,
    search_chunks,
)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search chunk embeddings with FAISS",
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to chunks JSONL file",
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to embeddings .npy file",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query text to search for",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return (default: 3)",
    )
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="Include previous and next chunks for each hit",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    chunks_path = Path(args.chunks)
    embeddings_path = Path(args.embeddings)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    records = load_chunk_records(chunks_path)
    embeddings = load_embeddings(embeddings_path)

    if embeddings.size == 0 or embeddings.shape[0] == 0:
        print("No embeddings found in the provided embeddings file.")
        return 0

    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(records)})"
        )

    results = search_chunks(
        args.query,
        records,
        embeddings,
        top_k=args.top_k,
        include_neighbors=args.with_context,
    )

    if not results:
        print("No results found.")
        return 0

    for idx, result in enumerate(results, start=1):
        print(f"\nResult #{idx}")
        print(f"  Score : {result['score']:.4f}")
        print(f"  Cosine: {result['cosine']:.4f}")
        if result.get("chunk_id") is not None:
            print(f"  Chunk ID : {result['chunk_id']}")
        if result.get("doc_id"):
            print(f"  Doc ID   : {result['doc_id']}")
        if result.get("doc_chunk_index") is not None:
            print(f"  DocIdx   : {result['doc_chunk_index']}")
        if result.get("page"):
            print(f"  Page     : {result['page']}")
        if result.get("page_end") and result.get("page_end") != result.get("page"):
            print(f"  PageEnd  : {result['page_end']}")
        if result.get("span"):
            print(f"  Span     : {result['span']}")
        if result.get("clean_span"):
            print(f"  CleanSpan: {result['clean_span']}")
        if args.with_context and result.get("previous"):
            print(f"  Prev  : {result['previous']}")
        print(f"  Text  : {result['text']}")
        if args.with_context and result.get("next"):
            print(f"  Next  : {result['next']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
