#!/usr/bin/env python3
"""
Lightweight FAISS-backed search over chunk embeddings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import faiss  # type: ignore
import numpy as np

from embeddings import embed_texts


def load_chunks(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D (got shape {vectors.shape})")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def search_chunks(
    query: str,
    chunks: Sequence[str],
    embeddings: np.ndarray,
    top_k: int = 3,
    include_neighbors: bool = False,
) -> List[dict]:
    if top_k <= 0:
        return []
    query_vec = embed_texts(query)
    if query_vec.size == 0:
        return []

    index = build_index(embeddings)
    D, I = index.search(query_vec, top_k)
    results: List[dict] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        payload = {
            "index": int(idx),
            "score": float(score),
            "chunk": chunks[idx],
        }
        if include_neighbors:
            payload["previous"] = chunks[idx - 1] if idx > 0 else None
            payload["next"] = chunks[idx + 1] if idx + 1 < len(chunks) else None
        results.append(payload)
    return results


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search chunk embeddings with FAISS",
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to *_chunks.txt file",
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to *_embeddings.npy file",
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

    chunks = load_chunks(chunks_path)
    embeddings = np.load(embeddings_path)

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) "
            f"do not match chunk count ({len(chunks)})"
        )

    results = search_chunks(
        args.query,
        chunks,
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
        print(f"  Index : {result['index']}")
        print(f"  Chunk : {result['chunk']}")
        if args.with_context:
            if result.get("previous"):
                print(f"  Prev  : {result['previous']}")
            if result.get("next"):
                print(f"  Next  : {result['next']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
