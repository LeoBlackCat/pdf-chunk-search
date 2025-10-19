#!/usr/bin/env python3
"""
Lightweight FAISS-backed search over chunk embeddings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

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

    recall_k = _recall_k(top_k, len(chunks))
    index = build_index(embeddings)
    D, I = index.search(query_vec, recall_k)

    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        candidates.append(
            {
                "index": int(idx),
                "text": chunks[idx],
                "cos": float(score),
                "vec": embeddings[idx],
            }
        )

    reranked = rerank(query, candidates, strict_if_quoted=True, topn=max(top_k, 8))
    final = mmr(reranked, lam=0.75, topn=top_k)

    results: List[dict] = []
    for item in final:
        payload = {
            "index": int(item["index"]),
            "score": float(item.get("rerank", item.get("cos", 0.0))),
            "cosine": float(item.get("cos", 0.0)),
            "chunk": item["text"],
        }
        if include_neighbors:
            idx = payload["index"]
            payload["previous"] = chunks[idx - 1] if idx > 0 else None
            payload["next"] = chunks[idx + 1] if idx + 1 < len(chunks) else None
        results.append(payload)
    return results


def _recall_k(top_k: int, total: int) -> int:
    recall = max(top_k * 4, 32)
    return min(total, max(top_k, recall))


def _extract_terms(query: str) -> Tuple[List[str], List[str], List[str]]:
    phrases = re.findall(r'"([^"]+)"', query)
    stripped = re.sub(r'"[^"]+"', " ", query)
    excluded = [w[1:].lower() for w in re.findall(r"-\w+", stripped)]
    terms = [
        w.lower()
        for w in re.findall(r"\w+", stripped)
        if not w.startswith("-") and len(w) > 2
    ]
    return phrases, terms, excluded


def _weights_for(query: str) -> Tuple[float, float, float, bool]:
    trimmed = query.strip()
    phrases = re.findall(r'"([^"]+)"', trimmed)
    bare = re.sub(r'"[^"]+"', " ", trimmed)
    tokens = re.findall(r"\w+", bare.lower())
    short = len(tokens) <= 3
    numeric = any(ch.isdigit() for ch in trimmed)

    w_sem = 0.7
    w_lex = 0.3
    phrase_bonus = 0.0
    if phrases:
        phrase_bonus = 0.15
    if short or numeric:
        w_sem, w_lex = 0.5, 0.5
    return w_sem, w_lex, phrase_bonus, bool(phrases)


def rerank(
    query: str,
    candidates: List[dict],
    strict_if_quoted: bool = True,
    topn: int = 10,
) -> List[dict]:
    if not candidates:
        return []
    phrases, terms, excludes = _extract_terms(query)
    w_sem, w_lex, phrase_bonus, has_quotes = _weights_for(query)

    filtered: List[dict] = []
    for cand in candidates:
        text = cand["text"]
        if excludes and any(re.search(rf"\b{re.escape(word)}\b", text, re.I) for word in excludes):
            continue
        if strict_if_quoted and has_quotes:
            tl = text.lower()
            if not all(p.lower() in tl for p in phrases):
                continue
        filtered.append(cand)

    if not filtered:
        filtered = candidates

    scored: List[Tuple[float, dict]] = []
    for cand in filtered:
        text = cand["text"]
        tl = text.lower()
        cos_score = float(cand.get("cos", 0.0))
        kw_hits = sum(tl.count(term) for term in terms)
        kw_norm = min(1.0, kw_hits / max(1.0, len(terms))) if terms else 0.0
        phrase_hits = sum(1 for phrase in phrases if phrase.lower() in tl)
        bonus = min(0.3, phrase_hits * (phrase_bonus or 0.0))
        score = w_sem * cos_score + w_lex * kw_norm + bonus
        cand_copy = dict(cand)
        cand_copy["rerank"] = float(score)
        scored.append((score, cand_copy))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = [cand for _, cand in scored[: max(1, topn)]]
    return top


def mmr(candidates: List[dict], lam: float = 0.75, topn: int = 8) -> List[dict]:
    if not candidates:
        return []
    if "vec" not in candidates[0]:
        return candidates[:topn]

    vectors = np.stack([cand["vec"] for cand in candidates]).astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors = vectors / norms
    sims = vectors @ vectors.T

    selected: List[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < topn:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_idx = None
        best_val = -1e9
        for ridx in remaining:
            relevance = candidates[ridx].get("rerank", candidates[ridx].get("cos", 0.0))
            diversity = max(sims[ridx, s] for s in selected)
            value = lam * relevance - (1 - lam) * diversity
            if value > best_val:
                best_val = value
                best_idx = ridx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


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
        print(f"  Cosine: {result['cosine']:.4f}")
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
