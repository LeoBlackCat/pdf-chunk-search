"""
Shared search utilities built on top of FAISS and the local embedding model.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np

from embeddings import embed_texts


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Construct an inner-product FAISS index for the provided vectors."""
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D (got shape {vectors.shape})")
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("No embeddings available to build an index")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def search_chunks(
    query: str,
    records: List[dict],
    embeddings: np.ndarray,
    top_k: int = 3,
    include_neighbors: bool = False,
) -> List[dict]:
    """Return top-k chunk matches for the provided query text."""
    if top_k <= 0:
        return []

    query_vec = embed_texts(query)
    if query_vec.size == 0:
        return []

    recall_k = _recall_k(top_k, len(records))
    index = build_index(embeddings)
    D, I = index.search(query_vec, recall_k)

    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(records):
            continue
        record = records[idx]
        candidates.append(
            {
                "index": record.get("id", int(idx)),
                "list_index": int(idx),
                "record": record,
                "text": record.get("text", ""),
                "cos": float(score),
                "vec": embeddings[idx],
            }
        )

    reranked = rerank(query, candidates, strict_if_quoted=True, topn=max(top_k, 8))
    final = mmr(reranked, lam=0.75, topn=top_k)

    results: List[dict] = []
    for item in final:
        record = item.get("record", {})
        payload = {
            "chunk_id": record.get("id", item.get("index")),
            "doc_id": record.get("doc_id"),
            "page": record.get("page"),
            "page_end": record.get("page_end"),
            "span": record.get("span"),
            "doc_chunk_index": record.get("doc_chunk_index"),
            "score": float(item.get("rerank", item.get("cos", 0.0))),
            "cosine": float(item.get("cos", 0.0)),
            "text": record.get("text", item.get("text", "")),
        }
        if clean_span := record.get("clean_span"):
            payload["clean_span"] = clean_span
        if include_neighbors:
            list_idx = item.get("list_index", 0)
            payload["previous"] = (
                records[list_idx - 1].get("text") if list_idx > 0 else None
            )
            payload["next"] = (
                records[list_idx + 1].get("text")
                if list_idx + 1 < len(records)
                else None
            )
        results.append(payload)
    return results


def rerank(
    query: str,
    candidates: List[dict],
    strict_if_quoted: bool = True,
    topn: int = 10,
) -> List[dict]:
    """Apply lightweight lexical reranking over semantic candidates."""
    if not candidates:
        return []
    phrases, terms, excludes = _extract_terms(query)
    w_sem, w_lex, phrase_bonus, has_quotes = _weights_for(query)

    filtered: List[dict] = []
    for cand in candidates:
        text = cand["text"]
        if excludes and any(
            re.search(rf"\b{re.escape(word)}\b", text, re.I) for word in excludes
        ):
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
    """Maximal Marginal Relevance pruning for semantic diversity."""
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
