# chunker.py
from __future__ import annotations
import re
from collections import deque
from dataclasses import dataclass
import os
from typing import Callable, Iterable, List, Tuple, Union

_USE_MLX = os.getenv("CHUNKER_USE_MLX", "1").lower() not in {"0", "false", "no"}
_tokenizer = None
_tokenizer_error: Exception | None = None

def _load_tokenizer():
    global _tokenizer, _tokenizer_error
    if _tokenizer is not None or _tokenizer_error is not None:
        return _tokenizer
    if not _USE_MLX:
        _tokenizer_error = RuntimeError("MLX disabled via CHUNKER_USE_MLX env var")
        return None
    try:
        from mlx_embeddings import load
        _, tok = load("mlx-community/all-MiniLM-L6-v2-4bit")
        _tokenizer = tok
        return _tokenizer
    except Exception as exc:
        _tokenizer_error = exc
        print(f"Warning: mlx_embeddings not available ({exc}), using character-based estimation")
        return None

def token_count(s: str) -> int:
    tokenizer = _load_tokenizer()
    if tokenizer is not None:
        try:
            return int(tokenizer.encode(s, return_tensors="mlx").shape[-1])
        except Exception as exc:
            global _tokenizer_error
            _tokenizer_error = exc
            print(f"Warning: mlx tokenizer encode failed ({exc}), falling back to char estimation")
    return max(1, len(s) // 4)

MAX_CHUNK_TOKENS = 256


@dataclass(frozen=True)
class Separator:
    token: Union[str, re.Pattern[str]]
    keep_with_previous: bool = True


DEFAULT_SEPARATORS: Tuple[Separator, ...] = (
    Separator(re.compile(r"(?<=[.!?。！？])\s+")),  # sentences first
    Separator(re.compile(r"\s*…\s*")),  # ellipsis as thought boundary
    Separator(re.compile(r"(?<=[,;:])\s+")),  # clause punctuation
    Separator(re.compile(r"\s*(?:—|–)\s*")),  # dash-separated thoughts
    Separator(re.compile(r"\s*\(\s*"), keep_with_previous=False),  # opening parenthesis
    Separator(re.compile(r"\s*\)\s*")),  # closing parenthesis
    Separator(
        re.compile(r"\s+(?=\b(?:and|but|or|nor|so|yet|for)\b)", flags=re.IGNORECASE)
    ),  # coordinating conjunctions
    Separator(re.compile(r"\n{2,}")),  # paragraphs
    Separator("\n"),  # single newlines
    Separator(","),  # stray commas
    Separator(" "),  # words
)


def split_text(
    txt: str,
    chunk_size: int = 500,
    chunk_overlap: int | None = None,
    separators: Iterable[Separator] = DEFAULT_SEPARATORS,
    length_fn: Callable[[str], int] = token_count,
) -> List[str]:
    """
    Split ``txt`` into chunks capped at ``chunk_size`` tokens (max 256) with a soft
    preference for higher-level boundaries (paragraphs → sentences → clauses → words).
    """
    chunk_size = max(1, min(chunk_size, MAX_CHUNK_TOKENS))
    if chunk_overlap is None:
        chunk_overlap = min(int(chunk_size * 0.15), chunk_size - 1)
    else:
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    separator_list = list(separators)
    segments = _recursive_split(txt, separator_list, chunk_size, length_fn)
    return _merge_segments(segments, chunk_size, chunk_overlap, length_fn)


def _recursive_split(
    text: str,
    separators: List[Separator],
    chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[str]:
    if not text:
        return []
    if length_fn(text) <= chunk_size:
        return [text]
    if not separators:
        return _word_wrap(text, chunk_size, length_fn)

    for idx, rule in enumerate(separators):
        splits = _split_with_rule(text, rule)
        if len(splits) == 1:
            continue
        pieces: List[str] = []
        next_rules = separators[idx + 1 :]
        for part in splits:
            if not part:
                continue
            pieces.extend(_recursive_split(part, next_rules, chunk_size, length_fn))
        return pieces

    return _word_wrap(text, chunk_size, length_fn)


def _split_with_rule(text: str, rule: Separator) -> List[str]:
    token = rule.token
    if isinstance(token, re.Pattern):
        parts: List[str] = []
        start = 0
        for match in token.finditer(text):
            if rule.keep_with_previous:
                end = match.end()
                if end == start:
                    continue
                parts.append(text[start:end])
                start = end
            else:
                split_at = match.start()
                if split_at > start:
                    parts.append(text[start:split_at])
                start = split_at
        if start < len(text):
            parts.append(text[start:])
        return [p for p in parts if p]

    # string-based separator
    sep = token
    if sep in ("\n\n", "\n"):
        fragments = text.split(sep)
        out: List[str] = []
        for idx, fragment in enumerate(fragments):
            if not fragment and idx != len(fragments) - 1:
                out.append(sep)
                continue
            piece = fragment
            if idx < len(fragments) - 1:
                piece += sep
            if piece:
                out.append(piece)
        return [p for p in out if p]

    parts: List[str] = []
    start = 0
    while True:
        pos = text.find(sep, start)
        if pos == -1:
            parts.append(text[start:])
            break
        end = pos + len(sep)
        if rule.keep_with_previous:
            parts.append(text[start:end])
            start = end
        else:
            if pos > start:
                parts.append(text[start:pos])
            start = pos
    return [p for p in parts if p]


def _word_wrap(
    text: str,
    limit: int,
    length_fn: Callable[[str], int],
) -> List[str]:
    if not text:
        return []
    if length_fn(text) <= limit:
        return [text]

    words = re.findall(r"\S+\s*", text)
    if not words:
        return [text]

    chunks: List[str] = []
    current = ""
    for word in words:
        word_tokens = length_fn(word)
        if not current:
            current = word
            if word_tokens > limit:
                chunks.append(word)
                current = ""
            continue
        candidate = current + word
        if length_fn(candidate) > limit:
            chunks.append(current)
            current = word
            if length_fn(current) > limit:
                chunks.append(current)
                current = ""
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _merge_segments(
    segments: List[str],
    chunk_size: int,
    chunk_overlap: int,
    length_fn: Callable[[str], int],
) -> List[str]:
    docs: List[str] = []
    current_text = ""

    def _finalize_current() -> None:
        nonlocal current_text
        if not current_text.strip():
            current_text = ""
            return
        docs.append(current_text.strip())
        overlap = _compute_overlap_tail(current_text, chunk_overlap, length_fn)
        current_text = overlap

    for segment in segments:
        if not segment or not segment.strip():
            continue
        pending = deque(
            _word_wrap(segment, chunk_size, length_fn)
            if length_fn(segment) > chunk_size
            else [segment]
        )
        while pending:
            piece = pending.popleft()
            if not piece.strip():
                continue
            candidate = current_text + piece
            candidate_tokens = length_fn(candidate)
            while current_text and candidate_tokens > chunk_size:
                _finalize_current()
                candidate = current_text + piece
                candidate_tokens = length_fn(candidate)
                if not current_text:
                    break
            if not current_text and candidate_tokens > chunk_size:
                sub_parts = _word_wrap(piece, chunk_size, length_fn)
                if len(sub_parts) == 1 and sub_parts[0] == piece:
                    docs.append(piece.strip())
                    current_text = ""
                else:
                    for sub in reversed([sp for sp in sub_parts if sp.strip()]):
                        pending.appendleft(sub)
                continue
            current_text = candidate

    if current_text.strip():
        docs.append(current_text.strip())
    return docs


def _compute_overlap_tail(
    text: str,
    overlap_tokens: int,
    length_fn: Callable[[str], int],
) -> str:
    if overlap_tokens <= 0 or not text.strip():
        return ""
    words = re.findall(r"\S+\s*", text)
    if not words:
        return ""
    collected: List[str] = []
    for word in reversed(words):
        collected.insert(0, word)
        if length_fn("".join(collected)) >= overlap_tokens:
            break
    return "".join(collected)
