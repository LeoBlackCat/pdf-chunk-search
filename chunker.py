# chunker.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Union

__all__ = ["split_text", "token_count"]

_USE_MLX = os.getenv("CHUNKER_USE_MLX", "1").lower() not in {"0", "false", "no"}
_tokenizer = None
_tokenizer_error: Exception | None = None

try:
    from llama_index.core.node_parser import SentenceSplitter as LlamaSentenceSplitter
except ImportError:  # pragma: no cover - optional dependency
    LlamaSentenceSplitter = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:  # pragma: no cover - optional dependency
        RecursiveCharacterTextSplitter = None


def _llama_token_stub(text: str) -> List[str]:
    count = token_count(text)
    return [""] * max(0, count)


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
    except Exception as exc:  # pragma: no cover - hard to trigger in tests
        _tokenizer_error = exc
        print(f"Warning: mlx_embeddings not available ({exc}), using character-based estimation")
        return None


def token_count(text: str) -> int:
    tokenizer = _load_tokenizer()
    if tokenizer is not None:
        try:
            return int(tokenizer.encode(text, return_tensors="mlx").shape[-1])
        except Exception as exc:  # pragma: no cover - defensive
            global _tokenizer_error
            _tokenizer_error = exc
            print(f"Warning: mlx tokenizer encode failed ({exc}), falling back to char estimation")
    return max(1, len(text) // 4)


MAX_CHUNK_TOKENS = 8192
DEFAULT_OVERLAP_TOKENS = 30

LANGCHAIN_SEPARATORS: List[str] = [
    "\n\n",
    "\n",
    ". ",
    ".",
    ", ",
    ",",
    " ",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\uff0c",
    "\uff0e",
    "\uff1a",
    "\uff1b",
    "\uff1f",
    "\uff01",
    "",
]

@dataclass(frozen=True)
class Separator:
    token: Union[str, re.Pattern[str]]
    keep_with_previous: bool = True


@dataclass(frozen=True)
class TextUnit:
    text: str
    level: Literal["sentence", "clause", "word"]
    tokens: int


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
CLAUSE_SEPARATORS: tuple[Separator, ...] = (
    Separator(re.compile(r"\s*\n\s*\n\s*"), keep_with_previous=False),  # double newline
    Separator(re.compile(r"\s*\n\s*"), keep_with_previous=False),  # single newline
    Separator(re.compile(r"\s*…\s*")),  # ellipsis
    Separator(re.compile(r"\s*(?:—|–)\s*")),  # em/en dash
    Separator(re.compile(r"(?<=[,;:])\s+")),  # punctuation-based clause breaks
    Separator(re.compile(r"\s*\(\s*"), keep_with_previous=False),  # opening parenthesis
    Separator(re.compile(r"\s*\)\s*")),  # closing parenthesis
    Separator(
        re.compile(r"\s+(?=\b(?:and|but|or|nor|so|yet|for)\b)", flags=re.IGNORECASE)
    ),  # coordinating conjunctions
)

SplitStrategy = Literal["smart", "sentence", "llama", "langchain"]


def split_text(
    txt: str,
    chunk_size: int = MAX_CHUNK_TOKENS,
    chunk_overlap: int | None = None,
    separators: Iterable[Separator] | None = None,  # retained for compatibility
    length_fn: Callable[[str], int] = token_count,
    strategy: SplitStrategy = "smart",
) -> List[str]:
    """
    Split text into chunks under a token ceiling while respecting natural boundaries.

    strategy == "smart":
        - Pack by sentences first.
        - If a sentence is too long, break into clauses.
        - If a clause is still too long, fall back to word groups (never mid-word).
        - Apply overlap (default 30 tokens) using the same hierarchy for context.

    strategy == "sentence":
        - Pack by sentences only (same fallbacks for oversized sentences).
        - No overlap.

    strategy == "llama":
        - Delegate to llama_index.core.node_parser.SentenceSplitter for chunking.
        - Uses the provided chunk_size and overlap budgets.
    """
    max_size = max(1, min(chunk_size, MAX_CHUNK_TOKENS))
    overlap = 0
    if separators:
        # Custom separator lists are no longer configurable; parameter kept for compatibility.
        pass
    if strategy in {"smart", "llama", "langchain"}:
        if chunk_overlap is None:
            overlap = min(DEFAULT_OVERLAP_TOKENS, max(0, max_size - 1))
        else:
            overlap = max(0, min(chunk_overlap, max(0, max_size - 1)))

    if strategy == "langchain":
        return _split_with_langchain(txt, max_size, overlap)

    if strategy == "llama":
        return _split_with_llama(txt, max_size, overlap)

    units = _build_units(txt, max_size, length_fn)
    if not units:
        return []

    chunks = _assemble_chunks(
        units=units,
        chunk_size=max_size,
        overlap_tokens=overlap,
        length_fn=length_fn,
    )
    return chunks


def _build_units(text: str, chunk_size: int, length_fn: Callable[[str], int]) -> List[TextUnit]:
    sentences = SENTENCE_SPLIT_RE.split(text)
    if not sentences:
        sentences = [text]

    units: List[TextUnit] = []
    for sentence in sentences:
        sentence_unit = _make_unit(sentence, "sentence", length_fn)
        if sentence_unit is None:
            continue
        if sentence_unit.tokens <= chunk_size:
            units.append(sentence_unit)
            continue

        clause_texts = _split_sentence_to_clauses(sentence_unit.text)
        if len(clause_texts) == 1 and clause_texts[0] == sentence_unit.text:
            units.extend(_split_clause_to_words(sentence_unit.text, chunk_size, length_fn))
            continue

        for clause in clause_texts:
            clause_unit = _make_unit(clause, "clause", length_fn)
            if clause_unit is None:
                continue
            if clause_unit.tokens <= chunk_size:
                units.append(clause_unit)
            else:
                units.extend(_split_clause_to_words(clause_unit.text, chunk_size, length_fn))

    return units


def _assemble_chunks(
    units: List[TextUnit],
    chunk_size: int,
    overlap_tokens: int,
    length_fn: Callable[[str], int],
) -> List[str]:
    chunks: List[str] = []
    current_units: List[TextUnit] = []
    current_text = ""
    current_tokens = 0

    def flush() -> None:
        nonlocal current_units, current_text, current_tokens
        if not current_units:
            return
        if not current_text:
            current_units = []
            current_tokens = 0
            current_text = ""
            return
        chunks.append(current_text)
        overlap_units = _extract_overlap_units(
            current_units,
            overlap_tokens,
            chunk_size,
            length_fn,
        )
        current_units = overlap_units
        current_text, current_tokens = _text_from_units(current_units, length_fn)

    for unit in units:
        if unit.tokens > chunk_size:
            if current_units:
                flush()
            chunks.append(unit.text)
            current_units = []
            current_tokens = 0
            current_text = ""
            continue

        candidate_units = current_units + [unit]
        candidate_text = _normalize(" ".join(u.text for u in candidate_units))
        candidate_tokens = length_fn(candidate_text)

        if current_units and candidate_tokens > chunk_size:
            flush()
            candidate_units = current_units + [unit]
            candidate_text = _normalize(" ".join(u.text for u in candidate_units))
            candidate_tokens = length_fn(candidate_text)

        if candidate_tokens > chunk_size and current_units:
            flush()
            candidate_units = current_units + [unit]
            candidate_text = _normalize(" ".join(u.text for u in candidate_units))
            candidate_tokens = length_fn(candidate_text)

        if candidate_tokens > chunk_size and not current_units:
            if unit.tokens > chunk_size:
                chunks.append(unit.text)
                current_units = []
                current_tokens = 0
                current_text = ""
            else:
                current_units = [unit]
                current_text = unit.text
                current_tokens = unit.tokens
            continue

        current_units = candidate_units
        current_text = candidate_text
        current_tokens = candidate_tokens

    flush()
    return chunks


def _extract_overlap_units(
    units: List[TextUnit],
    overlap_tokens: int,
    chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[TextUnit]:
    if overlap_tokens <= 0:
        return []

    collected: List[TextUnit] = []
    remaining = overlap_tokens
    for unit in reversed(units):
        if remaining <= 0:
            break
        if unit.tokens <= remaining:
            collected.append(unit)
            remaining -= unit.tokens
            continue
        tail = _overlap_from_unit(unit, remaining, chunk_size, length_fn)
        if tail:
            for piece in reversed(tail):
                if remaining <= 0:
                    break
                if piece.tokens <= remaining:
                    collected.append(piece)
                    remaining -= piece.tokens
            break
        # if we cannot take part of this unit (e.g., giant token span), move to earlier units

    collected.reverse()
    return collected


def _overlap_from_unit(
    unit: TextUnit,
    remaining: int,
    chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[TextUnit]:
    if remaining <= 0:
        return []
    if unit.tokens <= remaining:
        return [unit]

    if unit.level == "sentence":
        clause_texts = _split_sentence_to_clauses(unit.text)
        clause_units = [
            _make_unit(text, "clause", length_fn)
            for text in clause_texts
            if text.strip()
        ]
        clause_units = [c for c in clause_units if c is not None]
        if clause_units:
            tail = _collect_tail_units(clause_units, remaining, chunk_size, length_fn)
            if tail:
                return tail

    if unit.level in {"sentence", "clause"}:
        return _tail_word_units(unit.text, remaining, length_fn)

    if unit.level == "word":
        return _tail_word_units(unit.text, remaining, length_fn)

    return []


def _collect_tail_units(
    units: List[TextUnit],
    remaining: int,
    chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[TextUnit]:
    collected: List[TextUnit] = []
    for sub_unit in reversed(units):
        if remaining <= 0:
            break
        if sub_unit.tokens <= remaining:
            collected.append(sub_unit)
            remaining -= sub_unit.tokens
            continue
        tail = _tail_word_units(sub_unit.text, remaining, length_fn)
        if tail:
            collected.extend(tail)
        break
    collected.reverse()
    return collected


def _tail_word_units(
    text: str,
    remaining: int,
    length_fn: Callable[[str], int],
) -> List[TextUnit]:
    if remaining <= 0:
        return []

    words = re.findall(r"\S+", text)
    if not words:
        return []

    collected: List[str] = []
    total = 0
    for word in reversed(words):
        tokens = length_fn(word)
        if total and total + tokens > remaining:
            break
        if not collected and tokens > remaining:
            # Cannot take this word without exceeding the overlap budget.
            return []
        collected.insert(0, word)
        total += tokens
        if total >= remaining:
            break

    if not collected:
        return []
    overlap_text = _normalize(" ".join(collected))
    if not overlap_text:
        return []
    return [TextUnit(overlap_text, "word", length_fn(overlap_text))]


def _split_sentence_to_clauses(sentence: str) -> List[str]:
    segments: List[str] = [sentence]
    for separator in CLAUSE_SEPARATORS:
        next_segments: List[str] = []
        for segment in segments:
            parts = _split_with_rule(segment, separator)
            if len(parts) == 1:
                next_segments.append(segment)
            else:
                next_segments.extend(parts)
        segments = next_segments
    clauses = [_normalize(seg) for seg in segments if _normalize(seg)]
    return clauses if clauses else [_normalize(sentence)]


def _split_clause_to_words(
    clause: str,
    chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[TextUnit]:
    words = re.findall(r"\S+", clause)
    if not words:
        clause_unit = _make_unit(clause, "word", length_fn)
        return [clause_unit] if clause_unit else []

    groups: List[TextUnit] = []
    current: List[str] = []
    current_tokens = 0

    for word in words:
        tokens = length_fn(word)
        if current and current_tokens + tokens > chunk_size:
            group_text = _normalize(" ".join(current))
            if group_text:
                groups.append(TextUnit(group_text, "word", length_fn(group_text)))
            current = [word]
            current_tokens = tokens
        else:
            current.append(word)
            current_tokens += tokens

        if current_tokens > chunk_size and len(current) == 1:
            group_text = _normalize(current[0])
            if group_text:
                groups.append(TextUnit(group_text, "word", length_fn(group_text)))
            current = []
            current_tokens = 0

    if current:
        group_text = _normalize(" ".join(current))
        if group_text:
            groups.append(TextUnit(group_text, "word", length_fn(group_text)))

    if not groups:
        clause_unit = _make_unit(clause, "word", length_fn)
        if clause_unit:
            return [clause_unit]
        return []

    validated: List[TextUnit] = []
    for unit in groups:
        if unit.tokens <= chunk_size:
            validated.append(unit)
        else:
            validated.extend(_hard_wrap_units(unit.text, chunk_size, length_fn))
    return validated


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

    sep = token
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


def _hard_wrap_units(
    text: str,
    chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[TextUnit]:
    if length_fn(text) <= chunk_size:
        return [TextUnit(text, "word", length_fn(text))]

    pieces = _hard_wrap(text, chunk_size, length_fn)
    units: List[TextUnit] = []
    for piece in pieces:
        norm = _normalize(piece)
        if norm:
            units.append(TextUnit(norm, "word", length_fn(norm)))
    return units


def _hard_wrap(
    text: str,
    chunk_size: int,
    length_fn: Callable[[str], int],
) -> List[str]:
    if length_fn(text) <= chunk_size:
        return [text]

    tokens: List[str] = []
    current: List[str] = []
    for char in text:
        current.append(char)
        candidate = "".join(current)
        if length_fn(candidate) >= chunk_size:
            tokens.append(candidate)
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _make_unit(
    text: str,
    level: Literal["sentence", "clause", "word"],
    length_fn: Callable[[str], int],
) -> TextUnit | None:
    normalized = _normalize(text)
    if not normalized:
        return None
    return TextUnit(normalized, level, length_fn(normalized))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text_from_units(
    units: List[TextUnit],
    length_fn: Callable[[str], int],
) -> tuple[str, int]:
    if not units:
        return ("", 0)
    text = _normalize(" ".join(unit.text for unit in units))
    if not text:
        return ("", 0)
    return (text, length_fn(text))


def _split_with_llama(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if LlamaSentenceSplitter is None:
        raise ImportError(
            "llama-index is not installed. Install llama-index to use the 'llama' strategy."
        )
    splitter = LlamaSentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=_llama_token_stub,
        paragraph_separator="\n\n",
    )
    chunks = splitter.split_text(text)
    normalized = [_normalize(chunk) for chunk in chunks]
    return [chunk for chunk in normalized if chunk]


def _split_with_langchain(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if RecursiveCharacterTextSplitter is None:
        raise ImportError(
            "langchain is not installed. Install langchain to use the 'langchain' strategy."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_count,
        separators=LANGCHAIN_SEPARATORS,
    )
    chunks = splitter.split_text(text)
    normalized = [_normalize(chunk) for chunk in chunks]
    return [chunk for chunk in normalized if chunk]
