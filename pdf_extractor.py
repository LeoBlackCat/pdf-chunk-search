"""
PDF text extraction utilities using PyMuPDF.

This module borrows its logic from the clean_pdf.py helper referenced
by the project, but keeps only the purely local pieces for extraction.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

if load_dotenv is not None:  # load variables from .env automatically if available
    load_dotenv()


CLEANUP_MODEL = "gpt-5"
CLEANUP_SYSTEM_PROMPT = """# GOAL

Adjust the content below to make it clean and readable:
Remove repeated strings that do not add value to the text.

Remove any content unrelated to the text itself (e.g., metadata, artifacts, or extraction errors).

Format the output as unstructured but clear text.

Do not add extra text, introductions, conclusions, or commentary—only rewrite the provided content as it is.

Do not interpret, analyze, or alter the meaning, intent, or narrative of the text—just reformat it for clarity and readability.

Do not change the text structure, do not write conclusions about it. Your only job is to make it readable. 

Keep the text in its original language, regardless of what it is."""


@dataclass(frozen=True)
class PageContent:
    number: int
    raw: str
    clean: str


@dataclass(frozen=True)
class ExtractionResult:
    final_text: str
    clean_text: str
    pages: List[PageContent]
    clean_page_offsets: List[int]
    ai_used: bool
    extractor: str
def clean_pdf_text(text: str) -> str:
    """Normalize common PDF artifacts while keeping the text readable."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        """: "'",
        """: "'",
        '"': '"',
        "′": "'",
        "‚": ",",
        "„": '"',
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "…": "...",
        "•": "*",
        "°": " degrees ",
        "¹": "1",
        "²": "2",
        "³": "3",
        "©": "(c)",
        "®": "(R)",
        "™": "(TM)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    allowed_symbols = "()%=[]{}#$@!?.,;:+-*/^<>&|~"
    text = "".join(
        char
        for char in text
        if unicodedata.category(char)[0] != "C"
        or char in "\n\t "
        or char in allowed_symbols
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r"\n\t+", "\n", text)
    text = re.sub(r"\t+\n", "\n", text)
    text = re.sub(r"\t+", " ", text)

    def _collapse_ligature_gap(match: re.Match[str]) -> str:
        token = match.group(1)
        if any(ch.islower() for ch in token):
            return token
        return match.group(0)

    text = re.sub(
        r"((?:ffi|fi|fl))[ ]+(?=\w)",
        _collapse_ligature_gap,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+", "", text)
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
    text = re.sub(r"(\()\s+", r"\1", text)
    text = re.sub(r"\s+([.,])\s+", r"\1 ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u200e\u200f]", "", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Remove decorative or placeholder glyphs (e.g., standalone "z" dropcaps)
    text = re.sub(r"(?m)^\s*z\s*\n", "", text)
    text = re.sub(r"(?m)^z\s+", "", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> ExtractionResult:
    """
    Extract text from a PDF file using PyMuPDF and clean the output.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # pragma: no cover - mirrors upstream helper
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    extraction_flags = (
        fitz.TEXT_PRESERVE_LIGATURES
        | fitz.TEXT_PRESERVE_WHITESPACE
        | fitz.TEXT_PRESERVE_IMAGES
    )

    page_contents: List[PageContent] = []
    try:
        for number, page in enumerate(doc, start=1):
            raw_text = page.get_text(flags=extraction_flags)
            clean_text = clean_pdf_text(raw_text)
            page_contents.append(PageContent(number=number, raw=raw_text, clean=clean_text))
    finally:
        doc.close()

    clean_parts: List[str] = []
    offsets: List[int] = []
    current_offset = 0
    for page in page_contents:
        offsets.append(current_offset)
        clean_parts.append(page.clean)
        current_offset += len(page.clean)
    combined_clean = "".join(clean_parts)

    final_text, ai_used = _apply_ai_cleanup_if_configured(combined_clean)
    extractor_info = _build_extractor_info(extraction_flags)

    return ExtractionResult(
        final_text=final_text,
        clean_text=combined_clean,
        pages=page_contents,
        clean_page_offsets=offsets,
        ai_used=ai_used,
        extractor=extractor_info,
    )


def _apply_ai_cleanup_if_configured(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    if OpenAI is None:
        print("Skipping AI cleanup: openai package not installed")
        return text, False
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping AI cleanup: OPENAI_API_KEY not set")
        return text, False

    try:
        print(f"Running AI cleanup with OpenAI model {CLEANUP_MODEL}")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=CLEANUP_MODEL,
            messages=[
                {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        cleaned = response.choices[0].message.content
        if cleaned:
            print("✓ AI cleanup succeeded")
            return cleaned, True
        print("Warning: AI cleanup returned empty content, using raw cleaned text")
    except Exception as exc:  # pragma: no cover - network call
        print(f"Warning: AI cleanup failed ({exc}), using raw cleaned text")
    return text, False


def _build_extractor_info(flags: int) -> str:
    active: List[str] = []
    if flags & fitz.TEXT_PRESERVE_LIGATURES:
        active.append("LIG")
    if flags & fitz.TEXT_PRESERVE_WHITESPACE:
        active.append("WHITESPACE")
    if flags & fitz.TEXT_PRESERVE_IMAGES:
        active.append("IMAGES")
    flag_str = "|".join(active) if active else "DEFAULT"
    return f"pymupdf@{getattr(fitz, '__version__', 'unknown')} flags={flag_str}"


__all__ = ["extract_text_from_pdf", "clean_pdf_text", "ExtractionResult", "PageContent"]
