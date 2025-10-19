"""
PDF text extraction utilities using PyMuPDF.

This module borrows its logic from the clean_pdf.py helper referenced
by the project, but keeps only the purely local pieces for extraction.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import List

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


CLEANUP_MODEL = "gpt-5-a
CLEANUP_TEMPERATURE = 0.0
CLEANUP_MAX_TOKENS = 8000
CLEANUP_SYSTEM_PROMPT = """# GOAL

Adjust the content below to make it clean and readable:
Remove repeated strings that do not add value to the text.

Remove any content unrelated to the text itself (e.g., metadata, artifacts, or extraction errors).

Format the output as unstructured but clear text.

Do not add extra text, introductions, conclusions, or commentary—only rewrite the provided content as it is.

Do not interpret, analyze, or alter the meaning, intent, or narrative of the text—just reformat it for clarity and readability.

Do not change the text structure, do not write conclusions about it. Your only job is to make it readable. 

Keep the text in its original language, regardless of what it is."""


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
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF and clean the output.

    Args:
        pdf_path: File system path to the PDF.

    Returns:
        Cleaned text extracted from the document.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # pragma: no cover - mirrors upstream helper
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    full_text: List[str] = []
    extraction_flags = (
        fitz.TEXT_PRESERVE_LIGATURES
        | fitz.TEXT_PRESERVE_WHITESPACE
        | fitz.TEXT_PRESERVE_IMAGES
    )

    try:
        for page in doc:
            full_text.append(page.get_text(flags=extraction_flags))
    finally:
        doc.close()

    combined_text = "".join(full_text)
    cleaned = clean_pdf_text(combined_text)
    return _apply_ai_cleanup_if_configured(cleaned)


def _apply_ai_cleanup_if_configured(text: str) -> str:
    if not text:
        return text
    if OpenAI is None:
        return text
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return text

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=CLEANUP_MODEL,
            messages=[
                {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=CLEANUP_TEMPERATURE,
            max_tokens=CLEANUP_MAX_TOKENS,
        )
        cleaned = response.choices[0].message.content
        if cleaned:
            return cleaned
    except Exception as exc:  # pragma: no cover - network call
        print(f"Warning: AI cleanup failed ({exc}), using raw cleaned text")
    return text


__all__ = ["extract_text_from_pdf", "clean_pdf_text"]
