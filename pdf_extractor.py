"""
PDF text extraction utilities using PyMuPDF.

This module borrows its logic from the clean_pdf.py helper referenced
by the project, but keeps only the purely local pieces needed for
offline extraction (no OpenAI dependency).
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import List

import fitz  # PyMuPDF


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
    return clean_pdf_text(combined_text)


__all__ = ["extract_text_from_pdf", "clean_pdf_text"]
