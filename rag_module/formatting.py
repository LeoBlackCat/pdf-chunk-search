"""
Prompt and context formatting helpers shared across RAG entry points.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_PROMPTS: Dict[str, str] = {
    "standard": (
        "Today Date: {date}\n\n"
        "Given the following conversation, relevant context, and a follow up question, "
        "reply with an answer to the current question the user is asking. "
        "Return only your response to the question given the above information following "
        "the users instructions as needed.\n"
        "Context:\n"
    ),
    "context_only": (
        "Today Date: {date}\n\n"
        "You can only answer questions about the provided context.\n"
        "If you know the answer but it is not based on the provided context, do not provide the answer; "
        "state that the answer is not in the context provided.\n\n"
        "Context:\n"
    ),
}


def load_prompt_templates(prompt_file: Optional[Path]) -> Dict[str, str]:
    """Load prompt templates from YAML, falling back to defaults."""
    templates = dict(DEFAULT_PROMPTS)
    if prompt_file is None:
        return templates
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if yaml is None:
        raise ImportError("PyYAML is required to load prompt templates from a file.")
    with prompt_file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            "Prompt file must contain a mapping of prompt names to template strings."
        )
    for name, template in data.items():
        if not isinstance(name, str) or not isinstance(template, str):
            raise ValueError("Prompt file entries must map string names to string templates.")
        templates[name] = template
    return templates


def format_context_block(
    index: int,
    result: dict,
    chunk_lookup: Dict[int, dict],
    doc_lookup: Dict[str, dict],
) -> str:
    """Format a single retrieved chunk into the context block markup."""
    chunk_id = result.get("chunk_id")
    doc_id = result.get("doc_id")
    chunk = {}
    if chunk_id is not None:
        try:
            chunk = chunk_lookup.get(int(chunk_id), {})
        except (TypeError, ValueError):
            chunk = {}
    doc = doc_lookup.get(doc_id or "")

    source_name = ""
    if doc:
        source_name = doc.get("file_name") or doc.get("source_path") or ""
    if not source_name:
        source_name = str(doc_id or chunk_id or "unknown")

    published = ""
    if doc:
        published = doc.get("extracted_at") or ""

    page_display = ""
    pages = chunk.get("pages")
    if pages:
        if len(pages) == 1:
            page_display = f"page {pages[0]}"
        else:
            page_display = f"pages {pages[0]}–{pages[-1]}"
    elif result.get("page"):
        page_display = f"page {result['page']}"

    meta_lines = ["<document_metadata>", f"sourceDocument: {source_name}"]
    if published:
        meta_lines.append(f"extracted_at: {published}")
    if page_display:
        meta_lines.append(f"span: {page_display}")
    meta_lines.append("</document_metadata>")

    text = chunk.get("text") or result.get("text") or ""
    block = [
        f"[CONTEXT {index}]:",
        "\n".join(meta_lines),
        "",
        text.strip(),
        f"[END CONTEXT {index}]",
        "",
    ]
    return "\n".join(block)


def format_context_blocks(
    results: Iterable[dict],
    chunk_lookup: Dict[int, dict],
    doc_lookup: Dict[str, dict],
) -> List[str]:
    """Format all retrieved chunks in order."""
    return [
        format_context_block(idx, result, chunk_lookup, doc_lookup)
        for idx, result in enumerate(results)
    ]


def build_system_prompt(context_blocks: List[str], template: str) -> str:
    """Combine the prompt template and context into a system message."""
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    header = template.format(date=today)
    if not header.endswith("\n"):
        header += "\n"
    return header + "".join(context_blocks)
