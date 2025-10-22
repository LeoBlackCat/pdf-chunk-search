"""Shared RAG utilities for chunk loading, search, and prompt formatting."""

from .loader import (
    load_chunk_records,
    load_doc_records,
    load_embeddings,
)
from .search import (
    build_index,
    search_chunks,
    rerank,
    mmr,
)
from .formatting import (
    DEFAULT_PROMPTS,
    build_system_prompt,
    format_context_block,
    format_context_blocks,
    load_prompt_templates,
)
from .embeddings import (
    embed_texts,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "build_index",
    "build_system_prompt",
    "embed_texts",
    "format_context_block",
    "format_context_blocks",
    "load_chunk_records",
    "load_doc_records",
    "load_embeddings",
    "load_prompt_templates",
    "mmr",
    "rerank",
    "search_chunks",
]
