#!/usr/bin/env python3
"""
Chat with an LM Studio-served model using RAG output from local chunk files.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from chunk_search import load_chunk_records, search_chunks


def load_doc_records(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    lookup: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id")
            if doc_id:
                lookup[doc_id] = record
    return lookup


def format_context_block(
    index: int,
    result: dict,
    chunk_lookup: Dict[int, dict],
    doc_lookup: Dict[str, dict],
) -> str:
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
        raise ValueError("Prompt file must contain a mapping of prompt names to template strings.")
    for name, template in data.items():
        if not isinstance(name, str) or not isinstance(template, str):
            raise ValueError("Prompt file entries must map string names to string templates.")
        templates[name] = template
    return templates


def build_system_prompt(context_blocks: List[str], template: str) -> str:
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    header = template.format(date=today)
    if not header.endswith("\n"):
        header += "\n"
    return header + "".join(context_blocks)


def call_lmstudio(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    stream: bool = True,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "stream": bool(stream),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    response = requests.post(url, json=payload, stream=stream, timeout=180)
    response.raise_for_status()

    if not stream:
        data = response.json()
        return data["choices"][0]["message"]["content"]

    collected: List[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.strip().startswith("data:"):
            payload_text = line.split("data:", 1)[-1].strip()
            if payload_text == "[DONE]":
                break
            chunk = json.loads(payload_text)
            delta = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content")
            )
            if delta:
                print(delta, end="", flush=True)
                collected.append(delta)
    print()
    return "".join(collected)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-shot RAG chat against LM Studio using precomputed chunks",
    )
    parser.add_argument("--chunks", required=True, help="Path to chunk metadata JSONL")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    parser.add_argument(
        "--docs",
        required=True,
        help="Path to document metadata JSONL",
    )
    parser.add_argument("--query", required=True, help="User question to answer")
    parser.add_argument("--top-k", type=int, default=3, help="How many chunks to retrieve")
    parser.add_argument(
        "--api-base",
        default="http://localhost:1234/v1",
        help="LM Studio API base URL (default: http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama-3.1-8b-instruct",
        help="Model identifier to request from LM Studio",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable server-side streaming when calling LM Studio",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the RAG context blocks before sending the request",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional YAML file containing named prompt templates",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default="standard",
        help="Prompt template key to use (default: standard)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    chunks_path = Path(args.chunks)
    embeddings_path = Path(args.embeddings)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    records = load_chunk_records(chunks_path)
    if not records:
        raise RuntimeError("No chunk records found in the provided file.")

    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(records):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match chunk count ({len(records)})"
        )

    docs_path = Path(args.docs)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs file not found: {docs_path}")
    doc_lookup = load_doc_records(docs_path)

    chunk_lookup: Dict[int, dict] = {}
    for record in records:
        chunk_id = record.get("id")
        try:
            if chunk_id is not None:
                chunk_lookup[int(chunk_id)] = record
        except (TypeError, ValueError):
            continue

    results = search_chunks(
        args.query,
        records,
        embeddings,
        top_k=args.top_k,
        include_neighbors=False,
    )
    if not results:
        print("No matching chunks found; nothing to send to LM Studio.")
        return 0

    context_blocks = [
        format_context_block(idx, result, chunk_lookup, doc_lookup)
        for idx, result in enumerate(results)
    ]

    prompt_file = Path(args.prompt_file) if args.prompt_file else None
    prompt_templates = load_prompt_templates(prompt_file)
    if args.prompt_name not in prompt_templates:
        available = ", ".join(sorted(prompt_templates))
        raise ValueError(f"Prompt '{args.prompt_name}' not found. Available prompts: {available}")
    prompt_template = prompt_templates[args.prompt_name]

    if args.show_context:
        print("\n".join(context_blocks))
        print()

    print(f"User prompt: {args.query}")

    system_prompt = build_system_prompt(context_blocks, prompt_template)
    print(">>> Sending request to LM Studio...")
    response_text = call_lmstudio(
        api_base=args.api_base,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=args.query,
        temperature=args.temperature,
        stream=not args.no_stream,
    )

    if args.no_stream:
        print("\n---")
        print("LM Studio response:")
        print(response_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
