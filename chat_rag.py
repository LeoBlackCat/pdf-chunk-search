#!/usr/bin/env python3
"""
Chat with an LM Studio-served model using RAG output from local chunk files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests

from rag_module import (
    build_system_prompt,
    format_context_blocks,
    load_chunk_records,
    load_doc_records,
    load_embeddings,
    load_prompt_templates,
    search_chunks,
)


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

    embeddings = load_embeddings(embeddings_path)
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

    context_blocks = format_context_blocks(results, chunk_lookup, doc_lookup)

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
