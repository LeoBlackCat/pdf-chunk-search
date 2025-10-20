# Repository Guidelines

## Project Structure & Module Organization
- `pdf_chunker.py` orchestrates extraction, chunking, and embedding generation; keep new pipeline hooks modular beside `extract_and_chunk`.
- Supporting modules live at the repo root: `chunker.py` (token-aware splitting), `chunk_search.py` (FAISS queries), `pdf_extractor.py` (PyMuPDF wrapper and optional AI cleanup), and `embeddings.py` (MLX-backed embedders).
- Reference assets sit in `examples/` (sample PDFs) and `output/` (scratch artifacts—safe to clear). Automated checks live in `tests/`.
- Preserve relative paths required by tests (e.g., `examples/bk_tcco_000301.pdf`) and keep CLI entry points import-safe for reuse.

## Build, Test, and Development Commands
- Install deps into Python 3.11+: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Run the chunker end-to-end: `python pdf_chunker.py output/run examples/bk_tcco_000301.pdf --chunk-size 128`.
- Query generated vectors: `python chunk_search.py --chunks output/run_chunks.jsonl --embeddings output/run_embeddings.npy --query "African-American Vernacular English" --top-k 5 --with-context`.
- Execute the regression suite before submitting changes: `pytest`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; stick to type hints (`-> None`, `List[str]`) and dataclasses as in existing modules.
- Keep module-level constants uppercase (`EMBEDDING_MODEL_ID`) and CLI flags kebab-cased (`--chunk-size`).
- Prefer small, composable functions with docstrings when logic is non-obvious; use inline comments sparingly to clarify edge cases.

## Testing Guidelines
- Tests use `pytest`; mirror the pattern in `tests/test_pipeline.py` when adding scenarios (temp dirs + end-to-end subprocess checks).
- Name fixtures and helpers descriptively (`run_chunker`, `load_chunks`); keep sample documents under `examples/`.
- Validate metadata integrity (JSONL fields, spans) and numerical outputs (embedding shapes) whenever regressions are likely.
- For new RAG chat features, mock external services or gate tests behind environment variables to keep CI deterministic.

## Commit & Pull Request Guidelines
- Match the existing Conventional Commits style (`feat(search): …`, `test: …`); keep subjects ≤72 characters and written in the imperative mood.
- Include context in the body when behavior changes (inputs/outputs, new flags, migration steps).
- PRs should summarize user-visible impacts, list verification steps (commands run, tests passing), and link related issues. Attach CLI output snippets or screenshots only when they clarify UX changes.

## LM Studio & AnythingLLM Notes
- LM Studio exposes an OpenAI-compatible `/v1` API; post chat payloads with `stream=true`, the system prompt block, user turn, and `temperature` override.
- Ensure retrieved context is concatenated in the system message as AnythingLLM expects, and keep request logs scrubbed of sensitive document text before sharing.
