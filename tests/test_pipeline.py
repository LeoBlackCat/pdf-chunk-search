import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_chunker(tmp_path: Path, inputs: list[str]) -> tuple[Path, Path, Path]:
    output_prefix = tmp_path / "test_output"
    cmd = [
        sys.executable,
        "pdf_chunker.py",
        str(output_prefix),
        *[str(Path(inp)) for inp in inputs],
        "--chunk-size",
        "128",
    ]
    subprocess.run(cmd, check=True)

    docs_path = output_prefix.with_name(f"{output_prefix.name}_docs.jsonl")
    chunks_path = output_prefix.with_name(f"{output_prefix.name}_chunks.jsonl")
    embeddings_path = output_prefix.with_name(f"{output_prefix.name}_embeddings.npy")

    assert docs_path.exists(), "docs.jsonl not created"
    assert chunks_path.exists(), "chunks.jsonl not created"
    assert embeddings_path.exists(), "embeddings.npy not created"
    return docs_path, chunks_path, embeddings_path


def load_chunks(chunks_path: Path) -> list[dict]:
    lines = chunks_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_pipeline_with_search(tmp_path: Path):
    examples_dir = Path("examples")
    sample_pdf = examples_dir / "bk_tcco_000301.pdf"
    assert sample_pdf.exists(), "Sample PDF is missing in examples/"

    docs_path, chunks_path, embeddings_path = run_chunker(tmp_path, [sample_pdf])

    chunk_records = load_chunks(chunks_path)
    assert chunk_records, "No chunks produced"

    embeddings = np.load(embeddings_path)
    assert embeddings.shape[0] == len(chunk_records), "Embeddings do not align with chunk count"

    sample_chunk = chunk_records[0]
    assert "page" in sample_chunk and "pages" in sample_chunk and "page_end" in sample_chunk
    assert isinstance(sample_chunk["pages"], list)
    assert sample_chunk["pages"], "Chunk pages list is empty"
    assert sample_chunk["page"] == sample_chunk["pages"][0]
    assert sample_chunk["page_end"] == sample_chunk["pages"][-1]

    search_cmd = [
        sys.executable,
        "chunk_search.py",
        "--chunks",
        str(chunks_path),
        "--embeddings",
        str(embeddings_path),
        "--query",
        "African-American Vernacular",
        "--top-k",
        "10",
        "--with-context",
    ]
    result = subprocess.run(search_cmd, check=True, capture_output=True, text=True)
    output_lines = result.stdout.strip().splitlines()
    assert output_lines, "Search produced no output"

    # Find first result block and verify ordering of Prev/Text/Next
    prev_index = next((i for i, line in enumerate(output_lines) if line.startswith("  Prev  :")), None)
    text_index = next((i for i, line in enumerate(output_lines) if line.startswith("  Text  :")), None)
    next_index = next((i for i, line in enumerate(output_lines) if line.startswith("  Next  :")), None)

    assert text_index is not None, "Search result missing Text line"
    if prev_index is not None:
        assert prev_index < text_index, "Prev line should appear before Text"
    if next_index is not None:
        assert text_index < next_index, "Text line should appear before Next"

    joined_output = "\n".join(output_lines).lower()
    assert any(term in joined_output for term in ["african", "vernacular"]), "Query signal not present in output"
