"""
Helpers for loading RAG artefacts from disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_chunk_records(path: Path) -> List[dict]:
    """Load chunk metadata records from a JSON Lines file."""
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_doc_records(path: Path) -> Dict[str, dict]:
    """Load document metadata records keyed by doc_id."""
    lookup: Dict[str, dict] = {}
    if not path.exists():
        return lookup
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


def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from a .npy file."""
    return np.load(path)
