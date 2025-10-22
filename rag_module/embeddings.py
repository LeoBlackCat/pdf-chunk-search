"""
Embedding utilities powered by MLX.

Exposes a single helper `embed_texts` that returns L2-normalized NumPy vectors.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Union

import numpy as np

from mlx_embeddings import load, generate


def embed_texts(
    texts: Union[str, List[str]],
    model_id: str = "mlx-community/all-MiniLM-L6-v2-4bit",
) -> np.ndarray:
    """
    Convert text(s) to normalized embedding vectors (NumPy float32).

    Args:
        texts: A single string or list of strings.
        model_id: Which MLX embeddings model to use.

    Returns:
        Array of shape (N, D) with L2-normalized embeddings.
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model, tok = _get_model(model_id)
    out = generate(model, tok, texts=texts)
    vectors = np.array(out.text_embeds, dtype=np.float32)
    return vectors


@lru_cache(maxsize=2)
def _get_model(model_id: str):
    model, tok = load(model_id)
    return model, tok


__all__ = ["embed_texts"]
