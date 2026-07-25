"""Text embedding and ChromaDB client helpers shared by the memory manager.

Embeddings are produced with a deterministic hashing-trick bag-of-words
vectorizer rather than a downloaded ML model: it needs no network access or
bundled weights, and is fully reproducible across process restarts (unlike
Python's built-in `hash()`, which is randomized per-process), which matters
because embeddings computed today must still line up with a ChromaDB swap
store written in a previous run.
"""

from __future__ import annotations

import math
import re
import zlib
from pathlib import Path
from typing import List

import chromadb

DEFAULT_CHROMA_PATH = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
EMBEDDING_DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def embed_text(text: str) -> List[float]:
    """Deterministic, L2-normalized bag-of-words hashing embedding, so cosine
    similarity between two vectors reflects shared vocabulary between texts."""
    vector = [0.0] * EMBEDDING_DIM
    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        bucket = zlib.crc32(token.encode("utf-8")) % EMBEDDING_DIM
        vector[bucket] += 1.0

    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when a caller doesn't supply
    an explicit token_count."""
    return max(1, len(text) // 4)


def get_chroma_client(path: str = DEFAULT_CHROMA_PATH) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=path)
