"""Pluggable text-embedding backends, plus ChromaDB client helpers.

Mirrors the LLMDriver pattern from Phase 1: one `Embedder` interface with
several implementations, selected from `kernel/config.yaml` and falling back
automatically when the preferred backend is unreachable.

- `OllamaEmbedder` (default) calls a local Ollama embedding model
  (nomic-embed-text / mxbai-embed-large) via /api/embeddings. Real learned
  embeddings: local, free, no API key, no cloud dependency. This is what makes
  Semantic-LRU and the semantic file system genuinely *semantic* — similarity
  reflects meaning, so "a cat sat on the carpet" matches "the feline rested on
  the rug" despite sharing no words.
- `HashingEmbedder` is the offline fallback: a deterministic hashing-trick
  bag-of-words vectorizer needing no model, network, or setup, so a fresh clone
  and the test suite work with zero configuration. Its similarity only reflects
  SHARED VOCABULARY, not meaning — approximate, not truly semantic.

Which backend is active is logged at startup (see `get_embedder`) so it is
never ambiguous which one produced a given result.

DIMENSION NAMESPACING
---------------------
Backends produce different-dimension vectors (nomic-embed-text: 768, hashing:
256) and a ChromaDB collection is dimension-locked once created. Rather than
erroring and telling the user to delete ./chroma_db, collection names are
NAMESPACED by backend+dimension via `collection_name()`, e.g.
`ram_pages__ollama_nomic_embed_text_768` vs `ram_pages__hashing_256`.

That choice is deliberate: because fallback is automatic, the active backend
can change between runs with no user action at all (Ollama simply not running).
Hard-erroring there would turn a graceful degradation into a crash and defeat
the point of having a fallback. Namespacing is non-destructive — each backend's
store coexists on disk and switching back finds the old data intact. The
trade-off: data written under one backend is not visible (or migrated) under
another; re-index if you switch permanently.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import chromadb
import requests
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

HASHING_DIM = 256
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_TIMEOUT = 30
OLLAMA_PROBE_TIMEOUT = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_UNSAFE_NAME_RE = re.compile(r"[^a-z0-9]+")


def _load_config() -> dict:
    """Read kernel/config.yaml. Deliberately a local reader rather than reusing
    kernel.drivers.base.load_config: importing that would pull the whole LLM SDK
    stack (groq, google-generativeai) into the memory subsystem just to read a
    YAML file."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001 — a broken config must not break imports
        return {}


class Embedder(ABC):
    """One embedding backend. `name`/`dimension` identify it well enough to
    namespace a ChromaDB collection."""

    name: str = "base"
    semantic: bool = False

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, text: str) -> List[float]: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @property
    def namespace(self) -> str:
        """Collection-name suffix identifying this backend + its dimension."""
        raw = f"{self.name}_{self.dimension}"
        cleaned = _UNSAFE_NAME_RE.sub("_", raw.lower()).strip("_")
        return cleaned

    def describe(self) -> str:
        kind = "real learned embeddings" if self.semantic else "approximate, vocabulary-overlap only"
        return f"{type(self).__name__} ({self.name}, {self.dimension}-dim, {kind})"


class HashingEmbedder(Embedder):
    """Deterministic hashing-trick bag-of-words vectorizer.

    Needs no model, network, or API key, so it always works — but similarity
    only reflects shared vocabulary, NOT meaning. `zlib.crc32` is used rather
    than Python's built-in `hash()` because the latter is randomized per process,
    which would break embeddings written to ChromaDB by an earlier run.
    """

    name = "hashing"
    semantic = False

    def __init__(self, dimension: int = HASHING_DIM) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self) -> bool:
        return True

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            vector[zlib.crc32(token.encode("utf-8")) % self._dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class OllamaEmbedder(Embedder):
    """Real learned embeddings from a local Ollama embedding model."""

    semantic = True

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
    ) -> None:
        self.name = model
        self.model = model
        self.host = host.rstrip("/")
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # discovered by embedding a probe string; cached for the process
            self._dimension = len(self._request("dimension probe", OLLAMA_PROBE_TIMEOUT))
        return self._dimension

    def _request(self, text: str, timeout: int) -> List[float]:
        response = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"Ollama embeddings HTTP {response.status_code}: {response.text[:200]}"
            )
        embedding = response.json().get("embedding")
        if not embedding:
            raise RuntimeError(f"Ollama returned no embedding for model '{self.model}'")
        return [float(x) for x in embedding]

    def is_available(self) -> bool:
        """True only if Ollama responds AND the model can actually embed."""
        try:
            self.dimension  # probes and caches
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ollama embedder unavailable: %s", exc)
            return False

    def embed(self, text: str) -> List[float]:
        return self._request(text, OLLAMA_TIMEOUT)


def build_embedder(config: Optional[dict] = None) -> Embedder:
    """Select a backend from config, falling back to hashing when the preferred
    one is unreachable — the same graceful-degradation pattern as the LLM
    drivers. Always logs the backend actually in use."""
    cfg = config if config is not None else _load_config()
    embed_cfg = (cfg.get("embeddings") or {}) if isinstance(cfg, dict) else {}
    ollama_cfg = (cfg.get("ollama") or {}) if isinstance(cfg, dict) else {}

    backend = str(embed_cfg.get("backend", "ollama")).lower()
    model = embed_cfg.get("model", DEFAULT_OLLAMA_EMBED_MODEL)
    host = embed_cfg.get("host") or ollama_cfg.get("host") or DEFAULT_OLLAMA_HOST

    if backend == "hashing":
        embedder = HashingEmbedder()
        logger.info("embeddings: using %s (configured)", embedder.describe())
        return embedder

    if backend != "ollama":
        logger.warning(
            "embeddings: unknown backend '%s'; expected 'ollama' or 'hashing'. "
            "Falling back to hashing.",
            backend,
        )
        embedder = HashingEmbedder()
        logger.info("embeddings: using %s", embedder.describe())
        return embedder

    candidate = OllamaEmbedder(model=model, host=host)
    if candidate.is_available():
        logger.info("embeddings: using %s", candidate.describe())
        return candidate

    fallback = HashingEmbedder()
    logger.warning(
        "embeddings: Ollama unreachable at %s (model '%s') - falling back to %s. "
        "Similarity will reflect shared vocabulary, not meaning. "
        "Start Ollama and `ollama pull %s` for real semantic embeddings.",
        host,
        model,
        fallback.describe(),
        model,
    )
    return fallback


_active_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """The process-wide active embedder (selected once, on first use)."""
    global _active_embedder
    if _active_embedder is None:
        _active_embedder = build_embedder()
    return _active_embedder


def set_embedder(embedder: Optional[Embedder]) -> None:
    """Override the active embedder (used by tests). Pass None to re-select."""
    global _active_embedder
    _active_embedder = embedder


def embed_text(text: str) -> List[float]:
    """Embed text with the active backend. Kept as a module-level function so
    existing call sites are unchanged."""
    return get_embedder().embed(text)


def collection_name(base: str) -> str:
    """Namespace a ChromaDB collection by the active backend + dimension, so
    switching backends can never hit a dimension-locked collection."""
    return f"{base}__{get_embedder().namespace}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when a caller doesn't supply
    an explicit token_count."""
    return max(1, len(text) // 4)


def get_chroma_client(path: str = DEFAULT_CHROMA_PATH) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=path)


def release_chroma_clients() -> None:
    """Drop ChromaDB's process-global cache of live clients.

    Every `PersistentClient` stays registered — holding sqlite and HNSW file
    handles — until released, so any process that opens many of them (a
    multi-seed benchmark sweep, or a full test run) accumulates handles for its
    whole lifetime. Two consequences seen in practice:

    - intermittent "Error creating hnsw segment reader: Nothing found on disk"
      part-way through a long sweep;
    - on Windows, an open handle makes `shutil.rmtree` a **silent no-op**, so
      directories meant to be temporary are never actually deleted.

    Call this once a client is finished with, and always *before* removing its
    directory. Safe: it only releases clients, it never touches files.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001 — cleanup must never mask the caller's result
        pass
