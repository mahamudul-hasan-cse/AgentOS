import math
import re

import pytest

from kernel.memory.embeddings import (
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
    HashingEmbedder,
    OllamaEmbedder,
    _load_config,
    build_embedder,
    collection_name,
    set_embedder,
)

# Opt this module out of the suite-wide hashing pin (see tests/conftest.py):
# these tests are ABOUT the embedding backends, so they select their own and
# are the only tests permitted to talk to Ollama.
pytestmark = pytest.mark.real_embeddings

_TOKENS = re.compile(r"[a-z0-9]+")

# A semantically-related pair sharing ZERO tokens (not even stopwords like
# "the"/"on"/"a"). The zero-overlap property is what makes the semantic test
# meaningful: the hashing embedder scores these 0.0 because it only sees shared
# vocabulary, so only real learned embeddings can rank them above an unrelated
# sentence. test_related_pair_shares_no_words guards this invariant.
RELATED_A = "a cat sat on the carpet"
RELATED_B = "felines rest upon rugs"
UNRELATED = "quarterly earnings exceeded analyst expectations"


def cosine(a, b):
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0


def ollama_embedder_or_none():
    """The Ollama embedder if it is genuinely usable, else None.

    Resolves host/model from kernel/config.yaml exactly as the kernel does, so
    the skip decision reflects the real environment (a custom or unreachable
    host is honoured). Probing directly rather than via build_embedder() means
    these capability checks still run when config selects `backend: hashing`.
    """
    cfg = _load_config()
    embed_cfg = (cfg.get("embeddings") or {}) if isinstance(cfg, dict) else {}
    ollama_cfg = (cfg.get("ollama") or {}) if isinstance(cfg, dict) else {}
    embedder = OllamaEmbedder(
        model=embed_cfg.get("model", DEFAULT_OLLAMA_EMBED_MODEL),
        host=embed_cfg.get("host") or ollama_cfg.get("host") or DEFAULT_OLLAMA_HOST,
    )
    return embedder if embedder.is_available() else None


requires_ollama = pytest.mark.skipif(
    ollama_embedder_or_none() is None,
    reason=(
        "Ollama embeddings unavailable — start Ollama and "
        f"`ollama pull {DEFAULT_OLLAMA_EMBED_MODEL}` to run the semantic checks"
    ),
)


# (the active embedder is reset after every test by the autouse fixture in
# tests/conftest.py, so no local teardown is needed here)


# --- hashing fallback (always runs, no setup required) --------------------


def test_hashing_embedder_is_always_available_and_deterministic():
    embedder = HashingEmbedder()
    assert embedder.is_available() is True
    first = embedder.embed("the scheduler dispatches agent processes")
    second = embedder.embed("the scheduler dispatches agent processes")
    assert first == second  # stable across calls (and across processes: crc32)
    assert len(first) == embedder.dimension == 256
    assert abs(math.sqrt(sum(v * v for v in first)) - 1.0) < 1e-9  # L2-normalized


def test_falls_back_to_hashing_when_ollama_unreachable():
    # port 1 is not listening: stands in for "Ollama isn't running"
    config = {
        "embeddings": {
            "backend": "ollama",
            "model": "nomic-embed-text",
            "host": "http://127.0.0.1:1",
        }
    }
    embedder = build_embedder(config)
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.semantic is False
    # and it actually works, so a fresh clone/offline test run is unaffected
    assert len(embedder.embed("hello world")) == 256


def test_hashing_backend_can_be_selected_explicitly():
    embedder = build_embedder({"embeddings": {"backend": "hashing"}})
    assert isinstance(embedder, HashingEmbedder)


def test_unknown_backend_falls_back_to_hashing():
    embedder = build_embedder({"embeddings": {"backend": "not-a-backend"}})
    assert isinstance(embedder, HashingEmbedder)


def test_hashing_embedder_cannot_relate_the_zero_overlap_pair():
    """Documents *why* the semantic test matters: the hashing embedder scores
    the related pair identically to the unrelated one (both 0), because it only
    sees shared vocabulary. It would fail the semantic assertion below."""
    embedder = HashingEmbedder()
    related = cosine(embedder.embed(RELATED_A), embedder.embed(RELATED_B))
    unrelated = cosine(embedder.embed(RELATED_A), embedder.embed(UNRELATED))
    assert related == pytest.approx(0.0)
    assert unrelated == pytest.approx(0.0)
    assert not related > unrelated  # cannot distinguish -> not semantic


def test_related_pair_shares_no_words():
    """Guards the invariant the semantic test depends on."""
    assert not (set(_TOKENS.findall(RELATED_A)) & set(_TOKENS.findall(RELATED_B)))


# --- collection namespacing (dimension-lock safety) ----------------------


def test_collections_are_namespaced_by_backend_and_dimension():
    set_embedder(HashingEmbedder())
    hashing_name = collection_name("ram_pages")
    assert hashing_name == "ram_pages__hashing_256"

    class FakeOllama(HashingEmbedder):
        name = "nomic-embed-text"
        semantic = True

        @property
        def dimension(self):
            return 768

    set_embedder(FakeOllama())
    ollama_name = collection_name("ram_pages")
    assert ollama_name == "ram_pages__nomic_embed_text_768"
    # different backends never share a (dimension-locked) collection
    assert ollama_name != hashing_name


# --- real semantic embeddings (skipped when Ollama isn't available) -------


@requires_ollama
def test_ollama_embedder_returns_correctly_dimensioned_vectors():
    embedder = ollama_embedder_or_none()
    vector = embedder.embed("the scheduler dispatches agent processes")
    assert len(vector) == embedder.dimension
    assert embedder.dimension > 256, "expected a real model's dimensionality"
    assert all(isinstance(v, float) for v in vector[:10])
    assert embedder.semantic is True


@requires_ollama
def test_ollama_embeddings_are_genuinely_semantic():
    """The proof: two related sentences sharing NO words must score higher than
    two unrelated ones. The hashing embedder scores both 0.0 and fails this."""
    embedder = ollama_embedder_or_none()
    a, b, c = (embedder.embed(t) for t in (RELATED_A, RELATED_B, UNRELATED))

    related = cosine(a, b)
    unrelated = cosine(a, c)
    assert related > unrelated, (
        f"expected '{RELATED_A}' ~ '{RELATED_B}' ({related:.4f}) to beat "
        f"'{UNRELATED}' ({unrelated:.4f})"
    )
    # a clear margin, not a coin flip
    assert related - unrelated > 0.1


@requires_ollama
def test_build_embedder_prefers_ollama_when_available():
    embedder = build_embedder(
        {"embeddings": {"backend": "ollama", "model": DEFAULT_OLLAMA_EMBED_MODEL}}
    )
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.semantic is True
