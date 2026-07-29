"""Shared pytest configuration.

The kernel defaults to real Ollama embeddings, but letting every memory and
filesystem test issue real HTTP embedding calls made the suite roughly 17x
slower (~10s -> ~175s) while testing nothing those tests are actually about:
they exercise paging, eviction, page-fault handling, ACL scoping and syscall
plumbing — not embedding quality.

So the bulk suite is pinned to the deterministic, offline HashingEmbedder. That
keeps it fast and keeps a fresh clone runnable with zero setup and no network.

Tests genuinely about embedding backends opt out with the `real_embeddings`
marker and select their own backend; tests/test_embeddings.py applies it
module-wide (`pytestmark`), so it is the one place that may talk to Ollama.

ChromaDB isolation
------------------
`PageManager` and `SemanticFS` both default `chroma_path` to the repo's real
`./chroma_db`. Measured over one full run, that meant **43 clients opened on the
repo's own database** (42 of them `SemanticFS`), so tests were reading and
writing a single shared store that persists across runs — leftover
`ram_pages__hashing_256` / `swap_pages__hashing_256` documents were found in it.
Shared mutable state between tests, and between a test run and the dev server.

The autouse fixture below closes that off. It is deliberately autouse rather
than opt-in: the whole failure mode is a test that *forgot* to pass a path.

What NOT to do here
-------------------
Do **not** call `release_chroma_clients()` (ChromaDB's
`SharedSystemClient.clear_system_cache()`) between tests. It is tempting — the
client cache does grow to ~65 over a run, and with clients live `shutil.rmtree`
on a chroma directory is a silent no-op on Windows — but it drops Systems out
from under client and collection objects that are still referenced, and
ChromaDB's shared Rust-side state does not survive it. Measured over full-suite
runs:

    no cache clearing          0 failures / 25 runs
    clear after every test     1 failure  / 6 runs
      -> chromadb.errors.InternalError: Error creating hnsw segment reader:
         Nothing found on disk    (on a query, in a *freshly built* PageManager)

So clearing made the suite less reliable, not more. The benchmark harness can
call it safely because it does so at a point where it is genuinely finished with
every client for that run; a test suite has no such point until the session
ends. Directory cleanup is left to pytest's own `tmp_path` garbage collection.
"""

import pytest

from kernel.memory import embeddings as _embeddings
from kernel.memory import page_manager as _page_manager
from kernel.memory.embeddings import HashingEmbedder, set_embedder

try:  # optional: the filesystem module pulls in its own deps
    from kernel.filesystem import semantic_fs as _semantic_fs
except Exception:  # noqa: BLE001 — isolation must not depend on it importing
    _semantic_fs = None

# Modules that did `from ...embeddings import get_chroma_client` at import time
# and therefore hold their own binding that must be patched individually.
_CLIENT_CONSUMERS = [m for m in (_embeddings, _page_manager, _semantic_fs) if m is not None]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_embeddings: test manages its own embedding backend (may use Ollama)",
    )


@pytest.fixture(autouse=True)
def isolate_default_chroma_path(tmp_path, monkeypatch):
    """Redirect the repo's default ChromaDB path to a per-test directory.

    `chroma_path: str = DEFAULT_CHROMA_PATH` is bound as a default argument at
    import time, so reassigning the module constant would do nothing. Instead we
    intercept `get_chroma_client` — which is called at runtime with that already
    bound path — and reroute only the repo default, leaving tests that pass an
    explicit path untouched.
    """
    redirect = str(tmp_path / "default_chroma")
    real_get_client = _embeddings.get_chroma_client
    default_path = str(_embeddings.DEFAULT_CHROMA_PATH)

    def routed(path: str = _embeddings.DEFAULT_CHROMA_PATH):
        if str(path) == default_path:
            path = redirect
        return real_get_client(path)

    for module in _CLIENT_CONSUMERS:
        if hasattr(module, "get_chroma_client"):
            monkeypatch.setattr(module, "get_chroma_client", routed)
    yield


@pytest.fixture(autouse=True)
def pin_embedding_backend(request):
    """Pin the active embedder to the offline hashing backend, unless the test
    opts out via @pytest.mark.real_embeddings. Always reset afterwards so no
    backend choice leaks between tests."""
    if request.node.get_closest_marker("real_embeddings") is None:
        set_embedder(HashingEmbedder())
    yield
    set_embedder(None)
