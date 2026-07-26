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
"""

import pytest

from kernel.memory.embeddings import HashingEmbedder, set_embedder


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_embeddings: test manages its own embedding backend (may use Ollama)",
    )


@pytest.fixture(autouse=True)
def pin_embedding_backend(request):
    """Pin the active embedder to the offline hashing backend, unless the test
    opts out via @pytest.mark.real_embeddings. Always reset afterwards so no
    backend choice leaks between tests."""
    if request.node.get_closest_marker("real_embeddings") is None:
        set_embedder(HashingEmbedder())
    yield
    set_embedder(None)
