"""A lightweight semantic file system over a real on-disk directory.

Files are stored per-agent under a managed root (`./fs_root/<agent_id>/...`)
and simultaneously indexed into a dedicated ChromaDB collection so they can be
retrieved by natural-language query, not just by exact filename — a small
version of AIOS's LSFS idea.

How semantic the search really is depends on the active embedding backend
(kernel/memory/embeddings.py). With the default OllamaEmbedder it is genuinely
semantic: real learned embeddings from a local model, so a file can be found by
meaning even when the query shares no words with its content. If Ollama is
unreachable the kernel falls back to HashingEmbedder, whose similarity reflects
only **shared vocabulary** — search still works, but ranks by word overlap
rather than meaning. The backend actually in use is logged at startup.

Access is scoped per-agent: an agent only sees/searches its own files unless it
holds KERNEL privilege (checked via the Phase 6 AccessControl), in which case it
may target any agent's files. Cross-agent access by a USER agent raises
AccessDenied, which the syscall dispatcher surfaces as PERMISSION_DENIED.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from kernel.access_control import AccessControl, AccessDenied
from kernel.memory.embeddings import (
    DEFAULT_CHROMA_PATH,
    collection_name,
    embed_text,
    get_chroma_client,
)

DEFAULT_FS_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fs_root")
COLLECTION_NAME = "fs_files"
COSINE_SPACE_METADATA = {"hnsw:space": "cosine"}
SNIPPET_LEN = 160


class SemanticFS:
    def __init__(
        self,
        access_control: Optional[AccessControl] = None,
        fs_root: str = DEFAULT_FS_ROOT,
        chroma_path: str = DEFAULT_CHROMA_PATH,
    ) -> None:
        self.acl = access_control if access_control is not None else AccessControl()
        self.root = Path(fs_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = get_chroma_client(chroma_path)
        self.collection = self.client.get_or_create_collection(
            # namespaced by embedding backend + dimension (see embeddings.py)
            collection_name(COLLECTION_NAME), metadata=COSINE_SPACE_METADATA
        )

    # --- access + path helpers -------------------------------------------

    def _authorize(self, agent_id: str, target_agent_id: Optional[str]) -> str:
        """Resolve the file owner and enforce per-agent scoping. A USER agent
        may only touch its own files; a KERNEL agent may target any owner."""
        owner = target_agent_id or agent_id
        if owner != agent_id and not self.acl.registry.is_kernel(agent_id):
            raise AccessDenied(
                f"USER-level agent '{agent_id}' may not access files of '{owner}' "
                f"(requires KERNEL privilege)"
            )
        return owner

    @staticmethod
    def _validate_name(name: str, label: str) -> None:
        if not name or "/" in name or "\\" in name or ".." in name or name in (".", ""):
            raise ValueError(f"invalid {label}: {name!r}")

    def _agent_dir(self, owner: str) -> Path:
        self._validate_name(owner, "agent_id")
        return self.root / owner

    @staticmethod
    def _doc_id(owner: str, filename: str) -> str:
        return f"{owner}:{filename}"

    # --- operations -------------------------------------------------------

    def write_file(
        self, agent_id: str, filename: str, content: str, target_agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Write `content` to disk under the owner's directory AND index it into
        ChromaDB with {filename, path, created_at, agent_id} metadata."""
        owner = self._authorize(agent_id, target_agent_id)
        self._validate_name(filename, "filename")

        agent_dir = self._agent_dir(owner)
        agent_dir.mkdir(parents=True, exist_ok=True)
        file_path = agent_dir / filename
        file_path.write_text(content, encoding="utf-8")

        created_at = time.time()
        self.collection.upsert(
            ids=[self._doc_id(owner, filename)],
            documents=[content],
            embeddings=[embed_text(content)],
            metadatas=[
                {
                    "agent_id": owner,
                    "filename": filename,
                    "path": str(file_path),
                    "created_at": created_at,
                }
            ],
        )
        return {
            "agent_id": owner,
            "filename": filename,
            "path": str(file_path),
            "created_at": created_at,
        }

    def read_file(
        self, agent_id: str, filename: str, target_agent_id: Optional[str] = None
    ) -> str:
        """Read back the exact on-disk content of a file by filename."""
        owner = self._authorize(agent_id, target_agent_id)
        self._validate_name(filename, "filename")
        file_path = self._agent_dir(owner) / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"no such file '{filename}' for agent '{owner}'")
        return file_path.read_text(encoding="utf-8")

    def search_files(
        self,
        agent_id: str,
        query: str,
        top_k: int = 3,
        target_agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Natural-language search: embed `query` and return the owner's most
        similar files (by shared-vocabulary cosine similarity — see module
        docstring) as {filename, snippet, score}, most similar first."""
        owner = self._authorize(agent_id, target_agent_id)
        result = self.collection.query(
            query_embeddings=[embed_text(query)],
            where={"agent_id": owner},
            n_results=top_k,
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        matches: List[Dict[str, Any]] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "filename": meta["filename"],
                    "snippet": doc[:SNIPPET_LEN],
                    # cosine distance -> similarity; higher is more relevant
                    "score": round(1.0 - distance, 4),
                }
            )
        return matches

    def list_files(
        self, agent_id: str, target_agent_id: Optional[str] = None
    ) -> List[str]:
        """List all filenames the owner has written."""
        owner = self._authorize(agent_id, target_agent_id)
        agent_dir = self._agent_dir(owner)
        if not agent_dir.is_dir():
            return []
        return sorted(p.name for p in agent_dir.iterdir() if p.is_file())
