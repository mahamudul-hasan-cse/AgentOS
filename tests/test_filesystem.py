import asyncio

import pytest

from kernel.access_control import AccessControl, AccessDenied, AgentPrivilege
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


def run(coro):
    return asyncio.run(coro)


# Three files on distinct topics. Filenames are deliberately generic ("note_*")
# so a query can only match via file *content*, never via the filename.
FILES = {
    "note_a.txt": (
        "Plants perform photosynthesis, converting sunlight, water and carbon "
        "dioxide into glucose and oxygen using chlorophyll in their leaves."
    ),
    "note_b.txt": (
        "Round robin CPU scheduling gives each process a fixed time quantum "
        "before preempting it and switching to the next process in the queue."
    ),
    "note_c.txt": (
        "Ocean tides are caused by the gravitational pull of the moon and sun "
        "on the earth's water, producing regular high and low sea levels."
    ),
}


@pytest.fixture
def fs(tmp_path):
    from kernel.filesystem import SemanticFS

    acl = AccessControl()
    return SemanticFS(
        access_control=acl,
        fs_root=str(tmp_path / "fs_root"),
        chroma_path=str(tmp_path / "chroma_db"),
    )


def test_write_then_read_exact_content(fs):
    for name, content in FILES.items():
        fs.write_file("alice", name, content)

    assert fs.read_file("alice", "note_b.txt") == FILES["note_b.txt"]
    assert set(fs.list_files("alice")) == set(FILES.keys())


def test_natural_language_search_ranks_by_content_not_filename(fs):
    for name, content in FILES.items():
        fs.write_file("alice", name, content)

    # a natural-language question about photosynthesis. It shares NO words with
    # the filename "note_a.txt", but overlaps the file's *content* vocabulary.
    results = fs.search_files(
        "alice", "how do plants use sunlight and chlorophyll to make glucose?", top_k=3
    )

    assert results[0]["filename"] == "note_a.txt"
    assert "score" in results[0] and "snippet" in results[0]
    # the relevant file scores strictly higher than the unrelated ones
    assert results[0]["score"] > results[-1]["score"]


def test_user_cannot_access_another_agents_files_but_kernel_can(fs):
    fs.write_file("victim", "secret.txt", "the launch codes are 0000")

    # a USER-level agent cannot read or list another agent's files
    with pytest.raises(AccessDenied):
        fs.read_file("attacker", "secret.txt", target_agent_id="victim")
    with pytest.raises(AccessDenied):
        fs.list_files("attacker", target_agent_id="victim")
    with pytest.raises(AccessDenied):
        fs.search_files("attacker", "launch codes", target_agent_id="victim")

    # a KERNEL-level agent can
    fs.acl.registry.register("root", AgentPrivilege.KERNEL)
    assert fs.read_file("root", "secret.txt", target_agent_id="victim") == "the launch codes are 0000"
    assert fs.list_files("root", target_agent_id="victim") == ["secret.txt"]


def test_file_syscalls_flow_through_dispatcher_and_are_logged(tmp_path):
    from kernel.filesystem import SemanticFS

    acl = AccessControl()
    fs = SemanticFS(
        access_control=acl,
        fs_root=str(tmp_path / "fs_root"),
        chroma_path=str(tmp_path / "chroma_db"),
    )
    disp = SyscallDispatcher(
        access_control=acl,
        filesystem=fs,
        page_manager=PageManager(chroma_path=str(tmp_path / "mem_chroma")),
        driver_registry={"fake": FakeDriver},
    )

    async def scenario():
        w = await disp.dispatch(
            "alice", SyscallType.FILE_WRITE, filename="note_a.txt", content=FILES["note_a.txt"]
        )
        r = await disp.dispatch("alice", SyscallType.FILE_READ, filename="note_a.txt")
        s = await disp.dispatch(
            "alice", SyscallType.FILE_SEARCH, query="photosynthesis in plant leaves"
        )
        return w, r, s

    w, r, s = run(scenario())

    assert w.status == SyscallStatus.SUCCESS
    assert r.status == SyscallStatus.SUCCESS
    assert r.result["content"] == FILES["note_a.txt"]
    assert s.status == SyscallStatus.SUCCESS
    assert s.result["results"][0]["filename"] == "note_a.txt"

    logged_types = [entry.type for entry in disp.log]
    assert SyscallType.FILE_WRITE in logged_types
    assert SyscallType.FILE_READ in logged_types
    assert SyscallType.FILE_SEARCH in logged_types


def test_cross_agent_file_read_via_dispatcher_is_permission_denied(tmp_path):
    from kernel.filesystem import SemanticFS

    acl = AccessControl()
    fs = SemanticFS(
        access_control=acl,
        fs_root=str(tmp_path / "fs_root"),
        chroma_path=str(tmp_path / "chroma_db"),
    )
    disp = SyscallDispatcher(
        access_control=acl,
        filesystem=fs,
        page_manager=PageManager(chroma_path=str(tmp_path / "mem_chroma")),
        driver_registry={"fake": FakeDriver},
    )

    async def scenario():
        await disp.dispatch("victim", SyscallType.FILE_WRITE, filename="s.txt", content="secret")
        return await disp.dispatch(
            "attacker", SyscallType.FILE_READ, filename="s.txt", target_agent_id="victim"
        )

    syscall = run(scenario())
    assert syscall.status == SyscallStatus.PERMISSION_DENIED
    assert syscall.result["error_type"] == "PermissionDenied"
