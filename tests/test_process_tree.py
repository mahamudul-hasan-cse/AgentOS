import asyncio

from kernel.access_control import AccessControl, AgentPrivilege
from kernel.drivers.base import LLMDriver
from kernel.memory import PageManager
from kernel.scheduler import INIT_PID, ZOMBIE, Process
from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType


class FakeDriver(LLMDriver):
    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"echo: {prompt}"


def run(coro):
    return asyncio.run(coro)


def make_dispatcher(tmp_path, **kwargs) -> SyscallDispatcher:
    pm = PageManager(chroma_path=str(tmp_path / "chroma_db"))
    disp = SyscallDispatcher(
        page_manager=pm, driver_registry={"fake": FakeDriver}, record_state=False, **kwargs
    )
    disp.scheduler.ensure_init()
    return disp


def states(disp) -> dict:
    return {p.pid: p.state for p in disp.scheduler.queue}


# --- 1. spawning ---------------------------------------------------------


def test_spawned_child_has_correct_parent_pid(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        result = await disp.dispatch("worker", SyscallType.SPAWN_AGENT, pid="child-a")
        return disp, result

    disp, result = run(scenario())
    assert result.status == SyscallStatus.SUCCESS
    assert result.result["pid"] == "child-a"
    assert result.result["parent_pid"] == "worker"

    child = disp.scheduler.get("child-a")
    assert child is not None
    assert child.parent_pid == "worker"
    assert child.state == "ready"


def test_spawn_generates_a_pid_when_not_given(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        a = await disp.dispatch("w", SyscallType.SPAWN_AGENT)
        b = await disp.dispatch("w", SyscallType.SPAWN_AGENT)
        return a, b

    a, b = run(scenario())
    assert a.result["pid"] != b.result["pid"]  # unique


def test_child_gets_its_own_default_quota_not_the_parents(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        # give the parent a non-default quota
        disp.quota_manager.set_quota("boss", max_pages=99, max_calls_per_minute=99)
        await disp.dispatch("boss", SyscallType.SPAWN_AGENT, pid="minion")
        return disp

    disp = run(scenario())
    parent = disp.quota_manager.usage("boss")
    child = disp.quota_manager.usage("minion")
    assert parent["max_pages"] == 99
    assert child["max_pages"] == disp.quota_manager.default_max_pages
    assert child["max_pages"] != parent["max_pages"]


# --- 2. privilege inheritance / escalation -------------------------------


def test_child_inherits_parent_privilege(tmp_path):
    async def scenario():
        acl = AccessControl()
        acl.registry.register("root", AgentPrivilege.KERNEL)
        disp = make_dispatcher(tmp_path, access_control=acl)
        kernel_child = await disp.dispatch("root", SyscallType.SPAWN_AGENT, pid="kc")
        user_child = await disp.dispatch("plain", SyscallType.SPAWN_AGENT, pid="uc")
        return disp, kernel_child, user_child

    disp, kernel_child, user_child = run(scenario())
    assert kernel_child.result["privilege"] == "kernel"
    assert user_child.result["privilege"] == "user"
    assert disp.acl.registry.is_kernel("kc") is True
    assert disp.acl.registry.is_kernel("uc") is False


def test_user_cannot_spawn_kernel_child(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        denied = await disp.dispatch(
            "plain", SyscallType.SPAWN_AGENT, pid="sneaky", privilege="kernel"
        )
        # a KERNEL caller may of course do it
        disp.acl.registry.register("root", AgentPrivilege.KERNEL)
        allowed = await disp.dispatch(
            "root", SyscallType.SPAWN_AGENT, pid="legit", privilege="kernel"
        )
        return disp, denied, allowed

    disp, denied, allowed = run(scenario())
    assert denied.status == SyscallStatus.PERMISSION_DENIED
    assert disp.scheduler.get("sneaky") is None  # not created
    assert disp.acl.registry.is_kernel("sneaky") is False

    assert allowed.status == SyscallStatus.SUCCESS
    assert disp.acl.registry.is_kernel("legit") is True


# --- 3. zombies + WAIT ---------------------------------------------------


def test_terminated_child_becomes_zombie_until_parent_reaps_it(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        disp.scheduler.spawn("parent", parent_pid=INIT_PID)
        await disp.dispatch("parent", SyscallType.SPAWN_AGENT, pid="kid")

        term = await disp.dispatch("kid", SyscallType.TERMINATE_AGENT, pid="kid", exit_status=7)
        zombie_state = states(disp)

        reap = await disp.dispatch("parent", SyscallType.WAIT, pid="kid")
        after = states(disp)
        return term, zombie_state, reap, after

    term, zombie_state, reap, after = run(scenario())

    # still present, as a zombie holding its exit status
    assert term.result["zombie"] is True
    assert zombie_state["kid"] == ZOMBIE

    assert reap.status == SyscallStatus.SUCCESS
    assert reap.result["reaped"] is True
    assert reap.result["exit_status"] == 7
    # only now is it gone
    assert "kid" not in after


def test_wait_returns_nothing_for_a_non_parent(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        disp.scheduler.spawn("parent", parent_pid=INIT_PID)
        await disp.dispatch("parent", SyscallType.SPAWN_AGENT, pid="kid")
        await disp.dispatch("kid", SyscallType.TERMINATE_AGENT, pid="kid")
        # a stranger cannot reap someone else's child
        stranger = await disp.dispatch("stranger", SyscallType.WAIT, pid="kid")
        return disp, stranger

    disp, stranger = run(scenario())
    assert stranger.result["reaped"] is False
    assert disp.scheduler.get("kid").state == ZOMBIE  # still there


# --- 4. orphan reparenting ------------------------------------------------


def test_orphans_are_reparented_to_init_not_killed(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        disp.scheduler.spawn("parent", parent_pid=INIT_PID)
        await disp.dispatch("parent", SyscallType.SPAWN_AGENT, pid="kid1")
        await disp.dispatch("parent", SyscallType.SPAWN_AGENT, pid="kid2")

        disp.acl.registry.register("root", AgentPrivilege.KERNEL)
        term = await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid="parent")
        return disp, term

    disp, term = run(scenario())

    # the children SURVIVED
    assert disp.scheduler.get("kid1") is not None
    assert disp.scheduler.get("kid2") is not None
    assert disp.scheduler.get("kid1").state != ZOMBIE
    # ...adopted by init
    assert disp.scheduler.get("kid1").parent_pid == INIT_PID
    assert disp.scheduler.get("kid2").parent_pid == INIT_PID
    assert set(term.result["reparented_to_init"]) == {"kid1", "kid2"}


# --- 5. kill_tree ---------------------------------------------------------


def test_kill_tree_terminates_the_whole_subtree(tmp_path):
    async def scenario():
        acl = AccessControl()
        acl.registry.register("root", AgentPrivilege.KERNEL)
        disp = make_dispatcher(tmp_path, access_control=acl)
        disp.scheduler.spawn("top", parent_pid=INIT_PID)
        await disp.dispatch("top", SyscallType.SPAWN_AGENT, pid="mid")
        await disp.dispatch("mid", SyscallType.SPAWN_AGENT, pid="leaf1")
        await disp.dispatch("mid", SyscallType.SPAWN_AGENT, pid="leaf2")
        # an unrelated process that must NOT be touched
        disp.scheduler.spawn("bystander", parent_pid=INIT_PID)

        result = await disp.dispatch(
            "root", SyscallType.TERMINATE_AGENT, pid="mid", tree=True
        )
        return disp, result

    disp, result = run(scenario())
    assert result.status == SyscallStatus.SUCCESS
    assert set(result.result["killed"]) == {"mid", "leaf1", "leaf2"}

    for pid in ("leaf1", "leaf2"):
        assert disp.scheduler.get(pid) is None
    # 'mid' had a live parent ('top'), so it lingers as a zombie for top to reap
    mid = disp.scheduler.get("mid")
    assert mid is None or mid.state == ZOMBIE
    # untouched
    assert disp.scheduler.get("top") is not None
    assert disp.scheduler.get("bystander") is not None


def test_plain_terminate_does_not_cascade(tmp_path):
    """The contrast with kill_tree: children survive a normal terminate."""

    async def scenario():
        acl = AccessControl()
        acl.registry.register("root", AgentPrivilege.KERNEL)
        disp = make_dispatcher(tmp_path, access_control=acl)
        disp.scheduler.spawn("p", parent_pid=INIT_PID)
        await disp.dispatch("p", SyscallType.SPAWN_AGENT, pid="c")
        await disp.dispatch("root", SyscallType.TERMINATE_AGENT, pid="p")
        return disp

    disp = run(scenario())
    assert disp.scheduler.get("c") is not None
    assert disp.scheduler.get("c").parent_pid == INIT_PID


# --- 6. tree structure ----------------------------------------------------


def test_tree_structure_for_a_multi_level_hierarchy(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        disp.scheduler.spawn("a", parent_pid=INIT_PID)
        await disp.dispatch("a", SyscallType.SPAWN_AGENT, pid="b")
        await disp.dispatch("a", SyscallType.SPAWN_AGENT, pid="c")
        await disp.dispatch("b", SyscallType.SPAWN_AGENT, pid="d")
        return disp

    disp = run(scenario())
    tree = disp.scheduler.get_tree()

    assert tree["pid"] == INIT_PID
    kids = {node["pid"]: node for node in tree["children"]}
    assert "a" in kids

    a = kids["a"]
    a_children = {node["pid"]: node for node in a["children"]}
    assert set(a_children) == {"b", "c"}

    b_children = [node["pid"] for node in a_children["b"]["children"]]
    assert b_children == ["d"]
    assert a_children["c"]["children"] == []

    # every node carries its parent link
    assert a["parent_pid"] == INIT_PID
    assert a_children["b"]["parent_pid"] == "a"
    assert a_children["b"]["children"][0]["parent_pid"] == "b"


def test_parentless_processes_appear_under_init(tmp_path):
    """init is the ancestor of everything: a process created without a parent
    (as the scheduler demo and Gantt runs do) still shows up in the tree."""
    disp = make_dispatcher(tmp_path)
    disp.scheduler.add_process(Process(pid="legacy", arrival_time=0, estimated_burst=5))
    tree = disp.scheduler.get_tree()
    assert "legacy" in [node["pid"] for node in tree["children"]]


def test_zombie_is_visible_in_the_process_table(tmp_path):
    async def scenario():
        disp = make_dispatcher(tmp_path)
        disp.scheduler.spawn("parent", parent_pid=INIT_PID)
        await disp.dispatch("parent", SyscallType.SPAWN_AGENT, pid="kid")
        await disp.dispatch("kid", SyscallType.TERMINATE_AGENT, pid="kid", exit_status=3)
        return disp

    disp = run(scenario())
    table = {p.pid: p for p in disp.scheduler.queue}
    assert table["kid"].state == ZOMBIE
    assert table["kid"].exit_status == 3
    # and in the tree
    tree = disp.scheduler.get_tree()
    parent_node = next(n for n in tree["children"] if n["pid"] == "parent")
    kid_node = parent_node["children"][0]
    assert kid_node["state"] == ZOMBIE
    assert kid_node["exit_status"] == 3
