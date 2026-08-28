import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from copy import deepcopy

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import PipelineRunner, run_collaboration
from agents.kernel_assistant import ASSISTANT_PID, KernelAssistant
from kernel.memory import HashingEmbedder, get_embedder, set_embedder
from kernel.scheduler import DEFAULT_MLFQ_QUANTUMS, Process, Scheduler, UnknownAlgorithmError
from kernel.syscalls import Syscall, SyscallDispatcher, SyscallStatus, SyscallType

STARTUP_MEMORY_DEMO_TIMEOUT = 10.0
STARTUP_ASSISTANT_INDEX_TIMEOUT = 30.0
STARTUP_DEADLOCK_MONITOR_TIMEOUT = 5.0
STARTUP_EMBEDDINGS_ENV = "AGENTOS_STARTUP_EMBEDDINGS"
LEGACY_STARTUP_EMBEDDINGS_ENV = "AIOS_STARTUP_EMBEDDINGS"
STARTUP_EMBEDDINGS_HASHING = "hashing"
STARTUP_EMBEDDINGS_ACTIVE = "active"


def _apply_startup_embedding_policy_before_dispatcher() -> dict:
    """Select the startup embedder before Chroma collections are opened."""
    policy = os.environ.get(STARTUP_EMBEDDINGS_ENV) or os.environ.get(
        LEGACY_STARTUP_EMBEDDINGS_ENV, STARTUP_EMBEDDINGS_HASHING
    )
    policy = policy.strip().lower()

    if policy in {STARTUP_EMBEDDINGS_ACTIVE, "ollama", "semantic"}:
        active = get_embedder()
        return {
            "status": "skipped",
            "backend": active.describe(),
            "policy": policy,
            "reason": f"{STARTUP_EMBEDDINGS_ENV} keeps the active backend",
        }

    if policy != STARTUP_EMBEDDINGS_HASHING:
        logging.getLogger("uvicorn.error").warning(
            "startup: unknown %s=%r; using %s",
            STARTUP_EMBEDDINGS_ENV,
            policy,
            STARTUP_EMBEDDINGS_HASHING,
        )
        policy = STARTUP_EMBEDDINGS_HASHING

    active = get_embedder()
    if not active.semantic:
        return {
            "status": "skipped",
            "backend": active.describe(),
            "policy": policy,
            "reason": "active backend is already offline/non-semantic",
        }

    fallback = HashingEmbedder()
    set_embedder(fallback)
    return {
        "status": "complete",
        "from_backend": active.describe(),
        "to_backend": fallback.describe(),
        "policy": policy,
        "reason": "optional startup indexing must not depend on Ollama responsiveness",
    }


_startup_embedding_policy_result = _apply_startup_embedding_policy_before_dispatcher()

# Single choke point for all agent-kernel interaction. The dispatcher owns the
# PageManager (memory subsystem) and routes to the driver layer for LLM calls.
dispatcher = SyscallDispatcher()
page_manager = dispatcher.page_manager

# The live process queue is `dispatcher.scheduler` (the single source of truth
# shared with /scheduler/terminate). Only the derived scheduling artifacts — the
# most recent Gantt timeline and the algorithm that produced it — are cached
# here for the dashboard; the processes themselves are read live from the
# scheduler's queue.
scheduler_state: dict = {"timeline": [], "algorithm": None}
pipeline_state: dict = {
    "status": "idle",
    "current_stage": None,
    "stages": [],
    "final_report": None,
    "tester": None,
    "events": [],
}
startup_state: dict = {
    "status": "starting",
    "started_at": None,
    "ready_at": None,
    "steps": {},
}
_startup_background_tasks: set[asyncio.Task] = set()

def _process_to_dict(p: Process) -> dict:
    return {
        "pid": p.pid,
        "state": p.state,
        "arrival_time": p.arrival_time,
        "estimated_burst": p.estimated_burst,
        "remaining_burst": p.remaining_burst,
        "priority": p.priority,
        "parent_pid": p.parent_pid,
        "exit_status": p.exit_status,
    }


def _replace_queue(processes: list[Process]) -> None:
    """Swap in a new process queue while keeping init, the ancestor of every
    process, alive."""
    dispatcher.scheduler.queue = list(processes)
    dispatcher.scheduler.ensure_init()


def _seed_scheduler_demo() -> None:
    """Bootstrap dashboard scheduler panels with a sample queue + Gantt timeline.

    This is **demo/bootstrap state only** — not part of the agent execution path.
    Processes are installed directly on ``dispatcher.scheduler.queue`` so the
    process table is non-empty on first load. Agent workloads (pipeline,
    assistant) register processes through ``SPAWN_AGENT`` instead.
    """
    sample = [
        Process(pid="P1", arrival_time=0, estimated_burst=5, priority=1),
        Process(pid="P2", arrival_time=1, estimated_burst=3, priority=2),
        Process(pid="P3", arrival_time=2, estimated_burst=8, priority=0),
        Process(pid="P4", arrival_time=3, estimated_burst=2, priority=1),
    ]
    timeline = Scheduler([Process(pid=p.pid, arrival_time=p.arrival_time,
                                  estimated_burst=p.estimated_burst, priority=p.priority)
                          for p in sample]).run("round_robin", quantum=3)

    # a plausible in-flight snapshot so all four state badges are visible
    sample[0].state, sample[0].remaining_burst = "terminated", 0
    sample[1].state, sample[1].remaining_burst = "running", 1
    sample[2].state = "ready"
    sample[3].state = "waiting"

    # register into the live scheduler so the seeded queue is real (and killable)
    _replace_queue(sample)
    scheduler_state["timeline"] = [{"pid": s.pid, "start": s.start, "end": s.end} for s in timeline]
    scheduler_state["algorithm"] = "round_robin"


async def _seed_memory_demo() -> None:
    """Write a few pages for a 'demo' agent through the dispatcher so the memory
    panel shows RAM-vs-swap and the syscall trace has entries on first load."""
    for i in range(7):
        await dispatcher.dispatch(
            "demo",
            SyscallType.MEM_WRITE,
            page_id=f"demo-page-{i}",
            content=f"Demo conversation chunk {i}: notes about OS scheduling and paging.",
            token_count=100,
        )


# A built-in KERNEL-privileged admin identity so the KERNEL-only endpoints
# (e.g. POST /quotas) are usable out of the box; non-admin callers are still
# rejected by access control.
ADMIN_AGENT_ID = "root"

# The in-kernel chat assistant. Constructed here but only becomes a live
# process once register() runs in the lifespan below.
assistant = KernelAssistant(dispatcher)


def _log_embedding_backend() -> None:
    """Announce which embedding backend is actually in use.

    Routed through uvicorn's own logger because uvicorn does not surface INFO
    records from arbitrary module loggers — without this the choice between
    real (Ollama) and approximate (hashing) embeddings would be invisible at
    startup, which is exactly the ambiguity we want to avoid."""
    log = logging.getLogger("uvicorn.error")
    if not log.handlers and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    try:
        log.info("embeddings: active backend -> %s", get_embedder().describe())
    except Exception as exc:  # noqa: BLE001 — never block startup on logging
        log.warning("embeddings: could not determine active backend: %s", exc)


def _startup_log() -> logging.Logger:
    return logging.getLogger("uvicorn.error")


def _mark_startup_step(name: str, state: str, **extra: object) -> None:
    step = startup_state["steps"].setdefault(name, {})
    step.update({"status": state, "updated_at": time.time(), **extra})


async def _run_startup_step(name: str, awaitable, timeout: float):
    """Run an optional startup step with a hard budget.

    Startup steps must never make the API unreachable indefinitely. Failures and
    timeouts are recorded in /health and logged as warnings; callers decide
    whether the step should be awaited before serving or launched in background.
    """
    _mark_startup_step(name, "running", timeout_seconds=timeout)
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError:
        elapsed = round(time.perf_counter() - started, 2)
        _mark_startup_step(name, "timeout", elapsed_seconds=elapsed)
        _startup_log().warning(
            "startup: %s timed out after %.1fs; continuing with degraded/incomplete state",
            name,
            timeout,
        )
        return None
    except Exception as exc:  # noqa: BLE001 - optional startup work must not block boot
        elapsed = round(time.perf_counter() - started, 2)
        _mark_startup_step(
            name,
            "error",
            elapsed_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )
        _startup_log().warning("startup: %s failed; continuing: %s", name, exc)
        return None

    elapsed = round(time.perf_counter() - started, 2)
    if elapsed > timeout:
        _mark_startup_step(name, "over_budget", elapsed_seconds=elapsed)
        _startup_log().warning(
            "startup: %s completed in %.2fs, exceeding its %.1fs budget",
            name,
            elapsed,
            timeout,
        )
    else:
        _mark_startup_step(name, "complete", elapsed_seconds=elapsed)
    return result


def _track_startup_task(task: asyncio.Task) -> None:
    _startup_background_tasks.add(task)
    task.add_done_callback(_startup_background_tasks.discard)


def _configure_startup_embeddings() -> None:
    """Report the embedding policy selected before dispatcher construction."""
    result = dict(_startup_embedding_policy_result)
    status = str(result.pop("status", "unknown"))
    _mark_startup_step("embedding_backend_fallback", status, **result)
    if status != "complete":
        return

    _mark_startup_step(
        "embedding_backend_fallback",
        "complete",
        **result,
    )
    _startup_log().warning(
        "startup: switched optional boot indexing from %s to %s: %s",
        result.get("from_backend"),
        result.get("to_backend"),
        f"{STARTUP_EMBEDDINGS_ENV}={result.get('policy')}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    from kernel.access_control import AgentPrivilege

    memory_seed_task = None
    assistant_index_task = None
    startup_state["status"] = "starting"
    startup_state["started_at"] = time.time()
    startup_state["ready_at"] = None
    startup_state["steps"] = {}

    _log_embedding_backend()
    _configure_startup_embeddings()
    dispatcher.acl.registry.register(ADMIN_AGENT_ID, AgentPrivilege.KERNEL)
    _seed_scheduler_demo()
    _mark_startup_step(
        "memory_demo_seed",
        "queued",
        timeout_seconds=STARTUP_MEMORY_DEMO_TIMEOUT,
    )

    async def seed_memory_demo() -> None:
        await _run_startup_step(
            "memory_demo_seed", _seed_memory_demo(), STARTUP_MEMORY_DEMO_TIMEOUT
        )

    memory_seed_task = seed_memory_demo
    # Register the assistant as a real process and index the repo docs it
    # answers project questions from. Best-effort: a docs-indexing failure
    # must not stop the kernel booting.
    try:
        await assistant.register()
        _mark_startup_step("assistant_register", "complete")
        _mark_startup_step(
            "assistant_doc_index",
            "queued",
            timeout_seconds=STARTUP_ASSISTANT_INDEX_TIMEOUT,
        )
        log = logging.getLogger("uvicorn.error")
        log.info("assistant: registered as process '%s' (USER)", ASSISTANT_PID)

        async def index_assistant_docs() -> None:
            indexed = await _run_startup_step(
                "assistant_doc_index",
                assistant.index_documentation(),
                STARTUP_ASSISTANT_INDEX_TIMEOUT,
            )
            if indexed is not None:
                try:
                    indexed_total = len(dispatcher.filesystem.list_files(ASSISTANT_PID))
                except Exception:  # noqa: BLE001 - health metadata is best-effort
                    indexed_total = indexed.get("indexed", 0)
                _mark_startup_step(
                    "assistant_doc_index",
                    "complete",
                    indexed_documents=indexed_total,
                    indexed_this_run=indexed.get("indexed", 0),
                    missing=indexed.get("missing", []),
                )
                _startup_log().info(
                    "assistant: indexed %d doc chunks this run (%d total)",
                    indexed.get("indexed", 0),
                    indexed_total,
                )
            else:
                current = startup_state["steps"].get("assistant_doc_index", {})
                if current.get("status") not in ("timeout", "error"):
                    _mark_startup_step("assistant_doc_index", "partial")

        assistant_index_task = index_assistant_docs
    except Exception as exc:  # noqa: BLE001
        _mark_startup_step(
            "assistant_register",
            "error",
            error=f"{type(exc).__name__}: {exc}",
        )
        logging.getLogger("uvicorn.error").warning(
            "assistant: startup registration/indexing failed: %s", exc
        )
    # start the deadlock monitor iff avoidance is off (see _sync_deadlock_monitor)
    await _run_startup_step(
        "deadlock_monitor_sync",
        _sync_deadlock_monitor(),
        STARTUP_DEADLOCK_MONITOR_TIMEOUT,
    )
    startup_state["status"] = "serving"
    startup_state["ready_at"] = time.time()
    if memory_seed_task is not None:
        _track_startup_task(asyncio.create_task(memory_seed_task()))
    if assistant_index_task is not None:
        _track_startup_task(asyncio.create_task(assistant_index_task()))
    try:
        yield
    finally:
        for task in list(_startup_background_tasks):
            task.cancel()
        if _startup_background_tasks:
            await asyncio.gather(*_startup_background_tasks, return_exceptions=True)
        # always cancel the background task, even if startup raised
        await dispatcher.deadlock_detector.stop()


app = FastAPI(title="AgentOS", lifespan=lifespan)
# This is a local development kernel for a course project, not an
# internet-facing service, so a permissive policy across localhost ports is
# appropriate: the dashboard routinely moves to :3001+ when :3000 is taken, and
# pinning a single port turns that into a confusing "Failed to fetch". It stays
# LOCALHOST-ONLY rather than a wildcard — no remote origin is ever allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _raise_for_syscall(syscall: Syscall) -> None:
    """Translate a failed syscall record into an appropriate HTTP error."""
    if syscall.status == SyscallStatus.SUCCESS:
        return

    detail = (syscall.result or {}).get("error", "syscall failed")
    error_type = (syscall.result or {}).get("error_type")

    if syscall.status == SyscallStatus.PERMISSION_DENIED:
        raise HTTPException(status_code=403, detail=detail)
    if syscall.status == SyscallStatus.QUOTA_EXCEEDED:
        raise HTTPException(status_code=429, detail=detail)
    if syscall.status == SyscallStatus.NOT_IMPLEMENTED:
        raise HTTPException(status_code=501, detail=detail)
    if error_type == "ValueError":
        raise HTTPException(status_code=400, detail=detail)
    if error_type in ("KeyError", "FileNotFoundError"):
        raise HTTPException(status_code=404, detail=detail)
    if error_type == "ResourceUnavailable":
        raise HTTPException(status_code=503, detail=detail)
    raise HTTPException(status_code=502, detail=detail)


class GenerateRequest(BaseModel):
    prompt: str
    driver: str = "groq"
    model: str | None = None
    agent_id: str = "user"


class GenerateResponse(BaseModel):
    driver_used: str
    text: str


@app.get("/health")
def health() -> dict:
    """Liveness probe, used by the compose healthcheck.

    Also reports the active embedding backend, because "is it up" and "is it
    running in the mode I expect" are different questions — this is how you tell
    Ollama-only mode from a silent fallback to hashing embeddings.
    """
    embedder = get_embedder()
    return {
        "status": "ok",
        "embedding_backend": embedder.describe(),
        "semantic_embeddings": embedder.semantic,
        "startup": deepcopy(startup_state),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    syscall = await dispatcher.dispatch(
        request.agent_id,
        SyscallType.LLM_CALL,
        prompt=request.prompt,
        driver=request.driver,
        model=request.model,
    )
    _raise_for_syscall(syscall)
    return GenerateResponse(
        driver_used=syscall.result["driver_used"], text=syscall.result["text"]
    )


class ProcessIn(BaseModel):
    pid: str
    arrival_time: float
    estimated_burst: float
    priority: int = 0


class TimeSliceOut(BaseModel):
    pid: str
    start: float
    end: float


class GanttRequest(BaseModel):
    processes: list[ProcessIn]
    algorithm: str = "fcfs"
    quantum: float = 4.0
    mlfq_quantums: list[float] | None = None


class GanttResponse(BaseModel):
    algorithm: str
    timeline: list[TimeSliceOut]


def _new_process(p: "ProcessIn") -> Process:
    return Process(
        pid=p.pid,
        arrival_time=p.arrival_time,
        estimated_burst=p.estimated_burst,
        priority=p.priority,
    )


@app.post("/scheduler/gantt", response_model=GanttResponse)
def scheduler_gantt(request: GanttRequest) -> GanttResponse:
    # Offline CPU-scheduling simulation on throwaway process copies. Does not
    # mutate dispatcher.scheduler.queue — the live process table is unchanged.
    sim = Scheduler([_new_process(p) for p in request.processes])
    try:
        timeline = sim.run(
            request.algorithm,
            quantum=request.quantum,
            mlfq_quantums=request.mlfq_quantums or DEFAULT_MLFQ_QUANTUMS,
        )
    except UnknownAlgorithmError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    timeline_out = [{"pid": s.pid, "start": s.start, "end": s.end} for s in timeline]
    scheduler_state["timeline"] = timeline_out
    scheduler_state["algorithm"] = request.algorithm

    return GanttResponse(
        algorithm=request.algorithm,
        timeline=[TimeSliceOut(**s) for s in timeline_out],
    )


class SchedulerStateResponse(BaseModel):
    algorithm: str | None
    processes: list[dict]
    timeline: list[dict]


@app.get("/scheduler/state", response_model=SchedulerStateResponse)
def scheduler_state_endpoint() -> SchedulerStateResponse:
    """Current process queue (read live from the dispatcher's scheduler) plus the
    most recent Gantt timeline, for the dashboard's process table + chart. A
    process terminated via /scheduler/terminate is dropped from the queue here."""
    return SchedulerStateResponse(
        algorithm=scheduler_state["algorithm"],
        processes=[_process_to_dict(p) for p in dispatcher.scheduler.queue],
        timeline=scheduler_state["timeline"],
    )


@app.post("/scheduler/terminate/{pid}")
async def scheduler_terminate(pid: str, agent_id: str | None = None) -> dict:
    """SIGKILL a process: cancel its in-flight LLM_CALL and mark it terminated.
    `agent_id` is the caller (defaults to the pid itself, i.e. self-termination);
    terminating another agent's process requires a KERNEL-privileged caller.

    Children are NOT killed — they are reparented to init. Use
    /scheduler/kill-tree/{pid} for cascading termination."""
    caller = agent_id or pid
    syscall = await dispatcher.dispatch(caller, SyscallType.TERMINATE_AGENT, pid=pid)
    _raise_for_syscall(syscall)
    return syscall.result


@app.post("/scheduler/kill-tree/{pid}")
async def scheduler_kill_tree(pid: str, agent_id: str | None = None) -> dict:
    """Terminate a process AND all of its descendants (opt-in cascade)."""
    caller = agent_id or pid
    syscall = await dispatcher.dispatch(
        caller, SyscallType.TERMINATE_AGENT, pid=pid, tree=True
    )
    _raise_for_syscall(syscall)
    return syscall.result


class SpawnRequest(BaseModel):
    agent_id: str
    pid: str | None = None
    privilege: str | None = None
    estimated_burst: float = 0.0
    priority: int = 0


@app.post("/scheduler/spawn")
async def scheduler_spawn(request: SpawnRequest) -> dict:
    """fork(): create a child process owned by `agent_id`. The child inherits the
    caller's privilege unless a lower one is requested; asking for a higher one
    is refused as privilege escalation."""
    syscall = await dispatcher.dispatch(
        request.agent_id,
        SyscallType.SPAWN_AGENT,
        pid=request.pid,
        privilege=request.privilege,
        estimated_burst=request.estimated_burst,
        priority=request.priority,
    )
    _raise_for_syscall(syscall)
    return syscall.result


@app.post("/scheduler/wait/{pid}")
async def scheduler_wait(pid: str, child_pid: str | None = None) -> dict:
    """wait(): `pid` reaps one of its zombie children, retrieving the exit status
    and removing the zombie from the process table. Omit `child_pid` to reap any."""
    syscall = await dispatcher.dispatch(pid, SyscallType.WAIT, pid=child_pid)
    _raise_for_syscall(syscall)
    return syscall.result


@app.get("/scheduler/tree")
def scheduler_tree() -> dict:
    """The full process hierarchy as nested JSON, rooted at init."""
    return dispatcher.scheduler.get_tree()


class MemoryPageOut(BaseModel):
    page_id: str
    content: str
    token_count: int
    last_accessed: float | None = None


class MemoryWriteRequest(BaseModel):
    agent_id: str
    page_id: str
    content: str
    token_count: int | None = None
    policy: str | None = None


class MemoryWriteResponse(BaseModel):
    page: MemoryPageOut
    evicted_page_ids: list[str]


@app.post("/memory/write", response_model=MemoryWriteResponse)
async def memory_write(request: MemoryWriteRequest) -> MemoryWriteResponse:
    syscall = await dispatcher.dispatch(
        request.agent_id,
        SyscallType.MEM_WRITE,
        page_id=request.page_id,
        content=request.content,
        token_count=request.token_count,
        policy=request.policy,
    )
    _raise_for_syscall(syscall)
    return MemoryWriteResponse(
        page=MemoryPageOut(**syscall.result["page"]),
        evicted_page_ids=syscall.result["evicted_page_ids"],
    )


class MemoryQueryRequest(BaseModel):
    agent_id: str
    query_text: str
    policy: str | None = None


class MemoryQueryResponse(BaseModel):
    page: MemoryPageOut
    page_fault: bool
    evicted_page_id: str | None = None


@app.post("/memory/query", response_model=MemoryQueryResponse)
async def memory_query(request: MemoryQueryRequest) -> MemoryQueryResponse:
    syscall = await dispatcher.dispatch(
        request.agent_id,
        SyscallType.MEM_READ,
        query_text=request.query_text,
        policy=request.policy,
    )
    _raise_for_syscall(syscall)
    return MemoryQueryResponse(
        page=MemoryPageOut(**syscall.result["page"]),
        page_fault=syscall.result["page_fault"],
        evicted_page_id=syscall.result["evicted_page_id"],
    )


class MemoryStateResponse(BaseModel):
    agent_id: str
    ram_budget_tokens: int
    ram_tokens_used: int
    ram_pages: list[dict]
    swapped_pages: list[dict]
    #: copy-on-write accounting for this agent (shared vs private pages, COW
    #: faults) and for the whole kernel (frames, tokens saved vs a naive fork)
    cow: dict = {}
    cow_global: dict = {}


@app.get("/memory/state/{agent_id}", response_model=MemoryStateResponse)
def memory_state(agent_id: str) -> MemoryStateResponse:
    return MemoryStateResponse(**page_manager.state(agent_id))


class SyscallLogResponse(BaseModel):
    syscalls: list[dict]


@app.get("/syscalls/log", response_model=SyscallLogResponse)
def syscalls_log(limit: int | None = None) -> SyscallLogResponse:
    """Return the syscall trace, most recent first, for the dashboard's live
    strace-style view. Pass ?limit=N to cap the number of entries."""
    return SyscallLogResponse(
        syscalls=[s.as_dict() for s in dispatcher.get_log(limit)]
    )


class CollaborateRequest(BaseModel):
    topic: str
    driver: str = "groq"


class CollaborateResponse(BaseModel):
    topic: str
    blackboard: dict
    final_output: str


@app.post("/agents/collaborate", response_model=CollaborateResponse)
async def agents_collaborate(request: CollaborateRequest) -> CollaborateResponse:
    """Run the ResearcherAgent then the WriterAgent over the shared blackboard,
    returning the intermediate blackboard content and the Writer's final output."""
    result = await run_collaboration(dispatcher, request.topic, driver=request.driver)
    return CollaborateResponse(
        topic=result["topic"],
        blackboard=result["blackboard"],
        final_output=result["final_output"],
    )


class PipelineRunRequest(BaseModel):
    topic: str
    driver: str = "groq"


@app.post("/pipeline/run")
async def pipeline_run(request: PipelineRunRequest) -> dict:
    """Run the flagship research -> code -> test -> report pipeline.

    The endpoint returns once the run completes, while /pipeline/status exposes
    the latest stage status for the shell and dashboard to poll.
    """
    global pipeline_state
    pipeline_state = {
        "status": "starting",
        "current_stage": "coordinator",
        "stages": [],
        "final_report": None,
        "tester": None,
        "events": [],
    }

    def publish(update: dict) -> None:
        global pipeline_state
        pipeline_state = update

    runner = PipelineRunner(dispatcher, on_update=publish)
    result = await runner.run(request.topic, driver=request.driver)
    pipeline_state = result
    return {
        "process_tree": result["process_tree"],
        "final_report": result["final_report"],
        "tester": result["tester"],
        "run_id": result["run_id"],
        "status": result["status"],
        "stages": result["stages"],
        "events": result["events"],
        "sandbox_review_note": result["sandbox_review_note"],
    }


@app.get("/pipeline/status")
def pipeline_status() -> dict:
    """Latest pipeline stage status, suitable for live polling."""
    return pipeline_state


class ResourceStateResponse(BaseModel):
    providers: dict


@app.get("/resources/state", response_model=ResourceStateResponse)
def resources_state() -> ResourceStateResponse:
    """Per-provider rate-limit pool state: total capacity, current allocation,
    availability, peak usage, and whether the pool is in a safe state."""
    return ResourceStateResponse(providers=dispatcher.resource_manager.state())


class TimelineResponse(BaseModel):
    snapshots: list[dict]


@app.get("/replay/timeline", response_model=TimelineResponse)
def replay_timeline() -> TimelineResponse:
    """All recorded snapshots (id, timestamp, label, triggering syscall), oldest
    first — the data behind the dashboard's time-travel scrubber."""
    if dispatcher.recorder is None:
        return TimelineResponse(snapshots=[])
    return TimelineResponse(snapshots=dispatcher.recorder.timeline())


class SnapshotResponse(BaseModel):
    snapshot_id: int
    timestamp: float
    syscall_id: str | None
    label: str
    processes: list[dict]
    memory: dict
    resources: dict
    quotas: dict


@app.get("/replay/snapshot/{snapshot_id}", response_model=SnapshotResponse)
def replay_snapshot(snapshot_id: int) -> SnapshotResponse:
    """Full kernel state at the moment this snapshot was taken."""
    if dispatcher.recorder is None:
        raise HTTPException(status_code=404, detail="state recording is disabled")
    snapshot = dispatcher.recorder.get(snapshot_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"snapshot {snapshot_id} not found (evicted from the ring buffer or never taken)",
        )
    return SnapshotResponse(**snapshot.as_dict())


@app.get("/replay/diff/{id_a}/{id_b}")
def replay_diff(id_a: int, id_b: int) -> dict:
    """What changed between two snapshots: processes added/removed/state-changed,
    pages moved between RAM and swap, and resource/quota deltas."""
    if dispatcher.recorder is None:
        raise HTTPException(status_code=404, detail="state recording is disabled")
    try:
        return dispatcher.recorder.diff(id_a, id_b)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from e


class ResourceModeRequest(BaseModel):
    avoidance_enabled: bool


async def _sync_deadlock_monitor() -> bool:
    """Run the background detector only when it can actually do something.

    The two strategies are alternatives: while avoidance is on, the Banker's
    Algorithm prevents cycles upstream and a periodic scan would burn a syscall
    every interval to find nothing forever. So the monitor runs exactly when
    avoidance is OFF. Returns whether it is now running."""
    if dispatcher.resource_manager.avoidance_enabled:
        await dispatcher.deadlock_detector.stop()
        return False
    dispatcher.start_deadlock_monitor()
    return True


@app.post("/resources/mode")
async def resources_mode(request: ResourceModeRequest) -> dict:
    """Toggle deadlock AVOIDANCE (Banker's Algorithm).

    On (default): unsafe grants are refused, so deadlock essentially cannot
    form — and the detector correctly finds nothing. Off: slots are granted
    greedily (still bounded by capacity), letting real circular waits develop so
    detection and recovery can be demonstrated. The two are alternative
    strategies, not layers.

    Flipping the mode also starts/stops the background detection task."""
    enabled = dispatcher.resource_manager.set_avoidance(request.avoidance_enabled)
    monitoring = await _sync_deadlock_monitor()
    return {
        "avoidance_enabled": enabled,
        "strategy": "avoidance (Banker's Algorithm)" if enabled else "detection + recovery",
        "monitoring": monitoring,
        "interval_seconds": dispatcher.deadlock_detector.interval,
    }


@app.get("/deadlock/graph")
def deadlock_graph() -> dict:
    """The current wait-for graph: an edge A -> B means A is blocked on a
    resource B holds."""
    return dispatcher.deadlock_detector.build_graph().as_dict()


@app.get("/deadlock/status")
def deadlock_status() -> dict:
    """Whether a cycle currently exists, and its members."""
    return dispatcher.deadlock_detector.status()


@app.post("/deadlock/detect")
async def deadlock_detect(recover: bool = False) -> dict:
    """Force an immediate detection run. With ?recover=true, also break any
    cycle found by terminating a victim."""
    if recover:
        return await dispatcher.run_deadlock_scan()
    syscall = await dispatcher.dispatch(
        dispatcher.KERNEL_AGENT, SyscallType.DEADLOCK_DETECT
    )
    _raise_for_syscall(syscall)
    return syscall.result


class QuotaUsageResponse(BaseModel):
    agent_id: str
    #: the ENFORCED count — private pages only. A copy-on-write shared page
    #: costs no extra memory, so it is reported but never charged; see DESIGN
    #: DECISION 2 in kernel/memory/page_manager.py.
    pages_used: int
    pages_private: int = 0
    pages_shared: int = 0
    #: RSS-like view: private + shared
    pages_total: int = 0
    quota_charged_on: str = "private pages (shared pages are free)"
    max_pages: int
    calls_in_window: int
    max_calls_per_minute: int
    window_seconds: float


@app.get("/quotas/{agent_id}", response_model=QuotaUsageResponse)
def get_quota(agent_id: str) -> QuotaUsageResponse:
    """Current usage vs. limit for an agent's memory-page and LLM call-rate quotas."""
    return QuotaUsageResponse(**dispatcher.quota_manager.usage(agent_id))


class QuotaUpdateRequest(BaseModel):
    max_pages: int | None = None
    max_calls_per_minute: int | None = None


@app.post("/quotas/{agent_id}", response_model=QuotaUsageResponse)
async def set_quota(
    agent_id: str, request: QuotaUpdateRequest, caller: str = ADMIN_AGENT_ID
) -> QuotaUsageResponse:
    """Adjust an agent's quota (KERNEL-only). `caller` is the acting agent and
    defaults to the built-in admin identity; a non-KERNEL caller is rejected."""
    syscall = await dispatcher.dispatch(
        caller,
        SyscallType.SET_QUOTA,
        target_agent_id=agent_id,
        max_pages=request.max_pages,
        max_calls_per_minute=request.max_calls_per_minute,
    )
    _raise_for_syscall(syscall)
    return QuotaUsageResponse(**syscall.result)


class FsWriteRequest(BaseModel):
    agent_id: str
    filename: str
    content: str
    target_agent_id: str | None = None


class FsWriteResponse(BaseModel):
    agent_id: str
    filename: str
    path: str
    created_at: float


@app.post("/fs/write", response_model=FsWriteResponse)
async def fs_write(request: FsWriteRequest) -> FsWriteResponse:
    syscall = await dispatcher.dispatch(
        request.agent_id,
        SyscallType.FILE_WRITE,
        filename=request.filename,
        content=request.content,
        target_agent_id=request.target_agent_id,
    )
    _raise_for_syscall(syscall)
    return FsWriteResponse(**syscall.result)


class FsReadResponse(BaseModel):
    filename: str
    content: str


@app.get("/fs/read", response_model=FsReadResponse)
async def fs_read(
    agent_id: str, filename: str, target_agent_id: str | None = None
) -> FsReadResponse:
    syscall = await dispatcher.dispatch(
        agent_id,
        SyscallType.FILE_READ,
        filename=filename,
        target_agent_id=target_agent_id,
    )
    _raise_for_syscall(syscall)
    return FsReadResponse(**syscall.result)


class FsSearchRequest(BaseModel):
    agent_id: str
    query: str
    top_k: int = 3
    target_agent_id: str | None = None


class FsSearchResponse(BaseModel):
    query: str
    results: list[dict]


@app.post("/fs/search", response_model=FsSearchResponse)
async def fs_search(request: FsSearchRequest) -> FsSearchResponse:
    """Natural-language file search. Genuinely semantic when the Ollama
    embedding backend is active; falls back to shared-vocabulary similarity if
    it is unavailable (see kernel/memory/embeddings.py)."""
    syscall = await dispatcher.dispatch(
        request.agent_id,
        SyscallType.FILE_SEARCH,
        query=request.query,
        top_k=request.top_k,
        target_agent_id=request.target_agent_id,
    )
    _raise_for_syscall(syscall)
    return FsSearchResponse(**syscall.result)


class FsListResponse(BaseModel):
    agent_id: str
    files: list[str]


@app.get("/fs/list/{agent_id}", response_model=FsListResponse)
def fs_list(agent_id: str) -> FsListResponse:
    """List the files an agent has written. (State read, like /memory/state —
    not routed as a syscall.)"""
    try:
        files = dispatcher.filesystem.list_files(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return FsListResponse(agent_id=agent_id, files=files)


# --- kernel assistant --------------------------------------------------------
# The assistant is an in-kernel agent, not a service this API wraps. These
# endpoints are a thin transport over its syscalls: the process lives in
# dispatcher.scheduler, and everything it reads goes through dispatcher.dispatch.

class AssistantChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    driver: str = "groq"


class AssistantSyscall(BaseModel):
    syscall_id: str
    type: str
    target: str | None = None
    status: str
    latency_ms: float | None = None
    error: str | None = None


class AssistantChatResponse(BaseModel):
    answer: str
    syscalls: list[AssistantSyscall]
    process_alive: bool
    process_state: str | None = None


class AssistantStatusResponse(BaseModel):
    pid: str
    alive: bool
    state: str | None = None
    parent_pid: str | None = None
    privilege: str | None = None
    indexed_documents: int


@app.get("/assistant/status", response_model=AssistantStatusResponse)
def assistant_status() -> AssistantStatusResponse:
    """Whether process 'assistant' is alive. The chat panel polls this so it can
    degrade gracefully the moment the process is killed from the shell."""
    process = dispatcher.scheduler.get(ASSISTANT_PID)
    try:
        indexed = len(dispatcher.filesystem.list_files(ASSISTANT_PID))
    except Exception:  # noqa: BLE001 — status must never fail on a listing error
        indexed = 0
    return AssistantStatusResponse(
        pid=ASSISTANT_PID,
        alive=assistant.is_alive(),
        state=process.state if process is not None else None,
        parent_pid=process.parent_pid if process is not None else None,
        privilege=dispatcher.acl.registry.privilege(ASSISTANT_PID).value,
        indexed_documents=indexed,
    )


@app.post("/assistant/chat", response_model=AssistantChatResponse)
async def assistant_chat(request: AssistantChatRequest) -> AssistantChatResponse:
    """Ask the in-kernel assistant a question.

    Returns the answer together with every syscall it issued to produce it, so
    the caller can verify the grounding rather than take the answer on trust.
    Deliberately not streamed: the syscall list is the point, and it is only
    complete once the turn is.

    A dead assistant process is NOT an HTTP error — it is a normal, reportable
    state that the panel renders, so killing it from the shell degrades the UI
    instead of breaking it.
    """
    return AssistantChatResponse(
        **await assistant.answer(
            request.message, history=request.history, driver=request.driver
        )
    )


@app.post("/assistant/restart", response_model=AssistantStatusResponse)
async def assistant_restart() -> AssistantStatusResponse:
    """Re-register the assistant process after it has been killed."""
    await assistant.register()
    return assistant_status()
