from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import run_collaboration
from kernel.scheduler import DEFAULT_MLFQ_QUANTUMS, Process, Scheduler, UnknownAlgorithmError
from kernel.syscalls import Syscall, SyscallDispatcher, SyscallStatus, SyscallType

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


def _process_to_dict(p: Process) -> dict:
    return {
        "pid": p.pid,
        "state": p.state,
        "arrival_time": p.arrival_time,
        "estimated_burst": p.estimated_burst,
        "remaining_burst": p.remaining_burst,
        "priority": p.priority,
    }


def _seed_scheduler_demo() -> None:
    """Seed a representative queue + Gantt timeline so the dashboard's scheduler
    panels are populated on first load. The displayed queue is a mid-run
    snapshot showing every state badge; the timeline is a real Round Robin run."""
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
    dispatcher.scheduler.queue = sample
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    from kernel.access_control import AgentPrivilege

    dispatcher.acl.registry.register(ADMIN_AGENT_ID, AgentPrivilege.KERNEL)
    _seed_scheduler_demo()
    try:
        await _seed_memory_demo()
    except Exception:  # seeding is best-effort; never block startup on it
        pass
    yield


app = FastAPI(title="AgentOS-Lite", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    processes = [_new_process(p) for p in request.processes]

    # Run the simulation on throwaway copies so the processes we register keep
    # their pre-run (schedulable) state — the scheduling algorithms mutate
    # processes to "terminated" as they complete, and we want the live queue to
    # represent pending processes that can still be inspected and terminated.
    sim = Scheduler([_new_process(p) for p in request.processes])
    try:
        timeline = sim.run(
            request.algorithm,
            quantum=request.quantum,
            mlfq_quantums=request.mlfq_quantums or DEFAULT_MLFQ_QUANTUMS,
        )
    except UnknownAlgorithmError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Register the submitted processes into the dispatcher's scheduler — the
    # single source of truth that /scheduler/state and /scheduler/terminate read.
    dispatcher.scheduler.queue = processes

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


class TerminateResponse(BaseModel):
    pid: str
    cancelled_llm_call: bool
    process_found: bool
    memory_retained: bool


@app.post("/scheduler/terminate/{pid}", response_model=TerminateResponse)
async def scheduler_terminate(pid: str, agent_id: str | None = None) -> TerminateResponse:
    """SIGKILL a process: cancel its in-flight LLM_CALL and mark it terminated.
    `agent_id` is the caller (defaults to the pid itself, i.e. self-termination);
    terminating another agent's process requires a KERNEL-privileged caller."""
    caller = agent_id or pid
    syscall = await dispatcher.dispatch(caller, SyscallType.TERMINATE_AGENT, pid=pid)
    _raise_for_syscall(syscall)
    return TerminateResponse(**syscall.result)


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


class ResourceStateResponse(BaseModel):
    providers: dict


@app.get("/resources/state", response_model=ResourceStateResponse)
def resources_state() -> ResourceStateResponse:
    """Per-provider rate-limit pool state: total capacity, current allocation,
    availability, peak usage, and whether the pool is in a safe state."""
    return ResourceStateResponse(providers=dispatcher.resource_manager.state())


class QuotaUsageResponse(BaseModel):
    agent_id: str
    pages_used: int
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
    """Natural-language file search (approximate — shared-vocabulary similarity;
    see kernel/filesystem/semantic_fs.py)."""
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
