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

# Persistent snapshot of the scheduler for the dashboard: the current process
# queue plus the most recent /scheduler/gantt timeline. Updated whenever a
# schedule is run; seeded at startup so the dashboard shows a live example.
scheduler_state: dict = {"processes": [], "timeline": [], "algorithm": None}


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

    scheduler_state["processes"] = [_process_to_dict(p) for p in sample]
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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    if syscall.status == SyscallStatus.NOT_IMPLEMENTED:
        raise HTTPException(status_code=501, detail=detail)
    if error_type == "ValueError":
        raise HTTPException(status_code=400, detail=detail)
    if error_type == "KeyError":
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


@app.post("/scheduler/gantt", response_model=GanttResponse)
def scheduler_gantt(request: GanttRequest) -> GanttResponse:
    processes = [
        Process(
            pid=p.pid,
            arrival_time=p.arrival_time,
            estimated_burst=p.estimated_burst,
            priority=p.priority,
        )
        for p in request.processes
    ]

    scheduler = Scheduler(processes)
    try:
        timeline = scheduler.run(
            request.algorithm,
            quantum=request.quantum,
            mlfq_quantums=request.mlfq_quantums or DEFAULT_MLFQ_QUANTUMS,
        )
    except UnknownAlgorithmError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    timeline_out = [{"pid": s.pid, "start": s.start, "end": s.end} for s in timeline]
    # persist for the dashboard: processes are now in their post-run states
    scheduler_state["processes"] = [_process_to_dict(p) for p in processes]
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
    """Current process queue (pid, state, arrival_time, remaining_burst, ...) and
    the most recent Gantt timeline, for the dashboard's process table + chart."""
    return SchedulerStateResponse(
        algorithm=scheduler_state["algorithm"],
        processes=scheduler_state["processes"],
        timeline=scheduler_state["timeline"],
    )


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
