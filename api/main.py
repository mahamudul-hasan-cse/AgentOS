from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents import run_collaboration
from kernel.scheduler import DEFAULT_MLFQ_QUANTUMS, Process, Scheduler, UnknownAlgorithmError
from kernel.syscalls import Syscall, SyscallDispatcher, SyscallStatus, SyscallType

app = FastAPI(title="AgentOS-Lite")

# Single choke point for all agent-kernel interaction. The dispatcher owns the
# PageManager (memory subsystem) and routes to the driver layer for LLM calls.
dispatcher = SyscallDispatcher()
page_manager = dispatcher.page_manager


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

    return GanttResponse(
        algorithm=request.algorithm,
        timeline=[TimeSliceOut(pid=s.pid, start=s.start, end=s.end) for s in timeline],
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
