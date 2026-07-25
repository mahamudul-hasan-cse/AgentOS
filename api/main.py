from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kernel.drivers import DRIVER_REGISTRY, DriverConnectionError, DriverError, RateLimitError
from kernel.memory import PageManager
from kernel.scheduler import DEFAULT_MLFQ_QUANTUMS, Process, Scheduler, UnknownAlgorithmError

app = FastAPI(title="AgentOS-Lite")
page_manager = PageManager()


class GenerateRequest(BaseModel):
    prompt: str
    driver: str = "groq"
    model: str | None = None


class GenerateResponse(BaseModel):
    driver_used: str
    text: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    driver_cls = DRIVER_REGISTRY.get(request.driver)
    if driver_cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown driver '{request.driver}'. Available: {list(DRIVER_REGISTRY)}",
        )

    kwargs = {"model": request.model} if request.model else {}

    try:
        driver = driver_cls()
        text = await driver.generate(request.prompt, **kwargs)
        return GenerateResponse(driver_used=driver_cls.name, text=text)
    except (RateLimitError, DriverConnectionError) as primary_error:
        if request.driver == "ollama":
            raise HTTPException(status_code=502, detail=str(primary_error)) from primary_error
        try:
            fallback = DRIVER_REGISTRY["ollama"]()
            text = await fallback.generate(request.prompt, **kwargs)
            return GenerateResponse(driver_used="ollama", text=text)
        except DriverError as fallback_error:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Primary driver '{request.driver}' failed: {primary_error}. "
                    f"Fallback to ollama also failed: {fallback_error}"
                ),
            ) from fallback_error
    except DriverError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


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
def memory_write(request: MemoryWriteRequest) -> MemoryWriteResponse:
    try:
        page, evicted = page_manager.write_page(
            agent_id=request.agent_id,
            page_id=request.page_id,
            content=request.content,
            token_count=request.token_count,
            policy=request.policy,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return MemoryWriteResponse(
        page=MemoryPageOut(
            page_id=page.page_id,
            content=page.content,
            token_count=page.token_count,
            last_accessed=page.last_accessed,
        ),
        evicted_page_ids=evicted,
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
def memory_query(request: MemoryQueryRequest) -> MemoryQueryResponse:
    try:
        result = page_manager.read(request.agent_id, request.query_text, policy=request.policy)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return MemoryQueryResponse(
        page=MemoryPageOut(
            page_id=result.page.page_id,
            content=result.page.content,
            token_count=result.page.token_count,
            last_accessed=result.page.last_accessed,
        ),
        page_fault=result.page_fault,
        evicted_page_id=result.evicted_page_id,
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
