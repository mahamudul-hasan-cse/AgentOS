from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kernel.drivers import DRIVER_REGISTRY, DriverConnectionError, DriverError, RateLimitError

app = FastAPI(title="AgentOS-Lite")


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
