"""A real research -> code -> test -> report pipeline governed by the kernel.

The coordinator and every stage are scheduler processes. Stage hand-offs use
the blackboard and semantic filesystem, and every meaningful action goes through
the syscall dispatcher so the trace is inspectable end-to-end.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from kernel.access_control import AgentPrivilege
from kernel.scheduler import Process
from kernel.syscalls import Syscall, SyscallDispatcher, SyscallStatus, SyscallType

from .example_agents import KernelAgent

PIPELINE_AGENT_PREFIX = "pipeline"
ASSISTANT_PID = "assistant"
DEFAULT_STAGE_TIMEOUT_SECONDS = 90.0
DEFAULT_STAGE_QUOTAS = {
    "coordinator": {"max_pages": 6, "max_calls_per_minute": 2},
    "researcher": {"max_pages": 4, "max_calls_per_minute": 2},
    # The coder deliberately makes a second "refine" LLM call, so this quota
    # commonly produces QUOTA_EXCEEDED and demonstrates graceful degradation.
    "coder": {"max_pages": 4, "max_calls_per_minute": 1},
    "tester": {"max_pages": 3, "max_calls_per_minute": 1},
    "writer": {"max_pages": 4, "max_calls_per_minute": 1},
}


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass
class StageRecord:
    stage: str
    agent_id: str
    status: str = "pending"
    produced: Optional[str] = None
    file: Optional[str] = None
    driver_used: Optional[str] = None
    error: Optional[str] = None
    quota_events: List[Dict[str, Any]] = field(default_factory=list)
    resource_events: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "agent_id": self.agent_id,
            "status": self.status,
            "produced": self.produced,
            "file": self.file,
            "driver_used": self.driver_used,
            "error": self.error,
            "quota_events": self.quota_events,
            "resource_events": self.resource_events,
        }


class PipelineAgent(KernelAgent):
    def __init__(
        self,
        name: str,
        agent_id: str,
        dispatcher: SyscallDispatcher,
        run_id: str,
        status: StageRecord,
    ):
        super().__init__(name=name, agent_id=agent_id, dispatcher=dispatcher)
        self.run_id = run_id
        self.status = status

    async def _dispatch(self, syscall_type: SyscallType, **kwargs: Any) -> Syscall:
        syscall = await self.dispatcher.dispatch(self.agent_id, syscall_type, **kwargs)
        if syscall.status == SyscallStatus.QUOTA_EXCEEDED:
            self.status.quota_events.append(_event_from_syscall(syscall))
        elif syscall.status != SyscallStatus.SUCCESS:
            self.status.resource_events.append(_event_from_syscall(syscall))
        return syscall

    async def _llm(self, prompt: str, driver: str) -> Syscall:
        syscall = await self._dispatch(SyscallType.LLM_CALL, prompt=prompt, driver=driver)
        if syscall.status == SyscallStatus.SUCCESS:
            self.status.driver_used = syscall.result.get("driver_used")
        return syscall

    async def _blackboard_write(self, key: str, value: Any) -> None:
        syscall = await self._dispatch(SyscallType.BLACKBOARD_WRITE, key=key, value=value)
        _raise_stage(self.status.stage, syscall)

    async def _blackboard_read(self, key: str) -> Any:
        syscall = await self._dispatch(SyscallType.BLACKBOARD_READ, key=key)
        _raise_stage(self.status.stage, syscall)
        return syscall.result.get("value")

    async def _file_write(
        self, filename: str, content: str, target_agent_id: Optional[str] = None
    ) -> None:
        syscall = await self._dispatch(
            SyscallType.FILE_WRITE,
            filename=filename,
            content=content,
            target_agent_id=target_agent_id,
        )
        _raise_stage(self.status.stage, syscall)


class PipelineResearcherAgent(PipelineAgent):
    async def run(self, topic: str, driver: str) -> str:
        prompt = (
            f"Research this implementation task and return concise findings that a coder "
            f"can act on.\n\nTask:\n{topic}\n\nReturn 4-6 bullets."
        )
        llm = await self._llm(prompt, driver)
        _raise_stage(self.status.stage, llm)
        findings = llm.result["text"]
        key = _key(self.run_id, "research")
        filename = f"{self.run_id}_research.md"
        await self._blackboard_write(key, findings)
        await self._file_write(filename, findings)
        await self._file_write(filename, findings, target_agent_id=ASSISTANT_PID)
        self.status.produced = _preview(findings)
        self.status.file = filename
        return findings


class PipelineCoderAgent(PipelineAgent):
    async def run(self, topic: str, driver: str) -> Dict[str, str]:
        findings = await self._blackboard_read(_key(self.run_id, "research"))
        if not findings:
            raise PipelineStageError(self.status.stage, "no research findings found")

        prompt = (
            "Generate a complete, runnable Python 3 script for the task below. "
            "Keep it self-contained, deterministic, and safe for an offline subprocess. "
            "Return only the code or one fenced python code block.\n\n"
            f"Task:\n{topic}\n\nResearch findings:\n{findings}"
        )
        draft = await self._llm(prompt, driver)
        _raise_stage(self.status.stage, draft)
        code_text = draft.result["text"]

        refine = await self._llm(
            "Review the generated Python once for obvious syntax/runtime issues. "
            "If changes are needed, return the full corrected code; otherwise return the same code.\n\n"
            f"{code_text}",
            driver,
        )
        if refine.status == SyscallStatus.SUCCESS:
            code_text = refine.result["text"]
        elif refine.status != SyscallStatus.QUOTA_EXCEEDED:
            self.status.resource_events.append(_event_from_syscall(refine))

        code = extract_python_code(code_text)
        filename = f"{self.run_id}_generated.py"
        await self._file_write(filename, code)
        await self._file_write(filename, code, target_agent_id=ASSISTANT_PID)
        result = {"agent_id": self.agent_id, "filename": filename, "code": code}
        await self._blackboard_write(_key(self.run_id, "code"), result)
        self.status.produced = f"generated {len(code.splitlines())} line(s) of Python"
        self.status.file = filename
        return result


class PipelineTesterAgent(PipelineAgent):
    async def run(self) -> Dict[str, Any]:
        code_info = await self._blackboard_read(_key(self.run_id, "code"))
        if not code_info:
            raise PipelineStageError(self.status.stage, "no code artifact found")

        read = await self._dispatch(
            SyscallType.FILE_READ,
            filename=code_info["filename"],
            target_agent_id=code_info["agent_id"],
        )
        _raise_stage(self.status.stage, read)

        executed = await self._dispatch(
            SyscallType.TOOL_CALL,
            tool="python_sandbox",
            code=read.result["content"],
            timeout_seconds=3.0,
        )
        _raise_stage(self.status.stage, executed)
        result = {
            "passed": bool(executed.result.get("passed")),
            "exit_code": executed.result.get("exit_code"),
            "stdout": executed.result.get("stdout", ""),
            "stderr": executed.result.get("stderr", ""),
            "timeout": executed.result.get("timeout", False),
            "rejected": executed.result.get("rejected", False),
            "duration_ms": executed.result.get("duration_ms"),
            "sandbox": executed.result.get("sandbox", {}),
        }
        await self._blackboard_write(_key(self.run_id, "test"), result)
        self.status.produced = "pass" if result["passed"] else "fail"
        return result


class PipelineWriterAgent(PipelineAgent):
    async def run(self, topic: str, driver: str) -> str:
        research = await self._blackboard_read(_key(self.run_id, "research"))
        code = await self._blackboard_read(_key(self.run_id, "code"))
        test = await self._blackboard_read(_key(self.run_id, "test"))
        prompt = (
            "Write a final human-readable summary report for this multi-agent pipeline run. "
            "Be concise and include research, generated code artifact, test pass/fail, and any quota events.\n\n"
            f"Task: {topic}\n\nResearch:\n{research}\n\nCode artifact:\n{code}\n\nTest result:\n{test}"
        )
        llm = await self._llm(prompt, driver)
        if llm.status == SyscallStatus.SUCCESS:
            report = llm.result["text"]
        else:
            report = _fallback_report(topic, research, code, test, llm)
        filename = f"{self.run_id}_report.md"
        await self._file_write(filename, report)
        await self._file_write(filename, report, target_agent_id=ASSISTANT_PID)
        await self._blackboard_write(_key(self.run_id, "report"), report)
        self.status.produced = _preview(report)
        self.status.file = filename
        return report


class PipelineRunner:
    def __init__(
        self,
        dispatcher: SyscallDispatcher,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        stage_timeout_seconds: float = DEFAULT_STAGE_TIMEOUT_SECONDS,
    ) -> None:
        self.dispatcher = dispatcher
        self.on_update = on_update
        self.stage_timeout_seconds = stage_timeout_seconds

    async def run(
        self,
        topic: str,
        driver: str = "groq",
        quotas: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        run_id = "pipeline_" + uuid.uuid4().hex[:8]
        coordinator_id = f"{run_id}_coordinator"
        stage_order = ["researcher", "coder", "tester", "writer"]
        stage_ids = {stage: f"{run_id}_{stage}" for stage in stage_order}
        stages = {
            stage: StageRecord(stage=stage, agent_id=stage_ids[stage])
            for stage in stage_order
        }
        status = {
            "run_id": run_id,
            "topic": topic,
            "coordinator_id": coordinator_id,
            "status": "running",
            "current_stage": "coordinator",
            "stages": [stages[s].as_dict() for s in stage_order],
            "final_report": None,
            "tester": None,
            "events": [],
        }

        def publish(current_stage: Optional[str] = None) -> None:
            if current_stage is not None:
                status["current_stage"] = current_stage
            status["stages"] = [stages[s].as_dict() for s in stage_order]
            if self.on_update is not None:
                self.on_update(dict(status))

        publish()
        await self._spawn_processes(
            coordinator_id, stage_ids, quotas or DEFAULT_STAGE_QUOTAS, status
        )
        await self._coordinator_probe(coordinator_id, "start")

        agents = {
            "researcher": PipelineResearcherAgent(
                "PipelineResearcherAgent", stage_ids["researcher"], self.dispatcher, run_id, stages["researcher"]
            ),
            "coder": PipelineCoderAgent(
                "PipelineCoderAgent", stage_ids["coder"], self.dispatcher, run_id, stages["coder"]
            ),
            "tester": PipelineTesterAgent(
                "PipelineTesterAgent", stage_ids["tester"], self.dispatcher, run_id, stages["tester"]
            ),
            "writer": PipelineWriterAgent(
                "PipelineWriterAgent", stage_ids["writer"], self.dispatcher, run_id, stages["writer"]
            ),
        }

        report = ""
        tester_result: Optional[Dict[str, Any]] = None
        try:
            await self._run_stage("researcher", stages, publish, agents["researcher"].run(topic, driver))
            await self._run_stage("coder", stages, publish, agents["coder"].run(topic, driver))
            tester_result = await self._run_stage("tester", stages, publish, agents["tester"].run())
            status["tester"] = tester_result
            report = await self._run_stage("writer", stages, publish, agents["writer"].run(topic, driver))
            status["final_report"] = report
            status["status"] = "completed"
        except PipelineStageError as exc:
            stages[exc.stage].status = "failed"
            stages[exc.stage].error = str(exc)
            status["status"] = "failed"
            status["events"].append({"stage": exc.stage, "status": "failed", "message": str(exc)})
            if stages["writer"].status == "pending":
                try:
                    report = await self._run_stage(
                        "writer", stages, publish, agents["writer"].run(topic, driver)
                    )
                    status["final_report"] = report
                except Exception as writer_exc:  # noqa: BLE001
                    stages["writer"].status = "failed"
                    stages["writer"].error = str(writer_exc)
        finally:
            try:
                await self._coordinator_probe(coordinator_id, "finish")
            except PipelineStageError as exc:
                status["events"].append(
                    {"stage": "coordinator", "status": "failed", "message": str(exc)}
                )
            self._set_process_state(coordinator_id, "terminated")
            publish(None)

        result = {
            **status,
            "process_tree": self.dispatcher.scheduler.get_tree(),
            "syscalls": [s.as_dict() for s in self.dispatcher.get_log(limit=200)],
            "sandbox_review_note": (
                "Generated Python is executed by TOOL_CALL/python_sandbox using a restricted "
                "subprocess, timeout, scratch cwd, no shell, and AST deny-list. This is not "
                "a production-grade security sandbox."
            ),
        }
        if self.on_update is not None:
            self.on_update(result)
        return result

    async def _spawn_processes(
        self,
        coordinator_id: str,
        stage_ids: Dict[str, str],
        quotas: Dict[str, Dict[str, int]],
        status: Dict[str, Any],
    ) -> None:
        self.dispatcher.acl.registry.register(self.dispatcher.KERNEL_AGENT, AgentPrivilege.KERNEL)
        coord = await self.dispatcher.dispatch(
            self.dispatcher.KERNEL_AGENT,
            SyscallType.SPAWN_AGENT,
            pid=coordinator_id,
            privilege=AgentPrivilege.KERNEL.value,
            estimated_burst=4.0,
            priority=0,
        )
        _raise_stage("coordinator", coord)
        self._set_process_state(coordinator_id, "running")
        await self._set_quota(coordinator_id, coordinator_id, quotas["coordinator"])
        for stage, pid in stage_ids.items():
            spawned = await self.dispatcher.dispatch(
                coordinator_id,
                SyscallType.SPAWN_AGENT,
                pid=pid,
                privilege=AgentPrivilege.KERNEL.value,
                estimated_burst=2.0,
                priority=1,
            )
            _raise_stage(stage, spawned)
            await self._set_quota(coordinator_id, pid, quotas[stage])
        status["events"].append({"stage": "coordinator", "status": "spawned", "children": stage_ids})

    async def _set_quota(self, caller: str, target: str, quota: Dict[str, int]) -> None:
        syscall = await self.dispatcher.dispatch(
            caller,
            SyscallType.SET_QUOTA,
            target_agent_id=target,
            max_pages=quota.get("max_pages"),
            max_calls_per_minute=quota.get("max_calls_per_minute"),
        )
        _raise_stage("coordinator", syscall)

    async def _coordinator_probe(self, coordinator_id: str, label: str) -> None:
        deadlock = await self.dispatcher.dispatch(coordinator_id, SyscallType.DEADLOCK_DETECT)
        if deadlock.status != SyscallStatus.SUCCESS:
            raise PipelineStageError("coordinator", f"{label} probe failed: {deadlock.result}")

    async def _run_stage(
        self,
        stage: str,
        stages: Dict[str, StageRecord],
        publish: Callable[[Optional[str]], None],
        work: Any,
    ) -> Any:
        record = stages[stage]
        record.status = "running"
        self._set_process_state(record.agent_id, "running")
        publish(stage)
        try:
            result = await asyncio.wait_for(work, timeout=self.stage_timeout_seconds)
            record.status = "success"
            return result
        except asyncio.TimeoutError as exc:
            record.status = "failed"
            record.error = f"timeout after {self.stage_timeout_seconds:.1f}s"
            raise PipelineStageError(stage, record.error) from exc
        except PipelineStageError as exc:
            record.status = "failed"
            record.error = str(exc)
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            raise PipelineStageError(stage, record.error) from exc
        finally:
            process = self.dispatcher.scheduler.get(record.agent_id)
            if process is not None and process.state != "zombie":
                process.state = "terminated"
                process.remaining_burst = 0.0
            publish(stage)

    def _set_process_state(self, pid: str, state: str) -> None:
        process: Optional[Process] = self.dispatcher.scheduler.get(pid)
        if process is not None:
            process.state = state
            if state == "terminated":
                process.remaining_burst = 0.0


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    code = match.group(1) if match else text
    return code.strip() + "\n"


def _key(run_id: str, name: str) -> str:
    return f"{run_id}:{name}"


def _preview(text: Any, limit: int = 160) -> str:
    rendered = str(text).replace("\n", " ").strip()
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _event_from_syscall(syscall: Syscall) -> Dict[str, Any]:
    return {
        "syscall_id": syscall.syscall_id,
        "type": syscall.type.value,
        "status": syscall.status.value,
        "error": (syscall.result or {}).get("error") if isinstance(syscall.result, dict) else None,
    }


def _raise_stage(stage: str, syscall: Syscall) -> None:
    if syscall.status == SyscallStatus.SUCCESS:
        return
    error = (syscall.result or {}).get("error", syscall.status.value)
    raise PipelineStageError(stage, f"{syscall.type.value} returned {syscall.status.value}: {error}")


def _fallback_report(topic: str, research: Any, code: Any, test: Any, syscall: Syscall) -> str:
    error = (syscall.result or {}).get("error", syscall.status.value)
    return (
        f"# Pipeline report\n\n"
        f"Task: {topic}\n\n"
        f"Research summary: {_preview(research, 400)}\n\n"
        f"Code artifact: {_preview(code, 300)}\n\n"
        f"Tester result: {_preview(test, 300)}\n\n"
        f"Writer LLM_CALL could not complete ({syscall.status.value}: {error}), "
        f"so this deterministic fallback report was produced instead."
    )
