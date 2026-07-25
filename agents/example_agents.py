"""Example collaborating agents built on the agno framework.

Per the AgentOS-Lite architecture, agents interact with the kernel *only*
through syscalls — they never touch the driver, memory, or IPC subsystems
directly. Here agno's `Agent` supplies each agent's framework identity
(name/id), while every LLM call and every blackboard read/write is issued as
a syscall through the `SyscallDispatcher`, so all of it shows up in the
syscall trace.

The demo: a `ResearcherAgent` researches a topic (LLM_CALL) and posts its
findings to the shared blackboard (BLACKBOARD_WRITE); a `WriterAgent` reads
those findings back (BLACKBOARD_READ) and composes a short summary (LLM_CALL).
"""

from __future__ import annotations

from typing import Any, Dict

from agno.agent import Agent

from kernel.syscalls import SyscallDispatcher, SyscallStatus, SyscallType

BLACKBOARD_FINDINGS_KEY = "research_findings"


class KernelAgent:
    """Base class: an agno-identified agent whose only route to the kernel is
    the syscall dispatcher."""

    def __init__(self, name: str, agent_id: str, dispatcher: SyscallDispatcher):
        self.agent = Agent(name=name, id=agent_id)
        self.agent_id = agent_id
        self.dispatcher = dispatcher

    async def _llm_call(self, prompt: str, driver: str = "groq") -> str:
        syscall = await self.dispatcher.dispatch(
            self.agent_id, SyscallType.LLM_CALL, prompt=prompt, driver=driver
        )
        if syscall.status != SyscallStatus.SUCCESS:
            raise RuntimeError(
                f"{self.agent_id} LLM_CALL failed: {syscall.result}"
            )
        return syscall.result["text"]


class ResearcherAgent(KernelAgent):
    def __init__(self, dispatcher: SyscallDispatcher):
        super().__init__(name="ResearcherAgent", agent_id="researcher", dispatcher=dispatcher)

    async def research(self, topic: str, driver: str = "groq") -> str:
        prompt = (
            f"Research the topic '{topic}'. Provide 3-4 concise, factual findings "
            f"as short bullet points."
        )
        findings = await self._llm_call(prompt, driver=driver)
        await self.dispatcher.dispatch(
            self.agent_id,
            SyscallType.BLACKBOARD_WRITE,
            key=BLACKBOARD_FINDINGS_KEY,
            value=findings,
        )
        return findings


class WriterAgent(KernelAgent):
    def __init__(self, dispatcher: SyscallDispatcher):
        super().__init__(name="WriterAgent", agent_id="writer", dispatcher=dispatcher)

    async def write_summary(self, topic: str, driver: str = "groq") -> str:
        read = await self.dispatcher.dispatch(
            self.agent_id, SyscallType.BLACKBOARD_READ, key=BLACKBOARD_FINDINGS_KEY
        )
        findings = read.result["value"]
        if not findings:
            raise RuntimeError(
                "WriterAgent found no research findings on the blackboard"
            )
        prompt = (
            f"Using ONLY the following research findings, write a short (2-3 sentence) "
            f"summary about '{topic}'.\n\nResearch findings:\n{findings}"
        )
        return await self._llm_call(prompt, driver=driver)


async def run_collaboration(
    dispatcher: SyscallDispatcher, topic: str, driver: str = "groq"
) -> Dict[str, Any]:
    """Run Researcher then Writer in sequence over the shared blackboard."""
    researcher = ResearcherAgent(dispatcher)
    writer = WriterAgent(dispatcher)

    findings = await researcher.research(topic, driver=driver)
    # snapshot the blackboard *after* the researcher posts — this is the
    # intermediate collaboration state the Writer will build on.
    blackboard_snapshot = await dispatcher.blackboard.snapshot()

    final_output = await writer.write_summary(topic, driver=driver)

    return {
        "topic": topic,
        "findings": findings,
        "blackboard": blackboard_snapshot,
        "final_output": final_output,
    }
