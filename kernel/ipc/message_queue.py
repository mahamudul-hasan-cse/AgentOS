"""Inter-process communication for agents.

Two primitives, mirroring classic OS IPC:

- `MessageQueue` — per-agent async inboxes (direct message passing), one
  `asyncio.Queue` per agent.
- `Blackboard` — a shared key-value store guarded by an async lock, for
  scratchpad-style collaboration where one agent posts findings that another
  reads (e.g. a ResearcherAgent writes, a WriterAgent reads).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Message:
    from_agent: str
    to_agent: str
    content: Any
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class MessageQueue:
    """Per-agent async inboxes. Inboxes are created lazily on first use."""

    def __init__(self) -> None:
        self._inboxes: Dict[str, asyncio.Queue] = {}

    def _inbox(self, agent_id: str) -> asyncio.Queue:
        queue = self._inboxes.get(agent_id)
        if queue is None:
            queue = asyncio.Queue()
            self._inboxes[agent_id] = queue
        return queue

    async def send(self, to_agent: str, from_agent: str, content: Any) -> Message:
        message = Message(from_agent=from_agent, to_agent=to_agent, content=content)
        await self._inbox(to_agent).put(message)
        return message

    async def receive(self, agent_id: str, timeout: Optional[float] = 0.1) -> Optional[Message]:
        """Pull the next message for `agent_id`. Returns None if none arrives
        within `timeout` seconds (non-blocking-style behaviour for IPC_RECV).
        A timeout of None blocks until a message is available."""
        inbox = self._inbox(agent_id)
        if timeout is None:
            return await inbox.get()
        try:
            return await asyncio.wait_for(inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def pending(self, agent_id: str) -> int:
        queue = self._inboxes.get(agent_id)
        return queue.qsize() if queue is not None else 0


class Blackboard:
    """A shared key-value scratchpad guarded by an async lock."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def write(self, key: str, value: Any, agent_id: Optional[str] = None) -> None:
        async with self._lock:
            self._store[key] = value

    async def read(self, key: str) -> Optional[Any]:
        async with self._lock:
            return self._store.get(key)

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return dict(self._store)
