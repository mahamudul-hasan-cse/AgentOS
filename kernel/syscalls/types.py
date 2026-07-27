"""Syscall type/status enums and the Syscall record.

Every interaction between an agent and the kernel is modelled as a syscall,
mirroring the trap/interrupt mechanism of a real OS: agents never touch the
driver, memory, or IPC subsystems directly — they issue a syscall and the
dispatcher traps it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SyscallType(str, Enum):
    LLM_CALL = "LLM_CALL"
    MEM_READ = "MEM_READ"
    MEM_WRITE = "MEM_WRITE"
    TOOL_CALL = "TOOL_CALL"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    SPAWN_AGENT = "SPAWN_AGENT"
    TERMINATE_AGENT = "TERMINATE_AGENT"
    WAIT = "WAIT"
    IPC_SEND = "IPC_SEND"
    IPC_RECV = "IPC_RECV"
    BLACKBOARD_READ = "BLACKBOARD_READ"
    BLACKBOARD_WRITE = "BLACKBOARD_WRITE"
    FILE_SEARCH = "FILE_SEARCH"
    SET_QUOTA = "SET_QUOTA"


class SyscallStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    NOT_IMPLEMENTED = "not_implemented"
    PERMISSION_DENIED = "permission_denied"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass
class Syscall:
    syscall_id: str
    agent_id: str
    type: SyscallType
    args: Dict[str, Any]
    timestamp: float
    result: Any = None
    status: SyscallStatus = SyscallStatus.PENDING
    latency_ms: Optional[float] = None

    @classmethod
    def create(
        cls, agent_id: str, syscall_type: SyscallType, args: Dict[str, Any]
    ) -> "Syscall":
        return cls(
            syscall_id=str(uuid.uuid4()),
            agent_id=agent_id,
            type=syscall_type,
            args=args,
            timestamp=time.time(),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "syscall_id": self.syscall_id,
            "agent_id": self.agent_id,
            "type": self.type.value,
            "args": self.args,
            "timestamp": self.timestamp,
            "result": self.result,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
        }
