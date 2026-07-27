"""Two-level access control enforced inside the syscall dispatcher.

Mirrors kernel-mode vs. user-mode in a real OS:

- KERNEL-level agents may issue any syscall, including spawning other agents
  and reading/writing another agent's memory.
- USER-level agents are sandboxed: they may only issue the whitelisted syscalls
  below, and MEM_READ/MEM_WRITE only against their *own* memory.

Agents default to USER unless explicitly registered as KERNEL.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from kernel.syscalls.types import SyscallType


class AgentPrivilege(str, Enum):
    KERNEL = "kernel"
    USER = "user"


class AccessDenied(Exception):
    """Raised by AccessControl.enforce when a syscall violates the agent's
    privilege level. The dispatcher traps this into a PERMISSION_DENIED status."""


# Syscalls a USER-level agent is permitted to issue at all. MEM_READ/MEM_WRITE
# are additionally restricted to the agent's own memory (see enforce()).
USER_ALLOWED_SYSCALLS = frozenset(
    {
        SyscallType.LLM_CALL,
        SyscallType.MEM_READ,
        SyscallType.MEM_WRITE,
        SyscallType.IPC_SEND,
        SyscallType.IPC_RECV,
        SyscallType.BLACKBOARD_READ,
        SyscallType.BLACKBOARD_WRITE,
        SyscallType.FILE_READ,
        SyscallType.FILE_WRITE,
        SyscallType.FILE_SEARCH,
        # a USER agent may terminate its OWN process; terminating another
        # agent's process is additionally restricted to KERNEL (see enforce()).
        SyscallType.TERMINATE_AGENT,
        # A USER agent may fork a child, but only at its own privilege level:
        # privilege ESCALATION (a USER requesting a KERNEL child) is rejected by
        # the SPAWN_AGENT handler, which is the only place the requested level is
        # known. Spawning is how any agent delegates work, so forbidding it
        # outright would make the hierarchy a kernel-only feature.
        SyscallType.SPAWN_AGENT,
        # wait(): reaping is inherently scoped to the caller's own children, so
        # no extra check is needed here.
        SyscallType.WAIT,
    }
)


class AgentRegistry:
    """Maps agent_id -> privilege level. Unregistered agents are USER."""

    def __init__(self) -> None:
        self._privileges: Dict[str, AgentPrivilege] = {}

    def register(self, agent_id: str, privilege: AgentPrivilege = AgentPrivilege.USER) -> None:
        self._privileges[agent_id] = privilege

    def privilege(self, agent_id: str) -> AgentPrivilege:
        return self._privileges.get(agent_id, AgentPrivilege.USER)

    def is_kernel(self, agent_id: str) -> bool:
        return self.privilege(agent_id) == AgentPrivilege.KERNEL


class AccessControl:
    def __init__(self, registry: Optional[AgentRegistry] = None) -> None:
        self.registry = registry if registry is not None else AgentRegistry()

    def enforce(
        self,
        agent_id: str,
        syscall_type: SyscallType,
        target_agent_id: Optional[str] = None,
    ) -> None:
        """Raise AccessDenied if `agent_id` may not issue `syscall_type`.

        `target_agent_id` is the memory owner for MEM_READ/MEM_WRITE (or the
        process being killed for TERMINATE_AGENT); when it differs from the
        caller the operation crosses into another agent and requires KERNEL
        privilege.
        """
        if self.registry.is_kernel(agent_id):
            return  # kernel mode: unrestricted

        if syscall_type not in USER_ALLOWED_SYSCALLS:
            raise AccessDenied(
                f"USER-level agent '{agent_id}' is not permitted to issue "
                f"{syscall_type.value} (requires KERNEL privilege)"
            )

        if syscall_type in (SyscallType.MEM_READ, SyscallType.MEM_WRITE):
            if target_agent_id is not None and target_agent_id != agent_id:
                raise AccessDenied(
                    f"USER-level agent '{agent_id}' may not access the memory of "
                    f"'{target_agent_id}' (requires KERNEL privilege)"
                )

        if syscall_type == SyscallType.TERMINATE_AGENT:
            # for termination the "target" is the process/agent id being killed.
            if target_agent_id is not None and target_agent_id != agent_id:
                raise AccessDenied(
                    f"USER-level agent '{agent_id}' may not terminate process "
                    f"'{target_agent_id}' (requires KERNEL privilege)"
                )
