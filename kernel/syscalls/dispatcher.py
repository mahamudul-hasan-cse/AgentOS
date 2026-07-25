"""The single choke point through which all agent-kernel interaction flows.

`SyscallDispatcher.dispatch()` traps a syscall, routes it to the right kernel
subsystem, records the outcome (status + latency) in an in-memory log, and
returns the completed `Syscall` record. Handler failures never propagate out
of `dispatch()` — they are caught and reflected in the record's status so the
dispatcher itself can't be crashed by a misbehaving syscall.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from kernel.drivers import (
    DRIVER_REGISTRY,
    DriverConnectionError,
    DriverError,
    RateLimitError,
)
from kernel.ipc import Blackboard, MessageQueue
from kernel.memory import PageManager

from .types import Syscall, SyscallStatus, SyscallType

# NOTE: kernel.access_control depends on kernel.syscalls.types, so importing it
# at module load here would create a circular import. It is imported lazily
# inside the methods below instead.


class SyscallDispatcher:
    def __init__(
        self,
        page_manager: Optional[PageManager] = None,
        driver_registry: Optional[Dict[str, Any]] = None,
        message_queue: Optional[MessageQueue] = None,
        blackboard: Optional[Blackboard] = None,
        access_control: Optional["AccessControl"] = None,
        resource_manager: Optional["ResourceManager"] = None,
        filesystem: Optional["SemanticFS"] = None,
    ):
        # kernel.access_control and kernel.filesystem both depend (transitively)
        # on kernel.syscalls.types, so they are imported lazily here to avoid a
        # circular import at module load.
        from kernel.access_control import AccessControl, ResourceManager
        from kernel.filesystem import SemanticFS

        self.page_manager = page_manager if page_manager is not None else PageManager()
        self.driver_registry = (
            driver_registry if driver_registry is not None else DRIVER_REGISTRY
        )
        self.message_queue = message_queue if message_queue is not None else MessageQueue()
        self.blackboard = blackboard if blackboard is not None else Blackboard()
        self.acl = access_control if access_control is not None else AccessControl()
        self.resource_manager = (
            resource_manager if resource_manager is not None else ResourceManager()
        )
        # the filesystem shares this dispatcher's AccessControl so per-agent file
        # scoping uses the same privilege registry as syscall enforcement.
        self.filesystem = (
            filesystem if filesystem is not None else SemanticFS(access_control=self.acl)
        )
        self.log: List[Syscall] = []
        self._handlers: Dict[SyscallType, Callable] = {
            SyscallType.LLM_CALL: self._handle_llm_call,
            SyscallType.MEM_READ: self._handle_mem_read,
            SyscallType.MEM_WRITE: self._handle_mem_write,
            SyscallType.IPC_SEND: self._handle_ipc_send,
            SyscallType.IPC_RECV: self._handle_ipc_recv,
            SyscallType.BLACKBOARD_WRITE: self._handle_blackboard_write,
            SyscallType.BLACKBOARD_READ: self._handle_blackboard_read,
            SyscallType.FILE_WRITE: self._handle_file_write,
            SyscallType.FILE_READ: self._handle_file_read,
            SyscallType.FILE_SEARCH: self._handle_file_search,
        }

    async def dispatch(self, agent_id: str, syscall_type: SyscallType, **args) -> Syscall:
        from kernel.access_control import AccessDenied

        syscall_type = SyscallType(syscall_type)
        syscall = Syscall.create(agent_id, syscall_type, dict(args))
        start = time.perf_counter()
        try:
            # ENOSYS-before-EPERM: mirror real OS trap semantics where an
            # unknown/unimplemented syscall number fails with ENOSYS regardless
            # of the caller's privilege. So we resolve the handler FIRST and
            # return NOT_IMPLEMENTED for unhandled syscalls, and only enforce
            # access control on syscalls that actually exist — a privilege check
            # on a non-existent syscall would leak nothing useful anyway.
            handler = self._handlers.get(syscall_type)
            if handler is None:
                raise NotImplementedError(
                    f"Syscall '{syscall_type.value}' is not implemented yet"
                )
            self.acl.enforce(
                agent_id, syscall_type, target_agent_id=args.get("target_agent_id")
            )
            syscall.result = await handler(agent_id, **args)
            syscall.status = SyscallStatus.SUCCESS
        except AccessDenied as e:
            syscall.status = SyscallStatus.PERMISSION_DENIED
            syscall.result = {"error": str(e), "error_type": "PermissionDenied"}
        except NotImplementedError as e:
            syscall.status = SyscallStatus.NOT_IMPLEMENTED
            syscall.result = {"error": str(e), "error_type": "NotImplementedError"}
        except Exception as e:  # noqa: BLE001 — the dispatcher must never crash
            syscall.status = SyscallStatus.ERROR
            syscall.result = {"error": str(e), "error_type": type(e).__name__}
        finally:
            syscall.latency_ms = (time.perf_counter() - start) * 1000.0
            self.log.append(syscall)
        return syscall

    def get_log(self, limit: Optional[int] = None) -> List[Syscall]:
        """Return the syscall trace, most recent first."""
        entries = list(reversed(self.log))
        if limit is not None:
            entries = entries[:limit]
        return entries

    # --- syscall handlers -------------------------------------------------

    async def _handle_llm_call(
        self,
        agent_id: str,
        prompt: str,
        driver: str = "groq",
        model: Optional[str] = None,
        max_claim: Optional[int] = None,
        **_,
    ) -> Dict[str, Any]:
        from kernel.access_control import ResourceUnavailable

        kwargs = {"model": model} if model else {}

        # Try the requested provider first, then fall back to ollama. For each
        # candidate we must acquire a rate-limit slot from the resource manager
        # (Banker's Algorithm) before calling the driver; a refusal (capacity
        # reached or unsafe state) is treated like any other provider failure
        # and rolls over to the next candidate.
        candidates = [driver]
        if driver != "ollama" and "ollama" in self.driver_registry:
            candidates.append("ollama")

        last_error: Optional[Exception] = None
        for provider in candidates:
            driver_cls = self.driver_registry.get(provider)
            if driver_cls is None:
                if provider == driver:
                    raise ValueError(
                        f"Unknown driver '{driver}'. Available: {list(self.driver_registry)}"
                    )
                continue

            granted = await self.resource_manager.request(
                agent_id, provider, units=1, max_claim=max_claim
            )
            if not granted:
                last_error = ResourceUnavailable(
                    f"provider '{provider}' is at capacity or would enter an unsafe state"
                )
                continue

            try:
                text = await driver_cls().generate(prompt, **kwargs)
                return {"driver_used": driver_cls.name, "text": text}
            except (RateLimitError, DriverConnectionError) as e:
                last_error = e
                continue
            finally:
                await self.resource_manager.release(agent_id, provider, units=1)

        raise last_error or DriverError("No provider was able to serve the LLM_CALL")

    async def _handle_mem_write(
        self,
        agent_id: str,
        page_id: str,
        content: str,
        token_count: Optional[int] = None,
        policy: Optional[str] = None,
        target_agent_id: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        owner = target_agent_id or agent_id
        page, evicted = self.page_manager.write_page(
            owner, page_id, content, token_count=token_count, policy=policy
        )
        return {
            "page": {
                "page_id": page.page_id,
                "content": page.content,
                "token_count": page.token_count,
                "last_accessed": page.last_accessed,
            },
            "evicted_page_ids": evicted,
        }

    async def _handle_mem_read(
        self,
        agent_id: str,
        query_text: str,
        policy: Optional[str] = None,
        target_agent_id: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        owner = target_agent_id or agent_id
        result = self.page_manager.read(owner, query_text, policy=policy)
        return {
            "page": {
                "page_id": result.page.page_id,
                "content": result.page.content,
                "token_count": result.page.token_count,
                "last_accessed": result.page.last_accessed,
            },
            "page_fault": result.page_fault,
            "evicted_page_id": result.evicted_page_id,
        }

    async def _handle_ipc_send(
        self, agent_id: str, to_agent: str, content: Any, **_
    ) -> Dict[str, Any]:
        message = await self.message_queue.send(
            to_agent=to_agent, from_agent=agent_id, content=content
        )
        return {"delivered_to": to_agent, "message": message.as_dict()}

    async def _handle_ipc_recv(
        self, agent_id: str, timeout: float = 0.1, **_
    ) -> Dict[str, Any]:
        message = await self.message_queue.receive(agent_id, timeout=timeout)
        return {"message": message.as_dict() if message is not None else None}

    async def _handle_blackboard_write(
        self, agent_id: str, key: str, value: Any, **_
    ) -> Dict[str, Any]:
        await self.blackboard.write(key, value, agent_id=agent_id)
        return {"key": key, "value": value}

    async def _handle_blackboard_read(self, agent_id: str, key: str, **_) -> Dict[str, Any]:
        value = await self.blackboard.read(key)
        return {"key": key, "value": value}

    async def _handle_file_write(
        self,
        agent_id: str,
        filename: str,
        content: str,
        target_agent_id: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        return self.filesystem.write_file(
            agent_id, filename, content, target_agent_id=target_agent_id
        )

    async def _handle_file_read(
        self, agent_id: str, filename: str, target_agent_id: Optional[str] = None, **_
    ) -> Dict[str, Any]:
        content = self.filesystem.read_file(
            agent_id, filename, target_agent_id=target_agent_id
        )
        return {"filename": filename, "content": content}

    async def _handle_file_search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 3,
        target_agent_id: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        results = self.filesystem.search_files(
            agent_id, query, top_k=top_k, target_agent_id=target_agent_id
        )
        return {"query": query, "results": results}
